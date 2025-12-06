import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sys
import librosa
import traceback
import gc
from modeling_qwen2_5_omni_fastv import Qwen2_5OmniForConditionalGeneration
from processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# --- 配置区 ---
# 切片长度：1200秒 (20分钟)
CHAT_CHUNK_SEC = 1200
# 重叠长度：30秒
OVERLAP_SEC = 30

# 剪枝配置
PRUNE_LAYER_IDX = int(os.environ.get("PRUNE_LAYER_IDX", 1))
PRUNE_RATIO = float(os.environ.get("PRUNE_RATIO", 0.0))
PRUNE_METHOD = os.environ.get("PRUNE_METHOD", "random")

SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

SECRETARY_INSTRUCTION = (
    "你是一名专业的会议秘书。请根据音频内容生成一份详细的会议纪要。\n"
    "要求包含以下具体信息（如果提及）：\n"
    "1. 关键事实：具体的时间、地点、数字。\n"
    "2. 核心决策：会议最终定下来的结论是什么。\n"
    "3. 待办事项：谁（Who）在什么时间（When）之前需要做什么（What）。\n"
    "请不要只描述讨论过程，要直接列出结论事实。"
)

def _safe_set_config_attr(model, attr_name, value):
    """递归设置属性，确保覆盖所有 Config 副本"""
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, 'config'):
            setattr(model.thinker.config, attr_name, value)
        if hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
            setattr(model.thinker.model.config, attr_name, value)


def apply_pruning_config(model):
    """配置剪枝参数"""
    if PRUNE_RATIO >= 0:
        print(f"[Config] Pruning: Method={PRUNE_METHOD}, Ratio={PRUNE_RATIO}")
        sparse_config = {'prune_ratio': PRUNE_RATIO, 'prune_method': PRUNE_METHOD}

        _safe_set_config_attr(model, 'sparse_attention_config', sparse_config)
        _safe_set_config_attr(model, 'audio_layer_idx', PRUNE_LAYER_IDX)
        _safe_set_config_attr(model, 'audio_prune_ratio', PRUNE_RATIO)
        _safe_set_config_attr(model, 'random', (PRUNE_METHOD == "random"))
        _safe_set_config_attr(model, 'frame', (PRUNE_METHOD == "frame"))
        _safe_set_config_attr(model, 'yuxuan', (PRUNE_METHOD == "yuxuan"))

        # 初始化防止 NoneType
        _safe_set_config_attr(model, 'audio_token_num', 0)
        _safe_set_config_attr(model, 'audio_token_start', 0)


def split_audio(audio_path, target_sr=16000):
    """将音频切分为多个片段 (带重叠)"""
    try:
        audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Read Error: {e}")
        return []

    total_samples = len(audio_data)
    chunk_samples = CHAT_CHUNK_SEC * target_sr
    overlap_samples = OVERLAP_SEC * target_sr
    stride = chunk_samples - overlap_samples

    chunks = []
    if total_samples <= chunk_samples:
        chunks.append(audio_data.astype(np.float32))
        return chunks

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio_data[start:end].astype(np.float32)
        if len(chunk) > target_sr:
            chunks.append(chunk)
        if end >= total_samples:
            break
        start += stride

    return chunks


def run_chat_inference(model_path, input_file, output_file):
    print(f"启动多轮对话推理. Chunk={CHAT_CHUNK_SEC}s, Overlap={OVERLAP_SEC}s")

    # 1. 加载模型
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(model, 'disable_talker'): model.disable_talker()
        apply_pruning_config(model)
    except Exception as e:
        print(f"模型加载失败: {e}");
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, line in enumerate(tqdm(lines, desc="Processing")):
            try:
                data = json.loads(line)
                audio_path = data.get("audio")
                if not os.path.exists(audio_path): continue

                chunks = split_audio(audio_path)
                if not chunks: continue

                # 初始化对话历史
                messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]

                final_response = ""

                print(f"\n--- 开始处理样本 {idx}: {os.path.basename(audio_path)} (共 {len(chunks)} 段) ---")

                # 逐段输入，逐段生成
                for i, chunk in enumerate(chunks):
                    is_last_chunk = (i == len(chunks) - 1)

                    # 1. 构造用户输入
                    user_content = [{"type": "audio", "audio": chunk}]

                    if is_last_chunk:
                        prompt_text = f"这是最后一段音频。{SECRETARY_INSTRUCTION}"
                        user_content.append({"type": "text", "text": prompt_text})
                    else:
                        # 中间段落，只告诉模型接收
                        prompt_text = f"这是会议录音的第 {i + 1}/{len(chunks)} 部分，请先接收并保持记忆，回复'收到'即可，不要总结。"
                        user_content.append({"type": "text", "text": prompt_text})

                    messages.append({"role": "user", "content": user_content})

                    # 2. 预处理输入
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if isinstance(text, list): text = text[0]
                    # process_mm_info 会处理 msg 中的 numpy audio
                    auds, _, _ = process_mm_info(messages, use_audio_in_video=False)

                    inputs = processor(text=[text], audio=auds, return_tensors="pt", padding=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    # 3. 设置音频 Token 范围 (防报错)
                    if hasattr(model, 'thinker'):
                        try:
                            input_ids = inputs["input_ids"]
                            audio_token_id = 151646
                            if hasattr(model.thinker.config, 'audio_token_index'):
                                audio_token_id = model.thinker.config.audio_token_index
                            elif hasattr(model.thinker.config, 'audio_config'):
                                audio_token_id = getattr(model.thinker.config.audio_config, 'audio_token_index', 151646)

                            audio_token_id_tensor = torch.tensor(audio_token_id, device=input_ids.device,
                                                                 dtype=input_ids.dtype)
                            cols = (input_ids == audio_token_id_tensor).nonzero(as_tuple=True)[1]

                            if len(cols) > 0:
                                s = cols.min().item()
                                n = cols.max().item() - s + 1
                                _safe_set_config_attr(model, 'audio_token_start', int(s))
                                _safe_set_config_attr(model, 'audio_token_num', int(n))
                            else:
                                _safe_set_config_attr(model, 'audio_token_start', 0)
                                _safe_set_config_attr(model, 'audio_token_num', 0)
                        except:
                            _safe_set_config_attr(model, 'audio_token_start', 0)
                            _safe_set_config_attr(model, 'audio_token_num', 0)

                    # 4. 生成回复
                    with torch.no_grad():
                        generate_ids = model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            use_audio_in_video=False,
                            repetition_penalty=1.1
                        )

                    input_len = inputs["input_ids"].shape[1]
                    generated_seqs = generate_ids[:, input_len:]
                    response_text = processor.batch_decode(generated_seqs, skip_special_tokens=True)[0].strip()

                    # 5. 打印并记录回复
                    print(f"  [Turn {i + 1}] Model: {response_text}")

                    # 将模型的回复加入历史，用于下一轮
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

                    if is_last_chunk:
                        final_response = response_text

                    # 显存清理
                    del inputs, generate_ids
                    torch.cuda.empty_cache()

                # --- 循环结束，保存最终结果 ---
                output_item = {
                    "audio": audio_path,
                    "instruction": SECRETARY_INSTRUCTION, # 保存实际使用的指令
                    "ref_answer": data.get("ref_answer"),
                    "model_response": final_response
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                f_out.flush()

                gc.collect()

            except Exception as e:
                print(f"\n[Sample {idx} Error]: {e}")
                traceback.print_exc()
                err_item = data.copy()
                err_item["model_response"] = f"Evaluation Failed: {str(e)}"
                f_out.write(json.dumps(err_item, ensure_ascii=False) + '\n')
                f_out.flush()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    MODEL_PATH = "/mnt/users/hepeize-20251016/Model/Qwen2.5_Omni"
    INPUT_DATA = "/mnt/users/hepeize-20251016/Dataset/AudioMarathon_long/alimeeting_test.jsonl"
    OUTPUT_FILE = "/mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/results/alimeeting_prediction.jsonl"

    run_chat_inference(MODEL_PATH, INPUT_DATA, OUTPUT_FILE)