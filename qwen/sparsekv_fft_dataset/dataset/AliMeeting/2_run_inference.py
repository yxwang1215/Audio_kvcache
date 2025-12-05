import os
import json
import torch
from tqdm import tqdm
import sys
import traceback
import gc
from modeling_qwen2_5_omni_fastv import Qwen2_5OmniForConditionalGeneration
from processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# 剪枝配置 (Env 优先)
PRUNE_LAYER_IDX = int(os.environ.get("PRUNE_LAYER_IDX", 1))
PRUNE_RATIO = float(os.environ.get("PRUNE_RATIO", 0.0))
PRUNE_METHOD = os.environ.get("PRUNE_METHOD", "random")


def apply_pruning_config(model):
    """配置剪枝参数"""
    print(f"[Config] Pruning: Method={PRUNE_METHOD}, Ratio={PRUNE_RATIO}, Layer={PRUNE_LAYER_IDX}")
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
        config = model.thinker.model.config
        # 核心字典配置
        config.sparse_attention_config = {
            'prune_ratio': PRUNE_RATIO,
            'prune_method': PRUNE_METHOD
        }
        # 辅助属性配置
        config.audio_layer_idx = PRUNE_LAYER_IDX
        config.audio_prune_ratio = PRUNE_RATIO
        config.random = (PRUNE_METHOD == "random")
        config.frame = (PRUNE_METHOD == "frame")
        config.yuxuan = (PRUNE_METHOD == "yuxuan")

        # 初始化位置参数，防止 NoneType 报错
        if getattr(config, 'audio_token_num', None) is None: config.audio_token_num = 0
        if getattr(config, 'audio_token_start', None) is None: config.audio_token_start = 0


def run_inference_native(model_path, input_file, output_file):
    print(f"启动推理")

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
        print(f"模型加载挂了: {e}");
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. 循环处理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, line in enumerate(tqdm(lines, desc="Inference")):
            try:
                data = json.loads(line)
                audio_path = data.get("audio")

                instruction = (
                        "你是一名专业的会议秘书。请根据音频内容生成一份详细的会议纪要。\n"
                        "要求包含以下具体信息（如果提及）：\n"
                        "1. 关键事实：具体的时间、地点、数字。\n"
                        "2. 核心决策：会议最终定下来的结论是什么。\n"
                        "3. 待办事项：谁（Who）在什么时间（When）之前需要做什么（What）。\n"
                        "请不要只描述讨论过程，要直接列出结论事实。"
                )

                if not os.path.exists(audio_path):
                    print(f"文件不存在: {audio_path}")
                    continue

                msgs = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_path},  # 直接传路径字符串
                        {"type": "text", "text": instruction}
                    ]}
                ]

                # 预处理：process_mm_info 会自动读取音频并返回 tensor 列表
                audios, _, _ = process_mm_info(msgs, use_audio_in_video=False)

                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                if isinstance(text, list): text = text[0]

                inputs = processor(
                    text=[text],
                    audio=audios,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # 剪枝所需的 Token 位置计算
                if PRUNE_RATIO >= 0 and hasattr(model, 'thinker'):
                    try:
                        input_ids = inputs["input_ids"]
                        audio_token_id = 151646
                        # 尝试获取真实 ID
                        if hasattr(model.thinker.config, 'audio_token_index'):
                            audio_token_id = model.thinker.config.audio_token_index
                        elif hasattr(model.thinker.config, 'audio_config'):
                            audio_token_id = getattr(model.thinker.config.audio_config, 'audio_token_index', 151646)

                        audio_token_id_tensor = torch.tensor(audio_token_id, device=input_ids.device,
                                                             dtype=input_ids.dtype)
                        cols = (input_ids == audio_token_id_tensor).nonzero(as_tuple=True)[1]

                        if len(cols) > 0:
                            model.thinker.model.config.audio_token_start = cols.min().item()
                            model.thinker.model.config.audio_token_num = cols.max().item() - cols.min().item() + 1
                    except:
                        pass

                        # 生成
                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        use_audio_in_video=False,
                        repetition_penalty=1.1
                    )

                # 解码
                input_len = inputs["input_ids"].shape[1]
                generated_seqs = generate_ids[:, input_len:]
                response = processor.batch_decode(generated_seqs, skip_special_tokens=True)[0].strip()

                # 保存
                output_item = {
                    "audio": audio_path,
                    "instruction": instruction,
                    "ref_answer": data.get("ref_answer", ""),
                    "model_response": response
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                f_out.flush()

                # 显存清理
                del inputs, audios, generate_ids
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n[Error Sample {idx}]: {e}")
                traceback.print_exc()
                err_item = data.copy()
                err_item["model_response"] = f"Evaluation Failed: {str(e)}"
                f_out.write(json.dumps(err_item, ensure_ascii=False) + '\n')
                f_out.flush()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    MODEL_PATH = "/mnt/users/hepeize-20251016/Model/Qwen2.5-Omni-7B"
    INPUT_DATA = "/mnt/users/hepeize-20251016/Dataset/AudioMarathon_long/alimeeting_test.jsonl"
    OUTPUT_FILE = "/mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/results/alimeeting_prediction.jsonl"

    run_inference_native(MODEL_PATH, INPUT_DATA, OUTPUT_FILE)