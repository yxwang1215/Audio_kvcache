import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sys
import librosa
import math
import traceback
from modeling_qwen2_5_omni_fastv import Qwen2_5OmniForConditionalGeneration
from processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# --- 配置区 ---
CHUNK_DURATION_SEC = 1200
OVERLAP_SEC = 30

# 剪枝配置 (Map阶段开启)
PRUNE_LAYER_IDX = int(os.environ.get("PRUNE_LAYER_IDX", 1))
PRUNE_RATIO = float(os.environ.get("PRUNE_RATIO", 0.0))
PRUNE_METHOD = os.environ.get("PRUNE_METHOD", "frame")

SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."


def apply_token_pruning_config(model):
    """应用剪枝配置"""
    if PRUNE_RATIO > 0:
        print(f"[Map] 应用剪枝: Layer={PRUNE_LAYER_IDX}, Ratio={PRUNE_RATIO}")
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            config = model.thinker.model.config
            config.sparse_attention_config = {
                'prune_ratio': PRUNE_RATIO, 'prune_method': PRUNE_METHOD
            }
            config.audio_layer_idx = PRUNE_LAYER_IDX
            config.audio_prune_ratio = PRUNE_RATIO
            config.random = (PRUNE_METHOD == "random")
            config.frame = (PRUNE_METHOD == "frame")
            config.yuxuan = (PRUNE_METHOD == "yuxuan")
            # 初始化防止报错
            if getattr(config, 'audio_token_num', None) is None: config.audio_token_num = 0
            if getattr(config, 'audio_token_start', None) is None: config.audio_token_start = 0


def load_audio_chunks_with_overlap(audio_path, target_sr=16000):
    try:
        audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"音频加载失败: {audio_path}, {e}")
        return []

    total_samples = len(audio_data)
    chunk_samples = CHUNK_DURATION_SEC * target_sr
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
        if len(chunk) > target_sr * 5:
            chunks.append(chunk)
        if end >= total_samples: break
        start += stride
    return chunks


def run_model_inference(model, processor, inputs):
    """Map 阶段推理 (必须带音频)"""
    model_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                model_inputs[k] = v.to(device=model.device, dtype=model.dtype)
            else:
                model_inputs[k] = v.to(device=model.device, dtype=torch.long)
        else:
            model_inputs[k] = v

    if "input_ids" in model_inputs and "position_ids" not in model_inputs:
        seq_length = model_inputs["input_ids"].shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=model.device)
        model_inputs["position_ids"] = position_ids.unsqueeze(0).expand(model_inputs["input_ids"].shape[0], -1)

    # 动态设置 Range (仅针对有音频的情况)
    if PRUNE_RATIO > 0 and hasattr(model, 'thinker'):
        try:
            input_ids = model_inputs["input_ids"]
            audio_token_id = 151646
            # 探测 ID
            if hasattr(model.thinker.config, 'audio_token_index'):
                audio_token_id = model.thinker.config.audio_token_index
            elif hasattr(model.thinker.config, 'audio_config'):
                audio_token_id = getattr(model.thinker.config.audio_config, 'audio_token_index', 151646)

            audio_token_id_tensor = torch.tensor(audio_token_id, device=input_ids.device, dtype=input_ids.dtype)
            cols = (input_ids == audio_token_id_tensor).nonzero(as_tuple=True)[1]

            if len(cols) > 0:
                model.thinker.model.config.audio_token_start = cols.min().item()
                model.thinker.model.config.audio_token_num = cols.max().item() - cols.min().item() + 1
            else:
                model.thinker.model.config.audio_token_start = 0
                model.thinker.model.config.audio_token_num = 0
        except Exception:
            model.thinker.model.config.audio_token_start = 0
            model.thinker.model.config.audio_token_num = 0

    with torch.no_grad():
        generate_ids = model.generate(**model_inputs, max_new_tokens=1024, use_audio_in_video=False)

    input_len = model_inputs["input_ids"].shape[1]
    generated_seqs = generate_ids[:, input_len:]
    return processor.batch_decode(generated_seqs, skip_special_tokens=True)[0].strip()


def run_map_phase(model_path, input_file, output_file):
    print(f"启动 Map 阶段 (提取关键信息). Prune={PRUNE_RATIO}")

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(model, 'disable_talker'): model.disable_talker()
        apply_token_pruning_config(model)
    except Exception as e:
        print(f"模型加载失败: {e}");
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, line in enumerate(tqdm(lines, desc="Map Phase")):
            try:
                data = json.loads(line)
                audio_path = data.get("audio")
                if not os.path.exists(audio_path): continue

                chunks = load_audio_chunks_with_overlap(audio_path)
                num_chunks = len(chunks)
                map_results = []

                for i, chunk in enumerate(chunks):
                    map_prompt = (
                        f"这是会议录音的第 {i + 1}/{num_chunks} 部分。"
                        "请详细罗列这段录音中的关键信息，包括：1. 讨论的主要议题；2. 参与者的核心观点；3. 达成的结论或待办事项。"
                        "请直接列出要点，不要遗漏。"
                    )
                    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                            {"role": "user",
                             "content": [{"type": "audio", "audio": chunk}, {"type": "text", "text": map_prompt}]}]

                    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    if isinstance(text, list): text = text[0]
                    auds, _, _ = process_mm_info(msgs, use_audio_in_video=False)
                    inputs = processor(text=[text], audio=auds, return_tensors="pt", padding=True)

                    res = run_model_inference(model, processor, inputs)
                    map_results.append(res)
                    torch.cuda.empty_cache()

                # 保存中间结果：包含原始指令、原始参考答案、以及 Map 提取出的片段列表
                inter_item = {
                    "audio": audio_path,
                    "instruction": data.get("instruction", ""),
                    "ref_answer": data.get("ref_answer", ""),
                    "map_fragments": map_results  # 这里存的是列表
                }
                f_out.write(json.dumps(inter_item, ensure_ascii=False) + '\n')
                f_out.flush()

            except Exception as e:
                print(f"Error processing {idx}: {e}");
                traceback.print_exc()


if __name__ == '__main__':
    MODEL_PATH = "/mnt/users/hepeize-20251016/Model/Qwen2.5_Omni"
    INPUT_DATA = "/mnt/users/hepeize-20251016/Dataset/AudioMarathon_long/alimeeting_test.jsonl"
    # 中间结果文件
    OUTPUT_MAP = "/mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/results/intermediate_map.jsonl"

    run_map_phase(MODEL_PATH, INPUT_DATA, OUTPUT_MAP)