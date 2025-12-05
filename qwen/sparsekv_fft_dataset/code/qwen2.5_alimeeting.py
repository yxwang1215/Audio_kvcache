import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import math
import traceback
import gc
import re
import ast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import warnings

warnings.filterwarnings("ignore")

# Set up path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(current_dir)
sys.path.insert(0, parent_dir)

# Import model modules
USE_SPARSEKV = os.environ.get("USE_SPARSEKV", "false").lower() == "true"

if USE_SPARSEKV:
    from model.modeling_qwen2_5_omni_sparsekv import Qwen2_5OmniForConditionalGeneration
    print("Using sparsekv version of the model")
else:
    # Try importing from transformers or local file
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration
        print("Using standard version from transformers")
    except ImportError:
        # Fallback: try to import from dataset directory
        dataset_path = os.path.join(parent_dir, "dataset", "AliMeeting")
        if dataset_path not in sys.path:
            sys.path.insert(0, dataset_path)
        try:
            from modeling_qwen2_5_omni_fastv import Qwen2_5OmniForConditionalGeneration
            print("Using local modeling_qwen2_5_omni_fastv")
        except ImportError:
            raise ImportError("Cannot import Qwen2_5OmniForConditionalGeneration")

from model.processing_qwen2_5_omni import Qwen2_5OmniProcessor

# Try to import qwen_omni_utils
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    # Fallback: try to find it in common locations
    try:
        import importlib.util
        utils_paths = [
            os.path.join(parent_dir, "qwen_omni_utils.py"),
            "/mnt/users/wangyuxuan-20250915/Audio/qwen_omni_utils.py",
        ]
        for utils_path in utils_paths:
            if os.path.exists(utils_path):
                spec = importlib.util.spec_from_file_location("qwen_omni_utils", utils_path)
                qwen_omni_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(qwen_omni_utils)
                process_mm_info = qwen_omni_utils.process_mm_info
                break
        else:
            raise ImportError("Cannot find qwen_omni_utils")
    except Exception as e:
        print(f"Warning: Cannot import qwen_omni_utils: {e}")
        # Define a minimal fallback
        def process_mm_info(msgs, use_audio_in_video=False):
            audios = []
            for msg in msgs:
                if "content" in msg:
                    for item in msg["content"]:
                        if item.get("type") == "audio":
                            audio = item.get("audio")
                            if isinstance(audio, str):
                                # Load audio file
                                audio_data, sr = librosa.load(audio, sr=16000, mono=True)
                                audios.append(audio_data.astype(np.float32))
                            elif isinstance(audio, np.ndarray):
                                audios.append(audio.astype(np.float32))
            return audios, None, None

# ==================== 配置区 ====================
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# 剪枝配置
PRUNE_LAYER_IDX = int(os.environ.get("PRUNE_LAYER_IDX", 1))
PRUNE_RATIO = float(os.environ.get("PRUNE_RATIO", 0.0))
PRUNE_METHOD = os.environ.get("PRUNE_METHOD", "random")

# 音频切片配置
CHUNK_DURATION_SEC = int(os.environ.get("CHUNK_DURATION_SEC", 1200))  # 20分钟
OVERLAP_SEC = int(os.environ.get("OVERLAP_SEC", 30))  # 30秒重叠

# ==================== 工具函数 ====================

def parse_textgrid_full(file_path):
    """解析 TextGrid，提取所有 Tier 的内容，并按时间戳排序合并。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        intervals = []
        current_tier_name = ""

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == 'class = "IntervalTier"':
                while i < len(lines):
                    i += 1
                    name_line = lines[i].strip()
                    if name_line.startswith('name = "'):
                        current_tier_name = name_line.split('"')[1]
                        break

            if line.startswith('intervals ['):
                xmin = 0.0
                xmax = 0.0
                text = ""

                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('intervals ['):
                    sub_line = lines[j].strip()
                    if sub_line.startswith('xmin = '):
                        xmin = float(sub_line.split('=')[1].strip())
                    elif sub_line.startswith('xmax = '):
                        xmax = float(sub_line.split('=')[1].strip())
                    elif sub_line.startswith('text = "'):
                        text = sub_line[8:-1]
                        if text and text != "<NA>":
                            intervals.append({
                                'tier': current_tier_name,
                                'start': xmin,
                                'end': xmax,
                                'text': text
                            })
                    j += 1
            i += 1

        intervals.sort(key=lambda x: x['start'])
        transcript = [item['text'] for item in intervals]
        return " ".join(transcript)

    except Exception as e:
        print(f"TextGrid 解析失败: {file_path}, {e}")
        return ""


def get_audio_duration(file_path):
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception:
        return 0.0


def apply_token_pruning_config(model):
    """应用剪枝配置"""
    if PRUNE_RATIO > 0:
        print(f"[Config] 应用剪枝: Layer={PRUNE_LAYER_IDX}, Ratio={PRUNE_RATIO}, Method={PRUNE_METHOD}")
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
            if getattr(config, 'audio_token_num', None) is None: 
                config.audio_token_num = 0
            if getattr(config, 'audio_token_start', None) is None: 
                config.audio_token_start = 0


def _safe_set_config_attr(model, attr_name, value):
    """递归设置属性，确保覆盖所有 Config 副本"""
    if hasattr(model, 'thinker'):
        if hasattr(model.thinker, 'config'):
            setattr(model.thinker.config, attr_name, value)
        if hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
            setattr(model.thinker.model.config, attr_name, value)


def load_audio_chunks_with_overlap(audio_path, target_sr=16000):
    """将音频切分为多个片段 (带重叠)"""
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
        if end >= total_samples: 
            break
        start += stride
    return chunks


def set_audio_token_range(model, inputs):
    """设置音频 Token 范围"""
    if PRUNE_RATIO > 0 and hasattr(model, 'thinker'):
        try:
            input_ids = inputs["input_ids"]
            audio_token_id = 151646
            # 探测 ID
            if hasattr(model.thinker.config, 'audio_token_index'):
                audio_token_id = model.thinker.config.audio_token_index
            elif hasattr(model.thinker.config, 'audio_config'):
                audio_token_id = getattr(model.thinker.config.audio_config, 'audio_token_index', 151646)

            audio_token_id_tensor = torch.tensor(audio_token_id, device=input_ids.device, dtype=input_ids.dtype)
            cols = (input_ids == audio_token_id_tensor).nonzero(as_tuple=True)[1]

            if len(cols) > 0:
                start = cols.min().item()
                num = cols.max().item() - start + 1
                if hasattr(model.thinker, 'model') and hasattr(model.thinker.model, 'config'):
                    model.thinker.model.config.audio_token_start = start
                    model.thinker.model.config.audio_token_num = num
                _safe_set_config_attr(model, 'audio_token_start', start)
                _safe_set_config_attr(model, 'audio_token_num', num)
            else:
                _safe_set_config_attr(model, 'audio_token_start', 0)
                _safe_set_config_attr(model, 'audio_token_num', 0)
        except Exception:
            _safe_set_config_attr(model, 'audio_token_start', 0)
            _safe_set_config_attr(model, 'audio_token_num', 0)


# ==================== 数据集构建 ====================

def build_dataset(csv_path, output_path, root_dir):
    """构建 AliMeeting 数据集"""
    print(f"正在读取 CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    col_audio = 'Audio:FILE'
    col_text = 'Text:FILE'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = []
    count_far = 0
    count_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建数据"):
        rel_audio = str(row[col_audio]).strip()
        rel_text = str(row[col_text]).strip()

        # 只保留 far 音频
        if 'far' not in rel_audio.lower():
            count_skipped += 1
            continue

        abs_audio = os.path.join(root_dir, rel_audio)
        abs_text = os.path.join(root_dir, rel_text)

        if not os.path.exists(abs_audio):
            print(f"音频缺失: {abs_audio}")
            continue

        ref_text = ""
        if os.path.exists(abs_text):
            ref_text = parse_textgrid_full(abs_text)
        else:
            print(f"TextGrid 缺失: {abs_text}")

        item = {
            "meeting_id": os.path.splitext(os.path.basename(rel_audio))[0],
            "audio": abs_audio,
            "instruction": "请总结这次会议的主要内容、关键决策和后续行动计划。",
            "ref_answer": ref_text,
            "duration": get_audio_duration(abs_audio)
        }
        results.append(item)
        count_far += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"完成！共处理 {len(df)} 行。")
    print(f"保留 Far 音频: {count_far} 条")
    print(f"过滤 Near 音频: {count_skipped} 条")
    print(f"数据已写入: {output_path}")


# ==================== 推理函数 ====================

def run_inference_native(model_path, input_file, output_file):
    """基础推理（单次处理整个音频）"""
    print(f"启动推理")

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)

        if hasattr(model, 'disable_talker'): 
            model.disable_talker()
        apply_token_pruning_config(model)

    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

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
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": instruction}
                    ]}
                ]

                audios, _, _ = process_mm_info(msgs, use_audio_in_video=False)

                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                if isinstance(text, list): 
                    text = text[0]

                inputs = processor(
                    text=[text],
                    audio=audios,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                set_audio_token_range(model, inputs)

                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        use_audio_in_video=False,
                        repetition_penalty=1.1
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_seqs = generate_ids[:, input_len:]
                response = processor.batch_decode(generated_seqs, skip_special_tokens=True)[0].strip()

                output_item = {
                    "audio": audio_path,
                    "instruction": instruction,
                    "ref_answer": data.get("ref_answer", ""),
                    "model_response": response
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                f_out.flush()

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


def run_chat_inference(model_path, input_file, output_file):
    """多轮对话推理（分段处理长音频）"""
    print(f"启动多轮对话推理. Chunk={CHUNK_DURATION_SEC}s, Overlap={OVERLAP_SEC}s")

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(model, 'disable_talker'): 
            model.disable_talker()
        apply_token_pruning_config(model)
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    SECRETARY_INSTRUCTION = (
        "你是一名专业的会议秘书。请根据音频内容生成一份详细的会议纪要。\n"
        "要求包含以下具体信息（如果提及）：\n"
        "1. 关键事实：具体的时间、地点、数字。\n"
        "2. 核心决策：会议最终定下来的结论是什么。\n"
        "3. 待办事项：谁（Who）在什么时间（When）之前需要做什么（What）。\n"
        "请不要只描述讨论过程，要直接列出结论事实。"
    )

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, line in enumerate(tqdm(lines, desc="Processing")):
            try:
                data = json.loads(line)
                audio_path = data.get("audio")
                if not os.path.exists(audio_path): 
                    continue

                chunks = load_audio_chunks_with_overlap(audio_path)
                if not chunks: 
                    continue

                messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
                final_response = ""

                print(f"\n--- 开始处理样本 {idx}: {os.path.basename(audio_path)} (共 {len(chunks)} 段) ---")

                for i, chunk in enumerate(chunks):
                    is_last_chunk = (i == len(chunks) - 1)

                    user_content = [{"type": "audio", "audio": chunk}]

                    if is_last_chunk:
                        prompt_text = f"这是最后一段音频。{SECRETARY_INSTRUCTION}"
                        user_content.append({"type": "text", "text": prompt_text})
                    else:
                        prompt_text = f"这是会议录音的第 {i + 1}/{len(chunks)} 部分，请先接收并保持记忆，回复'收到'即可，不要总结。"
                        user_content.append({"type": "text", "text": prompt_text})

                    messages.append({"role": "user", "content": user_content})

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if isinstance(text, list): 
                        text = text[0]
                    auds, _, _ = process_mm_info(messages, use_audio_in_video=False)

                    inputs = processor(text=[text], audio=auds, return_tensors="pt", padding=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    set_audio_token_range(model, inputs)

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

                    print(f"  [Turn {i + 1}] Model: {response_text}")

                    messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

                    if is_last_chunk:
                        final_response = response_text

                    del inputs, generate_ids
                    torch.cuda.empty_cache()

                output_item = {
                    "audio": audio_path,
                    "instruction": SECRETARY_INSTRUCTION,
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


def run_map_phase(model_path, input_file, output_file):
    """Map 阶段：提取关键信息"""
    print(f"启动 Map 阶段 (提取关键信息). Prune={PRUNE_RATIO}")

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(model, 'disable_talker'): 
            model.disable_talker()
        apply_token_pruning_config(model)
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, line in enumerate(tqdm(lines, desc="Map Phase")):
            try:
                data = json.loads(line)
                audio_path = data.get("audio")
                if not os.path.exists(audio_path): 
                    continue

                chunks = load_audio_chunks_with_overlap(audio_path)
                num_chunks = len(chunks)
                map_results = []

                for i, chunk in enumerate(chunks):
                    map_prompt = (
                        f"这是会议录音的第 {i + 1}/{num_chunks} 部分。"
                        "请详细罗列这段录音中的关键信息，包括：1. 讨论的主要议题；2. 参与者的核心观点；3. 达成的结论或待办事项。"
                        "请直接列出要点，不要遗漏。"
                    )
                    msgs = [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user",
                         "content": [{"type": "audio", "audio": chunk}, {"type": "text", "text": map_prompt}]}
                    ]

                    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    if isinstance(text, list): 
                        text = text[0]
                    auds, _, _ = process_mm_info(msgs, use_audio_in_video=False)
                    inputs = processor(text=[text], audio=auds, return_tensors="pt", padding=True)

                    set_audio_token_range(model, inputs)

                    with torch.no_grad():
                        generate_ids = model.generate(**inputs, max_new_tokens=1024, use_audio_in_video=False)

                    input_len = inputs["input_ids"].shape[1]
                    generated_seqs = generate_ids[:, input_len:]
                    res = processor.batch_decode(generated_seqs, skip_special_tokens=True)[0].strip()
                    map_results.append(res)
                    torch.cuda.empty_cache()

                inter_item = {
                    "audio": audio_path,
                    "instruction": data.get("instruction", ""),
                    "ref_answer": data.get("ref_answer", ""),
                    "map_fragments": map_results
                }
                f_out.write(json.dumps(inter_item, ensure_ascii=False) + '\n')
                f_out.flush()

            except Exception as e:
                print(f"Error processing {idx}: {e}")
                traceback.print_exc()


def run_reduce_phase(model_path, input_map_file, output_final_file):
    """Reduce 阶段：纯文本汇总"""
    print("启动 Reduce 阶段 (纯文本汇总)...")

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(model, 'disable_talker'): 
            model.disable_talker()

        # 强制关闭所有可能残留的剪枝配置
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            cfg = model.thinker.model.config
            if hasattr(cfg, 'sparse_attention_config'):
                cfg.sparse_attention_config['prune_ratio'] = 0.0
            if hasattr(cfg, 'audio_prune_ratio'):
                cfg.audio_prune_ratio = 0.0
            if hasattr(cfg, 'audio_token_start'):
                cfg.audio_token_start = 0
                cfg.audio_token_num = 0

    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_final_file), exist_ok=True)

    with open(input_map_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_final_file, 'w', encoding='utf-8') as f_out:
        for idx, line in enumerate(tqdm(lines, desc="Reduce Phase")):
            try:
                data = json.loads(line)
                instruction = data.get("instruction", "请总结会议。")
                map_fragments = data.get("map_fragments", [])

                if not map_fragments:
                    final_summary = "无法提取会议内容。"
                elif len(map_fragments) == 1:
                    combined_context = map_fragments[0]
                else:
                    combined_context = "\n\n".join(
                        [f"【第{i + 1}段记录】:\n{frag}" for i, frag in enumerate(map_fragments)])

                reduce_prompt = (
                    "以下是从一场长会议录音中分段提取的关键信息记录：\n"
                    f"{'-' * 20}\n{combined_context}\n{'-' * 20}\n\n"
                    f"请根据上述所有记录，{instruction}\n"
                    "注意：请整合所有分段信息，生成一篇连贯、逻辑清晰的完整摘要，不要保留分段的痕迹。"
                )

                msgs = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": reduce_prompt}]}
                ]

                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                if isinstance(text, list): 
                    text = text[0]

                inputs = processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, use_audio_in_video=False)

                input_len = inputs["input_ids"].shape[1]
                generated_seqs = generated_ids[:, input_len:]
                final_summary = processor.batch_decode(generated_seqs, skip_special_tokens=True)[0].strip()

                output_item = {
                    "audio": data.get("audio"),
                    "instruction": instruction,
                    "ref_answer": data.get("ref_answer"),
                    "model_response": final_summary
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                f_out.flush()

            except Exception as e:
                print(f"Reduce Error {idx}: {e}")


# ==================== LLM 评测 ====================

PROMPT_EXTRACTION = """Stage 1: Facts Extraction Prompt

You are an expert at breaking down meeting transcripts into facts. Your task is to extract clear, factual statements with proper context.

IMPORTANT RULES:
1. Output must be a valid JSON list of objects
2. NEVER add information not in the transcript
3. Skip unclear or ambiguous content
4. Each fact must be atomic (single piece of information)
5. NO hallucination or inference
6. **LANGUAGE CONSTRAINT**: All output 'fact' and 'context' MUST be in Simplified Chinese (简体中文).

CONTENT GUIDELINES - STRICTLY FOLLOW:
1. INCLUDE:
- Clear, explicit statements
- Complete, meaningful information
- Actionable items or decisions
- Important discussion points
- Concrete facts or outcomes

2. EXCLUDE:
- Filler statements (e.g., "OK", "Right", "Mm-hmm", "嗯", "那个", "对")
- General acknowledgments
- Incomplete or unclear statements
- Transcription artifacts like {{disfmarker}} or {{vocalsound}}
- Redundant and Ambiguous information

3. For each included fact, provide:
- "fact": Single, atomic piece of information (in Chinese)
- "context": Comprehensive context with history (in Chinese)

Output Format: Return a JSON list of objects (no additional keys), e.g.,
[
  {{
    "fact": "团队同意在Q3发布产品",
    "context": "在分析了之前的延期原因和市场情况后，大家认为Q3是最佳时机"
  }}
]

Your Task:
Break down this transcript into atomic facts with context.
Remember: Must return a valid JSON list. Only include clear, explicit information. Skip all filler words, acknowledgments, and artifacts. Break compound statements into facts. Exclude unclear or ambiguous content.
**STRICTLY OUTPUT IN SIMPLIFIED CHINESE.**

Input Transcript:
{transcript}
"""

PROMPT_JUDGE = """Stage 4: Quality assurance

You are a checker agent evaluating a meeting summary against the ground truth facts.

EVALUATION CRITERIA & POINTS:

1. Content Accuracy (maximum 5 error points)
- Uses ONLY provided facts and contexts (Gold Facts).
- No hallucinated or inferred information.
- Facts are placed correctly.
- Context is preserved exactly.

2. Information Coverage (maximum 5 error points)
- High-importance facts (from Gold Facts) are included.
- Critical decisions and actions are covered.
- Essential context is present; nothing vital is missing.

3. Format Requirements (maximum 1 error points)
- Professional tone, clear and concise writing.
- Logical flow between points.
- No bullet points (unless specified otherwise), strictly follows narrative flow.

SCORING DEDUCTIONS:
Errors such as hallucinated content, missing high-importance fact, incorrect context usage, or outside word limit add one error point in the respective category.

OUTPUT FORMAT: Return a single JSON object with exactly:
**IMPORTANT FORMAT RULE**: Do NOT use double quotes (") INSIDE the feedback string. Use single quotes (') instead.
{{
  "accuracy_score": (0-5, start at 5 and deduct 1 per hallucination/error),
  "coverage_score": (0-5, start at 5 and deduct 1 per missing key fact),
  "format_score": (0-1),
  "final_confidence_score": (0-100 derived from above),
  "feedback": "Specific issues and suggestions in CHINESE (use single quotes)"
}}

Your Task:
Evaluate the following GENERATED SUMMARY strictly against the GOLD FACTS.
Provide feedback in Chinese.

GOLD FACTS (Ground Truth): {gold_facts}
GENERATED SUMMARY: {generated_summary}
"""


def extract_via_regex(text):
    """正则兜底提取"""
    result = {}
    patterns = {
        "accuracy_score": r'"accuracy_score"\s*:\s*(\d+)',
        "coverage_score": r'"coverage_score"\s*:\s*(\d+)',
        "format_score": r'"format_score"\s*:\s*(\d+)',
        "final_confidence_score": r'"final_confidence_score"\s*:\s*(\d+)'
    }
    for key, pat in patterns.items():
        match = re.search(pat, text)
        if match:
            result[key] = int(match.group(1))
        else:
            result[key] = 0

    try:
        if '"feedback":' in text:
            start_marker = '"feedback":'
            start_idx = text.find(start_marker) + len(start_marker)
            first_quote = text.find('"', start_idx)
            if first_quote != -1:
                content_start = first_quote + 1
                content_end = text.rfind('"')
                if content_end > content_start:
                    result["feedback"] = text[content_start:content_end]
                else:
                    result["feedback"] = text[content_start:]
    except:
        result["feedback"] = "Regex extraction failed"
    return result


def repair_and_parse_json(text):
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match: 
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except:
        pass

    try:
        return ast.literal_eval(text)
    except:
        pass

    if text.startswith("[") or "fact" in text:
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

    if "accuracy_score" in text:
        return extract_via_regex(text)

    if text.strip().startswith("[") and not text.strip().endswith("]"):
        try:
            return json.loads(text + "]")
        except:
            try:
                return json.loads(text + "}]")
            except:
                pass

    preview = text[:200] + "..." + text[-100:] if len(text) > 300 else text
    raise ValueError(f"JSON Parse Failed. Preview:\n{preview}")


def call_llm(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=4096,
            do_sample=False,
            temperature=0.1
        )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def print_final_statistics(output_file):
    """读取最终的 jsonl 文件，计算并打印平均分"""
    if not os.path.exists(output_file):
        print("未找到结果文件，无法统计。")
        return

    total_final = 0.0
    total_acc = 0.0
    total_cov = 0.0
    total_fmt = 0.0
    count = 0

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "frame_details" in data:
                    details = data["frame_details"]
                    final = float(details.get("final_confidence_score", 0))
                    acc = float(details.get("accuracy_score", 0))
                    cov = float(details.get("coverage_score", 0))
                    fmt = float(details.get("format_score", 0))

                    total_final += final
                    total_acc += acc
                    total_cov += cov
                    total_fmt += fmt
                    count += 1
            except:
                continue

    if count > 0:
        print("\n" + "=" * 40)
        print(f"📊 FRAME 评测最终统计 (共 {count} 条)")
        print("-" * 40)
        print(f"✅ 平均总分 (Final Score):   {total_final / count:.2f} / 100")
        print(f"🎯 平均准确分 (Accuracy):    {total_acc / count:.2f} / 5.0")
        print(f"覆盖分 (Coverage):    {total_cov / count:.2f} / 5.0")
        print(f"📝 平均格式分 (Format):      {total_fmt / count:.2f} / 1.0")
        print("=" * 40 + "\n")
    else:
        print("\n[Warn] 没有读取到有效的评测数据。")


def get_gold_facts(model, tokenizer, ref_text):
    try:
        ext_resp = call_llm(model, tokenizer, PROMPT_EXTRACTION.format(transcript=ref_text))
        raw_facts = repair_and_parse_json(ext_resp)
        if isinstance(raw_facts, list): 
            return raw_facts
        if isinstance(raw_facts, dict): 
            return [raw_facts]
        return []
    except Exception as e:
        print(f"[Extraction Error]: {e}")
        return []


def run_llm_judge(judge_model_path, input_file, output_file):
    """运行 LLM 评测"""
    print(f"Loading Judge Model: {judge_model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(judge_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            judge_model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True
        ).eval()
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(input_file):
        print("Input file not found.")
        return

    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for _ in f)
    print(f"Resuming from line {processed_count}...")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'a', encoding='utf-8') as f_out:

        lines = f_in.readlines()
        for idx, line in enumerate(tqdm(lines, desc="FRAME Evaluating")):
            if idx < processed_count: 
                continue

            try:
                data = json.loads(line)
                ref_text = data.get("ref_answer", "")
                model_summary = data.get("model_response", "")

                if not ref_text or not model_summary: 
                    continue

                gold_facts_list = get_gold_facts(model, tokenizer, ref_text)

                if not gold_facts_list:
                    output_item = data.copy()
                    output_item["judge_error"] = "No facts extracted"
                    f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                    f_out.flush()
                    continue

                gold_facts_str = json.dumps(gold_facts_list, ensure_ascii=False, indent=2)

                judge_resp = call_llm(model, tokenizer, PROMPT_JUDGE.format(
                    gold_facts=gold_facts_str,
                    generated_summary=model_summary
                ))

                eval_result = repair_and_parse_json(judge_resp)
                final_score = float(eval_result.get("final_confidence_score", 0))

                output_item = data.copy()
                output_item["frame_final_score"] = final_score
                output_item["frame_details"] = eval_result
                output_item["gold_facts"] = gold_facts_list

                f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                f_out.flush()

            except Exception as e:
                print(f"\n[Error processing line {idx}]: {e}")
                continue

            if (idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

    print_final_statistics(output_file)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="AliMeeting Evaluation")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["build", "inference", "chat", "map", "reduce", "judge"],
                       help="运行模式")
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型路径")
    parser.add_argument("--judge-model-path", type=str, default=None,
                       help="评测模型路径（仅用于 judge 模式）")
    parser.add_argument("--input-file", type=str, default=None,
                       help="输入文件路径")
    parser.add_argument("--output-file", type=str, default=None,
                       help="输出文件路径")
    parser.add_argument("--csv-path", type=str, default=None,
                       help="CSV 文件路径（仅用于 build 模式）")
    parser.add_argument("--root-dir", type=str, default=None,
                       help="根目录（仅用于 build 模式）")
    
    args = parser.parse_args()

    if args.mode == "build":
        if not all([args.csv_path, args.output_file, args.root_dir]):
            print("Error: build 模式需要 --csv-path, --output-file, --root-dir")
            sys.exit(1)
        build_dataset(args.csv_path, args.output_file, args.root_dir)
    
    elif args.mode == "inference":
        if not all([args.model_path, args.input_file, args.output_file]):
            print("Error: inference 模式需要 --model-path, --input-file, --output-file")
            sys.exit(1)
        run_inference_native(args.model_path, args.input_file, args.output_file)
    
    elif args.mode == "chat":
        if not all([args.model_path, args.input_file, args.output_file]):
            print("Error: chat 模式需要 --model-path, --input-file, --output-file")
            sys.exit(1)
        run_chat_inference(args.model_path, args.input_file, args.output_file)
    
    elif args.mode == "map":
        if not all([args.model_path, args.input_file, args.output_file]):
            print("Error: map 模式需要 --model-path, --input-file, --output-file")
            sys.exit(1)
        run_map_phase(args.model_path, args.input_file, args.output_file)
    
    elif args.mode == "reduce":
        if not all([args.model_path, args.input_file, args.output_file]):
            print("Error: reduce 模式需要 --model-path, --input-file, --output-file")
            sys.exit(1)
        run_reduce_phase(args.model_path, args.input_file, args.output_file)
    
    elif args.mode == "judge":
        if not all([args.judge_model_path, args.input_file, args.output_file]):
            print("Error: judge 模式需要 --judge-model-path, --input-file, --output-file")
            sys.exit(1)
        run_llm_judge(args.judge_model_path, args.input_file, args.output_file)


if __name__ == '__main__':
    main()

