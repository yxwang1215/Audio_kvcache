import os
import json
import torch
from tqdm import tqdm
import sys
import traceback
from modeling_qwen2_5_omni_fastv import Qwen2_5OmniForConditionalGeneration
from processing_qwen2_5_omni import Qwen2_5OmniProcessor

# --- 配置区 ---
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group."


def run_reduce_phase(model_path, input_map_file, output_final_file):
    print("启动 Reduce 阶段 (纯文本汇总)...")

    # 1. 加载模型
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2", trust_remote_code=True
        ).eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(model, 'disable_talker'): model.disable_talker()

        # 强制关闭所有可能残留的剪枝配置 (保险起见)
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
        print(f"模型加载失败: {e}");
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
                    # 如果只有一段，且本身已经是总结性的（取决于Map Prompt），可以直接用
                    # 但为了稳妥，还是让模型过一遍格式
                    combined_context = map_fragments[0]
                else:
                    combined_context = "\n\n".join(
                        [f"【第{i + 1}段记录】:\n{frag}" for i, frag in enumerate(map_fragments)])

                # Reduce Prompt
                reduce_prompt = (
                    "以下是从一场长会议录音中分段提取的关键信息记录：\n"
                    f"{'-' * 20}\n{combined_context}\n{'-' * 20}\n\n"
                    f"请根据上述所有记录，{instruction}\n"
                    "注意：请整合所有分段信息，生成一篇连贯、逻辑清晰的完整摘要，不要保留分段的痕迹。"
                )

                # 纯文本推理
                msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"type": "text", "text": reduce_prompt}]}]

                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                if isinstance(text, list): text = text[0]

                inputs = processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, use_audio_in_video=False)

                input_len = inputs["input_ids"].shape[1]
                generated_seqs = generated_ids[:, input_len:]
                final_summary = processor.batch_decode(generated_seqs, skip_special_tokens=True)[0].strip()

                # 写入最终结果
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


if __name__ == '__main__':
    MODEL_PATH = "/mnt/users/hepeize-20251016/Model/Qwen2.5_Omni"
    # 输入文件是 Map 阶段的输出
    INPUT_MAP = "/mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/results/intermediate_map.jsonl"
    # 最终输出
    OUTPUT_FINAL = "/mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/results/alimeeting_prediction.jsonl"

    run_reduce_phase(MODEL_PATH, INPUT_MAP, OUTPUT_FINAL)