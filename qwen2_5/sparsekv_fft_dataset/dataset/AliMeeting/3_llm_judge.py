import os
import json
import torch
import re
from tqdm import tqdm
import sys
import traceback
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 用户配置区 =================
JUDGE_MODEL_PATH = "/mnt/users/hepeize-20251016/Model/Qwen3-8B"
INPUT_PREDICTIONS_PATH = "/mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/results/alimeeting_prediction.jsonl"
OUTPUT_SCORES_PATH = "/mnt/users/hepeize-20251016/src/Qwen_2.5_Omni/AliMeeting/results/alimeeting_scores_frame_final.jsonl"

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


# ================= 工具函数 =================

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
    if match: text = match.group(1).strip()

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


# ================= 统计功能 =================

def print_final_statistics(output_file):
    """
    读取最终的 jsonl 文件，计算并打印平均分
    """
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
                # 只有成功评测的数据才有 frame_details
                if "frame_details" in data:
                    details = data["frame_details"]
                    # 兼容不同来源的字段名，正则提取的可能是数字，json解析的是数字
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


# ================= 核心流程 =================

def get_gold_facts(model, tokenizer, ref_text):
    try:
        ext_resp = call_llm(model, tokenizer, PROMPT_EXTRACTION.format(transcript=ref_text))
        raw_facts = repair_and_parse_json(ext_resp)
        if isinstance(raw_facts, list): return raw_facts
        if isinstance(raw_facts, dict): return [raw_facts]
        return []
    except Exception as e:
        print(f"[Extraction Error]: {e}")
        return []


def run_judging_frame(model_path, input_file, output_file):
    print(f"Loading Judge Model: {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto",
                                                     trust_remote_code=True).eval()
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
            if idx < processed_count: continue

            try:
                data = json.loads(line)
                ref_text = data.get("ref_answer", "")
                model_summary = data.get("model_response", "")

                if not ref_text or not model_summary: continue

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

    # 全部跑完后，打印统计信息
    print_final_statistics(output_file)


if __name__ == '__main__':
    run_judging_frame(JUDGE_MODEL_PATH, INPUT_PREDICTIONS_PATH, OUTPUT_SCORES_PATH)