import os
import json
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def parse_textgrid_full(file_path):
    """
    解析 TextGrid，提取所有 Tier 的内容，并按时间戳排序合并。
    """
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


def build_dataset(csv_path, output_path, root_dir):
    print(f"正在读取 CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    col_audio = 'Audio:FILE'
    col_text = 'Text:FILE'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = []
    # 计数器
    count_far = 0
    count_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建数据"):
        rel_audio = str(row[col_audio]).strip()
        rel_text = str(row[col_text]).strip()

        # 只保留 far 音频 (通常路径含 'far' 或文件名含 'MS')
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


if __name__ == "__main__":
    ROOT_DIR = "/mnt/users/hepeize-20251016"
    CSV_PATH = os.path.join(ROOT_DIR, "AliMeeting/Test_Ali.csv")
    OUTPUT_JSONL = os.path.join(ROOT_DIR, "Dataset/AudioMarathon_long/alimeeting_test.jsonl")

    build_dataset(CSV_PATH, OUTPUT_JSONL, ROOT_DIR)