# AliMeeting 长音频摘要评估流程 (Map-Reduce版)

本项目实现了一套基于 **Qwen2.5-Omni** 模型针对 **AliMeeting** 长会议音频进行自动摘要生成的完整评估流程。

为了解决长音频（20分钟+）导致的显存溢出、上下文丢失以及 Token 剪枝在纯文本阶段的兼容性问题，本流程采用了 **Map-Reduce（分块提取-汇总生成）** 策略，将音频理解（Map）与文本汇总（Reduce）解耦。

## 📋 核心特性

1.  **Map-Reduce 架构**：
      * **Map 阶段**：将长音频切片，并行提取关键信息（支持音频 Token 剪枝）。
      * **Reduce 阶段**：将提取的碎片信息拼接，进行纯文本汇总生成（强制关闭剪枝，避免报错）。
2.  **带重叠的切分 (Overlap Chunking)**：切分音频时保留 30 秒重叠，防止关键句子在切分点被截断。
3.  **Token 剪枝支持**：支持在 Map 阶段对音频 Token 进行剪枝（`PRUNE_RATIO`），以加速推理并减少计算量。
4.  **LLM-as-a-Judge**：内置自动化评分脚本，使用 Qwen3-8B 对生成结果进行多维度打分。

## 📂 文件说明

| 文件名 | 功能描述 |
| :--- | :--- |
| **`run_alimeeting_eval.sh`** | **主启动脚本**。控制整个流程的串行执行，负责设置 GPU 和剪枝参数等环境变量。 |
| `1_build_dataset.py` | **数据构建**。读取原始 QA JSON，定位远场混音音频（`MS*.wav`），生成标准化的 `alimeeting_test.jsonl`。 |
| `2_run_inference_map.py` | **Map 推理脚本**。读取音频，进行切分和剪枝推理，输出包含关键信息片段的 `intermediate_map.jsonl`。 |
| `2_run_inference_reduce.py` | **Reduce 推理脚本**。读取中间结果，进行纯文本汇总生成，输出最终摘要 `alimeeting_prediction.jsonl`。 |
| `3_llm_judge.py` | **评分脚本**。对比模型生成摘要与参考逐字稿，计算覆盖度、准确性和条理性得分，输出 `alimeeting_scores.jsonl`。 |

## 🚀 快速开始

### 1\. 环境配置

请确保已安装以下依赖库（基于 Qwen2.5-Omni 的环境）：

```bash
pip install torch transformers librosa soundfile tqdm
```

### 2\. 修改配置

打开 `run_alimeeting_eval.sh`，根据您的硬件和实验需求修改顶部的配置区：

```bash
# =================== 用户配置区 ===================
# 设置使用的 GPU 编号
export GPU_ID="7"

# 设置 Map 阶段的音频 Token 剪枝比例 (0.0 - 1.0)
# 0.0 = 不剪枝 (基线), 0.5 = 剪掉 50% 的音频 Token
export TARGET_PRUNE_RATIO=0.5 
# =================================================
```

### 3\. 运行流程

直接运行 Shell 脚本即可启动全流程：

```bash
bash run_alimeeting_eval.sh
```

## 🔄 详细流程解析

### Step 1: 数据构建 (`1_build_dataset.py`)

  * **输入**: `/Dataset/AudioMarathon/AliMeeting/meetingqa/Test_Ali_longform.json`
  * **处理**: 过滤无效数据，计算音频时长，确保指向包含所有说话人的混合音频（`MS*.wav`）。
  * **输出**: `alimeeting_test.jsonl`

### Step 2: Map 阶段 (`2_run_inference_map.py`)

  * **输入**: `alimeeting_test.jsonl`
  * **逻辑**:
      * 将音频按 1200秒（20分钟）切片，步长 1170秒（30秒重叠）。
      * **开启剪枝**（如果 `TARGET_PRUNE_RATIO > 0`）。
      * Prompt 指令模型提取“主要议题、核心观点、关键决策”。
  * **输出**: `intermediate_map.jsonl` (包含分段的文本记录列表)

### Step 3: Reduce 阶段 (`2_run_inference_reduce.py`)

  * **输入**: `intermediate_map.jsonl`
  * **逻辑**:
      * **强制关闭剪枝**（`PRUNE_RATIO=0.0`），避免在无音频输入时底层计算 `NoneType` 错误。
      * 拼接所有 Map 阶段产生的文本片段。
      * Prompt 指令模型将碎片信息整合成一篇连贯的摘要。
  * **输出**: `alimeeting_prediction.jsonl` (最终生成的摘要)

### Step 4: 裁判打分 (`3_llm_judge.py`)

  * **输入**: `alimeeting_prediction.jsonl`
  * **逻辑**: 使用 Qwen3-8B 模型作为裁判，根据参考逐字稿对生成结果进行打分（1-10分）。
  * **输出**: `alimeeting_scores.jsonl`

## 🛠️ 常见问题与调试

**Q: 为什么要把 Map 和 Reduce 分成两个脚本？**
A: 为了彻底隔离模型状态。Qwen2.5-Omni 在处理音频（需要剪枝）和纯文本（不需要剪枝）时，底层的 Config 状态容易在同一个进程中发生污染，导致 `TypeError: unsupported operand type(s) for +: 'NoneType'`。拆分后，Reduce 阶段作为一个干净的纯文本任务运行，稳定且高效。

**Q: 如何修改切片长度？**
A: 修改 `2_run_inference_map.py` 中的 `CHUNK_DURATION_SEC` 变量（默认 1200秒）。

**Q: 音频文件找不到？**
A: 请检查 `1_build_dataset.py` 中的 `DATASET_ROOT` 路径是否与服务器上的实际路径匹配。