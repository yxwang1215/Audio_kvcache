# 超参数搜索脚本使用说明

## 功能说明

`hyperparameter_search.py` 脚本用于自动搜索图中空白部分的超参数组合（alpha 和 cutoff），使用4个GPU（0,1,2,3）轮询分配任务。

## 使用方法

### 直接运行

```bash
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen/sparsekv_fft
python3 hyperparameter_search.py
```

### 后台运行（推荐）

```bash
cd /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/qwen/sparsekv_fft
nohup python3 hyperparameter_search.py > hyperparameter_search.log 2>&1 &
```

## 功能特点

1. **自动跳过已完成的实验**：如果结果文件已存在且状态为completed，会自动跳过
2. **4个GPU轮询分配**：自动在GPU 0,1,2,3之间轮询分配任务
3. **实时进度显示**：显示当前完成进度
4. **结果记录**：每次实验完成后立即记录到txt文件
5. **错误处理**：自动处理超时、错误等情况

## 输出文件

- `hyperparameter_search_results.txt`: 详细的实验日志，包含每次实验的时间戳、GPU、参数、准确率等信息
- `hyperparameter_search_results_summary.txt`: 实验完成后的摘要报告，按alpha分组显示所有结果

## 参数说明

脚本会自动搜索以下参数组合：

- **Alpha值**: 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9
- **Cutoff值**: 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8

脚本会自动跳过图中已有的组合，只搜索空白部分。

## 实验配置

所有实验使用以下固定配置：
- `budget=0.5`
- `filter_type=saliency`
- `USE_FFT_FILTER=true`
- `FFT_ADAPTIVE_CUTOFF=true`
- `RATIO=0.5`
- `WINDOW_SIZE=32`

## 结果存储

实验结果存储在：
```
/mnt/users/wangyuxuan-20250915/Audio/Results/LibriSpeech_Long_Results/SAH_FFT/LibriSpeech_budget0.5_fft_saliency_cutoff{cutoff}_alpha{alpha}/
```

每个实验的结果JSON文件为：`librispeech_results_qwen25.json`

## 注意事项

1. 确保conda环境 `Qwen3_2` 已正确配置
2. 确保有足够的磁盘空间存储结果
3. 每个实验可能需要较长时间（通常几十分钟到几小时），请耐心等待
4. 如果脚本被中断，重新运行时会自动跳过已完成的实验

## 查看进度

```bash
# 查看实时日志
tail -f hyperparameter_search_results.txt

# 查看后台运行日志
tail -f hyperparameter_search.log
```

