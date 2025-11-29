# Example 1: Using absolute budget value (>= 1.0)
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=0 \
USE_SPARSEKV=true \
BUDGET=4096 \
RATIO=0.5 \
WINDOW_SIZE=32 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_4096_ratio0.5_window32_avgpool \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_qwen2.5.py

# Example 2: Using budget as ratio of prefill tokens (< 1.0, e.g., 0.7 = 70%)
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=0 \
USE_SPARSEKV=true \
BUDGET=0.5 \
RATIO=0.5 \
WINDOW_SIZE=32 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_budget0.5 \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_qwen2.5.py

# Example 2.1: With audio-aware TopK selection enabled (from WHY_IMPROVEMENTS.md)

conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=1 \
USE_SPARSEKV=true \
USE_AUDIO_AWARE_TOPK=true \
AUDIO_BOUNDARY_BOOST=1.0 \
AUDIO_CONTINUITY_WEIGHT=0.1 \
AUDIO_SMOOTH_WINDOW=9 \
BUDGET=0.6 \
RATIO=0.5 \
WINDOW_SIZE=32 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_budget0.6_audio_aware_boundary1_continuity0.1_smooth9 \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv/LibriSpeech_qwen2.5.py

# Example 2.2: Using FFT-based spectral filtering (Spectral Cache Eviction)
# This replaces Conv1d smoothing with FFT-based frequency domain filtering
# Advantages: Global receptive field, adaptive cutoff, better separation of trends vs noise
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=1 \
USE_SPARSEKV=true \
USE_FFT_FILTER=true \
FFT_FILTER_TYPE=lowpass \
FFT_CUTOFF_RATIO=0.3 \
FFT_ALPHA=0.5 \
FFT_ADAPTIVE_CUTOFF=true \
USE_AUDIO_AWARE_TOPK=true \
AUDIO_BOUNDARY_BOOST=1.0 \
AUDIO_CONTINUITY_WEIGHT=0.1 \
AUDIO_SMOOTH_WINDOW=9 \
BUDGET=0.7 \
RATIO=0.5 \
WINDOW_SIZE=32 \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv_fft/LibriSpeech_budget0.7_fft_lowpass_alpha0.5 \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv_fft/LibriSpeech_qwen2.5.py

# Example 2.3: FFT with saliency detection (Spectral Residual method)
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=1 \
USE_SPARSEKV=true \
USE_FFT_FILTER=true \
FFT_FILTER_TYPE=saliency \
FFT_CUTOFF_RATIO=0.3 \
FFT_USE_SALIENCY=true \
FFT_ALPHA=0.6 \
FFT_ADAPTIVE_CUTOFF=true \
BUDGET=0.1 \
RATIO=0.5 \
WINDOW_SIZE=32 \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv_fft/LibriSpeech_budget0.1_fft_saliency_cutoff0.3_alpha0.6 \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv_fft/LibriSpeech_qwen2.5.py

# Example 3: Baseline without SparseMM
CUDA_VISIBLE_DEVICES=1 \
USE_SPARSEKV=false \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv_fft/LibriSpeech_vanilla \
python /mnt/users/wangyuxuan-20250915/Audio/AudioMarathon/src/Qwen_2.5_Omni/sparsekv_fft/LibriSpeech_qwen2.5.py