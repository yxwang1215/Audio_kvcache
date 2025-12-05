# Example 1: Using absolute budget value (>= 1.0)
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=6 \
USE_SPARSEKV=false \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/vanilla/LibriSpeech_vanilla \
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv/gemma_librispeech.py

# Example 2.1: Using SparseMM
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=6 \
USE_SPARSEKV=false \
BUDGET=0.8 \
RATIO=0.5 \
WINDOW_SIZE=32 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/Results/Gemma/sparsekv_SparseMM/LibriSpeech_budget0.8 \
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft/gemma_librispeech.py

conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=7 \
USE_SPARSEKV=true \
BUDGET=0.6 \
RATIO=0.5 \
WINDOW_SIZE=32 \
KERNEL_SIZE=5 \
POOLING=maxpool \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/Results/Gemma/sparsekv_SparseMM/LibriSpeech_budget0.6 \
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft/gemma_librispeech.py

# Example 2.2: Using FFT-based spectral filtering (Spectral Cache Eviction)
# This replaces Conv1d smoothing with FFT-based frequency domain filtering
# Advantages: Global receptive field, adaptive cutoff, better separation of trends vs noise
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=6 \
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
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft/LibriSpeech_budget0.7_fft_lowpass_alpha0.5 \
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft/gemma_librispeech.py

# Example 2.3: FFT with saliency detection (Spectral Residual method)
conda activate Qwen3_2 &&
CUDA_VISIBLE_DEVICES=7 \
USE_SPARSEKV=true \
USE_FFT_FILTER=true \
FFT_FILTER_TYPE=saliency \
FFT_CUTOFF_RATIO=0.3 \
FFT_USE_SALIENCY=true \
FFT_ALPHA=0.8 \
FFT_ADAPTIVE_CUTOFF=true \
BUDGET=0.5 \
RATIO=0.5 \
WINDOW_SIZE=32 \
HEAD_SCORE_FILE=aggregated \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft/LibriSpeech_budget0.5_fft_saliency_cutoff0.3_alpha0.8 \
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft/gemma_librispeech.py

# Example 3: Baseline without SparseMM
CUDA_VISIBLE_DEVICES=1 \
USE_SPARSEKV=false \
RESULTS_DIR=/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft/LibriSpeech_vanilla \
python /mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache/gemma/sparsekv_fft/gemma_librispeech.py