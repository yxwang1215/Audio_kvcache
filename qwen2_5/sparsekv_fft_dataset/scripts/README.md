# Scripts 分类说明

本目录包含用于 Qwen2.5-Omni 音频评估的脚本，已按功能分类组织。

## 目录结构

### 1. `audio_captioning/` - 音频字幕生成任务
包含用于音频字幕生成（Audio Captioning）评估的脚本：

- `run_audiocaps.sh` - AudioCaps 数据集评估
- `run_audioset.sh` - AudioSet (wavcaps-audioset) 数据集评估
- `run_freesound.sh` - FreeSound (wavcaps-freesound) 数据集评估
- `run_soundbible.sh` - SoundBible (wavcaps-soundbible) 数据集评估

### 2. `audio_qa/` - 音频问答任务
包含用于音频问答（Audio Question Answering）评估的脚本：

- `run_clothoaqa.sh` - ClothoAQA 数据集评估
- `run_llama-questions.sh` - LLaMA Questions 数据集评估
- `run_speech-triviaqa.sh` - Speech-TriviaQA 数据集评估
- `run_speech-web-questions.sh` - Speech-Web-Questions 数据集评估

### 3. `multimodal_eval/` - 多模态评估任务
包含用于多模态评估的脚本：

- `run_mmau_pro_instruction_following.sh` - MMAU-Pro 指令遵循评估
- `run_mmau_pro_open.sh` - MMAU-Pro 开放性问题评估

### 4. 其他已有分类目录

- `alimeeting/` - AliMeeting 相关脚本
- `ASR/` - 自动语音识别相关脚本
- `chatbot/` - 聊天机器人相关脚本
- `ST/` - 语音翻译相关脚本
- `others/` - 其他脚本

## 使用说明

所有脚本都遵循相同的配置模式：
- 支持 SparseKV 配置
- 支持 FFT 滤波配置
- 支持 Audio-aware TopK 配置
- 支持剪枝配置

运行前请根据需求修改脚本中的环境变量配置。

