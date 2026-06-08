# VY3D-Q3: Head-Score JSON Ablation (minimal)

**Reviewer:** Fix SSS; swap only head-importance JSON to test audio-aware supervision.

## Default setup (`run_vy3d_q3_all.sh`)

| Setting | Value |
|---------|--------|
| Model | Qwen2.5-Omni-3B only |
| Budget ρ | **0.8** |
| Head JSON | `audio_head/qwen2.5_omni/head_score/qwen2_5_3B/audio_head_scores_topk24_alpha1_beta1_gamma1.json` |
| Conditions | **AudioAware_SSS** vs **InverseAudioAware_SSS** vs **RandomInit_SSS_seed42** (no uniform by default) |
| Tasks | Spanish Multilingual LibriSpeech (20%) → CoVoST2 zh→en (20%) → KeSpeech ASR (20%) |

## Run

```bash
bash rebuttal/run_vy3d_q3_all.sh

# Single dataset only
GPU_ID=5 TASKS=kespeech bash rebuttal/run_vy3d_q3_all.sh

# Run Spanish multilingual ASR only
GPU_ID=5 TASKS=multilingual-spanish bash rebuttal/run_vy3d_q3_all.sh

# Optional LibriSpeech, only after AudioMarathon data is present locally
GPU_ID=5 TASKS=librispeech bash rebuttal/run_vy3d_q3_all.sh
```

Outputs: `Results/Qwen2_5_Omni_3B/ablation_vy3d_q3/<task>/{AudioAware_SSS,InverseAudioAware_SSS,RandomInit_SSS_seed42}/budget0.8/`

Generated JSONs are written next to the reference:
- `.../ablation_random/<stem>_inverse.json`
- `.../ablation_random/<stem>_random_seed42.json`

## Env overrides

- `AUDIO_HEAD_JSON` — different reference file
- `SEEDS="42 43"` — more random seeds
- `RUN_INVERSE=false` — skip inverse allocation control
- `RUN_UNIFORM=true` — add uniform-head control
- `MLS_SAMPLE_FRACTION=0.2` — multilingual Spanish subset ratio (used when `SAMPLE_LIMIT=0`)
- `COVOST2_SAMPLE_FRACTION=0.2` — CoVoST2 subset ratio (used when `SAMPLE_LIMIT=0`)
- `KESPEECH_SAMPLE_FRACTION=0.2` — KeSpeech subset ratio (used when `SAMPLE_LIMIT=0`)
- `BUDGET=0.6` — change retention ratio
