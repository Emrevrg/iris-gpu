# IRIS GPU

Lossless (bit-exact) compression for AI model weights, decoded on the GPU. Works across architectures and data types.

## Quick start
```bash
pip install torch safetensors nvidia-nvcomp-cu12
python iris_evrensel.py /path/to/model_dir        # ratio, GPU decode speed, bit-exact check
python iris_akis_olc.py /path/to/model_dir        # full-model load time vs. plain safetensors
```

## Measured (one H100)
| Model | Type / dtype | Size | Ratio | GPU decode | Bit-exact |
|---|---|---|---|---|---|
| Gemma 4 31B (fine-tuned) | language, bf16 | 62.5 GB | 0.675 | 30.2 GB/s | yes |
| Whisper large-v3 | audio, fp32+fp16 | 9.3 GB | 0.770 | 27.8 GB/s | yes |
| SigLIP2 so400m | vision, fp32 | 4.5 GB | 0.843 | 15.3 GB/s | yes |
| Qwen3-8B-FP8 | language, fp8 | 9.4 GB | 0.815 | 28.9 GB/s | yes |
| gpt-oss-20b | MoE, MXFP4 | 12.0 GB | 0.880 | 27.8 GB/s | yes |

Moving a full 62.5 GB model onto the GPU: 12.01 s with plain `safetensors` loading (the path AirLLM-style layer streaming uses) vs **2.11 s** with IRIS (5.7x). We measured the loading path, not the AirLLM package itself.

How it works: per tensor, the smaller of (a) the whole byte stream with nvCOMP ANS, or (b) the floating-point exponent field as its own plane (ANS) plus raw sign/mantissa. Decoding and reassembly run on the GPU.

License: MIT. Measured on one week of NVIDIA H100 time provided by portvMind (vMind) Public Cloud.