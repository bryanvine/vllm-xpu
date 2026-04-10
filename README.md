# vLLM XPU — Intel Arc/Battlemage Build

Patched fork of [vLLM v0.19.0](https://github.com/vllm-project/vllm/releases/tag/v0.19.0) with working GPTQ support on Intel XPU (Arc Pro B70 / Battlemage tested).

Upstream v0.19.0 removed the XPU code paths for GPTQ quantization, breaking all GPTQ models on Intel GPUs. This fork restores them. See [vllm-project/vllm#39474](https://github.com/vllm-project/vllm/issues/39474) for the upstream bug report.

## What's fixed

vLLM 0.19.0 removed `current_platform.is_xpu()` branches from `gptq.py` that v0.17 had, causing fallthrough to CUDA-only `vllm._C` ops (which don't exist in the XPU build). This fork:

1. Force-loads `vllm_xpu_kernels/_xpu_C.abi3.so` (not auto-loaded in v0.19)
2. Restores the XPU branch in `GPTQLinearMethod.process_weights_after_loading` using `vllm_xpu_kernels`
3. Restores the XPU branch in `GPTQLinearMethod.apply` using `torch.ops._xpu_C.int4_gemm_w4a16`

## Build the Docker image

```bash
git clone https://github.com/bryanvine/vllm-xpu.git
cd vllm-xpu
git checkout xpu-build-0.19.0
docker build -f docker/Dockerfile.xpu -t vllm-xpu:0.19.0 --shm-size=4g .
```

Build takes ~20-40 minutes. The resulting image is ~8GB compressed.

## Quick start

```bash
docker run --rm -it \
  --device /dev/dri:/dev/dri \
  --group-add render --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v /path/to/models:/models:ro \
  -p 8000:8000 \
  vllm-xpu:0.19.0 \
  vllm serve /models/YourModel \
    --host 0.0.0.0 --port 8000 \
    --dtype auto --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192
```

## Running models on Intel Arc Pro B70 (32GB VRAM)

### Qwen3.5-35B-A3B (MoE, GPTQ 4-bit)

Mixture-of-experts with 3B active parameters. Fits comfortably in 32GB with room for 8K+ context. Supports EAGLE3 speculative decoding for faster generation.

```bash
# Download model and EAGLE3 draft model
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --local-dir /models/Qwen3.5-35B-A3B-GPTQ-Int4
huggingface-cli download jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge --local-dir /models/Qwen3.5-35B-A3B-Eagle3-Specforge

# Serve with EAGLE3 speculative decoding
docker run --rm -it \
  --device /dev/dri:/dev/dri \
  --group-add render --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v /models:/models:ro \
  -p 8000:8000 \
  vllm-xpu:0.19.0 \
  vllm serve /models/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --served-model-name qwen3.5-35b \
    --host 0.0.0.0 --port 8000 \
    --dtype auto --trust-remote-code \
    --gpu-memory-utilization 0.97 \
    --max-model-len 8192 \
    --speculative-config '{"method":"eagle3","model":"/models/Qwen3.5-35B-A3B-Eagle3-Specforge","num_speculative_tokens":3,"draft_tensor_parallel_size":1}'

# Without speculative decoding
docker run --rm -it \
  --device /dev/dri:/dev/dri \
  --group-add render --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v /models:/models:ro \
  -p 8000:8000 \
  vllm-xpu:0.19.0 \
  vllm serve /models/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --served-model-name qwen3.5-35b \
    --host 0.0.0.0 --port 8000 \
    --dtype auto --trust-remote-code \
    --gpu-memory-utilization 0.97 \
    --max-model-len 8192
```

### Gemma 4 31B (GPTQ 4-bit)

Google's latest dense 31B model. At ~17.9GB in GPTQ-4bit, it fits in 32GB VRAM with context up to ~4-8K depending on `gpu-memory-utilization`.

```bash
# Download model
huggingface-cli download ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ --local-dir /models/gemma-4-31B-it-4bit-W4A16-GPTQ

# Serve
docker run --rm -it \
  --device /dev/dri:/dev/dri \
  --group-add render --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v /models:/models:ro \
  -p 8000:8000 \
  vllm-xpu:0.19.0 \
  vllm serve /models/gemma-4-31B-it-4bit-W4A16-GPTQ \
    --served-model-name gemma-4-31b \
    --host 0.0.0.0 --port 8000 \
    --dtype auto --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096
```

### Qwen3-30B-A3B (MoE, GPTQ 4-bit) — proven stable fallback

The original model this fork was tested against. Reliable on B70 with EAGLE3.

```bash
# Download model and EAGLE3 draft model
huggingface-cli download btbtyler09/Qwen3-30B-A3B-Instruct-2507-gptq-4bit --local-dir /models/Qwen3-30B-A3B-Instruct-2507-gptq-4bit
huggingface-cli download lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex --local-dir /models/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex

# Serve with EAGLE3 speculative decoding
docker run --rm -it \
  --device /dev/dri:/dev/dri \
  --group-add render --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v /models:/models:ro \
  -p 8000:8000 \
  vllm-xpu:0.19.0 \
  vllm serve /models/Qwen3-30B-A3B-Instruct-2507-gptq-4bit \
    --served-model-name qwen3-30b \
    --host 0.0.0.0 --port 8000 \
    --dtype auto --trust-remote-code \
    --gpu-memory-utilization 0.97 \
    --max-model-len 8192 \
    --speculative-config '{"method":"eagle3","model":"/models/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex","num_speculative_tokens":3,"draft_tensor_parallel_size":1}'
```

## Docker Compose

For a full setup with Open WebUI and SearXNG, use a `.env` file to swap models:

```yaml
# docker-compose.yml
services:
  vllm:
    image: vllm-xpu:0.19.0
    container_name: vllm-xpu
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - render
      - video
    security_opt:
      - seccomp=unconfined
    ipc: host
    volumes:
      - /models:/models:ro
    ports:
      - "8000:8000"
    environment:
      - VLLM_TARGET_DEVICE=xpu
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
    entrypoint: ["vllm", "serve"]
    command:
      - "${VLLM_MODEL}"
      - --served-model-name
      - "${VLLM_MODEL_ALIAS:-model}"
      - --host
      - "0.0.0.0"
      - --port
      - "8000"
      - --max-model-len
      - "${VLLM_MAX_MODEL_LEN:-8192}"
      - --gpu-memory-utilization
      - "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      - --dtype
      - auto
      - --trust-remote-code
    restart: unless-stopped
```

```bash
# .env — switch models by editing this file and restarting
VLLM_MODEL=/models/Qwen3.5-35B-A3B-GPTQ-Int4
VLLM_MODEL_ALIAS=qwen3.5-35b
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTILIZATION=0.97
```

## Test it

```bash
# Check the model loaded
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "What is 7 * 8?"}],
    "max_tokens": 64
  }'
```

## Hardware tested

- **GPU:** Intel Arc Pro B70 (Battlemage G31, 32GB VRAM, PCI ID `8086:e223`)
- **Host OS:** Ubuntu 25.10
- **Container base:** `intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04`

## Upstream

- Forked from [vllm-project/vllm](https://github.com/vllm-project/vllm) at tag `v0.19.0`
- Bug report: [vllm-project/vllm#39474](https://github.com/vllm-project/vllm/issues/39474)
