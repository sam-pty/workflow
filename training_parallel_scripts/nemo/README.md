# NeMo Llama-2-7B Benchmark Script

This script benchmarks the training throughput of the Llama-2-7B model using NVIDIA's NeMo framework. It allows you to configure parallelism strategies and batch sizes to optimize performance on multi-GPU systems.

## Features
- Supports Tensor, Pipeline, Context, DP and Fully Sharded Data Parallelism (FSDP)
- Configurable batch sizes and sequence length
- Uses synthetic data for benchmarking
- Parses and displays tokens/sec throughput per GPU

## Requirements
- NVIDIA GPUs (A100 recommended)
- CUDA and cuDNN installed
- PyTorch compatible with your CUDA version

## NeMo Installation
To install NeMo using Docker, run the following commands:

```bash
docker pull nvcr.io/nvidia/nemo:24.07

docker run --gpus all -it --rm --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace/experiments \
  nvcr.io/nvidia/nemo:24.07 bash
```


## Usage
1. Edit `run_benchmark.py` to set your desired configuration (batch sizes, parallelism, etc.).
2. Run the benchmark script:
   ```bash
   python run_benchmark.py
   ```
3. The script will launch NeMo's GPT pretraining with Llama-2-7B architecture overrides and print throughput metrics.

## Configuration Options
- `SEQ_LEN`: Sequence length for training
- `GLOBAL_BATCH_SIZE`: Total batch size across all GPUs
- `MICRO_BATCH_SIZE`: Per-GPU micro batch size
- `TP_SIZE`, `PP_SIZE`, `CP_SIZE`: Parallelism settings
- `USE_FSDP`: Boolean flag to enable Fully Sharded Data Parallel (FSDP) instead of TP/PP/CP; when True, uses automatic sharding for memory efficiency, or else uses standard DDP.
- `NUM_GPUS`: Number of GPUs to use
- `PRECISION`: Training precision (e.g., bf16-mixed)

## Output
The script streams training logs and parses step timing to display tokens/sec and tokens/sec/GPU in real time.

## Benchmark Results
Tested on NeMo 24.07, 4×A100 GPUs, Llama-2-7B, bf16, 4096 sequence length, 128 global batch size, 1 micro batch size.

| Parallelism (TP, PP, CP, DP) | Step Time (s) | Token/Sec/GPU | Throughput (token/s) |
|------------------------------|---------------|---------------|----------------------|
| TP=1, PP=1, CP=1, DP=4       | 36.90         | 3,552         | 14,208               |
| TP=2, PP=1, CP=1, DP=2       | 41.70         | 3,143         | 12,573               |
| TP=1, PP=2, CP=1, DP=2       | 38.20         | 3,431         | 13,725               |
| TP=1, PP=1, CP=2, DP=2       | 42.90         | 3,055         | 12,221               |

**Token/Sec/GPU is calculated as:**
```
Token/Sec/GPU = (global_batch_size * seq_len) / (step_time * GPUs)
```

## Notes
- Ensure your system has enough GPU memory for the chosen batch and sequence sizes.
- For real training, replace synthetic data settings with your dataset.

## References
- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/index.html)
