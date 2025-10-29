### Model Benchmarking with Accelerate
This guide explains how to use the benchmark.py script with Hugging Face accelerate to test various multi-GPU parallelism strategies.

### 1. Prerequisites

First, install the required libraries in your Python environment:

```bash
pip install transformers accelerate torch
```

### 2. How to Run & Test Parallelism
The accelerate library allows you to switch between parallelism strategies (like DDP, ZeRO, and FSDP) just by changing a configuration.

You will use `accelerate config` to set your strategy and `accelerate launch` to run the script.

For example:
```
accelerate config
```
```
In which compute environment are you running?
[0] This machine
Please select a choice: 0

Which type of machine are you using?
[0] No distributed training
[1] multi-CPU
[2] multi-GPU
Please select a choice: 2

How many different machines will you use? (1): 1
How many GPUs should be used for distributed training? (1): 2
Do you want to use DeepSpeed? [yes/NO]: NO
Do you want to use FSDP? [yes/NO]: NO
What is your main training process port? (29500): [Press Enter]
Do you want to use bfloat16 (bf16)? [yes/NO]: yes
```

### To test Global Batch Size 64 (32 per GPU):

```bash
accelerate launch benchmark.py --local_batch_size 32 --seq_len 4096
```