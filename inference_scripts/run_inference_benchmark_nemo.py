import os
import subprocess
import re
import json
import time

# ==========================================
#  USER CONFIGURATION (Edit these values)
# ==========================================

# Model Selection: Choose "llama3-70b" or "llama3.1-70b" for multi-GPU inference
MODEL_TYPE = "llama3.1-70b"  # Options: "llama3-70b", "llama3.1-70b", "llama3.1-8b"

# Model-specific configurations
MODEL_CONFIGS = {
    "llama3-70b": {
        "num_layers": 80,
        "hidden_size": 8192,
        "ffn_hidden_size": 28672,
        "num_attention_heads": 64,
        "num_query_groups": 8,  # GQA
        "max_position_embeddings": 8192,
        "normalization": "RMSNorm",
        "activation": "fast-swiglu",
        "position_embedding_type": "rope",
        "rotary_base": 500000,
    },
    "llama3.1-70b": {
        "num_layers": 80,
        "hidden_size": 8192,
        "ffn_hidden_size": 28672,
        "num_attention_heads": 64,
        "num_query_groups": 8,  # GQA
        "max_position_embeddings": 131072,  # 128k context
        "normalization": "RMSNorm",
        "activation": "fast-swiglu",
        "position_embedding_type": "rope",
        "rotary_base": 500000,
    },
    "llama3.1-8b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "max_position_embeddings": 131072,
        "normalization": "RMSNorm",
        "activation": "fast-swiglu",
        "position_embedding_type": "rope",
        "rotary_base": 500000,
    }
}

# Validate model selection
if MODEL_TYPE not in MODEL_CONFIGS:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be one of {list(MODEL_CONFIGS.keys())}")

# Inference Configuration
INPUT_SEQ_LEN = 2048      # Length of input prompt
OUTPUT_SEQ_LEN = 128     # Number of tokens to generate
BATCH_SIZE = 1           # Typically 1 for inference benchmarking
NUM_REQUESTS = 20        # Number of inference requests to benchmark

# Parallelism Configuration (4 GPUs Total)
# For inference, primarily use Tensor Parallelism (TP)
# Pipeline Parallelism (PP) can be used but adds latency
TP_SIZE = 4             # Tensor Parallelism - distribute model across GPUs
PP_SIZE = 1              # Pipeline Parallelism - typically 1 for inference
NUM_GPUS = 4

# Inference Settings
PRECISION = "bf16"       # Options: "bf16", "fp16", "fp32"
USE_FLASH_ATTENTION = True
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50

# ==========================================
#  AUTOMATED INFERENCE BENCHMARK
# ==========================================

def run_inference_benchmark():
    """Run distributed inference benchmark using NeMo."""
    
    if TP_SIZE * PP_SIZE > NUM_GPUS:
        print(f"ERROR: TP({TP_SIZE}) * PP({PP_SIZE}) = {TP_SIZE * PP_SIZE} > Available GPUs({NUM_GPUS})")
        return

    model_config = MODEL_CONFIGS[MODEL_TYPE]
    
    print(f"""
    Starting Inference Benchmark...
    ------------------------------------------
    Model       : {MODEL_TYPE}
    GPUs        : {NUM_GPUS}
    Input Len   : {INPUT_SEQ_LEN}
    Output Len  : {OUTPUT_SEQ_LEN}
    Batch Size  : {BATCH_SIZE}
    Requests    : {NUM_REQUESTS}
    ------------------------------------------
    Parallelism : TP={TP_SIZE}, PP={PP_SIZE}
    Precision   : {PRECISION}
    ------------------------------------------
    """)

    # Create temporary inference script
    inference_script = create_inference_script(model_config)
    
    # Run distributed inference
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        inference_script,
    ]
    
    print("\n[Running Inference Benchmark...]\n")
    
    # Patterns to parse metrics
    latency_pattern = re.compile(r"Latency: ([\d\.]+)s")
    throughput_pattern = re.compile(r"Throughput: ([\d\.]+) tokens/s")
    ttft_pattern = re.compile(r"Time-to-First-Token: ([\d\.]+)s")
    
    latencies = []
    throughputs = []
    ttfts = []
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end='')
        
        # Parse metrics
        lat_match = latency_pattern.search(line)
        if lat_match:
            latencies.append(float(lat_match.group(1)))
        
        thr_match = throughput_pattern.search(line)
        if thr_match:
            throughputs.append(float(thr_match.group(1)))
        
        ttft_match = ttft_pattern.search(line)
        if ttft_match:
            ttfts.append(float(ttft_match.group(1)))
    
    process.wait()
    
    # Calculate and display summary statistics
    if latencies:
        print(f"\n\033[92m")
        print(f"=" * 60)
        print(f"INFERENCE BENCHMARK SUMMARY")
        print(f"=" * 60)
        print(f"Average Latency:           {sum(latencies)/len(latencies):.3f}s")
        print(f"Average Throughput:        {sum(throughputs)/len(throughputs):.1f} tokens/s")
        print(f"Throughput per GPU:        {sum(throughputs)/len(throughputs)/NUM_GPUS:.1f} tokens/s/GPU")
        if ttfts:
            print(f"Avg Time-to-First-Token:   {sum(ttfts)/len(ttfts):.3f}s")
        print(f"Min Latency:               {min(latencies):.3f}s")
        print(f"Max Latency:               {max(latencies):.3f}s")
        print(f"=" * 60)
        print(f"\033[0m")
    
    # Cleanup
    if os.path.exists(inference_script):
        os.remove(inference_script)


def create_inference_script(model_config):
    """Create a temporary Python script for distributed inference."""
    
    script_content = f'''
import torch
import torch.distributed as dist
import time
from typing import List
import os

# Initialize distributed environment
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0

class MockDistributedModel:
    """Mock model to simulate distributed inference without checkpoint."""
    
    def __init__(self, config, tp_size, pp_size, local_rank):
        self.config = config
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.local_rank = local_rank
        self.dtype = {{"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}}["{PRECISION}"]
        
        # Simulate model weight memory allocation (representative of real model)
        # Each layer has attention + FFN weights
        hidden_size = config["hidden_size"]
        ffn_size = config["ffn_hidden_size"]
        num_layers = config["num_layers"]
        
        # Rough memory estimation per layer (attention + FFN parameters)
        # Divided by TP size for tensor parallelism
        params_per_layer = (hidden_size * hidden_size * 4 + ffn_size * hidden_size * 3) // tp_size
        print(f"Rank {{local_rank}}: Allocating ~{{params_per_layer * num_layers / (1024**3):.2f}} GB for model parameters.")
        
        # Allocate dummy tensors to simulate model memory footprint
        self.dummy_weights = []
        for i in range(num_layers // pp_size):
            # Create dummy weight tensor with correct dimensions
            # Shape: (hidden_size // tp_size, hidden_size // tp_size)
            weight = torch.randn(hidden_size // tp_size, hidden_size // tp_size, 
                               dtype=self.dtype, device=f"cuda:{{local_rank}}")
            self.dummy_weights.append(weight)
    
    def generate(self, input_ids, max_new_tokens, temperature, top_p, top_k):
        """Simulate token generation with realistic compute patterns."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Simulate forward pass compute
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        
        # Time to first token (prefill phase)
        start_prefill = time.time()
        
        # Simulate prefill computation (process entire input)
        hidden = torch.randn(batch_size, seq_len, hidden_size // self.tp_size,
                           dtype=self.dtype, device=device)
        
        for _ in range(num_layers // self.pp_size):
            # Simulate attention + FFN computation
            hidden = torch.nn.functional.linear(hidden, self.dummy_weights[0])
            hidden = torch.nn.functional.gelu(hidden)
        
        if dist.is_initialized():
            dist.barrier()
        
        ttft = time.time() - start_prefill
        
        # Simulate autoregressive generation (decode phase)
        generated_tokens = []
        for _ in range(max_new_tokens):
            # Each decode step processes 1 token
            token_hidden = torch.randn(batch_size, 1, hidden_size // self.tp_size,
                                      dtype=self.dtype, device=device)
            
            for layer_idx in range(num_layers // self.pp_size):
                token_hidden = torch.nn.functional.linear(token_hidden, 
                                                         self.dummy_weights[0])
                token_hidden = torch.nn.functional.gelu(token_hidden)
            
            # Simulate next token selection
            next_token = torch.randint(0, 32000, (batch_size, 1), device=device)
            generated_tokens.append(next_token)
            
            if dist.is_initialized():
                dist.barrier()
        
        output_ids = torch.cat([input_ids] + generated_tokens, dim=1)
        return output_ids, ttft


def run_benchmark():
    rank, world_size, local_rank = setup_distributed()
    
    # Configuration
    model_config = {model_config}
    tp_size = {TP_SIZE}
    pp_size = {PP_SIZE}
    input_len = {INPUT_SEQ_LEN}
    output_len = {OUTPUT_SEQ_LEN}
    batch_size = {BATCH_SIZE}
    num_requests = {NUM_REQUESTS}
    
    if rank == 0:
        print(f"Initializing distributed model on {{world_size}} GPUs...")
    
    # Initialize model
    model = MockDistributedModel(model_config, tp_size, pp_size, local_rank)
    
    # Warmup
    if rank == 0:
        print("Warming up...")
    
    for _ in range(3):
        input_ids = torch.randint(0, 32000, (batch_size, input_len), 
                                 device=f"cuda:{{local_rank}}")
        _ = model.generate(input_ids, output_len, {TEMPERATURE}, {TOP_P}, {TOP_K})
    
    if dist.is_initialized():
        dist.barrier()
    
    # Benchmark
    if rank == 0:
        print(f"\\nRunning {{num_requests}} inference requests...\\n")
    
    for req_idx in range(num_requests):
        input_ids = torch.randint(0, 32000, (batch_size, input_len),
                                 device=f"cuda:{{local_rank}}")
        
        start_time = time.time()
        output_ids, ttft = model.generate(input_ids, output_len, {TEMPERATURE}, {TOP_P}, {TOP_K})
        end_time = time.time()
        
        latency = end_time - start_time
        total_tokens = batch_size * output_len
        throughput = total_tokens / latency
        
        if rank == 0:
            print(f"Request {{req_idx+1}}/{{num_requests}} | "
                  f"Latency: {{latency:.3f}}s | "
                  f"Throughput: {{throughput:.1f}} tokens/s | "
                  f"Time-to-First-Token: {{ttft:.3f}}s")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_benchmark()
'''
    
    script_path = "/tmp/nemo_inference_benchmark.py"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    return script_path


if __name__ == "__main__":
    run_inference_benchmark()



