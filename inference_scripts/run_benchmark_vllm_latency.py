#!/usr/bin/env python3
import os
import sys
import time
import csv
from datetime import datetime
from typing import List, Dict
import argparse

from vllm import LLM, SamplingParams

def measure_latency(
    model_path: str,
    tensor_parallel_size: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    dtype: str = "bfloat16",
    max_model_len: int = 2048,
    enforce_eager: bool = True,
    max_num_seqs: int = 1
) -> Dict[str, float]:
    """Measure TTFT and TPOT for a given configuration."""
    
    # Create dummy input tokens
    prompt = "test " * input_len
    
    # Initialize LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.9
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True
    )
    
    # Warmup
    print("Warming up...")
    _ = llm.generate([prompt], sampling_params)
    
    # Actual measurements
    print(f"Running {num_prompts} benchmark iterations...")
    ttft_list = []
    tpot_list = []
    
    for i in range(num_prompts):
        start_time = time.perf_counter()
        
        # Track first token time
        first_token_time = None
        outputs = llm.generate([prompt], sampling_params)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Extract timing information
        output = outputs[0]
        num_output_tokens = len(output.outputs[0].token_ids)
        
        # TTFT approximation: assume uniform token generation if not available
        # For more accurate TTFT, we need streaming or custom callback
        estimated_ttft = total_time / (num_output_tokens + 1) if num_output_tokens > 0 else total_time
        
        # TPOT: time per output token (excluding first token)
        if num_output_tokens > 1:
            tpot = (total_time - estimated_ttft) / (num_output_tokens - 1)
        else:
            tpot = total_time / max(num_output_tokens, 1)
        
        ttft_list.append(estimated_ttft * 1000)  # Convert to ms
        tpot_list.append(tpot * 1000)  # Convert to ms
        
        print(f"  Iteration {i+1}/{num_prompts}: TTFT={estimated_ttft*1000:.2f}ms, TPOT={tpot*1000:.2f}ms")
    
    # Calculate averages
    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_tpot = sum(tpot_list) / len(tpot_list)
    
    # Calculate throughput metrics
    avg_total_time = (avg_ttft + avg_tpot * (output_len - 1)) / 1000  # Convert back to seconds
    requests_per_sec = 1 / avg_total_time if avg_total_time > 0 else 0
    total_tokens = input_len + output_len
    total_tokens_per_sec = total_tokens / avg_total_time if avg_total_time > 0 else 0
    output_tokens_per_sec = output_len / avg_total_time if avg_total_time > 0 else 0
    
    return {
        "ttft_ms": avg_ttft,
        "tpot_ms": avg_tpot,
        "requests_per_sec": requests_per_sec,
        "total_tokens_per_sec": total_tokens_per_sec,
        "output_tokens_per_sec": output_tokens_per_sec
    }

def main():
    # Configuration
    models = ["opt-13b", "opt-30b", "opt-66b"]
    model_base_path = "/data/pretrained_models/hf_models"
    
    # Benchmark parameters
    input_len = 1024
    output_len = 128
    num_prompts = 10
    dtype = "bfloat16"
    max_model_len = 2048
    max_num_seqs = 1
    
    # Output CSV
    output_csv = f"vllm_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Save original CUDA_VISIBLE_DEVICES
    original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    # Create CSV and write header
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            "timestamp", "model_name", "tensor_parallel_size", "cuda_devices",
            "input_len", "output_len", "num_prompts", "requests_per_sec",
            "total_tokens_per_sec", "output_tokens_per_sec", "ttft_ms", "tpot_ms"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Benchmark experiments
        experiments = [
            {"tp_size": 2, "cuda_devices": None, "cuda_label": "default"},
            {"tp_size": 4, "cuda_devices": None, "cuda_label": "default"},
            {"tp_size": 2, "cuda_devices": "1,2", "cuda_label": "1,2"},
        ]
        
        for model_name in models:
            model_path = f"{model_base_path}/{model_name}"
            print("=" * 50)
            print(f"Benchmarking model: {model_name}")
            print("=" * 50)
            
            for exp in experiments:
                # Set CUDA_VISIBLE_DEVICES
                if exp["cuda_devices"] is None:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = exp["cuda_devices"]
                
                print(f"\nRunning: TP={exp['tp_size']}, CUDA_DEVICES={exp['cuda_label']}")
                
                try:
                    metrics = measure_latency(
                        model_path=model_path,
                        tensor_parallel_size=exp["tp_size"],
                        input_len=input_len,
                        output_len=output_len,
                        num_prompts=num_prompts,
                        dtype=dtype,
                        max_model_len=max_model_len,
                        enforce_eager=True,
                        max_num_seqs=max_num_seqs
                    )
                    
                    # Write results
                    row = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                        "model_name": model_name,
                        "tensor_parallel_size": exp["tp_size"],
                        "cuda_devices": exp["cuda_label"],
                        "input_len": input_len,
                        "output_len": output_len,
                        "num_prompts": num_prompts,
                        "requests_per_sec": f"{metrics['requests_per_sec']:.4f}",
                        "total_tokens_per_sec": f"{metrics['total_tokens_per_sec']:.2f}",
                        "output_tokens_per_sec": f"{metrics['output_tokens_per_sec']:.2f}",
                        "ttft_ms": f"{metrics['ttft_ms']:.2f}",
                        "tpot_ms": f"{metrics['tpot_ms']:.2f}"
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    
                    print(f"\nResults:")
                    print(f"  TTFT: {metrics['ttft_ms']:.2f} ms")
                    print(f"  TPOT: {metrics['tpot_ms']:.2f} ms")
                    print(f"  Throughput: {metrics['requests_per_sec']:.4f} req/s")
                    
                except Exception as e:
                    print(f"Error during benchmark: {e}")
                    import traceback
                    traceback.print_exc()
                
                print("-" * 50)
    
    # Restore original CUDA_VISIBLE_DEVICES
    if original_cuda_devices is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
    
    print("=" * 50)
    print("Benchmarking complete!")
    print(f"Results saved to: {output_csv}")
    print("=" * 50)

if __name__ == "__main__":
    main()
