#!/bin/bash

# This script now serves as a wrapper that can run either:
# 1. The original vllm bench throughput command (without TTFT/TPOT)
# 2. The Python-based latency benchmark (with TTFT/TPOT)

# Uncomment the method you want to use:

# Method 1: Run Python script with accurate TTFT/TPOT measurements
# python3 "$(dirname "$0")/run_benchmark_vllm_latency.py"

# Method 2: Original vllm bench throughput (uncomment lines below and comment line above)
# Save original CUDA_VISIBLE_DEVICES
ORIGINAL_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# Output CSV file
OUTPUT_CSV="vllm_benchmark_results_$(date +%Y%m%d_%H%M%S).csv"

# Create CSV header
echo "timestamp,model_name,tensor_parallel_size,cuda_devices,input_len,output_len,num_prompts,requests_per_sec,total_tokens_per_sec,output_tokens_per_sec" > "$OUTPUT_CSV"

# Model configurations
declare -a MODELS=("opt-13b" "opt-30b" "opt-66b")
MODEL_BASE_PATH="/data/pretrained_models/hf_models"

# Benchmark parameters
INPUT_LEN=1024
OUTPUT_LEN=128
NUM_PROMPTS=10
DTYPE="bfloat16"
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=1

# Function to run benchmark and parse output
run_benchmark() {
    local model_path=$1
    local model_name=$2
    local tp_size=$3
    local cuda_devices=$4
    
    echo "Running benchmark for $model_name with TP=$tp_size, CUDA_DEVICES=$cuda_devices"
    
    # Run vllm benchmark and capture output
    output=$(vllm bench throughput \
        --model "$model_path" \
        --tensor-parallel-size "$tp_size" \
        --enforce-eager \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" 2>&1)
    
    echo "$output"
    
    # Extract metrics from output - looking for pattern: "Throughput: X requests/s, Y total tokens/s, Z output tokens/s"
    requests_per_sec=$(echo "$output" | grep -oP 'Throughput:\s+\K[\d.]+(?=\s+requests/s)' || echo "N/A")
    total_tokens_per_sec=$(echo "$output" | grep -oP '[\d.]+(?=\s+total tokens/s)' || echo "N/A")
    output_tokens_per_sec=$(echo "$output" | grep -oP '[\d.]+(?=\s+output tokens/s)' || echo "N/A")
    
    # Write to CSV
    timestamp=$(date +%Y-%m-%d_%H:%M:%S)
    echo "$timestamp,$model_name,$tp_size,$cuda_devices,$INPUT_LEN,$OUTPUT_LEN,$NUM_PROMPTS,$requests_per_sec,$total_tokens_per_sec,$output_tokens_per_sec" >> "$OUTPUT_CSV"
    
    echo "-------------------------------------------"
}

# Main benchmark loop
for model in "${MODELS[@]}"; do
    model_path="$MODEL_BASE_PATH/$model"
    
    echo "=========================================="
    echo "Benchmarking model: $model"
    echo "=========================================="
    
    # Experiment 1: TP=2 with default GPUs
    unset CUDA_VISIBLE_DEVICES
    run_benchmark "$model_path" "$model" 2 "default"
    
    # Experiment 2: TP=4 with default GPUs
    unset CUDA_VISIBLE_DEVICES
    run_benchmark "$model_path" "$model" 4 "default"
    
    # Experiment 3: TP=2 with selected GPUs (1,2)
    export CUDA_VISIBLE_DEVICES=1,2
    run_benchmark "$model_path" "$model" 2 "1,2"
    
    echo ""
done

# Restore original CUDA_VISIBLE_DEVICES
if [ -z "$ORIGINAL_CUDA_DEVICES" ]; then
    unset CUDA_VISIBLE_DEVICES
else
    export CUDA_VISIBLE_DEVICES="$ORIGINAL_CUDA_DEVICES"
fi

echo "=========================================="
echo "Benchmarking complete!"
echo "Results saved to: $OUTPUT_CSV"
echo "CUDA_VISIBLE_DEVICES restored to: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=========================================="
