from dataclasses import dataclass
import os
import json
import random
import subprocess
import re
from threading import Thread
from tensorrt_llm import LLM
from huggingface_hub import snapshot_download

SNAPSHOTS_FOLDER = "/data/pretrained_models/hf_snapshots" # "./hf_snapshots"
MODELS_FOLDER = "/data/pretrained_models/hf_models" #./models
DATASETS_FOLDER = "/data/dataset" #./datasets

HF_MODELS = {
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
    "Llama-3.1-70B": "unsloth/Meta-Llama-3.1-70B", # not recommended for low-memory setups
    # add more models as needed (if TensorRT-LLM already supports it!)
}

@dataclass
class BenchmarkConfig:
    model: str
    tp: int = 1
    pp: int = 1
    input_length: int = 16
    seq_length: int = 64
    batch_size: int = 8

def get_snapshot(model_id: str):
    snapshot_path = os.path.join(SNAPSHOTS_FOLDER, model_id.replace("/", "_"))
    if not os.path.exists(snapshot_path):
        print(f"Downloading model snapshot for {model_id}...")
        snapshot_download(
            repo_id=model_id, 
            local_dir=snapshot_path,
            allow_patterns=["*.json", "*.model", "*.txt"],  # Only configs & tokenizers
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth"], # Block weights
            local_dir_use_symlinks=False
        )
    return snapshot_path

# prepare TensorRT LLM engines for models of interest
def build_model(model: str, tp: int = 1, pp: int = 1):
    checkpoint_dir = os.path.join(MODELS_FOLDER, model, f"tp{tp}_pp{pp}")
    if not os.path.exists(checkpoint_dir):
        llm = LLM(
            model=get_snapshot(HF_MODELS[model]),
            # workspace="./tmp",
            fast_build=True,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            load_format='dummy', # use dummy weights for faster download
            enable_tqdm=True,
        )
        llm.save(checkpoint_dir)
    return checkpoint_dir

# generate prompts: vocab size (infer from model config), input length, sequence length, batch size
def generate_dataset(model: str, input_length: int, seq_length: int, batch_size: int):
    dataset_file = os.path.join(DATASETS_FOLDER, model, f"in{input_length}_seq{seq_length}_bs{batch_size}.txt")
    if not os.path.exists(dataset_file):
        # find any directory in MODELS_FOLDER/model 
        checkpoint_dir = None
        model_folder = os.path.join(MODELS_FOLDER, model)
        if os.path.exists(model_folder):
            for entry in os.listdir(model_folder):
                entry_path = os.path.join(model_folder, entry)
                if os.path.isdir(entry_path):
                    checkpoint_dir = entry_path
                    break
        else:
            raise FileNotFoundError(f"Model folder {model_folder} does not exist.")
        if checkpoint_dir is None:
            raise FileNotFoundError(f"No checkpoint directory found for model {model} in {MODELS_FOLDER}.")

        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found at {config_path}.")

        with open(config_path, "r") as f:
            config = json.load(f)
            vocab_size = config.get("pretrained_config", {}).get("vocab_size", None)
            if vocab_size is None:
                raise ValueError("Vocab size not found in model config.")
            build_config = config.get("build_config", {})
            max_input_length = build_config.get("max_input_length", None)
            if max_input_length is not None and input_length > max_input_length:
                raise ValueError(f"Input length {input_length} exceeds model's max input length {max_input_length}.")
            max_seq_length = build_config.get("max_sequence_length", None)
            if max_seq_length is not None and seq_length > max_seq_length:
                raise ValueError(f"Sequence length {seq_length} exceeds model's max sequence length {max_seq_length}.")
            max_batch_size = build_config.get("max_batch_size", None)
            if max_batch_size is not None and batch_size > max_batch_size:
                raise ValueError(f"Batch size {batch_size} exceeds model's max batch size {max_batch_size}.")
        
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        output_length = seq_length - input_length
        with open(dataset_file, "w") as f:
            for id in range(batch_size):
                input_logits = []
                for i in range(input_length):
                    input_logits.append(random.randint(0, vocab_size - 1))
                # dataset format: {"task_id":id,"input_ids":[...],"output_tokens":output_length}
                request_str = f"{{\"task_id\":{id},\"input_ids\":[{','.join(map(str, input_logits))}],\"output_tokens\":{output_length}}}\n"
                f.write(request_str)

        print(f"Generated dataset at {dataset_file}.")

    return dataset_file

# run trtllm-bench throughput with dataset and report token throughput
# def run_benchmark(config: BenchmarkConfig):
#     engine_dir = build_model(config.model, config.tp, config.pp)
#     dataset_file = generate_dataset(config.model, config.input_length, config.seq_length, config.batch_size)

#     command = f"trtllm-bench --model {HF_MODELS[config.model]} throughput \
#         --dataset {dataset_file} \
#         --engine_dir {engine_dir}"
    
#     print(f"Running benchmark command: {command}")
    
#     # Capture stdout and stderr without printing
#     result = subprocess.run(
#         command,
#         shell=True,
#         capture_output=True,
#         text=True
#     )
    
#     # Parse token throughput from output
#     token_throughput = None
#     for line in result.stdout.split('\n'):
#         # Match line like "Token Throughput (tokens/sec):  131.6885"
#         match = re.search(r'Token Throughput \(tokens/sec\):\s+([\d.]+)', line)
#         if match:
#             token_throughput = float(match.group(1))
#             print(f"Extracted Token Throughput: {token_throughput} tokens/sec")
#             break
    
#     if token_throughput is None:
#         print("Warning: Could not parse token throughput from output")
#         # Optionally print stderr for debugging if parsing fails
#         if result.stderr:
#             print("STDERR:", result.stderr)
    
#     return token_throughput


def run_benchmarks(configs: list[BenchmarkConfig]) -> list[tuple[BenchmarkConfig, float]]:
    """
    Run multiple benchmarks with pipelined execution.
    While a benchmark is running, the next model is being built to maximize throughput.
    Despite this, initial engine builds (for each model and parallelism mode) still dominate the benchmark runtime.
    """
    results = []
    
    if len(configs) == 0:
        return results
    
    # Pre-build the first model
    first_config = configs[0]
    print(f"Pre-building first model: {first_config.model}")
    first_engine_dir = build_model(first_config.model, first_config.tp, first_config.pp)
    first_dataset = generate_dataset(first_config.model, first_config.input_length, 
                                     first_config.seq_length, first_config.batch_size)
    
    for i, config in enumerate(configs):
        # Use pre-built engine and dataset for current config
        if i == 0:
            engine_dir = first_engine_dir
            dataset_file = first_dataset
        else:
            # This was built while previous benchmark was running
            engine_dir = build_model(config.model, config.tp, config.pp)
            dataset_file = generate_dataset(config.model, config.input_length, 
                                           config.seq_length, config.batch_size)
        
        # Start building next model in parallel while current benchmark runs
        next_build_thread = None
        if i + 1 < len(configs):
            next_config = configs[i + 1]
                
            def build_next():
                print(f"Building next model in background: {next_config.model}")
                build_model(next_config.model, next_config.tp, next_config.pp)
                generate_dataset(next_config.model, next_config.input_length, 
                               next_config.seq_length, next_config.batch_size)
            
            next_build_thread = Thread(target=build_next)
            next_build_thread.start()
        
        # Run current benchmark
        command = f"trtllm-bench --model {HF_MODELS[config.model]} throughput \
            --dataset {dataset_file} \
            --engine_dir {engine_dir}"
        
        print(f"\n{'='*80}")
        print(f"Benchmark {i+1}/{len(configs)}: {config.model}")
        print(f"  TP={config.tp}, PP={config.pp}, Input={config.input_length}, "
              f"Seq={config.seq_length}, BS={config.batch_size}")
        print(f"{'='*80}")
        print(f"Running benchmark command: {command}")
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Parse token throughput from output
        token_throughput = None
        for line in result.stdout.split('\n'):
            match = re.search(r'Token Throughput \(tokens/sec\):\s+([\d.]+)', line)
            if match:
                token_throughput = float(match.group(1))
                print(f"Extracted Token Throughput: {token_throughput} tokens/sec")
                break
        
        if token_throughput is None:
            print("Warning: Could not parse token throughput from output")
            if result.stderr:
                print("STDERR:", result.stderr)
        
        results.append((config, token_throughput))
        
        # Wait for next model build to complete before moving to next iteration
        if next_build_thread is not None:
            next_build_thread.join()
            print(f"Next model build completed")
    
    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    for config, throughput in results:
        print(f"{config.model} (TP={config.tp}, PP={config.pp}, "
              f"In={config.input_length}, Seq={config.seq_length}, BS={config.batch_size}): "
              f"{throughput if throughput else 'FAILED'} tokens/sec")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    os.makedirs(SNAPSHOTS_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(DATASETS_FOLDER, exist_ok=True)


    # Example benchmark run (tweak configurations as needed)
    configs = [
        BenchmarkConfig(
            model="Llama-3.1-8B",
            tp=1, pp=1,
            input_length=16, seq_length=64, batch_size=8
        ),
        BenchmarkConfig(
            model="Llama-3.1-8B",
            tp=1, pp=1,
            input_length=32, seq_length=128, batch_size=16
        ),
        BenchmarkConfig(
            model="Llama-3.1-8B",
            tp=1, pp=1,
            input_length=64, seq_length=256, batch_size=32
        ),
    ]
    
    results = run_benchmarks(configs)
    
    # Process results
    for config, throughput in results:
        if throughput:
            print(f"✓ {config.model}: {throughput:.2f} tokens/sec")
        else:
            print(f"✗ {config.model}: Failed")