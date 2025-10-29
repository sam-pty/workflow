import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.state import AcceleratorState

# --- Constants ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
WARMUP_STEPS = 5
TIMED_STEPS = 10

def main():
    parser = argparse.ArgumentParser(description="Benchmark a model for one step.")
    parser.add_argument(
        "--local_batch_size", 
        type=int, 
        required=True, 
        help="Batch size per GPU (micro-batch size)"
    )
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=4096, 
        help="Sequence length"
    )
    args = parser.parse_args()

    # 1. Initialize Accelerator
    # This will automatically handle device placement and parallelism
    accelerator = Accelerator()

    if accelerator.state.distributed_type == "DEEPSPEED":
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.local_batch_size
    
    # Global batch size is calculated for you
    global_batch_size = args.local_batch_size * accelerator.num_processes
    
    if accelerator.is_main_process:
        print(f"--- Model Benchmark ---")
        print(f"Model: {MODEL_NAME}")
        print(f"GPUs: {accelerator.num_processes}")
        print(f"Sequence Length: {args.seq_len}")
        print(f"Local Batch Size (per GPU): {args.local_batch_size}")
        print(f"Global Batch Size: {global_batch_size}")
        print(f"Using: {accelerator.state.distributed_type} with {accelerator.state.mixed_precision}")
        print("-----------------------")

    # 2. Load Model and Tokenizer
    # Using bfloat16 for 4090s. 
    # `from_pretrained` will be managed by `accelerate`
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        dtype=torch.bfloat16
    )
    
    vocab_size = model.config.vocab_size

    # 3. Create a Dummy Optimizer
    # We need this for a realistic backward pass and optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 4. Prepare model, optimizer with Accelerator
    # This is where the magic happens (wraps model in DDP, ZeRO, etc.)
    model, optimizer = accelerator.prepare(model, optimizer)

    # 5. Create Dummy Data
    # We create one batch of random data directly on the correct device
    dummy_input_ids = torch.randint(
        0, 
        vocab_size, 
        (args.local_batch_size, args.seq_len), 
        device=accelerator.device
    )
    dummy_labels = dummy_input_ids.clone()

    # --- Benchmarking Loop ---
    step_times = []
    
    for i in range(WARMUP_STEPS + TIMED_STEPS):
        # Wait for all processes to be ready
        accelerator.wait_for_everyone()
        torch.cuda.synchronize() # Synchronize before starting timer
        
        start_time = time.time()

        # Forward pass
        outputs = model(input_ids=dummy_input_ids, labels=dummy_labels)
        loss = outputs.loss

        # Backward pass
        accelerator.backward(loss)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize() # Synchronize after work is done
        end_time = time.time()

        if i >= WARMUP_STEPS:
            step_times.append(end_time - start_time)

    # --- Report Results ---
    if accelerator.is_main_process:
        avg_time = sum(step_times) / len(step_times)
        print("\n--- Results ---")
        print(f"Average time per batch: {avg_time:.4f} seconds")
        print(f"Throughput (samples/sec): {global_batch_size / avg_time:.2f}")

if __name__ == "__main__":
    main()