import os
import subprocess
import re

# ==========================================
#  USER CONFIGURATION (Edit these values)
# ==========================================
# Model Architecture: Llama-2-7b
SEQ_LEN = 4096
GLOBAL_BATCH_SIZE = 128  # Try 128, 64, etc.
MICRO_BATCH_SIZE = 1     # Start with 1 to avoid OOM, increase if memory allows

# Parallelism Configuration (4 GPUs Total)
# Ensure TP * PP * CP <= Number of GPUs (4)
TP_SIZE = 1   # Tensor Parallelism, better for a larger models
PP_SIZE = 1   # Pipeline Parallelism, better when bubble time to compute ratio is not poor (many GPUs with fast lainks) or batch size is large
CP_SIZE = 2   # Context Parallelism, better for long sequences
# DP_SIZE is calculated automatically: NUM_GPUS / (TP * PP * CP)

# Experiment Settings
# MAX_STEPS = 50           # Run enough steps to stabilize throughput
MAX_STEPS = 20          
NUM_GPUS = 4
PRECISION = "bf16-mixed" # A100s should use bf16

# ==========================================
#  AUTOMATED COMMAND BUILDER
# ==========================================

def run_experiment():
    # Calculate Gradient Accumulation to hit Global Batch Size
    # GBS = MBS * DP_SIZE * GRAD_ACC
    # Therefore: GRAD_ACC = GBS / (MBS * DP_SIZE)
    
    dp_size = NUM_GPUS / (TP_SIZE * PP_SIZE * CP_SIZE)
    if dp_size < 1 or not dp_size.is_integer():
        print(f"ERROR: Invalid Parallelism! TP({TP_SIZE}) * PP({PP_SIZE}) * CP({CP_SIZE}) > Total GPUs({NUM_GPUS})")
        return

    # grad_acc = GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * dp_size)
    
    # if not grad_acc.is_integer():
    #     print(f"WARNING: Global Batch Size ({GLOBAL_BATCH_SIZE}) is not divisible by (MBS * DP_SIZE).")
    #     print(f"Adjusting GBS to match closest valid number.")
    #     grad_acc = int(grad_acc)
    #     # Update GBS to reflect reality
    #     effective_gbs = int(grad_acc * MICRO_BATCH_SIZE * dp_size)
    # else:
    #     grad_acc = int(grad_acc)
    #     effective_gbs = GLOBAL_BATCH_SIZE

    # Calculate expected internal accumulation just for logging/verification
    # NeMo calculates this internally, we just print it for sanity check
    grad_acc_steps = GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * dp_size)

    print(f"""
    Starting Experiment...
    ------------------------------------------
    GPUs        : {NUM_GPUS}
    Sequence Len: {SEQ_LEN}
    Micro Batch : {MICRO_BATCH_SIZE}
    ------------------------------------------
    Parallelism : TP={TP_SIZE}, PP={PP_SIZE}, CP={CP_SIZE}, DP={int(dp_size)}
    Internal Acc Steps: {grad_acc_steps} (Managed by NeMo)
    ------------------------------------------
    """)

    # Base NeMo Command
    # We use a generic GPT config and override it to match Llama 2 7B architecture
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        "/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py",
        
        # Architecture Overrides for Llama 2 7B
        "--config-name=megatron_gpt_config",
        "model.mcore_gpt=True",
        "model.transformer_engine=True",
        f"model.encoder_seq_length={SEQ_LEN}",
        "model.num_layers=32",
        "model.hidden_size=4096",
        "model.ffn_hidden_size=11008",
        "model.num_attention_heads=32",
        "model.normalization=RMSNorm",
        "model.activation=fast-swiglu",
        "model.position_embedding_type=rope", # Rotary Embeddings
        
        # Data (Mock/Synthetic for Benchmarking)
        "model.data.data_impl=mock",
        "model.data.data_prefix=[]",
        
        # --- MEMORY OPTIMIZATIONS (CRITICAL FIXES) ---
        # 1. Distributed Optimizer (Shards optimizer state across DP ranks)
        "model.optim.name=distributed_fused_adam",
        
        # 2. Activation Checkpointing (Saves memory by recomputing activations)
        "model.activations_checkpoint_granularity=selective", 
        "model.activations_checkpoint_method=uniform",
        # ---------------------------------------------
        
        # Training Loop
        f"trainer.max_steps={MAX_STEPS}",
        f"trainer.val_check_interval={MAX_STEPS}", # Disable validation during bench
        "trainer.limit_val_batches=0",
        "trainer.log_every_n_steps=1",
        "+trainer.enable_progress_bar=True",  # Enable progress bar

        # Match Trainer devices to Torchrun processes
        f"trainer.devices={NUM_GPUS}",
        "trainer.num_nodes=1",
        "trainer.accelerator=gpu",
        
        # Parallelism Strategy
        f"model.tensor_model_parallel_size={TP_SIZE}",
        f"model.pipeline_model_parallel_size={PP_SIZE}",
        f"+model.context_parallel_size={CP_SIZE}",
        
        # Batch Sizes
        # f"model.global_batch_size={effective_gbs}",
        f"model.global_batch_size={GLOBAL_BATCH_SIZE}",
        f"model.micro_batch_size={MICRO_BATCH_SIZE}",
        # f"trainer.accumulate_grad_batches={grad_acc}",
        "trainer.accumulate_grad_batches=1",
        
        # Optimization
        f"trainer.precision={PRECISION}",
        "model.use_flash_attention=True",
        
        # Logging
        "exp_manager.create_checkpoint_callback=False", # Don't save checkpoints
        "exp_manager.name=llama2_bench",
        "exp_manager.resume_if_exists=False",

        "++model.dist_ckpt_format=torch_dist",
        "++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=False",

    ]
    
    # Run and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    print("\n[Streaming Output and Parsing Metrics...]\n")
    
    timing_pattern = re.compile(r"train_step_timing in s=([\d\.]+)")

    for line in process.stdout:
        print(line, end='') # Print live log
        
        match = timing_pattern.search(line)
        if match:
            try:
                step_time_seconds = float(match.group(1))
                
                if step_time_seconds > 0:
                    total_tokens_per_step = GLOBAL_BATCH_SIZE * SEQ_LEN
                    total_throughput = total_tokens_per_step / step_time_seconds
                    tokens_per_sec_per_gpu = total_throughput / NUM_GPUS
                    
                    # Print in Green
                    print(f"\033[92m") 
                    print(f">>> PARSED: {step_time_seconds:.2f}s/step | {total_throughput:,.0f} tokens/s | {tokens_per_sec_per_gpu:,.0f} tokens/s/GPU")
                    print(f"\033[0m") 
            except ValueError:
                pass

    process.wait()

    process.wait()


if __name__ == "__main__":
    run_experiment()