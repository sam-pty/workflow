import os
import subprocess
import re



# ==========================================
#  USER CONFIGURATION (Edit these values)
# ==========================================

# Model Selection: Choose "llama2-7b" or "llama3.1-8b"
MODEL_TYPE = "llama2-7b"  # Options: "llama2-7b", "llama3.1-8b"

# Model-specific configurations
MODEL_CONFIGS = {
    "llama2-7b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 11008,
        "num_attention_heads": 32,
        "num_query_groups": 32,  # GQA: same as num heads for MHA
        "max_position_embeddings": 4096,
        "normalization": "RMSNorm",
        "activation": "fast-swiglu",
        "position_embedding_type": "rope",
        "rotary_base": 10000,
    },
    "llama3.1-8b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_query_groups": 8,  # GQA: 8 KV heads for Llama 3.1
        "max_position_embeddings": 131072,  # 128k context
        "normalization": "RMSNorm",
        "activation": "fast-swiglu",
        "position_embedding_type": "rope",
        "rotary_base": 500000,  # Extended for long context
    }
}

# Validate model selection
if MODEL_TYPE not in MODEL_CONFIGS:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be one of {list(MODEL_CONFIGS.keys())}")

# Sequence Length Configuration
# For Llama 2 7B: recommended max 4096
# For Llama 3.1 8B: can go up to 128k (131072)
SEQ_LEN = 4096  # Adjust based on model and GPU memory
GLOBAL_BATCH_SIZE = 128  # Try 128, 64, etc.
MICRO_BATCH_SIZE = 1     # Start with 1 to avoid OOM, increase if memory allows

# Parallelism Configuration (4 GPUs Total)
# Ensure TP * PP * CP <= Number of GPUs (4)
TP_SIZE = 2   # Tensor Parallelism, better for a larger models
PP_SIZE = 1   # Pipeline Parallelism, better when bubble time to compute ratio is not poor (many GPUs with fast links) or batch size is large
CP_SIZE = 1   # Context Parallelism, better for long sequences
# DP_SIZE is calculated automatically: NUM_GPUS / (TP * PP * CP)

# Data Parallelism Type
USE_FSDP = False  # Set to True for FSDP, False for DDP (default)

# Experiment Settings
# MAX_STEPS = 50           # Run enough steps to stabilize throughput
MAX_STEPS = 20          
NUM_GPUS = 4
PRECISION = "bf16"  # A100s should use bf16; other options: "bf16-mixed", "fp16-mixed", "fp32", "16-mixed", "32-true"

# ==========================================
#  AUTOMATED COMMAND BUILDER
# ==========================================

def run_experiment():
    # env = os.environ.copy()
    # print(env)
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

    # Get selected model configuration
    model_config = MODEL_CONFIGS[MODEL_TYPE]
    
    # Calculate expected internal accumulation just for logging/verification
    # NeMo calculates this internally, we just print it for sanity check
    grad_acc_steps = GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * dp_size)

    dp_type = "FSDP" if USE_FSDP else "DDP"
    print(f"""
    Starting Experiment...
    ------------------------------------------
    Model       : {MODEL_TYPE}
    GPUs        : {NUM_GPUS}
    Sequence Len: {SEQ_LEN}
    Micro Batch : {MICRO_BATCH_SIZE}
    ------------------------------------------
    Parallelism : TP={TP_SIZE}, PP={PP_SIZE}, CP={CP_SIZE}, DP={int(dp_size)} ({dp_type})
    Internal Acc Steps: {grad_acc_steps} (Managed by NeMo)
    ------------------------------------------
    """)

    # Base NeMo Command
    # We use a generic GPT config and override it to match the selected model architecture
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        # "/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py",
        "faulty_wrapper.py",
        "--config-path=/opt/NeMo/examples/nlp/language_modeling/conf",
        
        # Architecture Overrides - Dynamic based on MODEL_TYPE
        "--config-name=megatron_gpt_config",
        "model.mcore_gpt=True",
        "model.transformer_engine=True",
        f"model.encoder_seq_length={SEQ_LEN}",
        f"model.num_layers={model_config['num_layers']}",
        f"model.hidden_size={model_config['hidden_size']}",
        f"model.ffn_hidden_size={model_config['ffn_hidden_size']}",
        f"model.num_attention_heads={model_config['num_attention_heads']}",
        f"model.num_query_groups={model_config['num_query_groups']}",  # GQA support
        f"model.max_position_embeddings={model_config['max_position_embeddings']}",
        f"model.normalization={model_config['normalization']}",
        f"model.activation={model_config['activation']}",
        f"model.position_embedding_type={model_config['position_embedding_type']}",
        f"++model.rotary_base={model_config['rotary_base']}",  # Use ++ to add new parameter
        
        # Data (Mock/Synthetic for Benchmarking)
        "model.data.data_impl=mock",
        "model.data.data_prefix=[]",
        
        # --- MEMORY OPTIMIZATIONS (CRITICAL FIXES) ---
        # 1. Distributed Optimizer (Shards optimizer state across DP ranks)
        # "model.optim.name=distributed_fused_adam",
        
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

    optim_name = "fused_adam" if USE_FSDP else "distributed_fused_adam"
    cmd.append(f"model.optim.name={optim_name}")
    
    # Add FSDP configuration if enabled
    if USE_FSDP:
        cmd.extend([
            "++model.fsdp=True",  # Enable FSDP
            "++model.fsdp_sharding_strategy=full",  # Options: full, hybrid, grad_op
        ])
    
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