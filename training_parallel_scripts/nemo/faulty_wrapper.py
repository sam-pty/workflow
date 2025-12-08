import sys
import os
import torch
import torch.distributed as dist

# ==========================================
#  FAULT INJECTION PATCH
# ==========================================
# Configuration (Match your desired topology)
TARGET_SRC = 0
TARGET_DST = 1       # 1 for NVLink, 2 for PCIe
NOISE_SIZE_MB = 500  # 500MB overhead per step

_original_all_reduce = dist.all_reduce

def faulty_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if dist.is_initialized():
        rank = dist.get_rank()
        try:
            # Create noise tensor once to save alloc time
            if not hasattr(faulty_all_reduce, 'noise_tensor'):
                faulty_all_reduce.noise_tensor = torch.zeros(
                    NOISE_SIZE_MB * 1024 * 1024, 
                    dtype=torch.uint8, 
                    device=f'cuda:{rank}'
                )

            # Blocking P2P Injection
            if rank == TARGET_SRC:
                dist.send(faulty_all_reduce.noise_tensor, TARGET_DST)
            elif rank == TARGET_DST:
                dist.recv(faulty_all_reduce.noise_tensor, TARGET_SRC)
        except Exception:
            pass # Handle init/setup edge cases gracefully

    return _original_all_reduce(tensor, op, group, async_op)

# INSTALL THE PATCH
dist.all_reduce = faulty_all_reduce
print(f"[Wrapper] Fault Injection Active: {NOISE_SIZE_MB}MB noise on Rank {TARGET_SRC}<->{TARGET_DST}")

# ==========================================
#  DELEGATE TO NEMO
# ==========================================
# Add NeMo examples to path so we can import the script directly
nemo_path = "/opt/NeMo/examples/nlp/language_modeling"
sys.path.append(nemo_path)

# Import the actual training script
# This works because we are now IN the training process
try:
    from megatron_gpt_pretraining import main
except ImportError:
    print(f"CRITICAL ERROR: Could not find megatron_gpt_pretraining.py at {nemo_path}")
    sys.exit(1)

if __name__ == '__main__':
    main()