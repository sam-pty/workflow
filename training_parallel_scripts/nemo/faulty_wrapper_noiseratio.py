'''
 # @ Author: Saptarshi Mitra
 # @ Create Time: 2025-12-09 00:42:43
 # @ Modified by: 
 # @ Modified time: 2025-12-09 00:42:45
 # @ Description:
 '''

import sys
import os
import torch
import torch.distributed as dist

# ==========================================
#  FAULT INJECTION CONFIGURATION
# ==========================================
# Topology Config
TARGET_SRC = 0
TARGET_DST = 1       # 1 for NVLink, 2 for PCIe

# Penalty Config
# N = 0.5 means for every 100MB of gradients, we inject 50MB of noise.
# N = 1.0 means for every 100MB of gradients, we inject 100MB of noise (50% effective BW).
PENALTY_FACTOR = 1.0

_original_all_reduce = dist.all_reduce
_original_all_gather = dist.all_gather
_original_reduce_scatter = dist.reduce_scatter
_original_all_to_all = dist.all_to_all

# Global buffer to avoid re-allocating memory every step
_noise_buffer = None

def get_noise_buffer(required_bytes, device):
    """
    Returns a slice of a pre-allocated buffer. 
    Resizes the global buffer if the requirement exceeds current capacity.
    """
    global _noise_buffer
    
    # 1. Initialize if empty
    if _noise_buffer is None:
        # Start with a reasonable default (e.g., 200MB) to reduce initial resizing
        _noise_buffer = torch.zeros(200 * 1024 * 1024, dtype=torch.uint8, device=device)
    
    # 2. Resize if too small
    if _noise_buffer.numel() < required_bytes:
        # Delete old buffer to free memory immediately
        del _noise_buffer
        # Allocate new larger buffer
        _noise_buffer = torch.zeros(required_bytes, dtype=torch.uint8, device=device)
        
    # 3. Return a view of the required size
    return _noise_buffer[:required_bytes]

def faulty_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if dist.is_initialized():
        rank = dist.get_rank()
        
        try:
            # --- CALCULATE RELATIVE NOISE SIZE ---
            # bytes = num_elements * bytes_per_element
            payload_bytes = tensor.numel() * tensor.element_size()
            noise_bytes = int(payload_bytes * PENALTY_FACTOR)
            
            # Only inject if there is actual data to penalize
            if noise_bytes > 0:
                if rank == TARGET_SRC:
                    noise = get_noise_buffer(noise_bytes, f'cuda:{rank}')
                    dist.send(noise, TARGET_DST)
                    # Force the stream to wait until send is done
                    torch.cuda.synchronize()
                    
                elif rank == TARGET_DST:
                    noise = get_noise_buffer(noise_bytes, f'cuda:{rank}')
                    dist.recv(noise, TARGET_SRC)
                    # Force the stream to wait until recv is done
                    torch.cuda.synchronize()

        except Exception as e:
            # Fallback for edge cases (e.g., uninitialized groups)
            pass

    # Proceed with the actual collective
    return _original_all_reduce(tensor, op, group, async_op)

def faulty_all_gather(tensor_list, tensor, group=None, async_op=False):
    if dist.is_initialized():
        rank = dist.get_rank()
        
        try:
            # Calculate payload bytes from input tensor
            payload_bytes = tensor.numel() * tensor.element_size()
            noise_bytes = int(payload_bytes * PENALTY_FACTOR)
            
            if noise_bytes > 0:
                if rank == TARGET_SRC:
                    noise = get_noise_buffer(noise_bytes, f'cuda:{rank}')
                    dist.send(noise, TARGET_DST)
                    torch.cuda.synchronize()
                    
                elif rank == TARGET_DST:
                    noise = get_noise_buffer(noise_bytes, f'cuda:{rank}')
                    dist.recv(noise, TARGET_SRC)
                    torch.cuda.synchronize()

        except Exception as e:
            pass

    return _original_all_gather(tensor_list, tensor, group, async_op)

def faulty_reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if dist.is_initialized():
        rank = dist.get_rank()
        
        try:
            # Calculate payload bytes from output tensor
            payload_bytes = output.numel() * output.element_size()
            noise_bytes = int(payload_bytes * PENALTY_FACTOR)
            
            if noise_bytes > 0:
                if rank == TARGET_SRC:
                    noise = get_noise_buffer(noise_bytes, f'cuda:{rank}')
                    dist.send(noise, TARGET_DST)
                    torch.cuda.synchronize()
                    
                elif rank == TARGET_DST:
                    noise = get_noise_buffer(noise_bytes, f'cuda:{rank}')
                    dist.recv(noise, TARGET_SRC)
                    torch.cuda.synchronize()

        except Exception as e:
            pass

    return _original_reduce_scatter(output, input_list, op, group, async_op)

def faulty_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    if dist.is_initialized():
        rank = dist.get_rank()
        
        try:
            # Calculate payload bytes from input tensor list
            payload_bytes = sum(t.numel() * t.element_size() for t in input_tensor_list)
            noise_bytes = int(payload_bytes * PENALTY_FACTOR)
            
            if noise_bytes > 0:
                if rank == TARGET_SRC:
                    noise = get_noise_buffer(noise_bytes, f'cuda:{rank}')
                    dist.send(noise, TARGET_DST)
                    torch.cuda.synchronize()
                    
                elif rank == TARGET_DST:
                    noise = get_noise_buffer(noise_bytes, f'cuda:{rank}')
                    dist.recv(noise, TARGET_SRC)
                    torch.cuda.synchronize()

        except Exception as e:
            pass

    return _original_all_to_all(output_tensor_list, input_tensor_list, group, async_op)

# INSTALL THE PATCHES
dist.all_reduce = faulty_all_reduce
dist.all_gather = faulty_all_gather
dist.reduce_scatter = faulty_reduce_scatter
dist.all_to_all = faulty_all_to_all
print(f"[Wrapper] Relative Fault Injection Active: Penalty Factor {PENALTY_FACTOR}x on Rank {TARGET_SRC}<->{TARGET_DST}")
print(f"[Wrapper] Patched Operations: all_reduce, all_gather, reduce_scatter, all_to_all")

# ==========================================
#  DELEGATE TO NEMO
# ==========================================
nemo_path = "/opt/NeMo/examples/nlp/language_modeling"
sys.path.append(nemo_path)

try:
    from megatron_gpt_pretraining import main
except ImportError:
    print(f"CRITICAL ERROR: Could not find megatron_gpt_pretraining.py at {nemo_path}")
    sys.exit(1)

if __name__ == '__main__':
    main()