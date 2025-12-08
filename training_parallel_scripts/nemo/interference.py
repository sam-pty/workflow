import torch
import time
import argparse
import sys

def inject_noise(gpu_src, gpu_dst, bandwidth_target_gbps):
    """
    Floods the link between gpu_src and gpu_dst with traffic 
    to simulate a soft fault/bandwidth degradation.
    """
    device_src = torch.device(f"cuda:{gpu_src}")
    device_dst = torch.device(f"cuda:{gpu_dst}")
    
    # Create a large tensor to saturate the link (1GB chunk)
    # Using ByteTensor to maximize throughput volume vs compute
    size_mb = 512
    num_elements = size_mb * 1024 * 1024
    data = torch.randint(0, 255, (num_elements,), dtype=torch.uint8, device=device_src)
    
    print(f"--- FAULT INJECTION STARTED ---")
    print(f"Link: GPU {gpu_src} <-> GPU {gpu_dst}")
    print(f"Payload: {size_mb} MB chunks")
    print(f"Target Throttling: MAX EFFORT (Simulating heavy packet loss)")
    
    total_bytes = 0
    start_time = time.time()
    
    try:
        while True:
            # P2P Copy: This travels over NVLink if available, or PCIe if not.
            # We copy src -> dst
            target_tensor = data.to(device_dst)
            
            # Optional: Copy back to create bi-directional congestion
            _ = target_tensor.to(device_src) 
            
            # Synchronization is required to ensure the bus is actually busy
            torch.cuda.synchronize()
            
            total_bytes += (size_mb * 1024 * 1024)
            
            # Optional: Sleep slightly if you want a specific quantitative bandwidth 
            # rather than "max interference", but for "fault" simulation, 
            # max interference is usually best.
            
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        gbps = (total_bytes * 8 / 1e9) / elapsed
        print(f"\n--- FAULT INJECTION STOPPED ---")
        print(f"Generated Interference Traffic: {gbps:.2f} Gbps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=0, help="Source GPU ID")
    parser.add_argument("--dst", type=int, default=1, help="Destination GPU ID")
    args = parser.parse_args()
    
    inject_noise(args.src, args.dst, 0)