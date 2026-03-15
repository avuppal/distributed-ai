import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import csv

def run_benchmark(rank, world_size, q):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    sizes = [1024**2, 10 * 1024**2, 100 * 1024**2, 500 * 1024**2]
    num_iters = 5
    
    results = []
    
    for size in sizes:
        tensor = torch.randn(size // 4)
        
        # Warmup
        dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
        
        dist.barrier()
        start_time = time.time()
        for _ in range(num_iters):
            t = tensor.clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        dist.barrier()
        end_time = time.time()
        
        avg_latency = ((end_time - start_time) / num_iters) * 1000  # ms
        tensor_size_mb = size / (1024**2)
        bandwidth_gbps = (tensor_size_mb * 8) / (avg_latency) # GBps
        
        if rank == 0:
            results.append({
                "world_size": world_size,
                "tensor_size_mb": tensor_size_mb,
                "latency_ms": avg_latency,
                "bandwidth_gbps": bandwidth_gbps
            })
            
    if rank == 0:
        q.put(results)
        
    dist.destroy_process_group()

def main():
    world_sizes = [2, 4, 8]
    all_results = []
    
    ctx = mp.get_context('spawn')
    
    for ws in world_sizes:
        print(f"Benchmarking world size {ws}...")
        q = ctx.Queue()
        
        processes = []
        for rank in range(ws):
            p = ctx.Process(target=run_benchmark, args=(rank, ws, q))
            p.start()
            processes.append(p)
            
        res = q.get()
        all_results.extend(res)
        
        for p in processes:
            p.join()
            
    print("\n--- Benchmark Results ---")
    print(f"{'World Size':<12} | {'Size (MB)':<12} | {'Latency (ms)':<15} | {'Bandwidth (GB/s)':<15}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['world_size']:<12} | {r['tensor_size_mb']:<12.1f} | {r['latency_ms']:<15.2f} | {r['bandwidth_gbps']:<15.2f}")

    with open("allreduce_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["world_size", "tensor_size_mb", "latency_ms", "bandwidth_gbps"])
        writer.writeheader()
        writer.writerows(all_results)

if __name__ == "__main__":
    main()
