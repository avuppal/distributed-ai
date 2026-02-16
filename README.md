# Distributed AI Training Simulator

**Project 3: Scaling AI Across Machines**

This repository simulates distributed training of AI models across multiple nodes (processes). It demonstrates key enterprise concepts like data parallelism, gradient synchronization, and fault tolerance—critical for training large models on clusters.

## Why This Project?
- **Enterprise Relevance:** Real AI (e.g., LLMs) trains on 100s of GPUs. This shows how.
- **Key Learning:** Understand networking bottlenecks (e.g., AllReduce over Ethernet) and why bandwidth > compute at scale.
- **RHEL-Friendly:** Runs on your server (CPU mode first; add GPUs later).

## Getting Started

### Prerequisites
- Python 3.8+ with PyTorch: `pip install torch torchvision torchaudio`
- For distributed: No extras needed (uses built-in `torch.distributed`).

### Running the Simulator
```bash
# Single process (baseline)
python src/distributed_trainer.py --world-size 1

# Simulate 4 nodes (processes) on one machine
torchrun --nproc_per_node=4 src/distributed_trainer.py
```

This trains a simple CNN on MNIST, syncing gradients across "nodes."

## Concepts Implemented
- **Data Parallelism:** Split dataset and average gradients (using `DistributedDataParallel`).
- **Backend:** Gloo (CPU-friendly) for simulation.
- **Benchmark:** Compare speedup vs. single-process.
- **Stretch:** Add Prometheus logging (link to Project 1) and fault injection.

Contributions welcome! 🚀
