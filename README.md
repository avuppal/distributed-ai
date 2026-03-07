# Distributed AI Training Simulator

**Project 3: Scaling AI Across Machines**

This repository simulates distributed training of AI models across multiple nodes (processes). It demonstrates key enterprise concepts: data parallelism, gradient synchronization, fault tolerance, and observability — all critical for training large models on clusters.

📐 **[Architecture Deep Dive →](docs/architecture.md)** — Mermaid diagrams covering system topology, Ring-AllReduce, the fault-tolerance state machine, and the observability stack.

## Why This Project?
- **Enterprise Relevance:** Real AI (e.g., LLMs) trains on 100s of GPUs. This shows the mechanics.
- **Key Learning:** Understand networking bottlenecks (AllReduce over Ethernet) and why bandwidth matters more than raw compute at scale.
- **CPU-Friendly:** Runs on any machine using the Gloo backend; plug in GPUs later.

## Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Running the Simulator
```bash
# Single process (baseline)
python src/distributed_trainer.py --world-size 1

# Simulate 4 nodes (processes) on one machine
torchrun --nproc_per_node=4 src/distributed_trainer.py

# Resume from a saved checkpoint
torchrun --nproc_per_node=4 src/distributed_trainer.py --resume
```

This trains a SimpleCNN on MNIST, syncing gradients across "nodes" and exporting
Prometheus metrics on ports `8000 + rank` (e.g. rank 0 → `:8000`).

## Observability

Each process exposes a Prometheus scrape endpoint. Metrics exported:

| Metric | Description |
|---|---|
| `training_loss{rank}` | Average cross-entropy loss for the epoch |
| `training_accuracy{rank}` | Top-1 accuracy % for the epoch |
| `samples_per_second{rank}` | Throughput (samples/sec) for the epoch |

Add these targets to your `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: distributed_trainer
    static_configs:
      - targets: ['localhost:8000', 'localhost:8001', 'localhost:8002', 'localhost:8003']
```

## Fault Tolerance

Checkpoints are written at the end of each epoch by rank 0:
```
checkpoints/checkpoint.pth.tar
```

The checkpoint stores model weights, optimizer state, and the current epoch so
training can resume exactly where it left off after a node failure.

## Running Tests

```bash
pytest tests/ -v
```

The suite covers:
- **Model correctness** — output shape, finite values, gradient flow through all layers
- **Checkpoint round-trip** — save/load of weights, optimizer state, and epoch counter
- **Training dynamics** — loss decreases after a gradient step
- **DistributedSampler** — no index overlap between ranks, shuffle changes per epoch

Tests skip automatically if PyTorch is not installed.

## Concepts Implemented
- **Data Parallelism** — dataset sharded across ranks via `DistributedSampler`; gradients averaged with `DistributedDataParallel`
- **Gloo backend** — CPU-compatible collective ops (AllReduce under the hood)
- **Prometheus metrics** — per-rank loss, accuracy, and throughput
- **Fault tolerance** — epoch-level checkpointing with resume support

Contributions welcome! 🚀

## Further Reading

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | System diagrams, Ring-AllReduce breakdown, scaling guide |
| [CHANGELOG.md](CHANGELOG.md) | Version history and upcoming features |
