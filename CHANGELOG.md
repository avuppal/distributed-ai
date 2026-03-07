# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- NCCL backend support for multi-GPU nodes
- Tensor-parallel sharding of `fc1`/`fc2` layers (Megatron-style)
- Async checkpoint writer (non-blocking epoch saves via background thread)
- Kubernetes `CronJob` example for rendezvous via etcd

---

## [1.2.0] — 2026-03-07

### Added
- `docs/architecture.md` — full architecture deep dive with Mermaid diagrams covering:
  system overview, initialization sequence, data-parallel training loop,
  Ring-AllReduce algorithm, fault-tolerance FSM, and observability stack

### Changed
- README updated with links to architecture docs and scaling guide

---

## [1.1.0] — 2026-03-06

### Added
- Prometheus observability: per-rank `training_loss`, `training_accuracy`, and `samples_per_second` Gauges exported on ports `8000 + rank`
- Epoch-level checkpointing (`checkpoints/checkpoint.pth.tar`) written by rank 0 only to avoid write contention
- `--resume` flag to reload model weights, optimizer state, and epoch counter after node failure
- Full pytest suite covering model correctness, checkpoint round-trip, training dynamics, and `DistributedSampler` index exclusivity

### Changed
- Switched process-group backend from `nccl` to `gloo` for CPU-portable simulation (no GPU required to run tests)

### Fixed
- `DistributedSampler` seed was not advancing per epoch — `sampler.set_epoch(epoch)` now called before every DataLoader iteration

---

## [1.0.0] — 2026-03-05

### Added
- Initial implementation of `distributed_trainer.py`
- Data-parallel training of `SimpleCNN` on MNIST using `DistributedDataParallel`
- `torchrun` launch instructions for simulating N-node clusters on a single machine
- `DistributedSampler` for disjoint data sharding across ranks
- Basic README with quickstart and concept explanations
