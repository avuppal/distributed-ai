# Distributed AI Training — Architecture Deep Dive

This document explains the design of the distributed training simulator: how processes are initialized, how data flows, how gradients are synchronized, and how the system recovers from failures.

---

## 1. System Overview

Four worker processes (ranks 0–3) simulate a 4-node training cluster. Rank 0 doubles as the **rendezvous coordinator** and the sole writer of checkpoints. All ranks export Prometheus metrics independently.

```mermaid
graph TD
    subgraph Host Machine
        TorchRun["torchrun (launcher)"]

        subgraph "Process Group (Gloo backend)"
            R0["Rank 0 🟢\ncoord + checkpoint writer\n:8000 metrics"]
            R1["Rank 1\n:8001 metrics"]
            R2["Rank 2\n:8002 metrics"]
            R3["Rank 3\n:8003 metrics"]
        end

        FS["Shared Filesystem\ncheckpoints/checkpoint.pth.tar"]
        MNIST["MNIST Dataset\n(downloaded once)"]
    end

    Prometheus["Prometheus Server\nscrape :8000–:8003"]
    Grafana["Grafana Dashboard"]

    TorchRun -->|spawn| R0
    TorchRun -->|spawn| R1
    TorchRun -->|spawn| R2
    TorchRun -->|spawn| R3

    R0 <-->|AllReduce / Gloo| R1
    R1 <-->|AllReduce / Gloo| R2
    R2 <-->|AllReduce / Gloo| R3
    R3 <-->|AllReduce / Gloo| R0

    R0 -->|write| FS
    R1 -.->|read on resume| FS
    R2 -.->|read on resume| FS
    R3 -.->|read on resume| FS

    R0 & R1 & R2 & R3 -->|pull shard| MNIST

    Prometheus -->|scrape| R0
    Prometheus -->|scrape| R1
    Prometheus -->|scrape| R2
    Prometheus -->|scrape| R3
    Prometheus --> Grafana
```

---

## 2. Initialization Sequence

Before any training can begin, every rank must complete a **three-phase initialization**:

```mermaid
sequenceDiagram
    participant L as torchrun (launcher)
    participant R0 as Rank 0
    participant R1 as Rank 1
    participant R2 as Rank 2
    participant R3 as Rank 3

    L->>R0: spawn (RANK=0, WORLD_SIZE=4)
    L->>R1: spawn (RANK=1, WORLD_SIZE=4)
    L->>R2: spawn (RANK=2, WORLD_SIZE=4)
    L->>R3: spawn (RANK=3, WORLD_SIZE=4)

    Note over R0,R3: Phase 1 — Rendezvous (env:// method)
    R0->>R0: bind MASTER_ADDR:MASTER_PORT
    R1->>R0: connect
    R2->>R0: connect
    R3->>R0: connect
    R0-->>R1: ack (rank assigned)
    R0-->>R2: ack (rank assigned)
    R0-->>R3: ack (rank assigned)

    Note over R0,R3: Phase 2 — Model & Data Setup
    R0->>R0: build SimpleCNN, wrap in DDP
    R1->>R1: build SimpleCNN, wrap in DDP
    R2->>R2: build SimpleCNN, wrap in DDP
    R3->>R3: build SimpleCNN, wrap in DDP
    R0->>R0: DistributedSampler(shard 0/4)
    R1->>R1: DistributedSampler(shard 1/4)
    R2->>R2: DistributedSampler(shard 2/4)
    R3->>R3: DistributedSampler(shard 3/4)

    Note over R0,R3: Phase 3 — Fault Tolerance Check
    R0->>R0: load_checkpoint() → resume if found
    R1->>R1: load_checkpoint() → resume if found
    R2->>R2: load_checkpoint() → resume if found
    R3->>R3: load_checkpoint() → resume if found
```

---

## 3. Training Loop — Data Parallelism

Each epoch follows a **forward → backward → AllReduce → optimizer step** cycle. The critical property of DDP is that `loss.backward()` *overlaps* gradient communication with the backward pass.

```mermaid
flowchart LR
    subgraph Rank0["Rank 0  (shard 0)"]
        direction TB
        D0["Mini-batch\nsamples 0–15"] --> FW0["Forward\npass"]
        FW0 --> BW0["Backward\npass"]
        BW0 --> GR0["∇W (local)"]
    end

    subgraph Rank1["Rank 1  (shard 1)"]
        direction TB
        D1["Mini-batch\nsamples 16–31"] --> FW1["Forward\npass"]
        FW1 --> BW1["Backward\npass"]
        BW1 --> GR1["∇W (local)"]
    end

    subgraph Rank2["Rank 2  (shard 2)"]
        direction TB
        D2["Mini-batch\nsamples 32–47"] --> FW2["Forward\npass"]
        FW2 --> BW2["Backward\npass"]
        BW2 --> GR2["∇W (local)"]
    end

    subgraph Rank3["Rank 3  (shard 3)"]
        direction TB
        D3["Mini-batch\nsamples 48–63"] --> FW3["Forward\npass"]
        FW3 --> BW3["Backward\npass"]
        BW3 --> GR3["∇W (local)"]
    end

    AR["🔄 AllReduce\ndist.all_reduce\nGloo backend\n∑ gradients ÷ world_size"]

    GR0 & GR1 & GR2 & GR3 --> AR

    AR -->|"∇W (averaged)"| OPT0["Optimizer step\nRank 0"]
    AR -->|"∇W (averaged)"| OPT1["Optimizer step\nRank 1"]
    AR -->|"∇W (averaged)"| OPT2["Optimizer step\nRank 2"]
    AR -->|"∇W (averaged)"| OPT3["Optimizer step\nRank 3"]
```

**Key guarantee:** After AllReduce, every rank has *identical* averaged gradients, so optimizer steps produce identical model weights — no manual synchronization needed afterwards.

---

## 4. Ring-AllReduce Algorithm

The Gloo backend implements **Ring-AllReduce**, which achieves near-100% bandwidth utilization regardless of the number of nodes (unlike parameter-server approaches that bottleneck at the PS).

```mermaid
graph LR
    subgraph "Scatter-Reduce phase  (N-1 steps)"
        direction LR
        R0a((Rank 0\nchunk 0)) -->|"send chunk 0"| R1a((Rank 1\nchunk 1))
        R1a -->|"send chunk 1"| R2a((Rank 2\nchunk 2))
        R2a -->|"send chunk 2"| R3a((Rank 3\nchunk 3))
        R3a -->|"send chunk 3"| R0a
    end

    subgraph "AllGather phase  (N-1 steps)"
        direction LR
        R0b((Rank 0\nfull ∇W)) -->|"broadcast"| R1b((Rank 1\nfull ∇W))
        R1b --> R2b((Rank 2\nfull ∇W))
        R2b --> R3b((Rank 3\nfull ∇W))
        R3b --> R0b
    end
```

**Bus bandwidth utilization:**
$$\text{BW} = \frac{2(N-1)}{2N} \cdot \text{link speed} \approx 100\%\ \text{for large } N$$

Compare to parameter-server AllReduce:
$$\text{BW}_{\text{PS}} = \frac{1}{N} \cdot \text{link speed}$$

At N=4, Ring-AllReduce uses **3× more** of the available bandwidth than a naïve PS approach.

---

## 5. Fault Tolerance — Checkpoint FSM

```mermaid
stateDiagram-v2
    [*] --> Starting

    Starting --> LoadCheckpoint : process launched
    LoadCheckpoint --> Epoch0 : no checkpoint found
    LoadCheckpoint --> EpochK : checkpoint found (epoch=K)

    Epoch0 --> Training : start epoch 0
    EpochK --> Training : resume epoch K

    Training --> EvalEpoch : mini-batches exhausted
    EvalEpoch --> WriteCheckpoint : rank == 0
    EvalEpoch --> NextEpoch : rank != 0

    WriteCheckpoint --> NextEpoch : checkpoint written
    NextEpoch --> Training : epoch < max_epochs
    NextEpoch --> Done : epoch == max_epochs

    Training --> Crashed : process killed / OOM
    Crashed --> LoadCheckpoint : restart with --resume

    Done --> [*]
```

**What the checkpoint stores:**

| Key | Value |
|-----|-------|
| `epoch` | Last completed epoch (resumption point) |
| `state_dict` | `model.module.state_dict()` — weights only, not DDP wrapper |
| `optimizer` | Full `Adam` state (momentum, variance buffers) |

Only rank 0 writes the checkpoint to avoid **write contention** on shared filesystems. All ranks read it independently on resume.

---

## 6. Observability Stack

```mermaid
graph LR
    subgraph "Each Worker (rank R)"
        Trainer["Training loop"]
        PClient["prometheus_client\nGauge.set()"]
        HTTP["HTTP :800R\n/metrics endpoint"]
        Trainer -->|"loss, accuracy,\nthroughput"| PClient
        PClient --> HTTP
    end

    Prom["Prometheus\nscrape_interval: 15s"]
    Graf["Grafana\ndashboard"]

    HTTP -->|pull| Prom
    Prom --> Graf
```

**Metrics exported per rank:**

| Metric name | Type | Description |
|-------------|------|-------------|
| `training_loss{rank="R"}` | Gauge | Mean cross-entropy loss for the epoch |
| `training_accuracy{rank="R"}` | Gauge | Top-1 accuracy (%) |
| `samples_per_second{rank="R"}` | Gauge | Mini-batch throughput |

Watching all ranks in Grafana lets you spot **straggler nodes** (one rank consistently slower → network bottleneck or CPU imbalance) — a critical operational signal in real clusters.

---

## 7. Scaling to Real Clusters

This simulator uses the **Gloo** CPU backend for portability. Production upgrades:

| Concern | Simulator | Production |
|---------|-----------|------------|
| Backend | `gloo` (CPU/Ethernet) | `nccl` (NVLink / InfiniBand) |
| Model | SimpleCNN (60 K params) | LLaMA-3 (8 B – 405 B params) |
| Parallelism | Data-parallel only | Data + Tensor + Pipeline (3D) |
| Checkpointing | Per-epoch, rank 0 | Async, distributed (Gemini/PyTorch FSDP) |
| Discovery | `env://` (manual) | Kubernetes + `c10d` etcd rendezvous |
| Monitoring | Prometheus Gauge | Weights & Biases / MLflow + custom kernels |

The architecture choices made here (DDP, DistributedSampler, ring communication, Prometheus observability) are direct analogues of what powers training at hyperscaler scale — just with smaller numbers.
