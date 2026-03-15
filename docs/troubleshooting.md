# NCCL Troubleshooting Guide

This guide covers the top 10 most common NCCL errors, their root causes, and how to fix them.

### 1. InfiniBand Misconfiguration (`NCCL_IB_DISABLE`)
* **Symptom:** `NCCL WARN Connect to ... failed` or poor performance.
* **Root Cause:** NCCL fails to detect or use InfiniBand (IB) correctly.
* **Fix:** Export `NCCL_IB_DISABLE=1` to force fallback to IP sockets, or verify IB configuration with `ibv_devinfo`.
* **Prevention:** Ensure network topology is properly defined and RoCE/IB drivers are correctly installed on all nodes.

### 2. Mismatched NCCL Versions Across Nodes
* **Symptom:** Unexplained hangs during initialization or `Unhandled system error`.
* **Root Cause:** Different machines in a multi-node setup are running different versions of the NCCL library.
* **Fix:** Standardize the NCCL version. Use a unified Docker image across all nodes.
* **Prevention:** Pin NCCL and CUDA toolkit versions in your deployment scripts.

### 3. Firewall Blocking Port Range 20000–21000
* **Symptom:** `NCCL WARN Bootstrap : no socket interface found` or connection timeouts.
* **Root Cause:** NCCL uses ephemeral ports for communication which are blocked by firewalls.
* **Fix:** Open the required port range (default: 20000–21000) or specify a custom range with `NCCL_SOCKET_IFNAME` and `NCCL_MIN_PORT`/`NCCL_MAX_PORT`.
* **Prevention:** Include these port rules in the baseline infrastructure provisioning.

### 4. CUDA Driver Version Mismatch
* **Symptom:** Kernel panics, `CUDA_ERROR_OUT_OF_MEMORY`, or generic initialization failures.
* **Root Cause:** The CUDA driver on the host is incompatible with the NCCL/CUDA toolkit version used by the container.
* **Fix:** Upgrade the host NVIDIA driver to meet the toolkit's minimum requirements.
* **Prevention:** Implement automated host-level compliance checks before deploying containers.

### 5. Deadlock from Uneven Batch Sizes (DDP Requirement)
* **Symptom:** Training script hangs indefinitely during `AllReduce` or `Broadcast`.
* **Root Cause:** One or more processes in a DistributedDataParallel (DDP) group have a different number of batches or operations, causing mismatched synchronization.
* **Fix:** Use `join()` context manager in PyTorch DDP or pad inputs to ensure identical batch counts across all ranks.
* **Prevention:** Add assertions at the start of the epoch to verify total batch counts match across ranks.

### 6. Timeout Tuning via `NCCL_TIMEOUT`
* **Symptom:** `Watchdog caught collective operation timeout`.
* **Root Cause:** A node is slower than others (straggler) or network congestion causes a collective operation to exceed the default timeout (usually 30 mins or less).
* **Fix:** Increase the timeout by setting `NCCL_TIMEOUT` (e.g., `export NCCL_TIMEOUT=3600`) or investigate the straggler node.
* **Prevention:** Monitor node-level metrics (CPU, IO wait) to proactively identify slow nodes.

### 7. Reading Logs with `NCCL_DEBUG=INFO`
* **Symptom:** You have a generic NCCL error but no useful context.
* **Root Cause:** The default NCCL log level hides the underlying initialization or topology building errors.
* **Fix:** Run with `NCCL_DEBUG=INFO` (and optionally `NCCL_DEBUG_SUBSYS=ALL`) to get detailed topology and connection logs.
* **Prevention:** Always enable `NCCL_DEBUG=INFO` in initial staging/testing runs before pushing to production.

### 8. GPU Direct RDMA Pitfalls
* **Symptom:** Poor multi-node performance despite having high-bandwidth interconnects, or PCI topology warnings.
* **Root Cause:** GPU Direct RDMA is not engaging because the network interface and GPU are not on the same PCIe switch.
* **Fix:** Check `nvidia-smi topo -m` to verify PCIe topology. Pin processes to the correct NUMA node and GPU.
* **Prevention:** Always align rank local IDs to the correct NUMA node and network interface (`NCCL_SOCKET_IFNAME`).

### 9. Mixed Precision Interaction with NCCL
* **Symptom:** NaNs or Infs during training, or `NCCL WARN Invalid datatype`.
* **Root Cause:** Reducing FP16 tensors across many nodes can overflow the accumulator if not handled correctly.
* **Fix:** Scale gradients before `AllReduce` or use FP32 for the reduction step if possible.
* **Prevention:** Use established mixed-precision libraries (like PyTorch AMP) which handle scaling automatically.

### 10. Hanging AllReduce Diagnosis with `gdb` / `py-spy`
* **Symptom:** Process hangs indefinitely without any error messages.
* **Root Cause:** Silent deadlock, often due to mismatched collective calls or Python GIL issues.
* **Fix:** Attach to the hanging process using `py-spy dump --pid <PID>` or `gdb` to inspect the call stack and identify which collective is blocking.
* **Prevention:** Implement comprehensive logging around collective calls in custom distributed layers.
