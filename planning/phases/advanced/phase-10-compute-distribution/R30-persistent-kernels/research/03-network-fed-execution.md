# R30 Research: Network-Fed Continuous Execution Pipeline

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

Design the full pipeline from network data arrival through persistent kernel processing to result output. This document covers the end-to-end architecture, buffering strategies, error handling, integration with existing OuterLink components (CUDA graphs, scatter-gather DMA), and performance analysis comparing persistent kernels to traditional repeated kernel launches.

---

## TL;DR

The network-fed execution pipeline has three stages: **ingest** (RDMA/DMA writes data to VRAM), **process** (persistent kernel consumes from ring buffer), and **emit** (results written to output buffer, optionally RDMA'd back). Double buffering overlaps all three stages. The key design insight: separate the "doorbell" (notification) path from the "data" (payload) path. The doorbell is a lightweight ring buffer descriptor; data goes directly to pre-allocated VRAM slots. This separation allows the NIC to write large payloads via bulk DMA while signaling readiness with a single small write. Performance comparison: persistent kernel pipeline achieves ~1.5-3us end-to-end latency per batch vs. ~10-25us for traditional launch-per-batch, a 5-10x improvement for latency-sensitive workloads.

---

## Pipeline Architecture

### Three-Stage Pipeline

```
Stage 1: INGEST              Stage 2: PROCESS             Stage 3: EMIT
(NIC -> VRAM)                (GPU Persistent Kernel)      (VRAM -> NIC/Host)

Remote Node                  GPU SMs                      Output Path
    |                            |                            |
    | RDMA Write (data)          | Read ring entry            | Write result to
    |  -> data_buffer[slot]      | Read data from slot        |   output_buffer[slot]
    |                            | Execute compute            |
    | RDMA Write (descriptor)    | Write result               | RDMA Write result
    |  -> ring.entries[idx]      |                            |   to remote node
    |                            |                            |   (or CPU reads)
    | RDMA Write (head++)        | Advance tail               |
    |  -> ring.head              |  -> ring.tail              | Signal output ready
    v                            v                            v
```

### Data Flow Diagram

```
+--------+     RDMA      +----------+     BAR1 DMA      +------------------+
| Remote |  ---------->  | ConnectX |  --------------->  | GPU VRAM         |
| Node   |               | NIC      |                    |                  |
+--------+               +----------+                    | Input Ring:      |
                                                         |  head/tail/descs |
                                                         |                  |
                                                         | Data Buffers:    |
                                                         |  [slot 0..N]     |
                                                         |                  |
                                                         | Persistent       |
                                                         | Kernel:          |
                                                         |  polls ring      |
                                                         |  processes data  |
                                                         |  writes output   |
                                                         |                  |
                                                         | Output Ring:     |
                                                         |  head/tail/descs |
                                                         |                  |
                                                         | Output Buffers:  |
                                                         |  [slot 0..M]     |
                                                         +------------------+
                                                                |
                                                         BAR1 DMA / cudaMemcpy
                                                                |
                                                                v
                                                         +----------+
                                                         | ConnectX | RDMA to
                                                         | NIC      | remote
                                                         +----------+
```

---

## Buffering Strategies

### Double Buffering

The minimum for overlap. While the kernel processes buffer A, the NIC fills buffer B. When the kernel finishes A, it switches to B, and the NIC starts filling A.

```
Time -->
NIC:    [Fill A] [Fill B] [Fill A] [Fill B] ...
GPU:    [idle]   [Proc A] [Proc B] [Proc A] ...
```

**Implementation with ring buffer:** The ring buffer naturally provides double buffering when the ring has >= 2 slots. The head/tail pointers track which slots are being written vs. read.

**Limitations:**
- If NIC fill time != GPU process time, one side stalls
- Minimum latency = max(fill_time, process_time)
- No absorption of burst traffic

### Triple Buffering

Adds a third buffer to absorb variance. While GPU processes buffer A, NIC has finished B and starts C. If GPU finishes A before NIC finishes C, it immediately starts B with no stall.

```
Time -->
NIC:    [Fill A] [Fill B] [Fill C] [Fill A] ...
GPU:    [idle]   [Proc A] [Proc B] [Proc C] ...
```

**Implementation:** Ring buffer with >= 3 slots. The extra slot absorbs jitter in either NIC or GPU timing.

### N-Way Buffering (Ring Buffer)

The ring buffer generalizes to N-way buffering. With N = 256 slots:
- Absorbs large bursts (up to 256 batches queued)
- Smooths variance in both producer and consumer rates
- Provides natural backpressure (ring full = producer must wait)

**The ring buffer from 02-doorbell-mechanisms.md with 256 entries IS the N-way buffer.** No separate buffering mechanism needed.

### Choosing Buffer Count

| Scenario | Recommended Slots | Rationale |
|---|---|---|
| Low-latency, steady stream | 4-8 | Minimal memory, enough for jitter |
| Bursty traffic | 32-64 | Absorbs bursts without backpressure |
| High-throughput bulk | 128-256 | Maximizes pipeline depth |
| Memory-constrained | 2-4 | Minimum for overlap |

For OuterLink's general case: **64 slots x 64KB = 4MB data buffer** provides good burst absorption while fitting easily in VRAM and BAR1 aperture.

---

## Use Cases and Workload Patterns

### 1. Inference Serving (Continuous Batching)

The most direct application. An inference server receives requests over the network, batches them, and feeds them to a persistent GPU kernel running the model.

**vLLM-style architecture adapted for OuterLink:**

```
Client Requests -> OuterLink Transport -> VRAM Ring Buffer
                                              |
                                    Persistent Kernel:
                                    - Read batch from ring
                                    - Tokenize / preprocess
                                    - Run model forward pass
                                    - Write output tokens
                                              |
                                    Output Ring Buffer -> OuterLink Transport -> Clients
```

Key insight from vLLM: **iteration-level batching** means the kernel doesn't wait for a full batch. As soon as any request completes, its slot is freed for the next request. The persistent kernel processes a dynamically changing set of active requests.

### 2. Data Stream Processing

Sensor data, financial market ticks, video frames — continuous streams that need real-time GPU processing.

**Pattern:**
- Fixed-size data chunks arrive at regular intervals
- Each chunk processed independently (no inter-chunk dependency)
- Results streamed back with minimal latency
- The persistent kernel maintains no state between chunks

**Example: Video frame processing**
- 1080p frame = ~6MB (YUV420)
- 30fps = one frame every 33ms
- GPU processing time per frame: ~1-5ms
- With persistent kernel: 0 launch overhead, ~1-5ms total
- Without: 5-20us launch + 1-5ms compute = slightly worse, but launch overhead is negligible for this workload

**Where persistent kernels shine:** High-frequency, low-latency data. Financial tick processing at 100K+ messages/second, where 5-20us launch overhead per message becomes significant.

### 3. Distributed Training (Gradient Aggregation)

Multiple GPU nodes compute gradients and aggregate them. Each node's persistent kernel:
1. Computes local gradients
2. Writes gradients to output ring
3. Receives remote gradients via input ring (RDMA)
4. Aggregates all gradients
5. Updates model weights
6. Repeats

The persistent kernel never exits between training iterations. Combined with RDMA, gradients flow directly between GPU VRAMs across nodes.

### 4. GPU-to-GPU Pipeline (Multi-Node)

A processing pipeline spans multiple GPU nodes:

```
Node A (Preprocessing) -> RDMA -> Node B (Inference) -> RDMA -> Node C (Postprocessing)
```

Each node runs a persistent kernel. Data flows through the pipeline via RDMA-connected ring buffers. No CPU involvement on the data path.

---

## Error Handling and Kernel Restart

### Error Categories

| Error | Detection | Recovery |
|---|---|---|
| **Bad input data** | Kernel validates, sets error flag in ring entry | Skip entry, log error, continue processing |
| **Numerical error** (NaN/Inf) | Kernel checks output | Write error to output ring, continue |
| **Memory access violation** | CUDA sticky error | Full context reset and kernel relaunch |
| **GPU hang** (kernel deadlock) | Host heartbeat timeout | Driver reset, full recovery |
| **NIC write failure** | Missing ring entries (gap in sequence) | Timeout, request retransmission from remote |

### Heartbeat Protocol

```cpp
// GPU persistent kernel heartbeat
__global__ void persistent_kernel(
    DoorbellRing* input_ring,
    volatile uint64_t* heartbeat,
    /* ... */
) {
    uint64_t iteration = 0;
    while (!shutdown_requested) {
        // Update heartbeat every N iterations
        if (iteration % 1000 == 0) {
            *heartbeat = clock64();  // GPU clock timestamp
            __threadfence_system();
        }

        // Normal processing...
        iteration++;
    }
}

// Host-side monitor (separate thread)
void monitor_kernel(volatile uint64_t* heartbeat_mapped) {
    uint64_t last_seen = *heartbeat_mapped;
    while (running) {
        sleep_ms(100);
        uint64_t current = *heartbeat_mapped;
        if (current == last_seen) {
            // Heartbeat stale — kernel may be hung
            trigger_recovery();
        }
        last_seen = current;
    }
}
```

### Recovery Sequence

When a fatal error is detected:

1. **Attempt graceful shutdown**: Set shutdown flag, wait 100ms
2. **If kernel responds**: Clean exit, collect any partial results
3. **If kernel unresponsive**:
   a. `cudaDeviceReset()` — destroys entire CUDA context
   b. Notify all connected OuterLink clients: "GPU node recovering"
   c. Re-initialize CUDA context
   d. Re-allocate VRAM buffers (ring buffers, data pools)
   e. Re-register memory for RDMA (re-map BAR1 regions)
   f. Re-launch persistent kernel
   g. Resume accepting data from clients
4. **Expected recovery time**: 100-500ms (dominated by CUDA context creation and RDMA re-registration)

### Graceful Degradation

If one GPU node in a pool fails:
- OuterLink client redirects work to other nodes (load balancer)
- Failed node recovers in background
- Once recovered, re-joins the pool
- No data loss if clients implement retry logic

---

## Integration with R13 CUDA Graphs

### Can a Persistent Kernel Be a CUDA Graph Node?

Technically yes — a kernel node in a CUDA graph can be a persistent kernel. But this defeats the purpose of CUDA graphs:

- CUDA graphs are designed for **repeated execution** of a fixed DAG
- A persistent kernel runs **once** and never returns
- Graph instantiation and replay overhead (~1-3us) is exactly what persistent kernels avoid

### Where They Complement Each Other

| Scenario | Best Approach |
|---|---|
| Fixed compute pipeline, repeated execution | CUDA Graph (R13) |
| Continuous streaming, variable data | Persistent Kernel (R30) |
| Fixed pipeline with streaming input | Persistent kernel that internally replays a graph |
| Multi-phase with different rates | Persistent kernel per phase, connected by rings |

**Hybrid pattern:** The persistent kernel can internally use CUDA graph replay for the compute-heavy portion:

```cpp
__global__ void persistent_kernel(DoorbellRing* ring, cudaGraphExec_t graph) {
    while (!shutdown) {
        wait_for_data(ring);
        // Internally replay a pre-compiled compute graph
        cudaGraphLaunch(graph, stream);  // Note: device-side graph launch (CC 9.0+)
        cudaStreamSynchronize(stream);
        emit_result(output_ring);
    }
}
```

However, device-side graph launch requires Compute Capability 9.0+ (Hopper). On Ampere (RTX 3090, CC 8.6), the persistent kernel must contain the compute logic directly or use standard kernel launches on device-created streams.

### Practical Integration for OuterLink

OuterLink should support **both** execution models:
1. **Graph mode (R13)**: For applications that submit discrete compute jobs. OuterLink intercepts graph submission, distributes across nodes, returns results.
2. **Persistent mode (R30)**: For applications that need continuous processing. OuterLink sets up the ring buffer pipeline, launches persistent kernels on target GPUs, routes data via RDMA.

The application (or OuterLink's client library) chooses which mode based on workload characteristics.

---

## Performance Analysis: Persistent vs. Repeated Launch

### Kernel Launch Overhead Breakdown

| Component | Time | Persistent Kernel? |
|---|---|---|
| CPU-side launch API call | 2-5us | Eliminated (kernel already running) |
| Driver command submission | 1-3us | Eliminated |
| GPU command processor pickup | 1-5us | Eliminated |
| SM scheduling/dispatch | 0.5-1us | Eliminated |
| **Total launch overhead** | **4.5-14us** | **0us** |

### End-to-End Latency Comparison

**Scenario: 64KB data batch, simple transform kernel (1us compute)**

| Path | Stage Latencies | Total |
|---|---|---|
| **Host-staged + relaunch** | RDMA to host (1us) + cudaMemcpy H2D (3us) + launch (10us) + compute (1us) + cudaMemcpy D2H (3us) | ~18us |
| **OpenDMA + relaunch** | RDMA to VRAM (1.5us) + launch (10us) + compute (1us) | ~12.5us |
| **OpenDMA + persistent** | RDMA to VRAM (1.5us) + doorbell detect (0.1us) + compute (1us) | ~2.6us |
| **Improvement** | | **~5-7x** |

### Throughput Comparison

**Scenario: Sustained stream of 64KB batches**

| Path | Bottleneck | Max Throughput |
|---|---|---|
| **Host-staged + relaunch** | cudaMemcpy + launch overhead | ~55K batches/sec (~3.5 GB/s) |
| **OpenDMA + relaunch** | Kernel launch overhead | ~80K batches/sec (~5.1 GB/s) |
| **OpenDMA + persistent** | NIC write bandwidth (100Gbps) | ~190K batches/sec (~12.2 GB/s) |
| **Improvement** | | **~2.4-3.5x** |

With persistent kernels, the bottleneck shifts from CPU/launch overhead to the raw network bandwidth — exactly what OuterLink is designed to maximize.

### When Persistent Kernels Do NOT Help

| Scenario | Why Not |
|---|---|
| Large compute kernels (>100ms) | Launch overhead is negligible vs. compute time |
| Infrequent batches (<10/sec) | Launch overhead amortized over long intervals |
| Multi-kernel pipelines | Need to launch different kernels for different stages |
| GPU sharing required | Persistent kernel monopolizes the GPU |

---

## Reference Architectures

### NVIDIA Triton Inference Server

Triton uses continuous batching (iteration-level scheduling) for LLM inference:
- Scheduler forms batches at each iteration step
- New requests join mid-execution as slots free up
- CUDA graph replay minimizes launch overhead for fixed-shape batches
- Does NOT use persistent kernels directly — relies on fast re-launch

**Lesson for OuterLink:** Triton's continuous batching is the application-level pattern. OuterLink provides the transport-level pattern (RDMA + doorbell). They are complementary, not competing.

### vLLM

vLLM implements PagedAttention with continuous batching:
- KV-cache managed as non-contiguous memory pages (like OS virtual memory)
- Sequences flattened into a single super-sequence per batch
- CUDA graphs recorded for fixed batch sizes, replayed for speed
- Iteration-level scheduling: new requests enter at any iteration boundary

**Lesson for OuterLink:** The paged memory management pattern is relevant for managing VRAM data buffers in the ring. Pre-allocate fixed-size pages, assign them to ring slots dynamically.

### NVIDIA DOCA GPUNetIO + DPDK

NVIDIA's official stack for inline GPU packet processing:
1. NIC receives packets via DPDK
2. Packets written to GPU memory (receive list in VRAM)
3. Persistent CUDA kernel polls the receive list
4. Kernel processes packets in-place
5. Results written to transmit list
6. NIC sends processed packets

This is the closest existing system to OuterLink's persistent kernel pipeline. The key difference: OuterLink uses OpenDMA (direct BAR1 access) instead of GPUDirect RDMA (requires Tesla/A100 hardware).

---

## Recommended Pipeline Design for OuterLink

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│ GPU VRAM Layout                                                  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Input Ring    │  │ Output Ring   │  │ Control Block         │  │
│  │ (256 entries) │  │ (256 entries) │  │  - heartbeat: u64    │  │
│  │ head/tail     │  │ head/tail     │  │  - shutdown: u32     │  │
│  │ descriptors   │  │ descriptors   │  │  - error_code: u32   │  │
│  └──────────────┘  └──────────────┘  │  - stats: {...}       │  │
│                                       └──────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Input Data Pool (64 slots x 64KB = 4MB)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Output Data Pool (64 slots x 64KB = 4MB)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Persistent Kernel (1-2 blocks/SM, polling input ring)     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Operational Flow

1. **Startup**: OuterLink server allocates VRAM structures, registers with RDMA, launches persistent kernel
2. **Ingest**: Remote nodes RDMA-write data + descriptors + head update to VRAM via BAR1
3. **Process**: Persistent kernel detects head change, reads descriptor, processes data, writes result to output pool + output ring
4. **Emit**: Host-side thread or second persistent kernel monitors output ring, RDMA-writes results to requesting node
5. **Shutdown**: Host sets shutdown flag, kernel exits cleanly, resources freed

### Configuration Parameters

| Parameter | Default | Tunable Range | Impact |
|---|---|---|---|
| `input_ring_size` | 256 | 4-4096 | Burst absorption capacity |
| `output_ring_size` | 256 | 4-4096 | Output queuing depth |
| `data_slot_size` | 64KB | 4KB-4MB | Max single-batch data size |
| `poll_interval_ns` | 100 | 0-10000 | Latency vs. power trade-off |
| `blocks_per_sm` | 1 | 1-16 | Parallelism vs. resource usage |
| `heartbeat_interval` | 1000 iterations | 100-10000 | Failure detection speed |

---

## Open Questions

- [ ] Should the output path use a second persistent kernel (GPU-initiated RDMA via doorbell to NIC) or a CPU-side monitor thread that reads the output ring and initiates RDMA sends?
- [ ] How to handle variable-size data that exceeds a single slot? Options: scatter-gather descriptors (R28), multi-slot entries, or separate large-transfer path.
- [ ] What is the optimal `__nanosleep()` interval for different workload profiles? Needs benchmarking with actual network traffic patterns.
- [ ] Can we support multiple concurrent persistent kernels on one GPU (different rings for different workloads) or must we multiplex through a single kernel?
- [ ] How does the persistent kernel interact with CUDA's memory pool allocator (`cudaMallocAsync`)? If the kernel needs dynamic memory during processing, can it use device-side `malloc`?

---

## Related Documents

- [01-persistent-kernel-patterns.md](./01-persistent-kernel-patterns.md) — Kernel design patterns
- [02-doorbell-mechanisms.md](./02-doorbell-mechanisms.md) — Ring buffer and doorbell design
- [R13: CUDA Graph Interception](../../R13-cuda-graph-interception/) — Graph-based execution model
- [R26: PTP Clock Sync](../../R26-ptp-clock-sync/) — Coordinated timing across nodes
- [R28: Scatter-Gather DMA](../../R28-scatter-gather-dma/) — Multi-region DMA transfers
- [vLLM Architecture Blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [NVIDIA DPDK/GPUdev Blog](https://developer.nvidia.com/blog/optimizing-inline-packet-processing-using-dpdk-and-gpudev-with-gpus/)
- [GPU-Centric Communication Landscape](https://arxiv.org/html/2409.09874v2)
- [NVSHMEM IBGDA Performance Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/performance.html)
