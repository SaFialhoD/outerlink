# R30 Research: VRAM-Based Doorbell Mechanisms

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

Analyze VRAM-based notification mechanisms that allow external sources (CPU, NIC, remote nodes) to signal a persistent GPU kernel that new data is available. This is the critical bridge between OuterLink's network transport and the GPU compute path — the doorbell mechanism determines how quickly the GPU reacts to incoming data.

---

## TL;DR

Three primary doorbell mechanisms exist: simple memory-mapped flags, atomic counters, and ring buffers. For OuterLink, an **atomic counter with a ring buffer** is the optimal design — the counter acts as a fast doorbell (NIC writes via RDMA to VRAM), while the ring buffer provides structured metadata about what data arrived and where it lives. The critical challenge is GPU cache coherency: when a NIC writes to VRAM via PCIe BAR1, a concurrently running GPU kernel may see stale cached data. The solution is `volatile` reads combined with `__threadfence_system()`, plus careful placement of doorbell memory in uncached regions. Existing systems (GPUrdma, NVSHMEM IBGDA, GPUnet) have validated this pattern in production.

---

## Doorbell Mechanism Options

### 1. Memory-Mapped Flag (Simple Boolean)

The simplest doorbell: a single integer in VRAM that the producer sets to indicate data readiness.

```cpp
// GPU side (persistent kernel)
__global__ void kernel(volatile int* doorbell, float* data) {
    while (true) {
        // Poll doorbell
        while (*doorbell == 0) {
            __nanosleep(100);
        }

        // Process data
        process(data, batch_size);
        __threadfence_system();

        // Signal completion
        *doorbell = 0;
    }
}

// CPU side (or NIC via RDMA write)
*doorbell = 1;  // Signal data ready
```

**Pros:**
- Minimal complexity
- Single memory location to write
- Easy to reason about ordering

**Cons:**
- No metadata (GPU doesn't know how much data, where it is)
- Single producer/single consumer only
- No queuing — if GPU is busy, producer must wait
- Lost signals if producer writes while GPU hasn't seen previous signal

**Verdict:** Too simple for OuterLink. Useful only for single-stream, single-buffer scenarios.

### 2. Atomic Counter (Producer-Consumer)

An atomic counter that tracks how many data batches are available. The producer increments, the consumer processes until caught up.

```cpp
// VRAM layout
struct DoorbellCounter {
    volatile uint64_t produced;  // Written by NIC/CPU
    volatile uint64_t consumed;  // Written by GPU
};

// GPU side
__global__ void kernel(DoorbellCounter* db, BatchDesc* batches) {
    uint64_t my_consumed = 0;
    while (true) {
        // Wait for new data
        uint64_t available;
        while ((available = db->produced) == my_consumed) {
            __nanosleep(100);
        }

        // Process all available batches
        for (uint64_t i = my_consumed; i < available; i++) {
            process(batches[i % RING_SIZE]);
        }

        my_consumed = available;
        __threadfence_system();
        db->consumed = my_consumed;
    }
}

// Producer (NIC writes via RDMA)
// 1. Write batch data to VRAM buffer
// 2. Write batch descriptor to batches[index]
// 3. Increment db->produced via atomic RDMA write
```

**Pros:**
- Supports batching (multiple signals before GPU processes)
- Natural flow control (producer sees consumed count)
- No lost signals — counter always reflects total produced
- Single atomic write for doorbell

**Cons:**
- Ordering: must ensure data write completes before counter increment
- Counter wrap-around handling (use 64-bit — effectively infinite)
- Still no rich metadata per batch

**Verdict:** Good foundation. Combine with a descriptor ring for metadata.

### 3. Ring Buffer in VRAM (Circular Queue)

A circular buffer with head/tail pointers and structured entries. Each entry describes one data batch.

```cpp
// VRAM layout
struct RingEntry {
    uint64_t data_offset;    // Offset into data buffer
    uint32_t data_size;      // Bytes in this batch
    uint32_t batch_id;       // Sequence number
    uint32_t flags;          // Metadata flags
    uint32_t padding;        // Alignment
};

struct DoorbellRing {
    volatile uint64_t head;              // Written by producer (NIC/CPU)
    volatile uint64_t tail;              // Written by consumer (GPU)
    RingEntry entries[RING_SIZE];        // Batch descriptors
    uint8_t data_buffer[DATA_BUF_SIZE]; // Actual data region
};

// GPU side
__global__ void kernel(DoorbellRing* ring) {
    uint64_t my_tail = 0;
    while (true) {
        while (ring->head == my_tail) {
            __nanosleep(100);
        }

        // Process entries from tail to head
        while (my_tail != ring->head) {
            RingEntry* entry = &ring->entries[my_tail % RING_SIZE];
            uint8_t* data = ring->data_buffer + entry->data_offset;
            process(data, entry->data_size);
            my_tail++;
        }

        __threadfence_system();
        ring->tail = my_tail;
    }
}
```

**Pros:**
- Rich metadata per batch (size, offset, flags, sequence number)
- Natural queuing with backpressure (ring full = producer waits)
- Multiple in-flight batches
- Decouples data placement from notification

**Cons:**
- More complex RDMA write sequence (data, then descriptor, then head pointer)
- Ring size must be pre-allocated
- Ordering constraints across multiple VRAM writes

**Verdict:** Best option for OuterLink. Provides the structure needed for variable-size batches, multiple data regions, and flow control.

---

## GPU-Side Polling Mechanisms

### Volatile Loads

The `volatile` keyword forces the compiler to issue actual memory loads instead of using cached register values:

```cpp
volatile int* doorbell = ...;
while (*doorbell == 0) { }  // Compiler MUST reload each iteration
```

Without `volatile`, the compiler may optimize to:
```cpp
int cached = *doorbell;
while (cached == 0) { }  // Infinite loop — never re-reads memory
```

### Memory Ordering with `__threadfence_system()`

When the NIC writes data to VRAM followed by a doorbell increment, the GPU must see both writes in order. The NIC's PCIe writes are ordered (PCIe guarantees write ordering within a single source), but the GPU's cache hierarchy may reorder observations.

```cpp
// GPU consumer must ensure it sees data AFTER seeing doorbell
while (*doorbell == old_value) { __nanosleep(100); }
__threadfence_system();  // Ensure all prior writes from NIC are visible
// Now safe to read data
```

The `__threadfence_system()` ensures that all memory writes from any source (including PCIe/NIC) that happened before the doorbell write are visible to this thread before it proceeds to read the data.

### Atomic Operations for Polling

Using `atomicAdd(doorbell, 0)` as a read forces an uncached load:

```cpp
while (atomicAdd(doorbell, 0) == 0) {
    __nanosleep(100);
}
```

This bypasses L1 cache entirely (atomics go through L2), providing stronger visibility guarantees than `volatile` alone. However, it is slower per operation (~10ns vs ~3ns for volatile load).

### Hardware-Assisted Polling: `cuStreamWaitValue32/64`

CUDA provides `cuStreamWaitValue32()` and `cuStreamWaitValue64()` which poll a memory location in hardware without tying up SMs:

```cpp
// CPU-side setup (not usable from within a kernel)
cuStreamWaitValue64(stream, (CUdeviceptr)doorbell, expected_value,
                    CU_STREAM_WAIT_VALUE_GEQ);
```

This is a **CPU-side API** — it makes a CUDA stream wait until a memory location reaches a value. The polling happens in the GPU's copy/command engine, not in SMs. This is ideal for non-persistent patterns but cannot be used inside a running kernel.

**OuterLink opportunity:** Use `cuStreamWaitValue` for the *initial* doorbell wait (before launching work), and kernel-side polling for subsequent batches within the persistent kernel.

---

## OpenDMA Integration: NIC Writes Doorbell to VRAM via BAR1

This is OuterLink's killer integration point. With OpenDMA, the ConnectX-5 NIC's DMA engine writes directly to GPU VRAM through the PCIe BAR1 aperture. The doorbell mechanism leverages this:

### Write Sequence

```
Remote Node                    Local NIC (ConnectX-5)         Local GPU VRAM
-----------                    --------------------           --------------
1. RDMA Write data    ------> DMA to BAR1 ----------------> data_buffer[offset]
2. RDMA Write desc    ------> DMA to BAR1 ----------------> ring.entries[idx]
3. RDMA Write head    ------> DMA to BAR1 ----------------> ring.head++
                                                              |
                                                              v
                                                         GPU kernel sees
                                                         head change, reads
                                                         descriptor, processes
                                                         data
```

### PCIe Ordering Guarantees

PCIe specification guarantees that writes from a single source (the NIC) to a single destination (the GPU's BAR1) arrive in order. This means:
1. Data write completes before descriptor write
2. Descriptor write completes before head pointer write
3. GPU sees consistent state when it observes the head pointer change

**Critical caveat from NVIDIA docs:** Despite PCIe ordering, a GPU kernel running concurrently with NIC writes may observe stale data due to GPU L1/L2 caching. The GPU must use volatile reads or atomics on the doorbell, and issue `__threadfence_system()` after observing the doorbell change before reading data.

### Latency Analysis

| Stage | Latency |
|---|---|
| Remote GPU VRAM to remote NIC | ~0.5-1us (PCIe DMA) |
| Wire transfer (100Gbps, 1KB) | ~0.1us |
| Local NIC to local GPU VRAM (BAR1) | ~0.5-1us (PCIe DMA) |
| GPU cache sees doorbell change | ~0.1-0.5us (L2 coherency) |
| Kernel wakeup from nanosleep poll | 0-0.1us (poll interval) |
| **Total end-to-end** | **~1.2-2.7us** |

Compare with host-staged path (without persistent kernel):
- RDMA to host: ~1us
- cudaMemcpy to GPU: ~3-5us
- Kernel launch: ~5-20us
- **Total: ~9-26us**

The persistent kernel + OpenDMA path is **4-10x faster** than host-staged with kernel relaunch.

---

## Existing Doorbell Implementations

### GPUrdma (Technion, 2016)

GPUrdma moves InfiniBand queue pair (QP) and completion queue (CQ) structures into GPU memory. The NIC's doorbell address is mapped into the GPU address space, allowing GPU threads to ring the NIC doorbell directly.

Key findings:
- Achieved 5us one-way latency between GPUs across the network
- 50Gbps bandwidth for messages >= 16KB
- Outperformed CPU RDMA by 4.5x for small packets (2-1024 bytes)
- Encountered correctness issues with persistent kernels due to GPU-NIC memory consistency

### NVSHMEM IBGDA (NVIDIA, 2022+)

NVSHMEM's IBGDA transport implements the full GPU-initiated doorbell pattern:

1. GPU SM creates a NIC Work Queue Entry (WQE) in GPU memory
2. GPU SM updates the Doorbell Record (DBR) in GPU memory
3. GPU SM notifies NIC by writing to NIC's doorbell register
4. NIC reads WQE from GPU memory via GPUDirect RDMA
5. NIC executes the RDMA transfer

**Optimization:** Batching multiple doorbells — ringing once with the latest sequence number triggers all pending operations. Reduces PCIe write traffic.

**Consistency issue:** NVSHMEM documents that GPU-NIC memory consistency is only enforced across kernel boundaries, not within persistent kernels. ROC_SHMEM (AMD) explicitly addresses this with stronger consistency guarantees.

### GPUnet (UT Austin, 2014)

GPUnet provides a socket-like API for GPU programs. Internally, it maps RDMA channels to GPU streams, maintaining socket tables in GPU memory. Each active socket holds flow-control metadata for receive and send buffers.

### NVIDIA DOCA GPUNetIO (2023+)

The modern successor: GPUNetIO (part of DOCA SDK for BlueField DPUs) allows the GPU to execute RDMA send and receive at any point during kernel execution, not just at kernel boundaries. This is specifically designed for persistent kernel patterns.

Available from DOCA v2.7+. Requires BlueField DPU (OuterLink uses ConnectX-5 NIC instead, which does not support GPUNetIO directly, but the doorbell pattern is the same).

---

## Memory Placement and Cache Coherency

### Where to Place the Doorbell

| Location | GPU Read Latency | NIC Write Method | Pros | Cons |
|---|---|---|---|---|
| **GPU VRAM (device memory)** | ~3ns (L1 hit), ~50ns (L2) | BAR1 DMA | Fast GPU access, NIC writes via BAR1 | Cache coherency challenges |
| **Host pinned memory** | ~500ns (PCIe round-trip) | Direct write | Simple, always coherent | High polling overhead for GPU |
| **GPU VRAM (uncached)** | ~50ns (always L2) | BAR1 DMA | No L1 stale data | Slightly slower reads |
| **Managed memory** | Variable | Via driver | Automatic migration | Unpredictable latency |

**Recommendation for OuterLink:** Place doorbell in GPU VRAM with `volatile` access pattern. The doorbell is a single cache line (64 bytes) — even at L2 latency (~50ns), polling cost is negligible. The key is that the GPU must not cache the doorbell value in L1 or registers.

### Ensuring Visibility of NIC Writes

When the NIC writes to GPU VRAM via BAR1, the data lands in the GPU's L2 cache (or bypasses to VRAM depending on the GPU architecture). The L1 cache on each SM may still hold stale data.

**Pattern for correct observation:**

```cpp
// Doorbell pointer declared volatile — prevents register caching
volatile uint64_t* head = &ring->head;

// Each poll re-reads from L2 (volatile prevents L1 optimization)
while (*head == my_tail) {
    __nanosleep(100);
}

// After seeing doorbell change, fence ensures data writes are also visible
__threadfence_system();

// Now safe to read data that the NIC wrote before updating head
RingEntry* entry = &ring->entries[my_tail % RING_SIZE];
```

### BAR1 Aperture Size Consideration

The RTX 3090 has a 256MB BAR1 aperture (or up to full VRAM with resizable BAR enabled). The doorbell ring structure is tiny (a few KB), but the data buffers can be large. With resizable BAR enabled:
- Full 24GB VRAM accessible via BAR1
- NIC can write data anywhere in VRAM
- Doorbell + data buffers all in VRAM, all accessible via BAR1

Without resizable BAR:
- Only 256MB accessible via BAR1
- Must carefully partition: doorbell ring + data buffers within 256MB
- Larger transfers may need host staging for overflow

---

## Recommended Doorbell Design for OuterLink

### Architecture

```
+------------------------------------------------------------------+
|                        GPU VRAM                                   |
|                                                                   |
|  +------------------+  +------------------------------------+    |
|  | Doorbell Ring     |  | Data Buffer Pool                   |    |
|  |                  |  |                                    |    |
|  | head: uint64     |  | [slot 0: 64KB] [slot 1: 64KB] ... |    |
|  | tail: uint64     |  | [slot N-1: 64KB]                   |    |
|  |                  |  |                                    |    |
|  | entries[0..N-1]: |  +------------------------------------+    |
|  |   data_slot: u32 |                                            |
|  |   data_size: u32 |  +------------------------------------+    |
|  |   batch_id: u64  |  | Output Buffer Pool                 |    |
|  |   flags: u32     |  | (for processed results)            |    |
|  +------------------+  +------------------------------------+    |
|                                                                   |
+------------------------------------------------------------------+
         ^                        ^
         |                        |
    NIC writes head          NIC writes data
    (RDMA atomic)            (RDMA write via BAR1)
```

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Ring size | 256 entries | Enough queuing depth for burst traffic |
| Data slot size | 64KB | Matches common DMA transfer granularity |
| Total data buffer | 16MB (256 x 64KB) | Fits easily in BAR1 aperture |
| Doorbell poll interval | 100ns (`__nanosleep`) | Sub-microsecond detection with low power |
| Head/tail width | 64-bit | No wrap-around concerns |

### Write Protocol (Producer — Remote Node)

1. RDMA Write: data to `data_buffer[slot]`
2. RDMA Write: descriptor to `ring.entries[head % RING_SIZE]`
3. RDMA Write: increment `ring.head` (or RDMA atomic add)
4. Steps 1-2-3 are ordered by PCIe write ordering within the same NIC

### Read Protocol (Consumer — GPU Persistent Kernel)

1. Poll `ring.head` via volatile read with `__nanosleep(100)`
2. On change: `__threadfence_system()`
3. Read descriptor from `ring.entries[tail % RING_SIZE]`
4. Process data from `data_buffer[entry.data_slot]`
5. Increment local tail, write back to `ring.tail`
6. `__threadfence_system()` to make tail visible to producer

---

## Open Questions

- [ ] Does RDMA atomic add to VRAM (via BAR1) work on ConnectX-5 with non-GPUDirect registration? Needs testing with OpenDMA path.
- [ ] What is the measured L2 cache latency for volatile reads of NIC-written VRAM on RTX 3090?
- [ ] Can we use CUDA's `cuStreamWaitValue64` to implement a hybrid doorbell that avoids SM polling for the initial wait?
- [ ] Ring buffer vs. multi-queue: should each OuterLink client connection get its own ring, or share a single ring with connection IDs in descriptors?

---

## Related Documents

- [01-persistent-kernel-patterns.md](./01-persistent-kernel-patterns.md) — Persistent kernel fundamentals
- [R28: Scatter-Gather DMA](../../R28-scatter-gather-dma/) — Multi-region DMA feeding the ring buffer
- [OpenDMA Design](../../../../docs/architecture/) — PCIe BAR1 direct VRAM access
- [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [GPUrdma Paper](https://marksilberstein.com/wp-content/uploads/2020/04/ross16net.pdf)
- [NVSHMEM IBGDA Documentation](https://docs.nvidia.com/nvshmem/api/using.html)
- [GPU-Centric Communication Landscape (2024)](https://arxiv.org/html/2409.09874v2)
- [NVIDIA DPDK/GPUdev Persistent Kernel Pattern](https://developer.nvidia.com/blog/optimizing-inline-packet-processing-using-dpdk-and-gpudev-with-gpus/)
