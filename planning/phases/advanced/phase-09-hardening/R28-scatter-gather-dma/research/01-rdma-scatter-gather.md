# R28 Research: RDMA Scatter-Gather Elements (SGE)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Complete
**Priority:** MEDIUM

## Purpose

Document how RDMA Scatter-Gather Elements work in libibverbs, ConnectX-5 hardware limits, performance characteristics of multi-SGE work requests vs separate transfers, and memory registration requirements. This is the foundation for OuterLink's non-contiguous VRAM transfer capability.

---

## 1. The ibv_sge Struct

Every RDMA work request references one or more scatter-gather elements (SGEs). Each SGE describes a contiguous memory region:

```c
struct ibv_sge {
    uint64_t addr;    // Virtual address of the buffer
    uint32_t length;  // Length of the buffer in bytes
    uint32_t lkey;    // Local key of the Memory Region (MR)
};
```

- **addr:** Start of a registered memory buffer. Must fall within a registered MR.
- **length:** Byte count to read/write from this address.
- **lkey:** Local key returned by `ibv_reg_mr()`. Authorizes the NIC to DMA to/from this address.

Multiple SGEs in a single work request tell the NIC to gather data from (send side) or scatter data to (receive side) several non-contiguous buffers as if they were one logical message.

---

## 2. SGE in Work Requests

### Send Side (ibv_post_send)

```c
struct ibv_send_wr {
    // ...
    struct ibv_sge *sg_list;    // Pointer to array of SGEs
    int             num_sge;    // Number of SGEs in the array
    enum ibv_wr_opcode opcode;  // SEND, RDMA_WRITE, RDMA_READ, etc.
    // ...
};
```

For **RDMA WRITE**: The NIC gathers data from multiple local SGEs and writes it to one contiguous remote address. This is a gather operation.

For **RDMA READ**: The NIC reads from one contiguous remote address and scatters data into multiple local SGEs. This is a scatter operation.

For **SEND**: The NIC gathers from multiple local SGEs into one message.

### Receive Side (ibv_post_recv)

```c
struct ibv_recv_wr {
    // ...
    struct ibv_sge *sg_list;
    int             num_sge;
    // ...
};
```

Incoming data is scattered across the listed SGEs in order: the first `sg_list[0].length` bytes go to `sg_list[0].addr`, the next `sg_list[1].length` bytes go to `sg_list[1].addr`, and so on.

### Key Rule

Every work request is one message, regardless of how many SGEs it has. A WR with 5 SGEs sends/receives exactly one message assembled from 5 fragments.

---

## 3. ConnectX-5 SGE Limits

### Device-Reported Maximums

Queried via `ibv_query_device()` in `struct ibv_device_attr`:

| Parameter | ConnectX-5 Value | Description |
|-----------|-----------------|-------------|
| `max_sge` | **30** | Max SGEs per WR for non-RD QPs (Send + Recv) |
| `max_sge_rd` | **30** | Max SGEs per WR for RD QPs |
| `max_qp_wr` | **16,351** | Max outstanding WRs per QP |

**Why 30 and not 32:** The mlx5 driver uses a Work Queue Element (WQE) segment-based format. Each WQE has a fixed control segment (16 bytes) and data segments (16 bytes each for SGE). The internal WQE size constrains the maximum SGE count to ~30 for ConnectX-5.

### Per-QP Configuration

When creating a QP, the actual SGE limit is set via `ibv_qp_init_attr`:

```c
struct ibv_qp_init_attr attr = {
    .cap = {
        .max_send_wr = 1024,
        .max_recv_wr = 1024,
        .max_send_sge = 16,   // Request up to 30
        .max_recv_sge = 16,   // Request up to 30
    },
    // ...
};
```

The device may return a lower value than requested. Always check the actual capability after QP creation.

### OuterLink Implications

With 30 SGEs per WR and 64KB pages (from R10), a single work request can reference:
- **30 x 64KB = 1.875 MB** of non-contiguous VRAM per WR
- With 16,351 WRs queued, theoretical maximum scatter-gather capacity per QP is enormous

For typical sparse tensor transfers, 30 SGEs per WR is sufficient for most fragmentation patterns.

---

## 4. Performance: N SGEs vs N Separate WRs

### General Guidance

The RDMA community consensus from extensive benchmarking:

1. **1 SGE per WR is fastest for raw throughput.** Each additional SGE requires the NIC to perform an extra memory read to fetch the SGE descriptor, then another read for the actual data. More SGEs = more memory accesses = lower throughput.

2. **Multiple SGEs in 1 WR vs. multiple WRs are NOT functionally equivalent.** N SGEs in 1 WR = 1 message. N WRs each with 1 SGE = N messages. The receiver sees different behavior.

3. **Chaining multiple WRs in one ibv_post_send() call** (linked list) is faster than calling ibv_post_send() N times, because the driver can batch-ring the doorbell once.

### Performance Hierarchy (Best to Worst)

| Approach | Doorbell Rings | NIC Memory Reads | Messages |
|----------|---------------|-------------------|----------|
| 1 WR, 1 SGE (contiguous) | 1 | 2 (WQE + data) | 1 |
| 1 WR, N SGEs (scatter-gather) | 1 | 1 + N (WQE + N SGE descriptors + N data reads) | 1 |
| N chained WRs, 1 SGE each | 1 | 2N (N WQEs + N data reads) | N |
| N separate ibv_post_send() calls | N | 2N | N |

### When Scatter-Gather Wins

Scatter-gather (multi-SGE) is preferred when:
- The receiver needs the data as **one logical message** (atomic delivery)
- Pre-packing into a contiguous buffer would require an extra CPU copy
- The fragments are small enough that NIC descriptor overhead < CPU copy cost
- Latency matters more than peak throughput (1 completion vs N completions)

### When Separate WRs Win

Separate WRs (chained, 1 SGE each) are preferred when:
- Each fragment is independently useful to the receiver
- Peak bandwidth matters (fewer NIC descriptor reads per byte)
- Fragments are large (the SGE descriptor overhead is negligible vs data)

### Alignment Effects

Memory alignment dramatically impacts performance:
- **4KB-aligned buffers:** ~350 clock cycles per ibv_post_send
- **8-byte-aligned buffers:** ~3000 clock cycles per ibv_post_send
- **Reason:** TLB misses inside the HCA. Hugepages (2MB) eliminate this entirely.

OuterLink's 64KB pages are naturally 4KB-aligned (and can be hugepage-backed), so alignment is not a concern.

---

## 5. Memory Registration Requirements

### Every SGE Buffer Must Be Registered

Each address in an SGE must fall within a Memory Region (MR) registered via `ibv_reg_mr()`. The NIC uses the lkey to validate access and translate virtual addresses.

### Registration Options for Scatter-Gather

| Strategy | Description | Overhead | Flexibility |
|----------|-------------|----------|-------------|
| **One large MR** | Register entire VRAM as one MR | 1 registration | Any address works, one lkey |
| **Per-page MRs** | Register each 64KB page separately | N registrations | Fine-grained control, N lkeys |
| **MR spanning regions** | Register contiguous chunks | M registrations (M << N) | Balance of overhead and control |

### Recommended: Single Large MR

For OuterLink, registering the entire BAR1 aperture (or the full VRAM range) as a single MR is optimal:
- One lkey for all SGEs — simplifies descriptor building
- Zero registration overhead at transfer time
- Pages within the MR can be non-contiguous in physical memory (virtual address space is contiguous)
- R10's page table already tracks which virtual pages are valid

### On-Demand Paging (ODP)

ConnectX-5 supports ODP (On-Demand Paging), which eliminates upfront registration entirely. Pages are registered on first access. This adds ~1us latency on first access but avoids pinning memory.

For OuterLink's GPU VRAM (always pinned via BAR1), ODP provides no benefit — standard registration is preferred.

---

## 6. Practical ibv_post_send with Scatter-Gather

### Example: Gather 4 Non-Contiguous VRAM Regions into One RDMA WRITE

```c
// Four non-contiguous 64KB pages in local VRAM
struct ibv_sge sge_list[4] = {
    { .addr = vram_page_0, .length = 65536, .lkey = mr->lkey },
    { .addr = vram_page_7, .length = 65536, .lkey = mr->lkey },
    { .addr = vram_page_3, .length = 65536, .lkey = mr->lkey },
    { .addr = vram_page_15, .length = 65536, .lkey = mr->lkey },
};

struct ibv_send_wr wr = {
    .sg_list = sge_list,
    .num_sge = 4,
    .opcode = IBV_WR_RDMA_WRITE,
    .send_flags = IBV_SEND_SIGNALED,
    .wr.rdma = {
        .remote_addr = remote_contiguous_buf,
        .rkey = remote_rkey,
    },
};

struct ibv_send_wr *bad_wr;
int ret = ibv_post_send(qp, &wr, &bad_wr);
```

This gathers 4 x 64KB = 256KB from 4 non-contiguous local addresses and writes them contiguously to the remote side — one doorbell, one completion.

### Reverse: Scatter on Receive

```c
struct ibv_sge recv_sge[4] = {
    { .addr = local_page_A, .length = 65536, .lkey = mr->lkey },
    { .addr = local_page_B, .length = 65536, .lkey = mr->lkey },
    { .addr = local_page_C, .length = 65536, .lkey = mr->lkey },
    { .addr = local_page_D, .length = 65536, .lkey = mr->lkey },
};

struct ibv_recv_wr recv_wr = {
    .sg_list = recv_sge,
    .num_sge = 4,
};

ibv_post_recv(qp, &recv_wr, &bad_recv_wr);
```

Incoming 256KB message is scattered across 4 non-contiguous local pages.

---

## 7. Verdict for OuterLink

### SGE Approach: Plan A

Use RDMA scatter-gather for non-contiguous VRAM transfers:
- 30 SGEs per WR on ConnectX-5 covers most fragmentation patterns
- 30 x 64KB = 1.875 MB per WR — sufficient for typical sparse tensor fragments
- Single MR over entire VRAM — one lkey for all SGEs
- NIC handles gather/scatter in hardware — no CPU involvement

### Fallback: Software Pre-Pack (Plan B)

If scatter-gather overhead exceeds benefit (very large transfers with many fragments):
- CPU/GPU kernel packs fragments into contiguous staging buffer
- Single-SGE RDMA WRITE of the packed buffer
- Receiver unpacks into non-contiguous destinations
- Higher latency (extra copy) but higher throughput for bulk transfers

### Decision Threshold

Use scatter-gather when: fragment_count <= 30 AND total_size < ~2MB
Use software pre-pack when: fragment_count > 30 OR fragments are extremely small (< 4KB each)

---

## Related Documents

- [R10 Memory Tiering](../../../R10-memory-tiering/) — 64KB page abstraction, page table
- [R14 Transport Compression](../../../R14-transport-compression/) — Compress gathered data before send
- [R17 Topology-Aware Scheduling](../../../R17-topology-scheduling/) — Multi-path for large scatter-gather
- [02-gpu-sparse-data.md](./02-gpu-sparse-data.md) — GPU sparse data patterns that drive scatter-gather
- [03-scatter-gather-pipeline.md](./03-scatter-gather-pipeline.md) — Full pipeline design

## Open Questions

- [ ] Exact `ibv_devinfo -v` output from Pedro's ConnectX-5 to confirm max_sge = 30
- [ ] Does UCX abstract scatter-gather, or do we need raw verbs for this?
- [ ] Performance delta: 30 SGEs in 1 WR vs software pre-pack on our specific hardware
