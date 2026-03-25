# R16 Research: BlueField Programming Models

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** APPROVED
**Purpose:** Document how to program BlueField DPUs — DOCA SDK, DPDK, Rust integration — and determine the best approach for OuterLink.

---

## 1. DOCA SDK Overview

DOCA (Data Center infrastructure On a Chip Architecture) is NVIDIA's official SDK for programming BlueField DPUs. It is analogous to what CUDA is for GPUs — a unified framework that abstracts hardware capabilities behind higher-level APIs.

### SDK Components

| Component | Purpose |
|---|---|
| **DOCA Core** | Base primitives: devices, memory maps, buffers, contexts, progress engine |
| **DOCA Flow** | Hardware packet processing pipeline (match, action, forward) |
| **DOCA DMA** | Hardware-accelerated memory copy (host↔DPU, local) |
| **DOCA Compress** | Hardware compression/decompression (deflate, LZ4 on BF-3) |
| **DOCA RDMA** | RDMA operations (send/recv, read/write) from DPU |
| **DOCA Comm Channel** | Host↔DPU control plane communication |
| **DOCA SHA / AES-GCM** | Cryptographic acceleration |
| **DOCA RegEx** | Hardware regular expression matching |
| **DOCA GPUNetIO** | GPU↔NIC direct communication (GPUDirect RDMA + GDRCopy) |
| **DOCA Erasure Coding** | Reed-Solomon erasure coding in hardware |
| **DOCA DPA** | Data Path Accelerator programming (BF-3 on-path cores) |

### SDK Architecture

```
┌──────────────────────────────────────────────┐
│              Your Application                │
├──────────────────────────────────────────────┤
│    DOCA Libraries (Flow, DMA, Compress...)   │
├──────────────────────────────────────────────┤
│              DOCA Core                        │
│   (Device, Mmap, Buf, Ctx, Progress Engine)  │
├──────────────────────────────────────────────┤
│         Hardware Abstraction Layer            │
├──────────────────────────────────────────────┤
│    ConnectX NIC │ ARM SoC │ HW Accelerators  │
└──────────────────────────────────────────────┘
```

### DOCA Core Primitives

**doca_dev (Device):** Represents a BlueField device instance. Applications discover devices and select based on capabilities or PCIe topology.

**doca_mmap (Memory Map):** Maps host or local memory for hardware access. Key mechanism for host↔DPU shared memory. The host exports an mmap, the DPU creates a remote mmap from the export descriptor, and then DMA/RDMA can operate on it.

**doca_buf / doca_buf_inventory (Buffers):** Buffer descriptors that point into mmap regions. Allocated from an inventory pool. Scatter-gather supported.

**doca_ctx (Context):** A hardware processing unit wrapper. Each library (DMA, Compress, etc.) provides its own context type. Contexts are attached to a progress engine.

**doca_pe (Progress Engine):** Async task completion mechanism. One PE per thread. Poll or event-driven. Multiple contexts can share one PE.

### Typical DOCA Code Flow

```
1. Discover device          → doca_devinfo_list_create()
2. Open device              → doca_dev_open()
3. Create mmap              → doca_mmap_create() + set_memrange() + start()
4. Create buf inventory     → doca_buf_inventory_create() + start()
5. Create context           → doca_dma_create() / doca_compress_create() / etc.
6. Attach context to PE     → doca_pe_connect_ctx()
7. Submit tasks             → doca_task_submit()
8. Progress (poll/wait)     → doca_pe_progress()
9. Handle completions       → callback functions
10. Cleanup                 → destroy in reverse order
```

---

## 2. DOCA Flow (Hardware Packet Processing)

DOCA Flow programs the NIC's hardware flow tables to match, modify, and forward packets entirely in hardware — no ARM core involvement for matched packets.

### Pipeline Model
- Define **pipes** with match criteria (L2-L4 headers, tunnel headers)
- Attach **actions** (modify, encap/decap, count, meter)
- Chain pipes together for multi-stage processing
- Unmatched packets (miss) escalate to ARM cores as exceptions

### Relevance to OuterLink
- Can identify OuterLink protocol packets in hardware and route them directly
- Matched packets bypass host entirely — DPU handles them
- Miss path (new connections, control messages) goes to ARM cores for software handling
- BF-3's APP cores (64-128 packet processing cores) handle complex per-packet logic at line rate

### Performance Characteristics
- Flow insertion: thousands of rules per second
- Matched packet latency: sub-microsecond (hardware steering)
- Number of pipes impacts latency — shorter pipelines = faster
- Large entry tables require main memory access, increasing latency

### gRPC Interface
DOCA Flow also has a gRPC server that lets the host program flow rules on the DPU remotely. Useful for initial bootstrapping before DPU-side software is ready.

---

## 3. DOCA DMA (Host Memory Access)

DOCA DMA enables the DPU to read/write host memory (and vice versa) using hardware-accelerated DMA engines.

### How It Works

1. **Host side** allocates memory, creates a `doca_mmap`, exports it via `doca_mmap_export()`
2. **Host side** sends the export descriptor + buffer address + length to DPU via Comm Channel
3. **DPU side** creates remote mmap from export descriptor via `doca_mmap_create_from_export()`
4. **DPU side** creates remote `doca_buf` pointing into the host's memory
5. **DPU side** submits DMA copy tasks between local and remote buffers
6. Hardware DMA engine performs the copy — no CPU involved

### Key Properties
- Asynchronous — submit task, poll for completion
- Works from DPU-to-host and host-to-DPU directions
- Hardware accelerated (uses the same DMA engines as RDMA)
- Maximum transfer size limited by hardware (varies by model)
- DPU must be in DPU mode (ECPF) for cross-domain DMA

### Relevance to OuterLink
- DPU can DMA data from host's CUDA pinned memory to its own buffers
- DPU can then transmit via RDMA to remote node — host CPU never touched the data
- Combined with GPUDirect RDMA, DPU's ConnectX can access GPU VRAM BAR1 directly

---

## 4. DOCA Compress (Hardware Compression)

Hardware compression offload engine on the BlueField SoC.

### Supported Algorithms

| Algorithm | BlueField-2 | BlueField-3 |
|---|---|---|
| **Deflate compress** | Yes (HW) | Yes (HW) |
| **Deflate decompress** | Yes (HW) | Yes (HW) |
| **LZ4 compress** | No | Yes (HW) |
| **LZ4 decompress** | No | Yes (HW) |
| **LZ4 stream** | No | Yes (HW) |
| **LZ4 block** | No | Yes (HW) |

### Performance
- HW compression is ~100x faster than software on the same ARM cores (measured up to 26.8x speedup for realistic workloads)
- Operates as a GGA (Generic Global Accelerator) — part of ARM complex but does not use NIC bandwidth
- Shares PCIe and memory bandwidth with NIC operations
- Completion time scales linearly with data size
- Scatter-gather support for compressing linked buffer lists
- Task batching for aggregating multiple compress operations

### LZ4 Constraint
- LZ4 source buffer must be in **local memory** (DPU DRAM), not remote host memory
- For OuterLink: data must be DMA'd to DPU first, then compressed, then transmitted. This is the natural flow anyway.

### Relevance to OuterLink
- Directly offloads R14's compression work
- BF-2: deflate only (still useful, widely compatible)
- BF-3: LZ4 at line rate — matches OuterLink's preferred compression (R14 selected LZ4 for CPU path)
- Checksum computed as side effect of compression — can verify data integrity

---

## 5. DPDK on BlueField

DPDK (Data Plane Development Kit) is an alternative to DOCA for raw packet processing.

### How It Works on BlueField
- BlueField's ConnectX NIC exposes an MLX5 PMD (Poll Mode Driver) to DPDK
- Packets can be received/transmitted using DPDK's standard ring-based API
- DPDK compressdev API (mlx5 driver) exposes the hardware compression engine
- Works alongside DOCA or standalone

### When to Use DPDK vs DOCA
| Criteria | DPDK | DOCA |
|---|---|---|
| Raw packet processing | Better (more control) | Good (DOCA Flow) |
| Host memory access | Manual (PCIe BARs) | DOCA DMA (managed) |
| Compression | compressdev API | DOCA Compress (richer) |
| Learning curve | Moderate (familiar to network devs) | Steep (NVIDIA-specific) |
| Forward compatibility | Standard API | Guaranteed across BF generations |
| Rust integration | DPDK-sys crates exist | FFI to C (custom) |

### Verdict
- DOCA is preferred for host↔DPU integration (DMA, Comm Channel)
- DPDK is viable for pure packet processing
- Can mix: use DOCA for host interaction + DPDK for packet I/O
- For OuterLink: DOCA is the right choice — we need DMA, Compress, and host integration more than raw packet I/O

---

## 6. Rust on ARM64 BlueField

### Cross-Compilation Target
- BlueField runs standard aarch64 Linux (Ubuntu or RHEL/CentOS based)
- Rust target: `aarch64-unknown-linux-gnu`
- Toolchain: `aarch64-linux-gnu-gcc` cross-compiler

### Build Strategy Options

**Option A: Cross-compile from x86_64 host**
```
# Add target
rustup target add aarch64-unknown-linux-gnu

# Install cross-linker
apt install gcc-aarch64-linux-gnu

# Build
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  cargo build --target aarch64-unknown-linux-gnu
```
- Pro: Fast build on powerful host
- Con: Linking against DOCA libs requires aarch64 sysroot with DOCA SDK installed

**Option B: Build natively on BlueField ARM cores**
- Pro: No cross-compilation issues, native linking
- Con: 8 A72 cores at 2 GHz — compilation will be slow
- Con: Limited eMMC storage (16-64 GB)

**Option C: Docker cross-compilation (recommended)**
- Use `cross-rs/cross` tool with custom Docker image containing DOCA SDK aarch64 libs
- Pro: Reproducible, no host pollution, handles sysroot automatically
- Pro: Can include DOCA headers + libraries in the Docker image

### FFI to DOCA

DOCA is a C library. OuterLink's Rust DPU component would use FFI:

```rust
// Low-level FFI bindings (generated with bindgen)
extern "C" {
    fn doca_dma_create(version: u32, dma: *mut *mut doca_dma) -> doca_error_t;
    fn doca_mmap_create(mmap: *mut *mut doca_mmap) -> doca_error_t;
    fn doca_pe_progress(pe: *mut doca_pe) -> doca_error_t;
    // ...
}

// Safe Rust wrapper
pub struct DocaDma { inner: *mut doca_dma }

impl DocaDma {
    pub fn new() -> Result<Self, DocaError> {
        let mut dma = std::ptr::null_mut();
        let err = unsafe { doca_dma_create(DOCA_VERSION, &mut dma) };
        if err != DOCA_SUCCESS { return Err(err.into()); }
        Ok(Self { inner: dma })
    }
}
```

### Crate Structure for DPU
```
outerlink-dpu/
├── doca-sys/          # Raw FFI bindings (bindgen)
├── doca-rs/           # Safe Rust wrappers
└── outerlink-dpu/     # OuterLink DPU service (transport, routing, compression)
```

### Considerations
- DOCA SDK headers are only available after installing DOCA on an aarch64 system (or extracting from the SDK package)
- `bindgen` can generate Rust bindings from DOCA C headers at build time
- DOCA uses opaque pointers extensively — maps well to Rust's ownership model
- The async progress engine model (submit task → poll PE) maps to Rust's async/await with a custom executor

---

## 7. BlueField Modes of Operation

### DPU Mode (Default — What OuterLink Needs)
- ARM cores own the NIC
- All host network traffic flows through DPU's virtual switch
- DPU can inspect, modify, forward, or drop packets
- Host sees a virtual NIC (representor)
- DPU runs full Linux + custom services
- This is the mode where OuterLink's transport logic runs on the DPU

### NIC Mode (Fallback)
- BlueField behaves as a standard ConnectX NIC
- ARM cores idle (BF-3) or minimal (BF-2)
- Host has full NIC control
- This is what OuterLink uses when DPU offload is disabled — same as using a standalone ConnectX card

### Zero-Trust / Restricted Mode
- Extension of DPU mode with additional host isolation
- Host cannot access DPU management
- Data center admin controls DPU via ARM cores or BMC
- Relevant for multi-tenant deployments of OuterLink

### Separated Host Mode (Deprecated)
- Both ARM cores and host have independent NIC functions
- Equal bandwidth share
- NVIDIA has deprecated this mode — do not use

### Mode Switching
- Configured via `mlxconfig` tool on the DPU
- Requires DPU reboot to take effect
- OuterLink installer would set DPU mode as part of setup

---

## 8. What Runs on ARM Cores

The BlueField ARM cores run a full Linux distribution. Anything that runs on ARM64 Linux runs on BlueField:

### Supported Environments
- **Base OS**: Ubuntu 22.04, RHEL 8/9, CentOS (NVIDIA provides BSP images)
- **Containers**: Docker, Podman, containerd — full OCI container support
- **Kubernetes**: DPF (DOCA Platform Framework) provides K8s integration
- **Custom binaries**: Any aarch64-linux-gnu binary
- **Rust binaries**: Cross-compiled or natively built

### Resource Constraints
- **CPU**: 8 cores (BF-2) or 16 cores (BF-3) — shared with DOCA runtime and OS
- **Memory**: 16-32 GB — shared with packet buffers, flow tables, OS
- **Storage**: 16-64 GB eMMC — limited, prefer NFS/network for large artifacts
- **No GPU**: ARM cores have no GPU compute — all ML/inference must be on host GPU

### What OuterLink Would Run on DPU
```
outerlink-dpu (Rust binary)
├── Transport manager (connection state, routing decisions)
├── DOCA DMA handler (host↔DPU data movement)
├── DOCA Compress handler (hardware compression/decompression)
├── DOCA Flow programmer (install/update packet steering rules)
├── Page table manager (track VRAM page ownership/location)
├── Prefetch scheduler (R11 logic at network edge)
└── Control plane (communicate with host-side outerlink-server)
```

---

## 9. Verdict: Programming Model Choice

### Recommended Approach for OuterLink

**Primary: DOCA SDK via Rust FFI**
- Use DOCA Core for device management, memory maps, buffers
- Use DOCA DMA for host memory access
- Use DOCA Compress for hardware compression
- Use DOCA Flow for packet steering (fast-path)
- Use DOCA Comm Channel for host↔DPU control plane

**Secondary: DPDK for packet I/O (if needed)**
- Only if DOCA Flow's packet steering is insufficient
- DPDK gives more control over raw packet processing
- Can coexist with DOCA

**Build: Docker cross-compilation**
- `cross-rs/cross` with custom Docker image containing DOCA SDK
- `bindgen` for generating Rust FFI bindings from DOCA headers
- CI/CD produces both x86_64 (host) and aarch64 (DPU) binaries

**Deployment: DPU mode with systemd service**
- `outerlink-dpu` binary deployed to DPU's filesystem
- Managed as a systemd service on the DPU's Linux
- Communicates with host-side `outerlink-server` via DOCA Comm Channel or shared memory

---

## Related Documents
- [01-bluefield-architecture.md](./01-bluefield-architecture.md) — Hardware specs and capabilities
- [03-outerlink-offload-design.md](./03-outerlink-offload-design.md) — What to offload and how
- [R14: Transport Compression](../../R14-transport-compression/) — Compression algorithms

## Open Questions
- [ ] What DOCA SDK version should we target? (2.9 LTS or 2.10 latest?)
- [ ] Can `bindgen` handle DOCA's header complexity, or do we need manual bindings for some APIs?
- [ ] Should `doca-sys` be a separate published crate or internal to OuterLink?
- [ ] How to handle DOCA SDK version differences between BF-2 and BF-3 in the same codebase?
- [ ] Is the DOCA Comm Channel fast enough for OuterLink's control plane, or should we use shared memory (NTB)?
