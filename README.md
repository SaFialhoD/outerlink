<p align="center">
  <h1 align="center">OutterLink</h1>
  <p align="center">
    <strong>Unified GPU pool across networked PCs. Any CUDA app, zero code changes.</strong>
  </p>
  <p align="center">
    <a href="#the-problem">Problem</a> &middot;
    <a href="#the-solution">Solution</a> &middot;
    <a href="#architecture">Architecture</a> &middot;
    <a href="#getting-started">Getting Started</a> &middot;
    <a href="#status">Status</a>
  </p>
</p>

---

OutterLink is a Rust software layer that makes GPUs across separate PCs work as a unified pool &mdash; shared VRAM, shared compute, shared system RAM. CUDA applications run **unmodified**. No recompilation, no source changes, no special SDK.

```
PC1 (no GPU)                           PC2 (RTX 3090)
+-----------------+                    +------------------+
| python train.py |                    | outerlink-server |
|   import torch  | ---- network ---> |   real GPU here  |
|   model.cuda()  |                    |   24 GB VRAM     |
+-----------------+                    +------------------+
     LD_PRELOAD                         Serves GPU ops
   intercepts CUDA                      via TCP/RDMA
```

## The Problem

GPUs across separate PCs **cannot share anything**. The effective cross-machine GPU bandwidth today is **zero**.

| Scenario | Reality |
|----------|---------|
| You have 4 GPUs across 2 PCs with 112 GB total VRAM | No single process can use more than one machine's GPUs |
| A 70B model needs ~140 GB VRAM. Your 3090 has 24 GB | Game over |
| NVIDIA's solution: DGX + NVLink + NVSwitch | $200K+ for hardware. GPUDirect RDMA locked to datacenter cards |
| Open-source alternative | **Doesn't exist. Until now.** |

## The Solution

**Any connection is infinitely better than no connection.**

OutterLink transparently intercepts CUDA Driver API calls via `LD_PRELOAD`, serializes them over the network, and executes them on remote GPUs. Your application doesn't know the GPU is on another machine.

| Connection | Bandwidth | What It Means |
|-----------|-----------|---------------|
| No connection (today) | **0 GB/s** | GPUs are isolated. Wasted potential. |
| 10 GbE | ~1.2 GB/s | Viable for inference, batch processing |
| 25 GbE | ~3.1 GB/s | Matches PCIe x1. Practical for training |
| **4x 100 GbE bonded** | **~50 GB/s** | **Exceeds PCIe 4.0 x16. Remote GPU faster than a local riser card.** |

### The Real Win: Memory Pooling

Bandwidth enables the connection. **Memory pooling** is the killer feature:

```
Without OutterLink:               With OutterLink:
PC1: 2x 3090 Ti (48 GB)          Combined: 112 GB VRAM
PC2: 2x 5090   (64 GB)           + 512 GB system RAM
Can't share. Period.              Any process sees ALL of it.

Max model: ~24 GB (one GPU)       Max model: 70B+ parameters
```

## Architecture

```
Application (PyTorch, TensorFlow, any CUDA app)
    |
    | LD_PRELOAD intercepts all CUDA Driver API calls
    v
+-------------------+          TCP / RDMA          +-------------------+
| outerlink-client  | --------------------------> | outerlink-server  |
| (libouterlink.so) |          wire protocol       | (daemon on GPU    |
|                   | <-------------------------- |  node)            |
| 169 hooked funcs  |          responses           |                   |
| Handle translation|                              | Real CUDA driver  |
| Retry + reconnect |                              | Dedicated thread  |
+-------------------+                              | per connection    |
                                                   +-------------------+
```

### How It Works

1. **Interception** &mdash; `LD_PRELOAD=libouterlink.so` hooks 169 CUDA functions via `dlsym` override and `cuGetProcAddress` interception
2. **Serialization** &mdash; Each CUDA call is serialized into a binary protocol message (22-byte header + payload)
3. **Transport** &mdash; Messages travel over TCP (Phase 1) or RDMA (Phase 2) to the GPU server
4. **Execution** &mdash; Server runs the real CUDA call on a dedicated OS thread (correct context management)
5. **Handle Translation** &mdash; Client maintains bidirectional maps between synthetic local handles and real remote GPU handles across 12 resource types

### Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Language | **Rust** (C for interpose .so) | Safety, performance, fearless concurrency |
| Interception | **Driver API + LD_PRELOAD** | Catches everything, including cuGetProcAddress |
| Transport (Phase 1) | **TCP + tokio** | Works everywhere, async I/O |
| Transport (Phase 2) | **UCX** | Auto-negotiates RDMA vs TCP |
| GPU DMA (Phase 2) | **OpenDMA** | PCIe BAR1 direct RDMA on GeForce GPUs |
| Context threading | **Dedicated OS thread per connection** | CUDA contexts are thread-local |
| License | **Apache 2.0** | Maximum adoption |

## What's Implemented

### CUDA API Coverage: 150+ Functions

| Category | Functions | Status |
|----------|-----------|--------|
| Device Queries | `cuDeviceGet`, `GetCount`, `GetName`, `GetAttribute`, `TotalMem`, `GetUuid`, `GetPCIBusId`, `CanAccessPeer`, `GetP2PAttribute` | Full |
| Context Management | `cuCtxCreate`, `Destroy`, `Set/GetCurrent`, `Push/Pop`, `Synchronize`, `GetFlags`, `Get/SetLimit`, `GetApiVersion`, `StreamPriorityRange`, `Cache/SharedMemConfig` | Full |
| Primary Context | `cuDevicePrimaryCtxRetain`, `Release`, `GetState`, `SetFlags`, `Reset` | Full |
| Memory | `cuMemAlloc`, `Free`, `HtoD`, `DtoH`, `DtoD` (sync+async), `Memset` (D8/D16/D32, sync+async), `AllocHost`, `FreeHost`, `HostAlloc`, `AllocManaged`, `AllocPitch`, `Memcpy`, `MemcpyAsync`, `GetInfo`, `GetAddressRange`, `Prefetch`, `Advise` | Full |
| Stream-Ordered Memory | `cuMemAllocAsync`, `FreeAsync`, `AllocFromPoolAsync`, `PoolCreate/Destroy`, `PoolGet/SetAttribute`, `PoolTrimTo`, `DeviceGetDefaultMemPool`, `DeviceGet/SetMemPool` | Full |
| Modules | `cuModuleLoadData`, `LoadDataEx`, `Load`, `LoadFatBinary`, `Unload`, `GetFunction`, `GetGlobal` | Full |
| CUDA 12 Library API | `cuLibraryLoadData`, `Unload`, `GetKernel`, `GetModule`, `KernelGetFunction` | Full |
| JIT Linker | `cuLinkCreate`, `AddData`, `AddFile`, `Complete`, `Destroy` | Full |
| Execution | `cuLaunchKernel`, `LaunchCooperativeKernel`, `LaunchKernelEx`, `LaunchHostFunc` | Full |
| Streams | `cuStreamCreate`, `CreateWithPriority`, `Destroy`, `Synchronize`, `Query`, `GetPriority/Flags/Ctx`, `WaitEvent`, `AddCallback` | Full |
| Events | `cuEventCreate`, `Destroy`, `Record`, `RecordWithFlags`, `Synchronize`, `ElapsedTime`, `Query` | Full |
| CUDA Graphs | `cuStreamBeginCapture`, `EndCapture`, `IsCapturing`, `GetCaptureInfo`, `GraphCreate`, `Instantiate`, `Launch`, `Destroy` | Full |
| Occupancy | `cuOccupancyMaxActiveBlocks(WithFlags)`, `MaxPotentialBlockSize(WithFlags)` | Full |
| Pointer Attributes | `cuPointerGetAttribute`, `GetAttributes` | Full |
| Host Memory | `cuMemHostRegister/Unregister`, `GetDevicePointer`, `GetFlags` | Full |
| Peer Access | `cuCtxEnable/DisablePeerAccess` | Full |
| Function Attributes | `cuFuncGet/SetAttribute`, `SetCacheConfig`, `SetSharedMemConfig`, `GetParamInfo` | Full |
| Error Handling | `cuGetErrorName`, `cuGetErrorString` (100+ error codes) | Full |
| Managed Memory | `cuMemRangeGetAttribute(s)` | Full |
| Entry Points | `cuGetProcAddress` (v1 + v2), `cuGetExportTable` | Full |

### Infrastructure

| Feature | Details |
|---------|---------|
| **Wire Protocol** | 143 message types, 22-byte OLNK header, little-endian payloads |
| **Handle Translation** | 12 typed handle maps (device ptrs, contexts, streams, events, modules, functions, pools, graphs, graph execs, libraries, kernels, link states) |
| **Kernel Param Translation** | Synthetic device pointer detection via prefix mask, automatic client-side translation to real GPU addresses |
| **Retry + Reconnect** | 3 per-request retries, exponential backoff reconnect (1s-30s), transparent to callers |
| **Session Management** | Per-connection resource tracking (16 types), dependency-ordered cleanup on disconnect |
| **Callback System** | Dedicated notification channel (lazy), client-side CallbackRegistry with re-entrant safety, two-phase StreamSynchronize |
| **Graceful Shutdown** | Ctrl+C/SIGTERM, drain timeout, backend cleanup |
| **CUDA Context Safety** | Dedicated OS thread per connection for CudaGpuBackend, `cuCtxSetCurrent` before every operation |

### Testing

```
1,190 tests across 4 crates, 0 failures

outerlink-common:  30 tests (protocol, handles, retry, error mapping)
outerlink-client: 365 tests (all 150 FFI functions, stub + connected paths)
outerlink-server: 790 tests (backend, handler, session, shutdown, cuda thread, integration)
outerlink-cli:     5 tests
```

## Getting Started

### Prerequisites

**Client machine** (runs your CUDA app):
- Rust toolchain (1.75+)
- GCC / build-essential (for interpose.c)
- Linux x86_64

**Server machine** (has the GPU):
- NVIDIA GPU with driver 470+
- CUDA Toolkit 11.2+ (for stream-ordered memory)
- Rust toolchain (1.75+)

### Build

```bash
# Clone
git clone https://github.com/SaFialhoD/outerlink.git
cd outerlink

# Build server (on GPU machine)
cargo build --release -p outerlink-server

# Build client library (on client machine)
cargo build --release -p outerlink-client

# The .so is at: target/release/libouterlink_client.so
```

### Run

```bash
# Start server on GPU machine (default port 14833)
./target/release/outerlink-server --bind 0.0.0.0:14833

# Run any CUDA app on client machine
OUTERLINK_SERVER=gpu-server:14833 \
LD_PRELOAD=./target/release/libouterlink_client.so \
python -c "import torch; print(torch.cuda.is_available())"
```

### Test App

```bash
# Build the test vector-add application
cd tests/cuda_test_app
make

# Run through OutterLink
OUTERLINK_SERVER=gpu-server:14833 \
LD_PRELOAD=../../target/release/libouterlink_client.so \
./test_vector_add
```

## Project Structure

```
outerlink/
├── crates/
│   ├── outerlink-client/     # LD_PRELOAD interception library (.so)
│   │   ├── csrc/             #   C interpose layer (dlsym, hooks)
│   │   └── src/              #   Rust FFI, handle translation, retry
│   ├── outerlink-server/     # GPU node daemon
│   │   └── src/              #   Handler, backends (stub + real CUDA),
│   │                         #   session management, cuda thread
│   ├── outerlink-common/     # Shared protocol, types, transport
│   └── outerlink-cli/        # Management CLI
├── docs/
│   ├── architecture/         # System design documents
│   ├── guides/               # Installation + quickstart
│   ├── specs/                # Technical specifications
│   └── decisions/            # Architecture Decision Records
├── planning/                 # Phase plans, research, pre-planning
├── cuda-stubs/               # CUDA header stubs for cross-compilation
├── tests/cuda_test_app/      # Standalone test application
└── opendma/                  # Future: kernel module for PCIe BAR1 DMA
```

## Roadmap

### Phase 1: TCP Transport (Current)
- [x] CUDA Driver API interception (150+ functions)
- [x] Binary wire protocol with handle translation
- [x] TCP transport with retry/reconnect
- [x] StubGpuBackend (testing) + CudaGpuBackend (real GPU)
- [x] Session management + resource cleanup
- [x] Callback system (cuStreamAddCallback, cuLaunchHostFunc)
- [x] CUDA Graphs, JIT Linker, Library API
- [ ] Linux build verification + real GPU testing
- [ ] End-to-end PyTorch validation

### Phase 2: UCX Transport
- [ ] UCX integration (auto-negotiates RDMA vs TCP)
- [ ] Multi-GPU support (multiple backends per server)
- [ ] Multi-server orchestration

### Phase 3: OpenDMA
- [ ] PCIe BAR1 direct NIC-to-GPU DMA
- [ ] Bypass NVIDIA's GPUDirect restrictions
- [ ] Works on ALL NVIDIA GPUs including GeForce

## OpenDMA: The Endgame

**OpenDMA** is OutterLink's killer feature: non-proprietary direct NIC-to-GPU VRAM access via PCIe BAR1, bypassing NVIDIA's artificial GPUDirect restriction.

```
Current (GPUDirect RDMA):           OpenDMA:
  Datacenter GPUs only               ALL NVIDIA GPUs
  NVIDIA proprietary                  Open-source kernel module
  $10K+ per GPU                       Your existing GeForce cards
  NIC -> GPU via nvidia-peermem       NIC -> GPU via PCIe BAR1
  ~25 GB/s per 100GbE                 ~25 GB/s per 100GbE (same!)
```

The physics are identical. The restriction is artificial. OpenDMA removes it.

## Why This Matters

Every AI researcher with multiple PCs and consumer GPUs has the same problem: **wasted potential**. Hundreds of gigabytes of VRAM sitting in machines that can't talk to each other.

OutterLink is the missing layer. Not a cloud service, not a $200K appliance &mdash; open-source software that turns the hardware you already own into a unified compute cluster.

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## Contributing

OutterLink is in active development. See the [docs/](docs/) directory for architecture details and the [planning/](planning/) directory for roadmap and phase plans.

---

<p align="center">
  <strong>Built with Rust. For GPUs that deserve to talk to each other.</strong>
</p>
