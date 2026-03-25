# R20 Research: NCCL Net Plugin API

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Document the exact C API surface that OuterLink must implement to register as a custom NCCL transport backend (`libnccl-net-outerlink.so`). This covers plugin discovery, versioning, struct definitions, function signatures, and the connection lifecycle.

---

## 1. Plugin Discovery and Loading

When NCCL initializes, it searches for a shared library to load as a network plugin:

1. **Library naming:** NCCL looks for `libnccl-net.so` by default. If `NCCL_NET_PLUGIN` is set, it looks for `libnccl-net-${NCCL_NET_PLUGIN}.so`. For OuterLink, users would set `NCCL_NET_PLUGIN=outerlink`, making NCCL load `libnccl-net-outerlink.so`.

2. **Symbol lookup:** After loading the library, NCCL looks for symbols named `ncclNet_vX` with X decreasing from the latest version it supports. It uses the first (highest version) symbol it finds. Plugins should export multiple version symbols for broad compatibility.

3. **CollNet (optional):** Plugins may also export `ncclCollNet_vX` symbols for in-network collective operations (e.g., SHARP). This is separate from the point-to-point `ncclNet` API.

4. **Version negotiation:** NCCL starts at its highest known version and falls back. If NCCL 2.29 knows v11, it tries `ncclNet_v11`, then `ncclNet_v10`, down to some minimum. Plugins export whichever versions they implement.

---

## 2. API Version History

The net plugin API has evolved through multiple versions. Each version is defined in a separate header (`net_v4.h`, `net_v6.h`, `net_v8.h`, etc.) in the NCCL source tree under `ext-net/example/nccl/`.

| Plugin API | NCCL Version | Key Changes |
|------------|-------------|-------------|
| v4 | Pre-2.12 | Original plugin API: init, devices, getProperties, listen, connect, accept, regMr, deregMr, isend, irecv, iflush, test, closeSend, closeRecv, closeListen |
| v5 | 2.12.x | Added grouped receives and tags |
| v6 | 2.17.x | Added `regMrDmaBuf` for DMA-BUF support, `latency` and `maxRecvs` to properties. Became the long-standing documented baseline |
| v7 | ~2.18.x | Incremental |
| v8 | ~2.19-2.21 | Added `getDeviceMr`, `irecvConsumed`, `regIsGlobal`/`netDeviceType`/`netDeviceVersion` to properties |
| v9 | ~2.22-2.24 | Changed `isend`/`irecv` size params from `int` to `size_t`. Added `makeVDevice` |
| v10 | ~2.25-2.27 | Added `ncclNetCommConfig_t` to `connect`. Added `ncclNetVDeviceProps_t` to `makeVDevice` |
| v11 | 2.28.3+ | Per-communicator init/finalize with `ctx`+`commId`. Multi-Request Net API (`maxMultiRequestSize`). App-aware networking. `finalize` function |

### Recommended Target for OuterLink

**Primary: v8** (widest compatibility with modern NCCL). **Also implement: v9, v10, v11** for future-proofing.

Rationale:
- v8 covers NCCL 2.19+, which is the minimum version in most current ML stacks
- AWS OFI NCCL maintains backward compatibility to NCCL 2.17.1 (v6), but v8 is a reasonable floor
- v11 is needed for NCCL 2.28+ features (per-communicator context, multi-request)
- Implementing v8 first, then adding thin shim layers for v9-v11 follows the NCCL example plugin pattern

---

## 3. ncclNet_v8_t Struct Definition

This is the primary struct to implement. The v8 API is the recommended baseline target.

```c
typedef struct {
  // Plugin name (for NCCL logging)
  const char* name;

  // Initialize the network plugin
  // logFunction: NCCL's logging callback
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);

  // Return number of network adapters/devices
  ncclResult_t (*devices)(int* ndev);

  // Get properties for device `dev`
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v8_t* props);

  // Create a listening endpoint on device `dev`
  // Fills `handle` (up to NCCL_NET_HANDLE_MAXSIZE bytes) for the remote side
  // Returns `listenComm` for later accept() calls
  // MUST NOT block. listenComm must not be NULL on success.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);

  // Connect to remote endpoint using `handle` from remote listen()
  // Returns `sendComm` (may be NULL if connection is pending — NCCL retries)
  // sendDevComm: optional device-side handle for GDR
  // MUST NOT block.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm,
                          ncclNetDeviceHandle_v8_t** sendDevComm);

  // Accept incoming connection on `listenComm`
  // Returns `recvComm` (may be NULL if pending — NCCL retries)
  // recvDevComm: optional device-side handle for GDR
  // MUST NOT block.
  ncclResult_t (*accept)(void* listenComm, void** recvComm,
                         ncclNetDeviceHandle_v8_t** recvDevComm);

  // Register a memory region for RDMA
  // type: NCCL_PTR_HOST or NCCL_PTR_CUDA
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type,
                        void** mhandle);

  // Register memory region with DMA-BUF fd (for GPU memory)
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, int size, int type,
                              uint64_t offset, int fd, void** mhandle);

  // Deregister a previously registered memory region
  ncclResult_t (*deregMr)(void* comm, void* mhandle);

  // Asynchronous send. Returns a request handle.
  // size: bytes to send (int in v8, size_t in v9+)
  // tag: message tag for matching
  // mhandle: from regMr
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag,
                        void* mhandle, void** request);

  // Asynchronous grouped receive. Returns a request handle.
  // n: number of buffers in this grouped receive
  // sizes: array of sizes (int* in v8, size_t* in v9+)
  // tags: array of tags for matching
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes,
                        int* tags, void** mhandles, void** request);

  // Flush received data (ensures GPU sees the data after RDMA write)
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes,
                         void** mhandles, void** request);

  // Test if an async operation (isend/irecv/iflush) has completed
  // done: set to 1 when complete
  // sizes: actual bytes transferred (on completion)
  ncclResult_t (*test)(void* request, int* done, int* sizes);

  // Close send/recv/listen comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Copy mhandle to device-accessible format (for GDR device-side API)
  ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);

  // Notify plugin that device has consumed a received buffer
  ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);

} ncclNet_v8_t;
```

### Key Differences in Newer Versions

**v9** (over v8):
- `isend` and `irecv` use `size_t` instead of `int` for sizes
- Adds `makeVDevice(int* d, ncclNetVDeviceProps_v9_t* props)` for virtual NIC support

**v10** (over v9):
- `connect` gains `ncclNetCommConfig_v10_t* config` parameter
- `init` gains `ncclProfilerCallback_t profFunction` parameter

**v11** (over v10):
- `init` becomes `init(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction)` — returns per-communicator context
- `listen` and `connect` gain `void* ctx` parameter
- Adds `finalize(void* ctx)` to clean up per-communicator state
- `ncclNetProperties_v11_t` adds `maxMultiRequestSize` for Multi-Request API

---

## 4. ncclNetProperties_v8_t

```c
typedef struct {
  char* name;            // NIC name (for logging)
  char* pciPath;         // PCI path (for topology detection)
  uint64_t guid;         // Globally unique ID for the NIC
  int ptrSupport;        // NCCL_PTR_HOST | NCCL_PTR_CUDA | NCCL_PTR_DMABUF
  int regIsGlobal;       // If 1, regMr is global (enables registration caching)
  int speed;             // Link speed in Mbps (100000 = 100Gbps)
  int port;              // Port number
  float latency;         // Network latency in microseconds
  int maxComms;          // Maximum number of connections
  int maxRecvs;          // Max buffers per grouped receive
  int netDeviceType;     // Device type for GDR device-side API
  int netDeviceVersion;  // Device API version for GDR
} ncclNetProperties_v8_t;
```

### Critical Fields for OuterLink

| Field | OuterLink Value | Notes |
|-------|----------------|-------|
| `name` | "outerlink" | Appears in NCCL logs |
| `pciPath` | PCIe path of ConnectX NIC or local GPU | Critical for topology detection |
| `guid` | Unique per OuterLink transport device | Must be globally unique across cluster |
| `ptrSupport` | `NCCL_PTR_HOST` initially, add `NCCL_PTR_CUDA` with OpenDMA | Determines whether NCCL passes GPU pointers directly |
| `regIsGlobal` | 0 initially, 1 when registration cache is implemented | Enables NCCL's buffer pre-registration |
| `speed` | Depends on transport: TCP=10000, ConnectX=100000, USB4=80000 | Mbps. Affects NCCL's algorithm tuning |
| `latency` | TCP~100us, RDMA~2us, OpenDMA~2us | Affects ring vs tree threshold |
| `maxComms` | 256+ (OuterLink can handle many connections) | |
| `maxRecvs` | 8 (typical) | Must handle maxRecvs * NCCL_NET_MAX_REQUESTS concurrent ops |

---

## 5. Connection Lifecycle

### Phase 1: Initialization

```
NCCL calls init() -> Plugin initializes OuterLink transport layer
NCCL calls devices() -> Plugin returns count of available transports
NCCL calls getProperties(dev) for each device -> Plugin reports capabilities
```

### Phase 2: Connection Establishment

```
Receiver: listen(dev, &handle, &listenComm)
  -> Plugin creates listening endpoint, fills handle with address info
  -> Handle is serialized (up to NCCL_NET_HANDLE_MAXSIZE bytes)

[NCCL bootstrap exchanges handle between ranks]

Sender: connect(dev, handle, &sendComm, &sendDevComm)
  -> Plugin connects to remote endpoint
  -> May return sendComm=NULL (pending) — NCCL retries

Receiver: accept(listenComm, &recvComm, &recvDevComm)
  -> Plugin accepts the incoming connection
  -> May return recvComm=NULL (pending) — NCCL retries
```

**Non-blocking requirement:** `listen`, `connect`, and `accept` MUST NOT block. If the connection is not ready, return `ncclSuccess` with a NULL comm pointer. NCCL will retry.

### Phase 3: Memory Registration

```
NCCL calls regMr(comm, data, size, type, &mhandle)
  -> Plugin prepares buffer for RDMA (pin memory, register with NIC, etc.)
  -> Returns opaque mhandle

[On cleanup] NCCL calls deregMr(comm, mhandle)
```

### Phase 4: Data Transfer

```
Sender: isend(sendComm, data, size, tag, mhandle, &request)
  -> Plugin initiates async send, returns request handle

Receiver: irecv(recvComm, n, data[], sizes[], tags[], mhandles[], &request)
  -> Plugin initiates async grouped receive

Both: test(request, &done, &sizes)
  -> Plugin checks completion. done=1 when finished.
  -> NCCL polls this in a tight loop

[After recv completes, if using RDMA]
Receiver: iflush(recvComm, n, data[], sizes[], mhandles[], &request)
  -> Plugin flushes data to ensure GPU visibility
  -> test() again until done
```

### Phase 5: Teardown

```
closeSend(sendComm)
closeRecv(recvComm)
closeListen(listenComm)
```

---

## 6. Concurrency Requirements

NCCL uses multiple channels (parallel execution contexts) and expects the plugin to handle concurrent operations:

- Each `sendComm` or `recvComm` must handle up to `NCCL_NET_MAX_REQUESTS` (typically 8) concurrent async operations
- With grouped receives (`maxRecvs`), the total concurrent operations per comm can be `maxRecvs * NCCL_NET_MAX_REQUESTS`
- Multiple channels means multiple independent send/recv comm pairs per peer connection
- The plugin must be thread-safe: NCCL may call plugin functions from different threads

---

## 7. CollNet API (Optional, Future)

The `ncclCollNet_vX` struct enables in-network collective operations. It shares many functions with `ncclNet` but adds collective-specific operations. This is how SHARP acceleration works.

For OuterLink, CollNet could be used to implement:
- Hardware-accelerated AllReduce using R29 RDMA multicast
- In-network aggregation for gradient reduction

CollNet is NOT required for initial implementation. The point-to-point `ncclNet` API is sufficient for all collective operations (NCCL decomposes them internally).

---

## 8. Handle Size and Serialization

`NCCL_NET_HANDLE_MAXSIZE` defines the maximum size of the handle buffer exchanged between ranks. The plugin must serialize its connection information (IP address, port, transport type, etc.) into this buffer.

For OuterLink, the handle would contain:
- Transport type (TCP, RDMA, USB4)
- Endpoint address (IP:port for TCP, GID+QPN for RDMA)
- Transport capabilities (compression support, OpenDMA support)

---

## 9. Return Codes

All plugin functions return `ncclResult_t`:

| Code | Meaning |
|------|---------|
| `ncclSuccess` | Operation succeeded (or pending for connect/accept) |
| `ncclSystemError` | System error (malloc failed, socket error, etc.) |
| `ncclInternalError` | Plugin internal error |
| `ncclInvalidArgument` | Bad argument passed by NCCL |

---

## 10. Symbol Export Pattern

The plugin exports versioned symbols as global `ncclNet_vX` variables:

```c
// Primary implementation at v8
ncclNet_v8_t ncclNetPlugin_v8 = {
  .name = "outerlink",
  .init = pluginInit,
  .devices = pluginDevices,
  // ... all function pointers
};

// Thin shim for v9 (size_t sizes, makeVDevice)
ncclNet_v9_t ncclNetPlugin_v9 = { ... };

// Export symbols NCCL looks for
ncclNet_v8_t* ncclNet_v8 = &ncclNetPlugin_v8;
ncclNet_v9_t* ncclNet_v9 = &ncclNetPlugin_v9;
// etc.
```

---

## Related Documents

- [02-existing-nccl-plugins.md](./02-existing-nccl-plugins.md) — Survey of existing plugins
- [03-nccl-topology-and-collectives.md](./03-nccl-topology-and-collectives.md) — How NCCL uses transport internally
- R14 Transport Compression — Compressed collective operations
- R17 Topology-Aware Scheduling — NCCL ring/tree topology integration
- R29 RDMA Multicast — Hardware-accelerated broadcast for CollNet

## Open Questions

1. **NCCL_NET_HANDLE_MAXSIZE value** — What is the exact size? Need to verify from NCCL headers. Likely 128 or 256 bytes. OuterLink handle must fit within this.

2. **Multi-transport handle encoding** — How to encode ConnectX + USB4 + TCP capability info in a single handle? May need a transport negotiation step.

3. **regMr with OuterLink virtual memory** — OuterLink virtualizes GPU memory. When NCCL calls regMr with a virtual address, how does the plugin resolve it to the actual remote GPU? Need coordination with the VRAM manager.

4. **iflush implementation for OpenDMA** — With direct NIC-to-GPU writes via PCIe BAR1, what flush mechanism is needed? PCIe write ordering should handle this, but need to verify.

5. **Thread safety model** — Does NCCL guarantee single-thread access per comm, or can multiple threads call isend/irecv on the same comm simultaneously? Affects our locking strategy.

6. **NCCL_NET_MAX_REQUESTS value** — What is the default? Likely 8. This determines how many concurrent async ops per comm we must support.

7. **v11 per-communicator context** — With v11, NCCL creates a separate context per communicator. How does this map to OuterLink's connection pooling? One transport connection per NCCL communicator, or shared?

## Sources

- NVIDIA NCCL ext-net README: https://github.com/NVIDIA/nccl/blob/master/ext-net/README.md
- NVIDIA NCCL example plugin: https://github.com/NVIDIA/nccl/blob/master/ext-net/example/plugin.c
- NCCL net.h header: https://github.com/NVIDIA/nccl/blob/master/ext-net/example/nccl/net.h
- RCCL Net Plugin API docs (AMD, mirrors NCCL API): https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/using-nccl.html
- NCCL Release Notes: https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html
- NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
