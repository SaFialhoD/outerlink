# Feature Plan: Critical Architecture Fixes for Real Hardware

Created: 2026-03-23
Author: architect-agent
Status: draft

## Overview

Six architectural bugs prevent OutterLink from functioning on real GPU hardware. Three are critical (segfaults, wrong CUDA context, corrupted kernel params) and three are important (TCP interleaving, infinite recursion, wire mismatch). This plan designs fixes for all six without introducing workarounds.

## Requirements

- [ ] Server pushes the correct CUDA context before every context-dependent backend call
- [ ] Kernel parameters containing synthetic device pointers are translated to real GPU addresses before launch
- [ ] cuFuncGetParamInfo is never called with synthetic handles in the C interpose layer
- [ ] Multi-threaded clients cannot corrupt each others TCP responses
- [ ] cuGetExportTable does not recurse infinitely through our dlsym hook
- [ ] cuModuleLoadFatBinary wire format matches between client and server

---

## Issue 1: Server Never Calls cuCtxSetCurrent Before CUDA Operations

### Root Cause

handle_request_full() in handler.rs updates session.current_ctx (a metadata field) but never calls the real CUDA driver cuCtxSetCurrent. Since tokio spawns the connection handler with join_set.spawn() (see server.rs:121), requests from different connections can be scheduled on any worker thread. CUDA contexts are thread-local, so thread A may have connection 1 context active while processing connection 2 request.

The CudaGpuBackend methods like ctx_synchronize, mem_alloc, launch_kernel, stream_create, etc. all operate on the current context implicitly but never set it.

### Design Decision: Dedicated CUDA Thread per Connection

**Rejected alternative: call cuCtxSetCurrent inside every CudaGpuBackend method.** This would require passing the context handle to ~60 trait methods, breaking the GpuBackend trait signature and the stub. It also creates a race: two tokio tasks on the same OS thread could interleave set_current and launch_kernel.

**Rejected alternative: backend.ensure_context(ctx) called by the handler.** Same race condition -- tokio can preempt between ensure_context and the backend call.

**Chosen approach: Pin each connection to a dedicated OS thread for CUDA calls.** The server already has per-connection state (ConnectionSession). We add a dedicated std::thread per connection that owns all CUDA driver calls for that session. The tokio task sends requests to this thread via a channel and receives responses back.

### Architecture

tokio task (connection loop) --mpsc::channel<CudaWork>--> std::thread cuda-worker-{session_id}

The worker thread: receives CudaWork, calls cuCtxSetCurrent if context changed, executes the backend closure, sends result back via oneshot.

### Specific Changes

**File: crates/outerlink-server/src/cuda_thread.rs (NEW)** -- CudaThread struct with sender channel (std::sync::mpsc::Sender) and join handle. CudaWork struct: required_ctx, boxed closure, oneshot reply. Worker loop. Constructor spawns OS thread with Arc<dyn GpuBackend>. Shutdown: drop sender, join.

**File: crates/outerlink-server/src/server.rs (MODIFY lines 121-222)** -- Spawn CudaThread per connection. Route context-dependent messages to it. Drop on close.

**File: crates/outerlink-server/src/handler.rs (MODIFY)** -- Add pub fn is_context_dependent(msg_type) -> bool. Context-free: Handshake, Init, DriverGetVersion, DeviceGet, DeviceGetCount, DeviceGetName, DeviceGetAttribute, DeviceTotalMem, DeviceGetUuid, DeviceCanAccessPeer, DeviceGetP2PAttribute, DeviceGetPCIBusId, DeviceGetByPCIBusId, GetErrorName, GetErrorString.

**File: crates/outerlink-server/src/gpu_backend.rs (MODIFY)** -- Add ctx_set_current to GpuBackend trait. Stub: Ok(()), Real: calls cuCtxSetCurrent FFI.

**Primary Context Path:** cuDevicePrimaryCtxRetain returns handle stored in session. cuCtxSetCurrent with that handle works via CudaThread. No special case.

**Thread Count:** One OS thread per connection. 2-8 node clusters = 1-7 workers.

---

## Issue 2: Kernel Parameters Contain Synthetic Device Pointers

### Root Cause

cuLaunchKernel param buffer contains raw bytes. Device pointer params hold synthetic values (0x0D00_xxxx). Client sends verbatim. Server passes to real CUDA expecting real GPU addresses. Function/stream handles ARE translated (ffi.rs:3835-3845), but device pointers in params are NOT.

### Design Decision: Client-Side Prefix-Based Translation

**Rejected A:** Query param types -- cuFuncGetParamInfo gives sizes not types.
**Rejected C:** Always use extra buffer -- still contains synthetic pointers.
**Chosen:** Client scans 8-byte params for DEVICEPTR_PREFIX and translates via handle map.

### Why This Works

DEVICEPTR_PREFIX = 0x0D00_0000_0000_0000 high byte. NVIDIA GPU VAs below 0x0008_xxxx. App scalars never in this range. No collision.

### Specific Changes

**crates/outerlink-common/src/handle.rs (line 12)** -- Export DEVICEPTR_PREFIX as pub.

**crates/outerlink-client/src/ffi.rs (ol_cuLaunchKernel lines 3880-3895)** -- In param loop, for 8-byte params: read u64, check DEVICEPTR mask, translate via device_ptrs.to_remote(), else leave unchanged. Same for cooperative/ex variants.

**Extra buffer limitation:** Cannot scan packed buffer without layout. Acceptable for Phase 1 (PyTorch eager uses kernelParams). Phase 3 FuncGetParamInfo enables later.

---

## Issue 3: cuFuncGetParamInfo Called With Synthetic Handles

### Root Cause

interpose.c:1362 calls real_cuFuncGetParamInfo with synthetic func handle. Segfaults on CUDA >= 12.3.

### Design Decision: Server-Side Query With Client Caching

**Rejected:** Skip entirely -- cannot serialize kernelParams without sizes.
**Rejected:** Cache at cuModuleGetFunction -- extra latency.
**Chosen:** New FuncGetParamInfo message. Server queries real CUDA. Client caches per-function.

### Specific Changes

**protocol.rs** -- Add FuncGetParamInfo = 0x00B0. Wire: req [8B func], resp [4B result][4B count][per param: 4B offset, 4B size].
**gpu_backend.rs** -- fn func_get_param_info. Stub: NotSupported.
**cuda_backend.rs** -- Resolve cuFuncGetParamInfo symbol (optional, CUDA 12.3+). Loop until INVALID_VALUE.
**handler.rs** -- Add FuncGetParamInfo handler.
**ffi.rs** -- Add ol_cuFuncGetParamInfo with DashMap cache. Add param_info_cache to OuterLinkClient.
**interpose.c (lines 1294-1395)** -- Replace real_cuFuncGetParamInfo with ol_cuFuncGetParamInfo.

---

## Issue 4: Multi-Threaded TCP Interleaving

### Root Cause

send_request_once (lib.rs:283) clones Arc, drops mutex, then does send+recv. Two threads interleave and get wrong responses.

### Fix: Hold Mutex for Full Request-Response

**lib.rs lines 283-320** -- Hold connection lock guard across send+recv instead of dropping after Arc clone. Add debug_assert for request_id match. Same for send_request_with_bulk_once.

---

## Issue 5: cuGetExportTable Infinite Recursion

### Root Cause

hook_cuGetExportTable uses dlsym() which we override. Our dlsym returns our hook. Stack overflow.

### Fix

**interpose.c line 1912** -- Change dlsym to real_dlsym with bootstrap guard. Same pattern as lines 1961-1963.

---

## Issue 6: ModuleLoadFatBinary Wire Mismatch

### Root Cause

Client sends [8B len][data]. Server passes entire payload (including len prefix) as fat binary. Corrupts binary.

### Fix

**handler.rs lines 1872-1884** -- Strip 8-byte length prefix, validate, pass payload[8..]. ModuleLoad and ModuleLoadData verified: no mismatch.

---

## Implementation Phases

### Phase 1: Safety Fixes (Issues 4, 5, 6) -- 1 session

- crates/outerlink-client/src/lib.rs -- mutex fix
- crates/outerlink-client/csrc/interpose.c -- real_dlsym fix
- crates/outerlink-server/src/handler.rs -- length prefix fix

### Phase 2: CUDA Thread Pinning (Issue 1) -- 1-2 sessions

- NEW: crates/outerlink-server/src/cuda_thread.rs
- MODIFY: gpu_backend.rs, cuda_backend.rs, server.rs, handler.rs, lib.rs

### Phase 3: Param Info + Pointer Translation (Issues 2, 3) -- 1-2 sessions

- MODIFY: protocol.rs, handle.rs, gpu_backend.rs, cuda_backend.rs, handler.rs, ffi.rs, interpose.c

### Phase 4: Integration Testing

- NEW: context_pinning_tests.rs, kernel_param_tests.rs

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CudaThread channel latency | Medium | ~100ns vs ~100us network RTT |
| Device ptr prefix collision | High | Verify NVIDIA VA ranges; runtime assert |
| cuFuncGetParamInfo < CUDA 12.3 | Medium | Falls back to no-params path |
| Mutex serializes all CUDA calls | Low | Acceptable; serial from CUDA perspective |
| CudaThread blocks tokio | Low | tokio::sync::oneshot is await-able |

## Open Questions

- [ ] Bounded vs unbounded channel for CudaThread? Rec: bounded(64)
- [ ] Extra buffer device pointers: defer? Rec: yes, PyTorch uses kernelParams
- [ ] Request-ID validation now? Rec: yes, as debug_assert

## Success Criteria

1. Vector addition runs unmodified on real hardware, two nodes
2. No segfaults from synthetic handles
3. Multi-threaded PyTorch: no corrupted results from TCP interleaving
4. Two concurrent connections: no context confusion

## Related Documents

- docs/architecture/00-project-vision.md
- planning/pre-planning/02-FINAL-PREPLAN.md
- crates/outerlink-common/src/handle.rs (DEVICEPTR_PREFIX, line 12)
- crates/outerlink-server/src/handler.rs (7571 lines)
- crates/outerlink-client/csrc/interpose.c
