# R20: NCCL Backend — Pre-Plan

**Date Created:** 2026-03-25
**Date Last Updated:** 2026-03-25
**Status:** DRAFT

## Purpose

Define the scope, dependencies, decisions needed, risks, and open questions for implementing an NCCL transport plugin (`libnccl-net-outerlink.so`) that registers OuterLink as a custom NCCL backend. This is the gateway to the entire ML ecosystem — PyTorch, TensorFlow, JAX, DeepSpeed, Megatron-LM all use NCCL for distributed communication.

---

## 1. Scope Definition

### What R20 Covers

1. **NCCL Net Plugin (`libnccl-net-outerlink.so`):**
   - Implements `ncclNet_v8` through `ncclNet_v11` API structs
   - Exports versioned symbols (`ncclNet_v8`, `ncclNet_v9`, `ncclNet_v10`, `ncclNet_v11`)
   - Adapts NCCL's point-to-point transport API to OuterLink's transport layer
   - Supports all OuterLink transports: TCP, RDMA (ConnectX-5), USB4, OpenDMA

2. **Plugin Discovery Integration:**
   - Named `libnccl-net-outerlink.so` (activated via `NCCL_NET_PLUGIN=outerlink`)
   - Installs to standard library paths
   - Works alongside other NCCL plugins (no conflicts)

3. **Transport Property Reporting:**
   - Reports each OuterLink transport as a separate NCCL device
   - Accurate PCI paths for topology detection
   - Correct speed/latency for algorithm tuning

4. **Connection Lifecycle:**
   - listen/connect/accept mapping to OuterLink transport connections
   - Memory registration (regMr) coordinated with OuterLink's VRAM manager
   - Async send/recv/test mapped to OuterLink's async transport operations

5. **Testing and Validation:**
   - Pass nccl-tests suite (all_reduce_perf, all_gather_perf, etc.)
   - Validate with PyTorch distributed training
   - Performance benchmarks vs native NCCL IB transport

### What R20 Does NOT Cover

- **CollNet (in-network collectives):** Deferred to future phase. ncclNet point-to-point API is sufficient for all collectives.
- **NCCL tuner plugin:** Separate plugin for algorithm/protocol tuning. Consider in future phase.
- **NCCL profiler plugin:** v10+ supports profiler callbacks. Not in scope for initial implementation.
- **Device-side API (GIN):** NCCL 2.28 GPU-initiated networking. Advanced feature for future consideration.
- **Compression in NCCL path:** R14's transport compression integration with NCCL is a separate concern.

---

## 2. Dependency Mapping

### What R20 Depends On

| Dependency | Status | Why Needed |
|-----------|--------|------------|
| P6 Core Transport | IN PROGRESS | R20 is a thin adapter over OuterLink's transport layer. Without working transport, the plugin has nothing to delegate to. |
| VRAM Manager (P7) | PLANNED | regMr must coordinate with OuterLink's virtual memory system for GPU pointer handling |
| CUDA Driver API interception (P1) | DONE | OuterLink client must be running for the plugin to connect to |
| Build system | DONE | Cargo workspace exists. Plugin needs a new crate or C build target |

### What Depends on R20

| Dependent | Why |
|-----------|-----|
| PyTorch/DeepSpeed integration | NCCL plugin is THE way ML frameworks use OuterLink |
| R14 Compressed Collectives | Compression hooks may integrate through the NCCL plugin path |
| R29 RDMA Multicast | Future CollNet implementation builds on ncclNet |
| R17 Topology-Aware Scheduling | NCCL topology detection feeds into OuterLink's scheduling |
| User adoption | 90%+ of users will interact with OuterLink through NCCL |

### External Dependencies

| Dependency | Version | Notes |
|-----------|---------|-------|
| NCCL headers | 2.19+ (for v8 API) | Copy into OuterLink tree, no build-time NCCL dependency |
| CUDA toolkit | 11.0+ | For CUDA types used in plugin API |
| Linux kernel | 5.4+ | For DMA-BUF support in regMrDmaBuf |

---

## 3. Decision Inventory

### Decision 1: Primary API Version Target

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| **v6 (baseline)** | Widest compatibility (NCCL 2.17+) | Missing device MR, registration caching support |
| **v8 (recommended)** | Good compatibility (NCCL 2.19+), has GDR device API, regIsGlobal | Missing size_t sizes, virtual device support |
| **v9** | size_t sizes (>2GB transfers), virtual devices | Narrower compatibility (NCCL 2.22+) |
| **v11 (latest)** | Per-communicator context, multi-request API | Only NCCL 2.28+ |

**Recommendation:** Implement v8 as the primary API, with shim layers for v9, v10, v11. This follows the aws-ofi-nccl pattern and provides the best compatibility-to-features tradeoff. The v8->v9 shim is trivial (int to size_t casts). The v9->v10->v11 shims add progressively more context parameters.

**Status:** NEEDS DECISION

### Decision 2: Implementation Language

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| **Rust with C FFI** | Consistent with OuterLink codebase, memory safety, direct access to transport layer | C FFI boilerplate, must ensure no panics cross FFI boundary, cbindgen/bindgen complexity |
| **Pure C** | Follows NCCL example plugin pattern, zero FFI overhead, easiest for NCCL developers to review | Second language in codebase, cannot directly use Rust transport layer, separate build system |
| **C shim + Rust core** | Thin C layer for symbol export, Rust for all logic | Best of both but adds complexity |

**Recommendation:** Rust with C FFI (`cdylib` crate type). The plugin is a thin adapter over OuterLink's Rust transport layer. Writing it in Rust avoids duplicating transport logic. The C FFI surface is well-defined (20 function pointers) and stable. Use `#[no_mangle] pub extern "C"` for exported functions and `#[repr(C)]` for structs.

**Status:** NEEDS DECISION

### Decision 3: NCCL Header Management

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| **Copy headers into OuterLink tree** | No external build dependency, follows aws-ofi-nccl pattern | Must update when targeting new API versions |
| **Depend on NCCL installation** | Always current headers | Adds build dependency, complicates distribution |
| **Generate Rust bindings from headers** | Type-safe FFI, catches API changes at compile time | bindgen complexity, generated code is verbose |

**Recommendation:** Copy the versioned `net_vX.h` headers into OuterLink's tree (under `crates/outerlink-nccl-plugin/nccl-headers/`). Write Rust struct definitions manually matching the C definitions. This is the cleanest approach: no bindgen dependency, no NCCL build dependency, and the struct definitions change infrequently.

**Status:** NEEDS DECISION

### Decision 4: Multi-Transport Device Model

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| **One device per transport** | NCCL handles multi-device topology natively, channels distributed across transports | May confuse NCCL if transports have very different characteristics |
| **Single virtual device** | Simpler, OuterLink internally selects best transport per transfer | NCCL cannot optimize topology across transports, loses multi-path benefit |
| **Aggregate virtual device** | Report combined bandwidth, OuterLink manages transport selection | Hides real topology from NCCL |

**Recommendation:** One device per transport. Report ConnectX as device 0 (speed=100000, latency=2.0), USB4 as device 1 (speed=80000, latency=5.0), TCP as device 2 (speed=10000, latency=100). NCCL's topology engine is sophisticated and handles heterogeneous devices. This lets NCCL assign channels optimally across transports.

**Status:** NEEDS DECISION

### Decision 5: GPU Pointer Support Timeline

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| **NCCL_PTR_HOST only (Phase 1)** | Simplest, works immediately with host-staged transport | Extra copy: GPU->host->network->host->GPU |
| **NCCL_PTR_HOST + NCCL_PTR_CUDA immediately** | Better performance, less copying | Requires working GPUDirect or OpenDMA |

**Recommendation:** Start with `NCCL_PTR_HOST` only. Add `NCCL_PTR_CUDA` when RDMA (Phase 2) or OpenDMA (Phase 5) is ready. This matches OuterLink's phased transport development.

**Status:** NEEDS DECISION

---

## 4. Risk Assessment

### High Risk

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **NCCL internal API changes** | Plugin breaks on NCCL update | Medium | Multi-version support (v8-v11). Monitor NCCL releases. Follow versioning pattern that makes old plugins still work. |
| **Performance below NCCL expectations** | NCCL falls back to TCP socket, defeating the purpose | Medium | Accurate property reporting (speed, latency). Pre-allocate request pools. Minimize plugin overhead (async operations, no blocking). |
| **regMr coordination with VRAM manager** | Memory registration fails or is slow for virtualized GPU memory | High | Design regMr to delegate to VRAM manager. Implement registration cache early. Test with real GPU workloads. |

### Medium Risk

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **Non-blocking requirement violations** | NCCL hangs if plugin blocks in listen/connect/accept | Medium | All transport operations must be async. Connect returns NULL if pending. Unit test non-blocking behavior. |
| **Concurrency bugs** | Data corruption under NCCL's multi-channel access | Medium | Thread-safety testing. Each comm object has independent state. No shared mutable state without synchronization. |
| **Handle serialization overflow** | OuterLink handle exceeds NCCL_NET_HANDLE_MAXSIZE | Low | Verify handle max size. Compress handle data if needed. Keep handle lean. |
| **PCI path accuracy** | NCCL makes bad topology decisions | Medium | Test with `NCCL_DEBUG=GRAPH` to see topology detection. Verify PCI paths on real hardware. |

### Low Risk

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **Plugin loading conflicts** | Other NCCL plugins interfere | Low | `NCCL_NET_PLUGIN=outerlink` is explicit. Only one plugin loaded at a time. |
| **Rust FFI panics** | Undefined behavior if Rust panics across C boundary | Low | Catch all panics at FFI boundary with `catch_unwind`. Return ncclInternalError on panic. |

---

## 5. Implementation Phases (Within R20)

### Phase A: Skeleton Plugin (1-2 weeks)

- Create `crates/outerlink-nccl-plugin/` crate (cdylib)
- Define Rust structs matching ncclNet_v8_t
- Export `ncclNet_v8` symbol with stub implementations
- Verify NCCL discovers and loads the plugin
- All functions return `ncclSuccess` with dummy data
- **Deliverable:** `NCCL_DEBUG=INFO` shows "OuterLink plugin loaded"

### Phase B: TCP Transport Backend (2-3 weeks)

- Implement listen/connect/accept over OuterLink TCP transport
- Implement regMr (pin host memory)
- Implement isend/irecv/test using OuterLink async transport
- Report `NCCL_PTR_HOST`, accurate speed/latency
- **Deliverable:** Pass `nccl-tests all_reduce_perf` over TCP

### Phase C: Multi-Version Support (1 week)

- Add v9, v10, v11 shim layers
- Export all version symbols
- Test with multiple NCCL versions
- **Deliverable:** Same binary works with NCCL 2.19 through 2.29

### Phase D: RDMA Transport Backend (2-3 weeks)

- Add ConnectX-5 RDMA path in the plugin
- Implement regMr with IB verbs memory registration
- Report `NCCL_PTR_CUDA` when GPUDirect is available
- Implement iflush for GPU memory visibility
- **Deliverable:** Pass nccl-tests over RDMA with GPUDirect

### Phase E: Multi-Transport and USB4 (2 weeks)

- Report multiple devices (ConnectX + USB4 + TCP)
- Verify NCCL channel distribution across devices
- Test combined bandwidth
- **Deliverable:** nccl-tests shows combined multi-transport bandwidth

### Phase F: OpenDMA Integration (follows Phase 5)

- Add OpenDMA path when PCIe BAR1 transport is ready
- True zero-copy GPU-to-GPU transfers
- Lowest latency path
- **Deliverable:** nccl-tests with OpenDMA showing <5us latency

---

## 6. Validation Criteria

### Must Pass

- [ ] NCCL discovers and loads `libnccl-net-outerlink.so` via `NCCL_NET_PLUGIN=outerlink`
- [ ] `nccl-tests all_reduce_perf` completes without errors (2+ nodes)
- [ ] `nccl-tests all_gather_perf` completes without errors
- [ ] `nccl-tests broadcast_perf` completes without errors
- [ ] `nccl-tests sendrecv_perf` completes without errors
- [ ] PyTorch `torch.distributed` initializes with NCCL backend over OuterLink
- [ ] DeepSpeed ZeRO training runs to completion
- [ ] No data corruption (verified with checksum validation)
- [ ] Plugin handles connection failures gracefully (no hang, returns error)

### Performance Targets

| Metric | TCP Target | RDMA Target | OpenDMA Target |
|--------|-----------|-------------|----------------|
| AllReduce bandwidth (8 GPU, 1GB) | >5 Gbps | >80 Gbps | >90 Gbps |
| AllReduce latency (8 GPU, 8B) | <1 ms | <50 us | <20 us |
| Plugin overhead per operation | <10 us | <2 us | <1 us |

---

## 7. Crate Structure

```
crates/outerlink-nccl-plugin/
  Cargo.toml              # crate-type = ["cdylib"]
  nccl-headers/           # Copied NCCL header definitions (net_v8.h etc.)
  src/
    lib.rs                # FFI exports, symbol definitions
    ffi_types.rs          # #[repr(C)] struct definitions matching NCCL
    plugin.rs             # Core plugin logic
    transport_adapter.rs  # Maps NCCL ops to OuterLink transport
    device_manager.rs     # Multi-transport device enumeration
    memory_registry.rs    # regMr implementation and caching
    connection.rs         # listen/connect/accept lifecycle
    async_ops.rs          # isend/irecv/test request management
    version_shims.rs      # v9/v10/v11 compatibility layers
```

---

## 8. Open Questions

### Architecture

1. **Should the plugin communicate with OuterLink server directly, or through the client library?** If through the client library, the plugin depends on LD_PRELOAD being active. If direct, it needs its own transport connection to the server.

2. **Connection pooling:** NCCL creates many channels, each with send/recv comms. Should the plugin create a dedicated OuterLink transport connection per NCCL comm, or pool connections?

3. **Handle format:** What exactly goes in the NCCL handle? IP:port? Transport capability bitmap? How to encode multi-transport endpoints?

### Performance

4. **Request pool sizing:** How many pre-allocated requests per comm? NCCL_NET_MAX_REQUESTS * maxRecvs = minimum. What's the optimal pool size?

5. **regMr caching strategy:** When `regIsGlobal=1`, the plugin maintains a registration cache. What eviction policy? LRU? Size-based?

6. **Zero-copy threshold:** For small messages, is it faster to copy data than to register/deregister memory for RDMA? What's the crossover size?

### Compatibility

7. **Minimum NCCL version to test against:** v8 API implies NCCL 2.19+. But should we test against 2.17 (v6) as well for maximum compatibility?

8. **CUDA version requirements:** Does the plugin need CUDA runtime, or only driver API? Can it work with any CUDA version?

9. **ARM support:** OuterLink targets x86_64 Linux. Any ARM (Grace Hopper) considerations for the NCCL plugin?

---

## Related Documents

- [research/01-nccl-net-plugin-api.md](./research/01-nccl-net-plugin-api.md) — Exact API surface
- [research/02-existing-nccl-plugins.md](./research/02-existing-nccl-plugins.md) — Existing plugin survey
- [research/03-nccl-topology-and-collectives.md](./research/03-nccl-topology-and-collectives.md) — NCCL internals
- R14 Transport Compression — Compressed collective operations
- R17 Topology-Aware Scheduling — NCCL topology integration
- R29 RDMA Multicast — Hardware broadcast for future CollNet
- `planning/pre-planning/02-FINAL-PREPLAN.md` — Overall project pre-plan
- `docs/architecture/00-project-vision.md` — Project vision
