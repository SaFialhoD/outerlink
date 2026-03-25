# R23: Heterogeneous GPU Mixing — Pre-Plan

**Created:** 2026-03-25
**Updated:** 2026-03-25
**Status:** Draft
**Purpose:** Define WHAT needs to be planned for heterogeneous GPU support before writing the detailed plan.

## 1. Scope Definition

### What R23 Covers

- Runtime detection and profiling of GPU hardware capabilities across pool nodes
- Performance normalization and GPU scoring for the scheduler
- CUDA binary compatibility checking (cubin + PTX) before kernel dispatch
- Workload-aware task-to-GPU matching (compute-bound, memory-bound, capacity-bound)
- Integration with R17 topology-aware scheduling (adding capability dimension)
- Driver version compatibility enforcement across nodes
- User-facing GPU preference/exclusion policies
- GPU Equivalent Unit (GEU) abstraction for multi-user fairness

### What R23 Does NOT Cover

- The transport layer (R5/OpenDMA handles data movement)
- Memory tiering policies (R10 handles VRAM vs host RAM placement)
- Network topology scoring (R17 handles this; R23 adds GPU capability scoring)
- NCCL topology reporting (R20 handles multi-device communication)
- Time-sliced GPU sharing between users (R24 covers this)
- CUDA Graph-level scheduling (R13 covers graph decomposition)

### Interaction Points with Other Phases

| Phase/Research | R23 Provides | R23 Consumes |
|---------------|-------------|-------------|
| R10 Memory Tiering | GPU bandwidth profiles for tier placement | Memory tier topology for scheduling |
| R13 CUDA Graph | Per-node GPU capabilities for graph partitioning | Graph structure for batch scheduling |
| R17 Topology Scheduling | GPU capability scores as scheduling input | Network topology, data locality scores |
| R20 NCCL Backend | Heterogeneous device list for topology reporting | Collective communication performance data |
| R24 Time-Sliced Sharing | GPU performance profiles for quota allocation | Time-slice boundaries affecting scheduling |

## 2. Key Technical Decisions Needed

### D1: Minimum Supported Compute Capability

**Options:**
- CC 5.2 (Maxwell 2nd gen) — broadest compatibility, includes GTX 900 series
- CC 7.5 (Turing) — covers RTX 2000+ series, has Tensor Cores
- CC 8.6 (Ampere) — covers RTX 3000+ series, modern features

**Research finding:** CC 7.5 (Turing) is the recommended minimum. It covers all RTX GPUs (2000/3000/4000/5000), includes Tensor Cores, and excludes only very old hardware that wouldn't meaningfully contribute to a pool. Supporting Maxwell/Pascal adds complexity (different SM architectures, no Tensor Cores) for minimal user benefit.

**Decision needed:** Confirm CC 7.5 minimum or adjust based on user demand analysis.

### D2: Performance Normalization Method

**Options:**
1. Static TFLOPS-only — simplest, available immediately
2. Benchmark-calibrated — run micro-benchmarks at GPU registration
3. Workload-class-specific — different scores for compute/memory/tensor workloads
4. Runtime-adaptive — continuously adjust based on observed kernel times

**Research finding:** Hybrid approach recommended. Start with static TFLOPS weighting (available immediately), refine with startup benchmarks (per GPU model, cached), and add runtime adaptation as a future enhancement. Gavel's research shows 1.4-3.5x improvement from heterogeneity-aware vs agnostic scheduling.

**Decision needed:** Define the benchmark suite (which benchmarks, how long, when to run).

### D3: Kernel Dispatch Compatibility Strategy

**Options:**
1. Check cubin compatibility only — reject if no matching cubin
2. Check cubin + allow PTX JIT — broader compatibility but potential latency spike
3. Pre-warm JIT caches — trigger compilation for all pool GPUs at module load

**Research finding:** Option 3 (pre-warm) is ideal. When OuterLink intercepts `cuModuleLoad`, extract PTX and trigger async JIT compilation on all pool GPUs. This avoids first-launch latency spikes. Fallback to option 2 for unanticipated GPU additions.

**Decision needed:** Confirm pre-warming strategy. Determine how to handle architecture-conditional PTX (`compute_90a` etc.) which is not forward-compatible.

### D4: Scheduler Integration Architecture

**Options:**
1. R23 as a standalone scoring plugin called by the main scheduler
2. R23 merged into R17's scoring function as additional dimensions
3. R23 as a pre-filter (eliminate incompatible GPUs) before R17 scores

**Research finding:** Option combining 1 and 3: R23 first filters by hard constraints (CC, VRAM, binary compatibility, driver version), then provides a capability score that R17's scheduler incorporates via weighted sum.

**Decision needed:** Define the API between R23's capability scorer and R17's topology scheduler.

### D5: Driver Version Policy

**Options:**
1. Strict: all nodes must have identical driver version
2. Moderate: all nodes same major version, minor can vary
3. Permissive: any driver, scheduler checks compatibility per-kernel

**Research finding:** Option 2 (moderate) is practical. Same major version ensures CUDA minor compatibility. The scheduler additionally checks per-node CUDA version against kernel requirements.

**Decision needed:** How to handle the upgrade path when a new CUDA major version comes out (12.x → 13.x transition).

## 3. Unknowns and Risks

### Unknown 1: Workload Classification Accuracy

**Risk:** Misclassifying a memory-bound kernel as compute-bound sends it to a high-TFLOPS, low-bandwidth GPU, causing poor performance.

**Mitigation path:** Start with conservative classification (assume memory-bound by default, since most kernels are). Add heuristics based on kernel grid dimensions, argument sizes, and kernel name patterns. Refine with runtime observation.

### Unknown 2: JIT Compilation Overhead on Consumer GPUs

**Risk:** PTX JIT compilation on GeForce GPUs may take significantly longer than on data center GPUs, causing noticeable delays.

**Mitigation path:** Benchmark JIT compilation time for representative kernels across GPU generations. If overhead is significant, the pre-warming strategy (D3) becomes critical.

### Unknown 3: Multi-Generation Tensor Core Compatibility

**Risk:** A kernel optimized for 4th-gen Tensor Cores (FP8) may not have a fallback path for 3rd-gen (only FP16/BF16). The scheduler must either find a compatible GPU or gracefully degrade.

**Mitigation path:** Inspect kernel PTX for precision-specific Tensor Core instructions. Maintain a feature requirement list per kernel module. Report to the user which GPUs can run each kernel at full speed vs degraded.

### Unknown 4: Straggler Behavior Under Thermal Throttling

**Risk:** A GPU in a poorly cooled case performs well initially but throttles under sustained load, disrupting scheduling assumptions.

**Mitigation path:** Monitor GPU temperature and clock speed via NVML. Detect throttling events and dynamically reduce that GPU's assigned workload. Include thermal headroom in the GPU profile.

### Unknown 5: Resizable BAR Adoption

**Risk:** Many users may not have ReBAR enabled (requires BIOS changes, VBIOS updates on RTX 3000 series). This limits OpenDMA effectiveness.

**Mitigation path:** Detect BAR1 size at registration. Provide clear user guidance for enabling ReBAR. Mark non-ReBAR GPUs as "host-staged only" rather than excluding them.

## 4. Dependencies

### Hard Dependencies (Must Exist Before R23)

| Dependency | What R23 Needs | Status |
|-----------|---------------|--------|
| P10 Multi-Node | Working multi-node GPU pool | Not started |
| GPU Registration | Mechanism to register GPUs with coordinator | Part of P10 |
| Kernel Interception | cuModuleLoad, cuLaunchKernel intercepts | Phase 1 |
| CUDA Driver API access | cuDeviceGetAttribute, etc. | Phase 1 |

### Soft Dependencies (Can Proceed in Parallel)

| Dependency | Interaction | Status |
|-----------|-------------|--------|
| R17 Topology Scheduling | R23 adds capability scores to R17's scorer | Pre-plan done |
| R10 Memory Tiering | R23 informs tier placement with bandwidth data | Pre-plan done |
| R13 CUDA Graph | R23 provides per-GPU capabilities for graph partitioning | Pre-plan done |
| R20 NCCL Backend | R23 feeds GPU list to NCCL topology engine | Pre-plan done |

## 5. Deliverables for the Full Plan

The detailed plan (plan.md) should define:

1. **GpuProfile struct** — Final Rust struct for GPU capability representation
2. **Benchmark suite specification** — Which benchmarks, expected duration, caching strategy
3. **Capability scorer API** — Input/output contract between R23 and the main scheduler
4. **Binary compatibility checker** — How to inspect fatbins, extract PTX, verify CC compatibility
5. **Driver compatibility enforcer** — Registration-time checks, per-kernel dispatch checks
6. **User preference system** — Environment variables, config file format, API
7. **GEU calculation** — Weights for compute/memory/capacity, reference GPU definition
8. **Monitoring and alerts** — What to monitor (utilization, throttling, errors), how to alert
9. **Test plan** — Unit tests for scorer, integration tests for heterogeneous dispatch, chaos tests for mixed-driver scenarios

## 6. Estimated Effort

| Component | Complexity | Estimate |
|-----------|-----------|----------|
| GpuProfile and capability query | Low | 1-2 days |
| Benchmark suite | Medium | 2-3 days |
| Capability scorer | Medium | 2-3 days |
| Binary compatibility checker | High | 3-5 days |
| Driver compatibility enforcer | Low | 1 day |
| Integration with R17 scheduler | Medium | 2-3 days |
| User preference system | Low | 1-2 days |
| Testing | High | 3-5 days |
| **Total** | | **15-24 days** |

This assumes P10 multi-node and kernel interception are already working.

## 7. Success Criteria

R23 is complete when:
1. Any GeForce GPU (CC 7.5+) can join the pool and be characterized automatically
2. The scheduler routes kernels only to compatible GPUs (CC, binary, driver)
3. Work distribution across heterogeneous GPUs is proportional to their capabilities
4. Adding a slower GPU to the pool never DECREASES total throughput (smart partitioning)
5. Users can express GPU preferences and exclusions
6. Driver version mismatches are detected and reported at registration time
7. Performance within 90% of theoretical maximum for proportional work distribution

## Related Documents

- [research/01-gpu-capability-landscape.md](./research/01-gpu-capability-landscape.md) — GPU hardware diversity
- [research/02-heterogeneous-scheduling.md](./research/02-heterogeneous-scheduling.md) — Scheduling approaches
- [research/03-practical-mixing-scenarios.md](./research/03-practical-mixing-scenarios.md) — Real-world use cases
- [R17 Topology-Aware Scheduling](../../../phase-08-network-optimization/R17-topology-scheduling/README.md)
- [R10 Memory Tiering](../../phase-07-memory-intelligence/R10-memory-tiering/README.md)

## Open Questions

- [ ] Should GEU weights be globally fixed or tunable per-pool?
- [ ] How to handle live GPU addition/removal without disrupting running workloads?
- [ ] What telemetry should R23 expose for debugging scheduling decisions?
- [ ] Is there value in a "GPU compatibility simulator" that predicts whether a given set of GPUs will work well together before the user sets up the pool?
