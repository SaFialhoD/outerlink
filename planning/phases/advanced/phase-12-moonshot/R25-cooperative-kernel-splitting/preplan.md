# R25: Cooperative Kernel Splitting — Pre-Plan

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** DRAFT
**Purpose:** Define WHAT needs to be planned for kernel splitting implementation, based on the completed research phase.

---

## 1. Scope Definition

### What R25 Delivers
A system that intercepts a single CUDA kernel launch, partitions its thread block grid across multiple physical GPUs, and launches smaller grids on each GPU — transparently to the application.

### What R25 Does NOT Deliver
- Full coherency for cross-GPU memory (that's R18)
- Network page faults for on-demand data migration (that's R19)
- Graph-level splitting (that's R13 — and R13 should be built FIRST)
- Support for ALL kernels (only GREEN and YELLOW classified kernels)

### Hard Prerequisite
**R13 (CUDA Graph Interception) must be functional before R25 is useful.** Graph-level splitting handles the majority of multi-GPU distribution. R25 is an optimization on top of R13 for the subset of kernels where intra-kernel parallelism exceeds what one GPU can offer.

---

## 2. Components to Plan

### Component 1: PTX Interceptor & Parser
**What:** Intercept `cuModuleLoadData` / `cuModuleLoadFatBinary`, extract PTX, parse it into a structured representation.
**Why:** Foundation for both classification and blockIdx rewriting.
**Unknowns:**
- How many real apps ship PTX vs SASS-only? (determines feasibility of the entire approach)
- Can we strip SASS from fat binaries to force PTX JIT?
- PTX parsing complexity — full parser or regex-based pattern matching?

### Component 2: Kernel Classifier
**What:** Analyze parsed PTX to classify kernels as GREEN / YELLOW / RED.
**Why:** Determines which kernels can be split and how.
**Detection targets:**
- Atomic operations on global memory (type: add vs CAS vs exchange)
- Cooperative group sync calls
- Memory access patterns (blockIdx-linear, data-dependent, indirect)
- Read vs write classification for pointer arguments
**Unknowns:**
- What accuracy can static analysis achieve? (need to test against real kernels)
- How to handle PTX inline assembly in C++ kernels?

### Component 3: PTX Transformer
**What:** Modify PTX to inject blockIdx offset parameter and redirect atomics.
**Why:** The core mechanism that makes splitting correct.
**Modifications needed:**
- Add `__blkoff_x/y/z` parameters to kernel entry
- Replace `%ctaid.x/y/z` reads with offset-adjusted version
- For YELLOW kernels: redirect atomic targets to per-GPU local copies
**Unknowns:**
- Can we transform PTX reliably without a full compiler? (text manipulation vs AST)
- How to handle multiple `%ctaid` uses in complex control flow?
- Verify transformed PTX compiles correctly on target GPU

### Component 4: Split Launch Orchestrator
**What:** The runtime system that decides how to split a kernel and coordinates the launches.
**Why:** Turns classification + transformation into actual multi-GPU execution.
**Responsibilities:**
- Decide split ratio (equal for homogeneous GPUs, weighted for heterogeneous)
- Set up per-GPU kernel arguments (pointer remapping, offset injection)
- Launch on all GPUs simultaneously
- Synchronize completion
- Execute merge step for YELLOW kernels
**Unknowns:**
- How to synchronize kernel launches across GPUs with minimal skew? (R26 integration)
- How to handle kernel launch failures on one GPU?
- Stream management: new streams per split, or reuse?

### Component 5: Data Distribution Advisor
**What:** Determines which data each GPU needs before a split kernel launch.
**Why:** Kernel splitting without data locality is useless (network transfer eats the compute savings).
**Responsibilities:**
- Map blocks to data regions based on PTX analysis
- Recommend data placement to minimize cross-GPU traffic
- Interface with R13 (graph-level data placement decisions)
**Unknowns:**
- How much data movement planning can happen at module load time vs launch time?
- Can we piggyback on R13's data flow analysis?

---

## 3. Dependencies

| Dependency | Type | Impact |
|-----------|------|--------|
| R13 CUDA Graph Interception | **HARD** | Must exist before R25 is useful. Graph context drives splitting decisions. |
| R26 PTP Clock Sync | **SOFT** | Improves launch synchronization. R25 works without it, just with more skew. |
| R18 Virtual NVLink | **SOFT** | Enables splitting YELLOW/RED kernels with cross-GPU data. Without it, only GREEN kernels. |
| R17 Topology-Aware Scheduling | **SOFT** | Picks optimal GPU pair for splitting. Without it, arbitrary assignment. |
| R23 Heterogeneous GPU Mixing | **SOFT** | Enables weighted splits for mixed GPU types. Without it, equal splits only. |
| P6 CUDA Interception (core) | **HARD** | Must have working `cuModuleLoadData` and `cuLaunchKernel` interception. |

---

## 4. Risk Assessment

### Risk 1: SASS-Only Modules (HIGH)
**Problem:** If major libraries (cuBLAS, cuDNN, PyTorch custom kernels) ship only precompiled SASS with no PTX, we cannot analyze or modify them.
**Mitigation:** Test empirically — load cuBLAS/cuDNN fat binaries and check for PTX sections. If absent, investigate stripping SASS to force PTX JIT. If that fails, these kernels are unsplittable (use graph-level splitting from R13 instead).
**Contingency:** For library calls (cuBLAS GEMM, cuDNN convolution), detect them at the API level and use their built-in multi-GPU paths (cuBLAS-XT) instead of splitting ourselves.

### Risk 2: PTX Transformation Correctness (HIGH)
**Problem:** Modifying PTX incorrectly produces wrong results or crashes. Silent data corruption is the worst outcome.
**Mitigation:** Extensive testing — run original kernel, run split kernel on 1 GPU (with offset 0), compare results bit-for-bit. Build a test suite of representative kernels.
**Contingency:** If PTX transformation proves too fragile, fall back to the memory remapping approach (Strategy B) for simple kernels only.

### Risk 3: Low Kernel Coverage (MEDIUM)
**Problem:** If the classifier marks 80%+ of kernels as RED, the effort of building the splitter has low payoff.
**Mitigation:** Build the classifier (Phase A) FIRST. Measure the GREEN/YELLOW/RED distribution before investing in the splitter.
**Contingency:** If coverage is low, R25 becomes a niche optimization for specific workloads (large GEMM, elementwise) rather than a general feature.

### Risk 4: Communication Overhead Exceeds Compute Savings (MEDIUM)
**Problem:** For many kernels, the overhead of data distribution + result merge + launch coordination exceeds the compute saved by splitting.
**Mitigation:** The performance model (03-practical-feasibility.md Section 3) predicts this. Only split when expected net benefit is positive. Conservative thresholds.
**Contingency:** If overhead is consistently high, R25 becomes a "data already distributed" optimization only — activated when R13 has already placed data across GPUs and there's a single hot kernel.

### Risk 5: Complexity vs Reward (MEDIUM)
**Problem:** This is the most complex feature in the entire project. The reward (10-30% additional throughput on top of graph splitting) may not justify the engineering investment.
**Mitigation:** Phased approach. Phase A (classifier) is cheap. Phase B (trivial split) is moderate. We can stop at any phase if the ROI doesn't justify continuing.
**Contingency:** Document the research and classification engine as reusable components. Even if full splitting isn't built, the PTX analysis infrastructure has value for other features (profiling, optimization hints).

---

## 5. Phased Implementation Plan

### Phase A: Classification Engine (4-6 weeks)
- PTX interceptor and parser
- Kernel classifier (GREEN/YELLOW/RED)
- Logging and statistics
- **Deliverable:** Report showing kernel classification distribution for PyTorch inference + training workloads
- **Go/No-Go:** If >30% of compute-heavy kernels are GREEN, proceed to Phase B

### Phase B: Trivial Kernel Splitting (6-8 weeks)
- PTX blockIdx offset transformer
- Split launch orchestrator (equal split, 2 GPUs)
- Pointer remapping for pre-distributed data
- End-to-end test: split elementwise kernel on 2 GPUs
- **Deliverable:** Working split for GREEN kernels, benchmarks showing actual speedup
- **Go/No-Go:** If measured efficiency >60% for GEMM-class kernels, proceed to Phase C

### Phase C: Atomic-Aware Splitting (4-6 weeks)
- Atomic detection and redirection in PTX
- Per-GPU local accumulator allocation
- Merge step implementation (host-side and kernel-based)
- **Deliverable:** YELLOW kernel support, including reductions and histograms

### Phase D: Graph-Integrated Splitting (6-8 weeks, depends on R13)
- Interface with R13's computation graph
- Graph-context-driven splitting decisions
- Data locality optimization using graph data flow
- Hotspot detection (which kernel to split)
- **Deliverable:** Fully integrated kernel splitting within the graph execution engine

### Total Estimated Effort: 20-28 weeks (5-7 months)

This is a substantial investment. The phased approach ensures we can stop early if the ROI doesn't materialize.

---

## 6. Success Criteria

| Criterion | Target |
|-----------|--------|
| Kernel classification coverage | >80% of kernels classifiable |
| GREEN kernel percentage (by compute time) | >30% for typical workloads |
| Split GEMM efficiency (2 GPUs, 100Gbps) | >60% |
| Split elementwise efficiency (2 GPUs) | >85% |
| End-to-end training speedup (with R13) | +10% over graph splitting alone |
| Zero correctness regressions | Bit-identical results for all split kernels |

---

## 7. What Needs to Be Planned in Detail

Before moving to a full plan, these items need design documents:

1. **PTX parser specification** — What subset of PTX do we parse? Full grammar or pattern matching?
2. **Classifier rule set** — Exact rules for GREEN/YELLOW/RED, with examples
3. **PTX transformer specification** — Exact transformations applied, with before/after examples
4. **Split orchestrator protocol** — Message sequence for coordinated multi-GPU launch
5. **Merge protocol** — How partial results are combined, for each atomic type
6. **R13 integration interface** — API between graph engine and kernel splitter

---

## Open Questions

1. **When should we build the classifier?** It's cheap and informative. Could build it during R13 development to gather data early.
2. **Should we target NVIDIA's CUTLASS library instead of cuBLAS?** CUTLASS ships as PTX-compilable templates, making it splittable. cuBLAS ships as SASS binary.
3. **Multi-node splitting?** The design assumes 2 GPUs on the same network. Could it extend to 4+ GPUs across multiple nodes? Theoretically yes, but communication overhead scales with GPU count.
4. **Is there a simpler PTX modification?** Instead of rewriting `%ctaid`, could we use a wrapper kernel that adjusts blockIdx before calling the real kernel? Would add one function call of overhead per thread.

---

## Related Documents

- [research/01-cuda-execution-model.md](research/01-cuda-execution-model.md)
- [research/02-kernel-splitting-strategies.md](research/02-kernel-splitting-strategies.md)
- [research/03-practical-feasibility.md](research/03-practical-feasibility.md)
- R13: CUDA Graph Interception (hard dependency)
- R18: Virtual NVLink (soft dependency)
- R26: PTP Clock Sync (soft dependency)
