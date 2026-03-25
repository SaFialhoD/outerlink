# Advanced Features: Organization & Process

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** ACTIVE
**Parent:** [04-advanced-features-preplan.md](../../pre-planning/04-advanced-features-preplan.md)

## Purpose

This document defines HOW we plan and execute the advanced features (R10-R30, Phases 7-12). The process is intentionally slow and thorough — every topic gets deep research and ultra-detailed planning BEFORE any code is written. The goal is one-shot implementation: when we code, there's zero ambiguity.

---

## The Process (Per Topic)

Every research topic (R10-R30) follows this exact lifecycle:

```
STEP 1: RESEARCH          <- Understand the problem space deeply
    |   - Read papers, docs, existing implementations
    |   - Document findings in research/ subfolder
    |   - Identify unknowns, risks, options
    |   - May produce prototypes or experiments
    |
STEP 2: PRE-PLAN          <- Organize what we know, identify what we still need
    |   - Scope definition
    |   - Dependency mapping
    |   - Decision inventory (what must we choose?)
    |   - Risk assessment
    |   - Open questions list
    |
STEP 3: DETAILED PLAN     <- Ultra-detailed implementation blueprint
    |   - File-by-file, function-by-function specification
    |   - Interface contracts with other components
    |   - Test plan (what we test, how, acceptance criteria)
    |   - Error handling strategy
    |   - Performance targets with numbers
    |   - Migration path (how existing code adapts)
    |   - One-shot implementation ready — no guesswork
    |
STEP 4: IMPLEMENTATION    <- Code it (one-shot, following the plan exactly)
    |   - Implementation agent follows the plan
    |   - Review agent validates against the plan
    |   - Fix any deviations
    |
STEP 5: VERIFICATION      <- Prove it works
        - Run test plan from Step 3
        - Benchmark against targets
        - Document results in side-docs/
        - Update progress tracker
```

---

## Folder Structure (Per Topic)

```
R{N}-{topic-name}/
├── README.md              <- Status, summary, quick links
├── preplan.md             <- Step 2: scope, dependencies, decisions, risks
├── plan.md                <- Step 3: ultra-detailed implementation plan (created later)
├── progress.md            <- Tracks status through the lifecycle
├── research/              <- Step 1: all research documents
│   ├── 01-{finding}.md
│   ├── 02-{finding}.md
│   └── ...
└── side-docs/             <- Notes, experiments, diagrams, benchmarks
    ├── notes/
    └── experiments/
```

---

## Phase Structure

Topics are grouped into phases by dependency:

```
planning/phases/advanced/
├── 00-ORGANIZATION.md                          <- THIS DOCUMENT
│
├── phase-07-memory-intelligence/               <- FOUNDATION (do first)
│   ├── R10-memory-tiering/
│   ├── R14-transport-compression/
│   └── R20-nccl-backend/
│
├── phase-08-smart-memory/                      <- MAGIC (builds on Phase 7)
│   ├── R11-speculative-prefetching/
│   ├── R12-memory-deduplication/
│   ├── R17-topology-aware-scheduling/
│   └── R19-network-page-faults/
│
├── phase-09-hardening/                         <- PRODUCTION (builds on Phase 8)
│   ├── R15-fault-tolerance/
│   ├── R21-gpu-direct-storage/
│   ├── R26-ptp-clock-sync/
│   ├── R28-scatter-gather-dma/
│   └── R29-rdma-multicast/
│
├── phase-10-compute-distribution/              <- GPU OS (builds on Phase 9)
│   ├── R13-cuda-graph-interception/
│   ├── R16-bluefield-dpu-offload/
│   ├── R23-heterogeneous-gpu-mixing/
│   └── R30-persistent-kernels/
│
├── phase-11-product-layer/                     <- PRODUCT (builds on Phase 10)
│   ├── R22-live-migration/
│   ├── R24-time-sliced-sharing/
│   └── R27-rocm-hip-interception/
│
└── phase-12-moonshot/                          <- HOLY GRAIL (builds on everything)
    ├── R18-virtual-nvlink/
    └── R25-cooperative-kernel-splitting/
```

---

## Status Tracking

Each topic has one of these statuses:

| Status | Meaning |
|--------|---------|
| `NOT STARTED` | Folder exists, no work done |
| `RESEARCHING` | Actively gathering information (Step 1) |
| `RESEARCH COMPLETE` | Research done, ready for pre-plan |
| `PRE-PLANNING` | Writing the pre-plan (Step 2) |
| `PRE-PLAN COMPLETE` | Pre-plan done, ready for detailed plan |
| `PLANNING` | Writing the detailed plan (Step 3) |
| `PLAN COMPLETE` | Plan done, ready for implementation |
| `IMPLEMENTING` | Code being written (Step 4) |
| `VERIFYING` | Testing and benchmarking (Step 5) |
| `COMPLETE` | Done, merged, documented |

### Current Status Overview

| Phase | Topic | Status |
|-------|-------|--------|
| **7** | R10 Memory Tiering | NOT STARTED |
| **7** | R14 Transport Compression | NOT STARTED |
| **7** | R20 NCCL Backend | NOT STARTED |
| **8** | R11 Speculative Prefetching | NOT STARTED |
| **8** | R12 Memory Deduplication | NOT STARTED |
| **8** | R17 Topology-Aware Scheduling | NOT STARTED |
| **8** | R19 Network Page Faults | NOT STARTED |
| **9** | R15 Fault Tolerance | NOT STARTED |
| **9** | R21 GPU Direct Storage | NOT STARTED |
| **9** | R26 PTP Clock Sync | NOT STARTED |
| **9** | R28 Scatter-Gather DMA | NOT STARTED |
| **9** | R29 RDMA Multicast | NOT STARTED |
| **10** | R13 CUDA Graph Interception | NOT STARTED |
| **10** | R16 BlueField DPU Offload | NOT STARTED |
| **10** | R23 Heterogeneous GPU Mixing | NOT STARTED |
| **10** | R30 Persistent Kernels | NOT STARTED |
| **11** | R22 Live Migration | NOT STARTED |
| **11** | R24 Time-Sliced Sharing | NOT STARTED |
| **11** | R27 ROCm/HIP Interception | NOT STARTED |
| **12** | R18 Virtual NVLink | NOT STARTED |
| **12** | R25 Cooperative Kernel Splitting | NOT STARTED |

---

## Rules

1. **No skipping steps** — Every topic goes through all 5 steps in order. No "let's just code it."
2. **Research before pre-plan** — You can't scope what you don't understand.
3. **Pre-plan before plan** — You can't detail what you haven't organized.
4. **Plan before code** — One-shot implementation requires zero ambiguity.
5. **Review after code** — No implementation goes unreviewed (per CLAUDE.md).
6. **Document as you go** — Side-docs capture everything that doesn't fit in formal docs. Notes, experiments, dead ends, surprises.
7. **Update progress.md** — Every status change gets recorded with date and summary.
8. **Cross-reference aggressively** — If R12 discovers something relevant to R10, link it.
9. **Open questions are valuable** — Track them, don't ignore them. They prevent surprises later.
10. **Slow is fast** — Rushing planning creates debugging. Thorough planning creates one-shot implementation.

---

## Related Documents

- [04-advanced-features-preplan.md](../../pre-planning/04-advanced-features-preplan.md) — Master pre-plan with dependency graph
- [02-FINAL-PREPLAN.md](../../pre-planning/02-FINAL-PREPLAN.md) — Original pre-plan (P1-P13)
- [Documentation Framework](../../../.claude/rules/documentation-framework.md) — Project doc standards
- [Project Discipline](../../../.claude/rules/project-discipline.md) — Planning rules
