# Planning Index

## Pre-Planning
*Plan what we need to plan - scope, unknowns, dependencies*

| Document | Status | Description |
|----------|--------|-------------|
| [Pre-Planning Master](pre-planning/00-master-preplan.md) | Updated | Original pre-plan with research results |
| [Hardware Inventory](pre-planning/01-hardware-inventory.md) | Draft | All hardware specs and setup checklist |
| [FINAL PRE-PLAN](pre-planning/02-FINAL-PREPLAN.md) | **COMPLETE** | Exhaustive map: research, decisions, plans, risks, hardware |
| [Contingency Plans](pre-planning/03-contingency-plans.md) | **COMPLETE** | Plan B for every critical path |

## Research
*Findings on existing tools, libraries, and protocols*

| # | Topic | Document | Status |
|---|-------|----------|--------|
| R1 | Existing GPU sharing/remoting projects | [R1](research/R1-existing-projects.md) | Complete |
| R2 | SoftRoCE / rdma_rxe capabilities | [R2](research/R2-softroce-rdma.md) | Complete |
| R3 | CUDA interception strategies | [R3](research/R3-cuda-interception.md) | Complete |
| R4 | ConnectX-5 + Transport Stack | [R4](research/R4-connectx5-transport-stack.md) | Complete |
| R5 | GPUDirect RDMA on GeForce | [R5](research/R5-gpudirect-geforce-restriction.md) | Complete |
| R6 | NVLink as Cross-PC Bridge | [R6](research/R6-nvlink-cross-pc.md) | Complete |
| R7 | Non-Proprietary GPU DMA (OpenDMA) | [R7](research/R7-non-proprietary-gpu-dma.md) | Complete |
| - | Research Synthesis | [Synthesis](research/SYNTHESIS-initial-research.md) | Complete |
| - | **Full Consolidation** | [Consolidation](research/CONSOLIDATION-all-research.md) | **Complete** |

## Phases
*All 13 plans - implementation ready*

| Phase | Document | Status | Description | Estimated Scope |
|-------|----------|--------|-------------|----------------|
| P1 | [GitHub Repo Setup](phases/P1-github-repo-setup.md) | **READY** | Repo, CI, README, labels | ~20 files |
| P2 | [Dev Environment](phases/P2-dev-environment.md) | **READY** | Linux, NVIDIA, ConnectX-5, Rust, BIOS | Guide doc |
| P3 | [CI/CD Pipeline](phases/P3-cicd-pipeline.md) | **READY** | GitHub Actions, CUDA stubs, test matrix | 3 workflows |
| P4 | [Project Skeleton](phases/P4-project-skeleton.md) | **READY** | Rust workspace, crates, FFI, C interposition | ~15 files |
| P5 | [Phase 1: PoC](phases/P5-poc-plan.md) | **READY** | 23 CUDA functions, protocol, handle translation | ~20 files |
| P6 | [Phase 2: Core Transport](phases/P6-core-transport.md) | **READY** | Binary protocol, memory transfers, kernel launch | ~24 files, ~7150 LOC |
| P7 | [Phase 3: CUDA Completeness](phases/P7-cuda-completeness.md) | **READY** | 222 functions, NVML, PyTorch compat | ~18 files, ~7700 LOC |
| P8 | [Phase 4: Performance](phases/P8-performance.md) | **READY** | io_uring, batching, UCX, RDMA pipeline | ~12 files |
| P9 | [Phase 5: OpenDMA](phases/P9-opendma.md) | **READY** | BAR1 RDMA kernel module, tinygrad patches | ~20 files, ~4000 LOC |
| P10 | [Phase 6: Multi-Node](phases/P10-multi-node.md) | **READY** | Node discovery, GPU pool, NVLink scheduling | ~15 files |
| P11 | [Testing Strategy](phases/P11-testing-strategy.md) | **READY** | Unit, integration, E2E, mock GPU, CI | ~93 tests |
| P12 | [Benchmarking Plan](phases/P12-benchmarking-plan.md) | **READY** | Metrics, microbenchmarks, baselines, reporting | 5 benchmarks |
| P13 | [Documentation Standards](phases/P13-documentation-standards.md) | **READY** | API docs, user guides, dev docs, rules | Standards doc |

## Decisions

| # | Decision | Status | Document |
|---|----------|--------|----------|
| ADR-001 | License: Apache 2.0 | Accepted | [ADR-001](../docs/decisions/ADR-001-license.md) |
| ADR-002 | OpenDMA naming | Accepted | [ADR-002](../docs/decisions/ADR-002-opendma-naming.md) |
| D9 | Serialization: Custom binary (not protobuf) | Decided in P6 | [P6](phases/P6-core-transport.md) |
| D14 | Handle translation: DashMap concurrent maps | Decided in P6 | [P6](phases/P6-core-transport.md) |
