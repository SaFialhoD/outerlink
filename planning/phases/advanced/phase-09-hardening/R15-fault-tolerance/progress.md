# R15: Fault Tolerance — Progress

| Date | Status | Summary |
|------|--------|---------|
| 2026-03-25 | NOT STARTED | Folder created, awaiting research phase |
| 2026-03-25 | RESEARCH COMPLETE | 3 research documents completed, pre-plan drafted |

## Research Phase (Complete)

- **01-erasure-coding-algorithms.md** — Evaluated Reed-Solomon, XOR parity, Fountain/Raptor codes, LDPC. Recommended hybrid RS + XOR via ISA-L on CPU. GPU-accelerated EC available but unnecessary (CPU saturates 100Gbps wire). Carbink (OSDI '22) validated RS(4,2) with ISA-L for far memory.
- **02-distributed-checkpointing.md** — Evaluated PyTorch DCP, DeepSpeed UCP, Gemini (SOSP '23), CheckFreq, LowDiff. Recommended Gemini-style in-memory checkpointing with incremental deltas. 13x recovery speedup over disk-based approaches. Checkpoint hierarchy: VRAM -> local DRAM -> remote DRAM -> NVMe.
- **03-failure-detection-recovery.md** — Designed multi-layer detection (ibv_async_event + phi accrual + TCP fallback). Defined recovery pipeline: detect (<1s) -> fence -> assess -> reconstruct (<30s) -> resume. Covered partial failures (GPU crash, NIC failure, process crash). Hot spare node design.

## Pre-Plan (Draft)

- **preplan.md** — Scope, dependencies, 5 key decisions, risk matrix, 5-phase implementation (13-18 weeks estimated). Success criteria: <30s recovery, <5% normal overhead, <3% checkpoint overhead.

## Next Steps

- [ ] Review pre-plan, finalize key decisions (especially parity storage location and checkpoint strategy)
- [ ] Create detailed plan.md with implementation tasks
- [ ] Prototype ISA-L Rust bindings
- [ ] Prototype phi accrual failure detector
