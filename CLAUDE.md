# OuterLink

## What Is This

OuterLink is a Rust software layer that makes GPUs across separate PCs work as a unified pool - shared VRAM, shared compute, shared system RAM. Any CUDA application runs unmodified. Written in Rust, licensed Apache 2.0.

**OpenDMA** is OuterLink's killer feature: non-proprietary direct NIC-to-GPU VRAM access via PCIe BAR1, bypassing NVIDIA's artificial GPUDirect restriction. Works on ALL NVIDIA GPUs including GeForce.

## Project Phase

**Current: Planning Phase**

Pre-planning and research are complete (7 research documents, consolidation, contingency plans). Now planning implementation phases starting with GitHub repo setup and dev environment.

## Mindset

**Be ambitious. Always aim for the best path, not the easiest.** We have contingency plans (Plan B, C, D) but we always pursue Plan A first - the one that gives us the best result. Innovation is real. What others said was impossible (GPU RDMA on GeForce) has a viable path through PCIe BAR1 + open kernel patches. We don't settle for "good enough" when "breakthrough" is within reach.

Contingencies exist for safety, not as the default. When making decisions, choose the option that makes OuterLink the project no one else could build.

## Architecture

```
App -> LD_PRELOAD -> OuterLink Client (.so) -> Transport -> OuterLink Server -> Real GPU
```

### Two Data Paths

**Host-Staged (Phase 1):** GPU -> cudaMemcpy -> pinned host -> network -> pinned host -> cudaMemcpy -> GPU

**OpenDMA (Phase 5):** GPU VRAM <-> ConnectX-5 DMA engine <-> wire <-> ConnectX-5 DMA engine <-> GPU VRAM (zero CPU, ~2us)

## Key Technical Decisions

- **Language:** Rust (C for interception .so and kernel module)
- **CUDA interception:** Driver API + LD_PRELOAD + cuGetProcAddress (222+ functions, HAMi-core pattern)
- **Transport Phase 1:** TCP + io_uring + CUDA pinned memory
- **Transport Phase 2:** UCX (auto-negotiates RDMA vs TCP)
- **GPU DMA Phase 1:** Host-staged transfers
- **GPU DMA Phase 2:** OpenDMA (PCIe BAR1 direct RDMA)
- **License:** Apache 2.0 (ADR-001)

## Folder Structure

```
outterlink/
├── planning/
│   ├── pre-planning/    # Pre-plans, hardware inventory, contingencies
│   ├── phases/          # Detailed phase plans (P1-P13)
│   └── research/        # Research docs (R1-R7) + consolidation
├── docs/
│   ├── architecture/    # System design, project vision
│   ├── guides/          # Setup and usage guides
│   ├── specs/           # Technical specifications
│   └── decisions/       # Architecture Decision Records (ADRs)
├── side-docs/
│   ├── references/      # External links and papers
│   ├── notes/           # Working notes and brainstorms
│   └── diagrams/        # Visual diagrams
├── crates/              # Rust source
│   ├── outerlink-client/   # LD_PRELOAD interception library
│   ├── outerlink-server/   # GPU node daemon
│   ├── outerlink-common/   # Shared protocol, types, transport
│   └── outerlink-cli/      # Management CLI
└── opendma/             # OpenDMA kernel module (future, C)
```

## Key Documents

- Vision: `docs/architecture/00-project-vision.md`
- Final Pre-Plan: `planning/pre-planning/02-FINAL-PREPLAN.md`
- Contingency Plans: `planning/pre-planning/03-contingency-plans.md`
- Research Consolidation: `planning/research/CONSOLIDATION-all-research.md`
- Research: `planning/research/R1-R7`
- ADRs: `docs/decisions/ADR-001-license.md`, `ADR-002-opendma-naming.md`

## Agent Workflow

**Every implementation agent MUST be followed by a review+fix agent.** This is non-negotiable:

1. Spawn implementation agent (in worktree)
2. When done, spawn review agent (critic or general-purpose) to review the new code
3. Fix any issues found by the review
4. Only then merge to master

This applies to all code-writing agents. No implementation goes unreviewed.

## Rules

- Documentation framework: `.claude/rules/documentation-framework.md`
- Project discipline: `.claude/rules/project-discipline.md`
