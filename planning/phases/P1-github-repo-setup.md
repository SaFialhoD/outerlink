# P1: GitHub Repository Setup

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - Do First

## Goal

Create and configure the OutterLink GitHub repository with proper structure, documentation, CI skeleton, and community standards so development can begin cleanly.

## Milestone

- Public GitHub repo exists at `github.com/<owner>/outterlink`
- README clearly explains what OutterLink is and what OpenDMA is
- Apache 2.0 LICENSE file present
- Rust workspace builds (`cargo build` succeeds)
- GitHub Actions runs `cargo check` + `cargo clippy` + `cargo test`
- Issue templates and labels configured
- Branch protection on `main`

## Prerequisites

- [x] License decided: Apache 2.0 (ADR-001)
- [x] Project vision documented
- [x] Pre-planning complete
- [ ] Pedro's GitHub account/org name (needed for repo creation)

---

## Components

### 1. Repository Settings

| Setting | Value |
|---------|-------|
| Name | `outterlink` |
| Visibility | Public |
| Description | "Unified GPU pool across networked PCs. OpenDMA: non-proprietary direct NIC-to-GPU VRAM access." |
| Topics | `gpu`, `cuda`, `rdma`, `rust`, `nvlink`, `distributed-computing`, `gpu-sharing`, `opendma` |
| Default branch | `main` |
| Wiki | Disabled (docs live in repo) |
| Issues | Enabled |
| Discussions | Enabled |
| Projects | Enabled |

### 2. Branch Strategy

| Branch | Purpose | Protection |
|--------|---------|-----------|
| `main` | Stable, always builds | Require PR, require CI pass, no force push |
| `dev` | Active development | Require CI pass |
| `feature/*` | Feature branches | None |
| `research/*` | Experimental/research branches | None |

### 3. Files for Initial Commit

```
outterlink/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                    # Rust CI pipeline
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── research_topic.md
│   ├── pull_request_template.md
│   └── FUNDING.yml                   # Optional
├── crates/
│   ├── outterlink-common/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs                # Shared types, transport trait
│   ├── outterlink-server/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs               # Server entry point (skeleton)
│   ├── outterlink-client/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs                # Client library (skeleton)
│   └── outterlink-cli/
│       ├── Cargo.toml
│       └── src/
│           └── main.rs               # CLI tool (skeleton)
├── planning/                          # All existing planning docs
├── docs/                              # All existing docs
├── side-docs/                         # All existing side docs
├── .gitignore
├── Cargo.toml                         # Workspace root
├── CLAUDE.md                          # Claude rules (existing)
├── LICENSE                            # Apache 2.0
├── README.md                          # Project README
└── CONTRIBUTING.md                    # Contribution guide
```

### 4. README.md Content Outline

```
# OutterLink

One-line: Unified GPU pool across networked PCs.

## What Is OutterLink?
- Makes GPUs across separate PCs work as one pool
- Any CUDA app runs unmodified
- Written in Rust, Apache 2.0

## OpenDMA
- Non-proprietary direct NIC-to-GPU VRAM access
- Bypasses NVIDIA's artificial GPUDirect restriction
- Works on ALL NVIDIA GPUs including GeForce
- Zero CPU involvement, ~2us latency

## How It Works
- Architecture diagram
- Bandwidth table

## Status
- Current phase
- Roadmap

## Getting Started
- Prerequisites
- Installation
- Quick start

## Contributing
- Link to CONTRIBUTING.md

## License
Apache 2.0
```

### 5. GitHub Actions CI (`ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - run: cargo fmt --all -- --check
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo test --all
      - run: cargo build --all
```

### 6. Issue Labels

| Label | Color | Description |
|-------|-------|-------------|
| `phase-0` | blue | Project setup |
| `phase-1-poc` | green | Proof of concept |
| `phase-2-core` | yellow | Core transport |
| `phase-3-cuda` | orange | CUDA completeness |
| `phase-4-perf` | red | Performance |
| `phase-5-opendma` | purple | OpenDMA |
| `phase-6-scale` | dark blue | Multi-node scaling |
| `research` | gray | Research task |
| `bug` | red | Bug report |
| `enhancement` | blue | Feature request |
| `documentation` | light blue | Docs improvement |
| `good-first-issue` | green | Good for newcomers |

### 7. .gitignore

```
# Rust
target/
Cargo.lock  # Keep for binaries, ignore for libraries? TBD
*.swp
*.swo

# CUDA
*.o
*.cubin
*.ptx
*.fatbin

# Kernel modules
*.ko
*.mod.c
*.mod
modules.order
Module.symvers

# IDE
.vscode/
.idea/
*.code-workspace

# OS
.DS_Store
Thumbs.db

# Environment
.env
*.log
```

---

## Test Plan

| Test | Expected |
|------|----------|
| `cargo build --all` | Compiles without errors |
| `cargo test --all` | All tests pass (skeleton tests) |
| `cargo clippy --all-targets` | No warnings |
| `cargo fmt --all -- --check` | Properly formatted |
| GitHub Actions CI | Green check on push |

## Risks

| Risk | Mitigation |
|------|-----------|
| GitHub org/account naming conflict | Check availability first |
| CI fails on Rust version | Pin Rust stable toolchain |

## Related Documents

- [ADR-001: License](../../docs/decisions/ADR-001-license.md)
- [Final Pre-Plan](../pre-planning/02-FINAL-PREPLAN.md)
- [Project Vision](../../docs/architecture/00-project-vision.md)
