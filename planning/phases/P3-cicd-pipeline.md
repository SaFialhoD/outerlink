# P3: CI/CD Pipeline

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** MEDIUM - Parallel with P4

## Goal

Establish a GitHub Actions CI/CD pipeline that validates every PR (Rust lint/test/build, C code compilation, kernel module compile-check) and automates release binary builds and documentation deployment -- all without requiring a physical GPU.

## Milestone

- `ci.yml` runs on every push/PR: fmt, clippy, test, build for all Rust crates
- C interception `.so` compiles in CI
- Kernel module compile-check passes against target kernel headers
- CUDA-dependent code compiles without a GPU via feature flags and stubs
- `release.yml` produces Linux x86_64 release binaries as GitHub Release artifacts
- `docs.yml` generates and deploys `cargo doc` output to GitHub Pages
- CI completes in under 10 minutes for the common path

## Prerequisites

- [x] P1: GitHub Repository Setup (repo exists, basic CI skeleton)
- [ ] P4: Project Skeleton (Cargo workspace, crate structure) -- can iterate together

---

## 1. CUDA Without a GPU in CI

GitHub Actions runners have no GPU. All CUDA-dependent code must compile and pass unit tests without `libcuda.so` or any NVIDIA hardware.

### Strategy: Feature Flags + Compile-Time Stubs

Every crate that touches CUDA uses a Cargo feature flag to gate real CUDA calls:

```toml
# crates/outterlink-common/Cargo.toml
[features]
default = []
cuda = []  # Enables real CUDA FFI bindings
```

```rust
// crates/outterlink-common/src/cuda_ffi.rs

#[cfg(feature = "cuda")]
mod real {
    // Link to real libcuda.so at runtime
    extern "C" {
        pub fn cuInit(flags: u32) -> i32;
        pub fn cuDeviceGetCount(count: *mut i32) -> i32;
        // ... all Driver API functions
    }
}

#[cfg(not(feature = "cuda"))]
mod stub {
    // Compile-time stubs that return CUDA_SUCCESS (0)
    // These are ONLY for CI compilation -- never for real execution
    pub unsafe fn cuInit(_flags: u32) -> i32 { 0 }
    pub unsafe fn cuDeviceGetCount(count: *mut i32) -> i32 {
        unsafe { *count = 0; }
        0
    }
    // ... all Driver API functions
}

#[cfg(feature = "cuda")]
pub use real::*;
#[cfg(not(feature = "cuda"))]
pub use stub::*;
```

### How This Works in Practice

| Environment | Feature Flag | CUDA Calls | Behavior |
|-------------|-------------|------------|----------|
| CI (GitHub Actions) | `default` (no `cuda`) | Stubs | Compiles, unit tests pass, no GPU needed |
| Dev machine (no GPU) | `default` (no `cuda`) | Stubs | Same as CI |
| Dev machine (with GPU) | `--features cuda` | Real FFI | Links to real `libcuda.so` |
| Release build | `--features cuda` | Real FFI | Production binary |

### Rules

- All Rust unit tests MUST pass without the `cuda` feature. Tests that need real CUDA are integration tests gated behind `#[cfg(feature = "cuda")]`.
- The stub module must have the exact same function signatures as the real FFI. A proc-macro or build script can enforce this (future improvement).
- The server binary requires `--features cuda` for any real execution. CI only verifies it compiles.

---

## 2. C Code Compilation in CI

The LD_PRELOAD interception `.so` is written in C because it must be a shared library that overrides `dlsym` and `cuGetProcAddress` at the libc level.

### Build System

The C code lives in `crates/outterlink-client/csrc/` and is compiled via a `build.rs` or a `Makefile` invoked from Cargo.

```makefile
# crates/outterlink-client/csrc/Makefile

CC = gcc
CFLAGS = -shared -fPIC -Wall -Wextra -Werror -O2
LDFLAGS = -ldl

# CUDA headers are optional -- use stub headers in CI
ifdef CUDA_HOME
    CFLAGS += -I$(CUDA_HOME)/include
else
    CFLAGS += -I./stub-headers
endif

TARGET = liboutterlink_intercept.so
SRCS = intercept.c dlsym_hook.c cuda_dispatch.c

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
```

### Stub CUDA Headers for CI

CI does not have the CUDA toolkit installed. We ship minimal stub headers that define only the types and constants we reference:

```c
// crates/outterlink-client/csrc/stub-headers/cuda.h

#ifndef CUDA_STUB_H
#define CUDA_STUB_H

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_INITIALIZED 3

// Driver API function pointer types
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGetCount_t)(int*);
// ... extend as interception grows

#endif
```

These headers are checked into the repo alongside the C source. They contain ONLY type definitions and constants -- no function implementations.

---

## 3. Kernel Module Compile-Check

The OpenDMA kernel module (`opendma/module/`) is C code that builds against kernel headers. It cannot run in CI, but it can compile-check.

### Approach

Install the `linux-headers` package for the target kernel version and run `make -C /lib/modules/$(uname -r)/build M=$(pwd) modules` in a check mode.

```yaml
# In ci.yml kernel-module job
- name: Install kernel headers
  run: |
    sudo apt-get update
    sudo apt-get install -y linux-headers-$(uname -r) build-essential

- name: Compile-check kernel module
  working-directory: opendma/module
  run: |
    make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
    # Clean up .ko -- we only care that compilation succeeded
    make -C /lib/modules/$(uname -r)/build M=$(pwd) clean
```

### Kernel Module Makefile

```makefile
# opendma/module/Makefile

obj-m += opendma.o
opendma-objs := main.o bar1_rdma.o

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```

### Limitations

- CI uses the Ubuntu runner's kernel version, which may differ from the target deployment kernel. This checks compilation, not ABI compatibility.
- When the module reaches maturity, add DKMS packaging that builds against a user's running kernel at install time.

---

## 4. Test Matrix

### Rust Versions

| Version | Purpose | When |
|---------|---------|------|
| `stable` | Primary build target | Every PR |
| `nightly` | Catch upcoming breakage, enable nightly features | Weekly scheduled run |
| MSRV (to be set, e.g., `1.80`) | Ensure minimum supported version | Every PR |

### Ubuntu Versions

| Version | Kernel | Why |
|---------|--------|-----|
| `ubuntu-24.04` | 6.8 | Primary -- matches P2 recommendation |
| `ubuntu-22.04` | 5.15 | Fallback -- widely deployed |

### CUDA SDK Versions (for release builds only)

| Version | Why |
|---------|-----|
| CUDA 12.6 | Current stable target |
| CUDA 12.8 | Latest -- catch forward-compatibility issues |

CUDA is NOT installed for the basic CI job. Only the `release.yml` workflow installs the CUDA toolkit to build production binaries with `--features cuda`.

---

## 5. Caching

### Cargo Registry + Build Cache

```yaml
- name: Cache Cargo registry
  uses: actions/cache@v4
  with:
    path: |
      ~/.cargo/registry/index
      ~/.cargo/registry/cache
      ~/.cargo/git/db
    key: cargo-registry-${{ hashFiles('**/Cargo.lock') }}
    restore-keys: |
      cargo-registry-

- name: Cache Cargo build
  uses: actions/cache@v4
  with:
    path: target
    key: cargo-build-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}-${{ hashFiles('**/*.rs') }}
    restore-keys: |
      cargo-build-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}-
      cargo-build-${{ runner.os }}-
```

### Expected Impact

| Without cache | With cache |
|---------------|-----------|
| ~5-8 min (full dependency compile) | ~1-2 min (incremental) |

### sccache (Optional Future Improvement)

For even faster builds, `sccache` can share compilation artifacts across jobs:

```yaml
- name: Install sccache
  uses: mozilla-actions/sccache-action@v0.0.6
- name: Configure sccache
  run: |
    echo "SCCACHE_GHA_ENABLED=true" >> $GITHUB_ENV
    echo "RUSTC_WRAPPER=sccache" >> $GITHUB_ENV
```

This is optional for the initial setup. Add when CI times become a bottleneck.

---

## 6. Workflow Files

### 6.1 `ci.yml` -- Runs on Every Push and PR

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  # ---- Rust checks (no CUDA feature) ----
  fmt:
    name: Formatting
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --all -- --check

  clippy:
    name: Clippy
    runs-on: ubuntu-24.04
    needs: fmt
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry/index
            ~/.cargo/registry/cache
            ~/.cargo/git/db
            target
          key: clippy-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: clippy-${{ runner.os }}-
      - run: cargo clippy --all-targets --all-features -- -D warnings

  test:
    name: Tests (${{ matrix.rust }})
    runs-on: ubuntu-24.04
    needs: fmt
    strategy:
      matrix:
        rust: [stable, "1.80"]  # stable + MSRV
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry/index
            ~/.cargo/registry/cache
            ~/.cargo/git/db
            target
          key: test-${{ matrix.rust }}-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: test-${{ matrix.rust }}-${{ runner.os }}-
      - run: cargo test --all

  build:
    name: Build
    runs-on: ubuntu-24.04
    needs: [clippy, test]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry/index
            ~/.cargo/registry/cache
            ~/.cargo/git/db
            target
          key: build-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: build-${{ runner.os }}-
      - run: cargo build --all

  # ---- C interception library ----
  c-intercept:
    name: C Interception Library
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Install build tools
        run: sudo apt-get update && sudo apt-get install -y build-essential
      - name: Build interception .so (stub headers)
        working-directory: crates/outterlink-client/csrc
        run: make

  # ---- Kernel module compile-check ----
  kernel-module:
    name: Kernel Module Check
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Install kernel headers
        run: |
          sudo apt-get update
          sudo apt-get install -y linux-headers-$(uname -r) build-essential
      - name: Compile-check kernel module
        working-directory: opendma/module
        run: |
          make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
          make -C /lib/modules/$(uname -r)/build M=$(pwd) clean

  # ---- Nightly Rust (allowed to fail) ----
  nightly:
    name: Nightly Rust
    runs-on: ubuntu-24.04
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: cargo test --all
      - run: cargo build --all
```

### 6.2 `release.yml` -- Build Release Binaries

Triggered on version tags (`v*`). Installs the CUDA toolkit, builds with `--features cuda`, and uploads artifacts to a GitHub Release.

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

permissions:
  contents: write

env:
  CARGO_TERM_COLOR: always

jobs:
  build-release:
    name: Build Release (${{ matrix.cuda }})
    runs-on: ubuntu-24.04
    strategy:
      matrix:
        cuda: ["12.6"]
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable

      - name: Install CUDA Toolkit
        uses: Jimver/cuda-toolkit@v0.2.19
        with:
          cuda: ${{ matrix.cuda }}
          method: network
          sub-packages: '["nvcc", "cudart-dev"]'

      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry/index
            ~/.cargo/registry/cache
            ~/.cargo/git/db
            target
          key: release-${{ runner.os }}-cuda${{ matrix.cuda }}-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: release-${{ runner.os }}-cuda${{ matrix.cuda }}-

      - name: Build release binaries
        run: cargo build --release --features cuda
        env:
          CUDA_HOME: ${{ env.CUDA_PATH }}

      - name: Build C interception library
        working-directory: crates/outterlink-client/csrc
        run: make CUDA_HOME=${{ env.CUDA_PATH }}
        env:
          CFLAGS_EXTRA: "-O2"

      - name: Package artifacts
        run: |
          mkdir -p dist
          cp target/release/outterlink-server dist/
          cp target/release/outterlink-cli dist/
          cp crates/outterlink-client/csrc/liboutterlink_intercept.so dist/
          tar czf outterlink-${{ github.ref_name }}-linux-x86_64-cuda${{ matrix.cuda }}.tar.gz -C dist .

      - name: Upload release artifacts
        uses: softprops/action-gh-release@v2
        with:
          files: outterlink-*.tar.gz
          generate_release_notes: true
```

### 6.3 `docs.yml` -- Generate and Deploy API Docs

```yaml
name: Documentation

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  pages: write
  id-token: write
  contents: read

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build-docs:
    name: Build API Docs
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - name: Cache Cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry/index
            ~/.cargo/registry/cache
            ~/.cargo/git/db
            target
          key: docs-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: docs-${{ runner.os }}-
      - name: Build docs
        run: cargo doc --no-deps --all-features --document-private-items
        env:
          RUSTDOCFLAGS: "--cfg docsrs -D warnings"
      - name: Add redirect to main crate
        run: echo '<meta http-equiv="refresh" content="0;url=outterlink_common/index.html">' > target/doc/index.html
      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: target/doc

  deploy-docs:
    name: Deploy to GitHub Pages
    needs: build-docs
    runs-on: ubuntu-24.04
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy
        id: deployment
        uses: actions/deploy-pages@v4
```

---

## 7. Branch Protection Rules

Configure in GitHub repo Settings > Branches:

### `main` branch

| Rule | Value |
|------|-------|
| Require PR before merging | Yes |
| Required status checks | `fmt`, `clippy`, `test (stable)`, `build`, `c-intercept` |
| Require branches to be up to date | Yes |
| Require linear history | Yes (squash merge) |
| Allow force pushes | No |
| Allow deletions | No |

### `dev` branch

| Rule | Value |
|------|-------|
| Required status checks | `fmt`, `clippy`, `test (stable)` |
| Allow force pushes | No |

---

## 8. Future Additions (Not in Initial Setup)

| Addition | When | Why |
|----------|------|-----|
| `sccache` for shared build cache | When CI > 5 min | Faster builds |
| Code coverage with `cargo-tarpaulin` | After P5 (PoC) | Track test quality |
| Benchmarking CI (`cargo bench`) | After P8 (Performance) | Prevent performance regressions |
| CUDA integration test runner (self-hosted) | After P5 (PoC) | Test real GPU path |
| Security audit (`cargo audit`) | After first release | Dependency vulnerabilities |
| Multi-arch builds (aarch64) | If ARM support needed | Broader hardware support |

### Self-Hosted Runner for GPU Tests

When integration tests that require real CUDA hardware exist (post-P5), set up a self-hosted GitHub Actions runner on one of the development PCs:

```yaml
# Future: gpu-integration.yml
jobs:
  gpu-tests:
    runs-on: self-hosted  # Runs on PC with GPU
    steps:
      - uses: actions/checkout@v4
      - run: cargo test --features cuda --test integration_tests
```

This is NOT part of the initial CI setup. It comes after P5 when there are actual GPU-dependent integration tests.

---

## Test Plan

| Test | Expected |
|------|----------|
| Push to `dev` branch | `ci.yml` runs, all jobs green |
| Open PR to `main` | `ci.yml` runs, required checks block merge until green |
| Tag `v0.1.0` | `release.yml` runs, tar.gz appears on GitHub Release page |
| Push to `main` | `docs.yml` runs, API docs deployed to GitHub Pages |
| Rust code with `cargo fmt` violation | `fmt` job fails |
| Rust code with clippy warning | `clippy` job fails |
| C code with `-Werror` violation | `c-intercept` job fails |
| Kernel module syntax error | `kernel-module` job fails |

## Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CUDA stub headers drift from real CUDA API | Tests pass but real build fails | Release workflow builds with real CUDA -- catches drift |
| Kernel module compile-check passes on CI kernel but fails on target | False confidence | Document target kernel version, test on real hardware |
| CI cache grows unbounded | Slow cache restore | GitHub evicts caches >10GB automatically; cache keys include Cargo.lock hash |
| `Jimver/cuda-toolkit` action breaks | Release workflow fails | Pin action version, vendor CUDA install script as fallback |
| GitHub Actions rate limits on forks | External contributors blocked | Allow manual re-run, document "expected" |

## Estimated Scope

| Component | Files | Complexity |
|-----------|-------|-----------|
| `.github/workflows/ci.yml` | 1 | Medium |
| `.github/workflows/release.yml` | 1 | Medium |
| `.github/workflows/docs.yml` | 1 | Low |
| CUDA stub headers | 2-3 files | Low |
| Feature flag wiring in Cargo.toml | 4 files (one per crate) | Low |
| C Makefile | 1 | Low |
| Kernel module Makefile | 1 | Low (already exists for P9) |

## Related Documents

- [P1: GitHub Repository Setup](P1-github-repo-setup.md)
- [P2: Development Environment](P2-dev-environment.md)
- [Final Pre-Plan](../pre-planning/02-FINAL-PREPLAN.md)
- [R3: CUDA Interception](../research/R3-cuda-interception.md)
