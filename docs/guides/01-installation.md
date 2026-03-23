# OuterLink Installation Guide

**Created:** 2026-03-23
**Last Updated:** 2026-03-23
**Status:** Draft

## Purpose

Step-by-step instructions for building OuterLink from source on two networked Linux machines: a client machine (runs CUDA applications) and a GPU machine (hosts the NVIDIA GPU and runs the server daemon).

---

## Prerequisites

### Both Machines

| Requirement | Version | Notes |
|-------------|---------|-------|
| Linux | Ubuntu 20.04+ or equivalent glibc-based distro | LD_PRELOAD requires Linux/glibc |
| Rust toolchain | stable, edition 2021 | Install via `rustup` |
| Git | any recent | For cloning the repo |

Install Rust if needed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup show  # should print "stable"
```

### Client Machine (Application Machine)

| Requirement | Notes |
|-------------|-------|
| GCC or Clang | For compiling `interpose.c` (the C interposition layer) |
| `build-essential` | `sudo apt install build-essential` on Ubuntu/Debian |
| `libdl` + `libpthread` | Standard on all Linux distros, linked by the build script |

The client machine does **not** need an NVIDIA GPU or CUDA toolkit installed. It only needs the Rust toolchain and a C compiler. The build script (`crates/outerlink-client/build.rs`) uses the `cc` crate to compile `csrc/interpose.c` and link it into the shared library.

### GPU Machine (Server Machine)

| Requirement | Version | Notes |
|-------------|---------|-------|
| NVIDIA driver | 520+ | Must match or exceed CUDA toolkit version |
| CUDA Toolkit | 11.0+ (12.0+ recommended) | 12.0+ required for Library API (`cuLibraryLoadData`, `cuKernelGetFunction`) and CUDA Graph APIs |
| `libcuda.so` | Installed with driver | The server loads it at runtime via `libloading` |

Verify CUDA is available on the GPU machine:

```bash
nvidia-smi                                         # Should show your GPU(s)
nvcc --version                                     # Should show CUDA version
ls /usr/lib/x86_64-linux-gnu/libcuda.so* 2>/dev/null || \
  ls /usr/local/cuda/lib64/libcuda.so* 2>/dev/null   # Should find the library
```

---

## Building

### Clone the Repository

```bash
git clone https://github.com/SaFialhoD/outerlink.git
cd outerlink
```

### Build the Server (runs on the GPU machine)

```bash
cargo build --release -p outerlink-server
```

Output: `target/release/outerlink-server`

This produces a statically-linked binary (except for system libs). It can be copied to any Linux machine with the NVIDIA driver installed.

### Build the Client Library (runs on the application machine)

```bash
cargo build --release -p outerlink-client
```

Output: `target/release/libouterlink_client.so`

The build script automatically:
1. Compiles `csrc/interpose.c` with `-fvisibility=default -Wall -Wextra -O2`
2. Links `libdl` and `libpthread`
3. Produces a `cdylib` (.so) suitable for `LD_PRELOAD`

**Note:** The C interposition layer only compiles on Linux (`target_os == "linux"`). On Windows, the Rust code compiles normally but the interposition layer is skipped (no LD_PRELOAD on Windows).

### Build the Test Application (optional)

A minimal vector-add test program is provided in `tests/cuda_test_app/`:

```bash
cd tests/cuda_test_app

# Build against real CUDA headers (if CUDA toolkit is installed):
make native

# Or build with OuterLink's stub CUDA headers (no toolkit needed):
make native CUDA_PATH=/nonexistent
```

This produces `test_vector_add`, a C program that exercises: `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuCtxCreate_v2`, `cuMemAlloc_v2`, `cuMemcpyHtoD_v2`, `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, `cuCtxSynchronize`, `cuMemcpyDtoH_v2`, `cuMemFree_v2`, `cuModuleUnload`, `cuCtxDestroy_v2`.

### Build Everything

```bash
cargo build --release --workspace
```

This builds all four crates:
- `outerlink-server` -- GPU node daemon
- `outerlink-client` -- LD_PRELOAD interception library
- `outerlink-common` -- shared protocol, types, transport
- `outerlink-cli` -- management CLI (skeleton, no production commands yet)

---

## Verifying the Build

### Server Binary

```bash
./target/release/outerlink-server --help
```

Expected output:
```
OuterLink GPU node daemon

Usage: outerlink-server [OPTIONS]

Options:
  -l, --listen <LISTEN>  Address to listen on (ip:port) [default: 0.0.0.0:14833]
  -v, --verbose          Enable verbose (debug-level) logging
      --real-gpu         Use the real CUDA GPU backend (requires NVIDIA driver)
  -h, --help             Print help
```

### Client Library

```bash
file target/release/libouterlink_client.so
```

Expected: `ELF 64-bit LSB shared object, x86-64, ...`

Verify the interposition symbols are exported:

```bash
nm -D target/release/libouterlink_client.so | grep -c "hook_cu\|^T.*dlsym"
```

Should show a positive count (the hook functions and dlsym override are visible).

### Run Unit Tests (no GPU required)

```bash
cargo test --workspace -- --test-threads=1
```

All tests use `StubGpuBackend` by default and require no NVIDIA hardware. The `--test-threads=1` flag avoids TCP port conflicts between concurrent server tests.

### Run Real GPU Tests (requires NVIDIA GPU)

```bash
cargo test -p outerlink-server --features real-gpu-test --test real_gpu_test -- --nocapture
```

---

## Deployment

### Copy Binaries to Target Machines

**GPU machine:**
```bash
scp target/release/outerlink-server gpu-machine:~/outerlink/
```

**Application machine:**
```bash
scp target/release/libouterlink_client.so app-machine:~/outerlink/
```

### Firewall

Open port 14833 TCP (or whichever port you configure) on the GPU machine:

```bash
sudo ufw allow 14833/tcp
```

OuterLink uses **two TCP connections per client session** (one for requests, one for callbacks), both to the same port. No additional ports need to be opened.

---

## Project Layout Reference

```
outerlink/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── outerlink-server/               # GPU node daemon
│   │   ├── src/main.rs                 # CLI entry point (clap)
│   │   ├── src/server.rs               # TCP accept loop, graceful shutdown
│   │   ├── src/session.rs              # Per-connection state, resource cleanup
│   │   ├── src/handler.rs              # Protocol dispatch -> GpuBackend
│   │   ├── src/gpu_backend.rs          # GpuBackend trait + StubGpuBackend
│   │   └── src/cuda_backend.rs         # Real CUDA driver via libloading
│   ├── outerlink-client/               # LD_PRELOAD interception library
│   │   ├── csrc/interpose.c            # C layer: hooks dlsym + cuGetProcAddress
│   │   ├── csrc/interpose.h            # Declarations for ol_* Rust FFI functions
│   │   ├── build.rs                    # Compiles interpose.c via cc crate
│   │   ├── src/lib.rs                  # OuterLinkClient struct + connection logic
│   │   ├── src/ffi.rs                  # #[no_mangle] extern "C" FFI functions
│   │   └── src/callback.rs             # CallbackRegistry for stream callbacks
│   ├── outerlink-common/               # Shared types and transport
│   │   ├── src/protocol.rs             # OLNK wire protocol (22-byte header)
│   │   ├── src/handle.rs               # HandleMap + HandleStore
│   │   ├── src/transport.rs            # TransportConnection trait
│   │   ├── src/tcp_transport.rs        # TcpTransportConnection impl
│   │   ├── src/retry.rs                # RetryConfig (exponential backoff)
│   │   ├── src/cuda_types.rs           # CuResult enum, CUDA type definitions
│   │   └── src/error.rs                # OuterLinkError enum
│   └── outerlink-cli/                  # Management CLI (skeleton)
├── cuda-stubs/                         # Minimal cuda.h for building without CUDA toolkit
├── tests/cuda_test_app/                # Standalone C test (vector add)
│   ├── test_vector_add.c
│   ├── Makefile
│   └── README.md
└── opendma/                            # Future: OpenDMA kernel module (C)
```

---

## Related Documents

- [System Architecture](../architecture/01-system-architecture.md)
- [Quickstart Guide](02-quickstart.md)
- [Testing on Linux](02-testing-on-linux.md)
- [Project Vision](../architecture/00-project-vision.md)

## Open Questions

- [ ] Package manager / installer script for one-command setup?
- [ ] systemd unit file for running the server as a daemon?
- [ ] Cross-compilation support (e.g., build on x86, deploy to ARM)?
- [ ] Container (Docker) support for the server?
- [ ] Minimum supported NVIDIA driver version for the full CUDA 12+ feature set?
