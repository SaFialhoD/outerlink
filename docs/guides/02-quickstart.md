# OuterLink Quickstart Guide

**Created:** 2026-03-23
**Last Updated:** 2026-03-23
**Status:** Draft

## Purpose

End-to-end walkthrough: start the server on the GPU machine, set up the client on the application machine, run CUDA applications through OuterLink, and troubleshoot common issues.

**Prerequisite:** You have already built both `outerlink-server` and `libouterlink_client.so` following the [Installation Guide](01-installation.md).

---

## 1. Start the Server on the GPU Machine

### Stub Mode (no GPU required, for testing)

```bash
./outerlink-server --listen 0.0.0.0:14833
```

Stub mode responds to all CUDA calls with plausible fake values (simulates an RTX 3090). Use this to verify connectivity and protocol correctness without touching real GPU hardware.

### Real GPU Mode

```bash
./outerlink-server --listen 0.0.0.0:14833 --real-gpu
```

This loads `libcuda.so` via `libloading` at runtime and forwards all CUDA operations to the real NVIDIA driver. The NVIDIA driver must be installed and `nvidia-smi` must work.

### With Verbose Logging

```bash
./outerlink-server --listen 0.0.0.0:14833 --real-gpu --verbose
```

Verbose mode enables `debug`-level tracing output. You will see every incoming request, the dispatched `GpuBackend` call, and the response sent back.

### Select a Specific GPU

```bash
CUDA_VISIBLE_DEVICES=0 ./outerlink-server --listen 0.0.0.0:14833 --real-gpu
```

### Expected Startup Output

```
INFO outerlink_server: OuterLink Server starting on 0.0.0.0:14833
INFO outerlink_server: OpenDMA: non-proprietary GPU direct access
INFO outerlink_server: Loading real CUDA GPU backend...
INFO outerlink_server: GPU backend initialised (real GPU mode)
INFO outerlink_server: Listening on 0.0.0.0:14833
INFO outerlink_server::server: server accept loop starting addr=0.0.0.0:14833
```

### Verify It Is Listening

```bash
ss -tlnp | grep 14833
```

---

## 2. Set Up the Client on the Application Machine

### Set the Server Address

```bash
export OUTERLINK_SERVER=<gpu-machine-ip>:14833
```

Examples:
```bash
export OUTERLINK_SERVER=192.168.1.50:14833    # LAN IP
export OUTERLINK_SERVER=10.0.0.2:14833        # Private network
export OUTERLINK_SERVER=localhost:14833        # Same machine (testing)
```

If `OUTERLINK_SERVER` is not set, the client defaults to `localhost:14833`.

### Verify Network Connectivity

```bash
# From the application machine:
nc -zv <gpu-machine-ip> 14833
```

Expected: `Connection to <ip> 14833 port [tcp/*] succeeded!`

---

## 3. Run the Test Application

The included test app performs a complete vector-add workflow using only CUDA Driver API calls.

### Build the Test App

```bash
cd tests/cuda_test_app
make native
```

If the CUDA toolkit is not installed locally, the Makefile automatically uses OuterLink's stub headers from `cuda-stubs/`.

### Run Through OuterLink

```bash
OUTERLINK_SERVER=<gpu-machine-ip>:14833 \
LD_PRELOAD=../../target/release/libouterlink_client.so \
./test_vector_add
```

Or use the Makefile shortcut:

```bash
make interpose OUTERLINK_SERVER=<gpu-machine-ip>:14833 \
               OUTERLINK_LIB=../../target/release/libouterlink_client.so
```

### Expected Output (Real GPU Mode)

```
=== OuterLink CUDA Driver API Test: Vector Add ===

[1/12] cuInit
[2/12] cuDeviceGetCount
        Found 1 device(s)
[3/12] cuDeviceGet (ordinal=0)
        Device: NVIDIA GeForce RTX 3090
[4/12] cuCtxCreate_v2
[5/12] cuMemAlloc_v2 (3 buffers, 4096 bytes each)
[6/12] cuMemcpyHtoD_v2 (A and B)
[7/12] cuModuleLoadData (embedded PTX, 757 bytes)
[8/12] cuModuleGetFunction ("vector_add")
[9/12] cuLaunchKernel (grid=4, block=256, params=4)
[10/12] cuCtxSynchronize
[11/12] cuMemcpyDtoH_v2 (C)

--- Verification ---
  PASS: All 1024 elements correct.
  Sample: C[0]=0.0, C[1]=3.0, C[1023]=3069.0

[12/12] Cleanup

=== Test complete (exit code: 0) ===
```

### Server-Side Logs (Verbose Mode)

When a client connects, the server shows:

```
INFO outerlink_server::server: new connection peer=192.168.1.100:54321
```

Each CUDA call appears as a handled request in verbose mode.

---

## 4. Run PyTorch Through OuterLink

### Quick Smoke Test

```bash
LD_PRELOAD=/path/to/libouterlink_client.so \
OUTERLINK_SERVER=<gpu-machine-ip>:14833 \
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Device name:', torch.cuda.get_device_name(0))
print('Total memory:', torch.cuda.get_device_properties(0).total_mem // (1024**3), 'GB')
"
```

Expected output (real GPU mode):
```
CUDA available: True
Device count: 1
Device name: NVIDIA GeForce RTX 3090
Total memory: 24 GB
```

### Run a Training Script

```bash
LD_PRELOAD=/path/to/libouterlink_client.so \
OUTERLINK_SERVER=<gpu-machine-ip>:14833 \
python3 my_training_script.py
```

### Shell Alias (Optional)

Add to your `~/.bashrc` for convenience:

```bash
alias outerlink='LD_PRELOAD=/path/to/libouterlink_client.so OUTERLINK_SERVER=<gpu-machine-ip>:14833'
```

Then run any CUDA application:

```bash
outerlink python3 my_model.py
outerlink ./my_cuda_binary
```

### Wrapper Script (Optional)

Create `~/bin/outerlink-run`:

```bash
#!/bin/bash
# OuterLink wrapper: runs any command with CUDA interception
export OUTERLINK_SERVER="${OUTERLINK_SERVER:-192.168.1.50:14833}"
export LD_PRELOAD="/path/to/libouterlink_client.so"
exec "$@"
```

```bash
chmod +x ~/bin/outerlink-run
outerlink-run python3 my_model.py
```

---

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTERLINK_SERVER` | `localhost:14833` | Server address in `host:port` format |
| `LD_PRELOAD` | (none) | Must point to `libouterlink_client.so` to activate interception |
| `CUDA_VISIBLE_DEVICES` | (all GPUs) | Standard CUDA variable, controls which GPUs the server exposes |
| `RUST_LOG` | (none) | Set to `debug` or `trace` for detailed client-side logging (e.g., `RUST_LOG=outerlink_client=debug`) |

**Server CLI flags:**

| Flag | Description |
|------|-------------|
| `--listen <addr:port>` | Bind address (default: `0.0.0.0:14833`) |
| `--real-gpu` | Use real CUDA GPU backend (without this, stub mode is used) |
| `--verbose` / `-v` | Enable debug-level logging |

---

## 6. Troubleshooting

### Client cannot connect to server

**Symptom:** Warning at startup:
```
WARN outerlink_client: OuterLink: failed to connect to server: connection refused
```

**Causes and fixes:**
1. Server not running: start it with `./outerlink-server --listen 0.0.0.0:14833`
2. Wrong address: verify `OUTERLINK_SERVER` matches the server's IP and port
3. Firewall blocking: open port 14833 (`sudo ufw allow 14833/tcp`)
4. Server bound to `127.0.0.1` instead of `0.0.0.0`: use `--listen 0.0.0.0:14833` for remote access

**Test connectivity:**
```bash
nc -zv <gpu-machine-ip> 14833
```

### Application falls back to stub mode

**Symptom:** PyTorch reports `CUDA available: True` but device name is "NVIDIA GeForce RTX 3090" even when the actual GPU is different, or no real computation happens.

**Cause:** The client could not connect to the server. It falls back to returning plausible stub values so the application does not crash.

**Fix:** Ensure `OUTERLINK_SERVER` is set correctly and the server is reachable. Check for the connection warning in stderr.

### Connection timeout

**Symptom:**
```
WARN outerlink_client: connection timed out addr=192.168.1.50:14833 timeout_secs=10
```

**Cause:** The server is unreachable (network issue, wrong IP, server not started). The client waits up to 10 seconds for the initial TCP connect.

**Fix:** Verify network path, check that the server process is running.

### Server starts but GPU backend fails

**Symptom:**
```
ERROR outerlink_server: GPU backend init failed: ErrorNotInitialized
```

**Cause:** `--real-gpu` was passed but `libcuda.so` cannot be loaded or `cuInit()` fails.

**Fix:**
1. Verify NVIDIA driver: `nvidia-smi`
2. Verify `libcuda.so` exists: `ldconfig -p | grep libcuda`
3. Check driver version compatibility with your CUDA toolkit

### LD_PRELOAD has no effect

**Symptom:** Application runs but uses a local GPU instead of the remote one.

**Causes:**
1. Wrong library path: ensure `LD_PRELOAD` points to the actual `libouterlink_client.so` file
2. SUID binaries ignore `LD_PRELOAD` for security reasons
3. Static linking: if the application statically links `libcuda.a`, interception will not work (extremely rare)

**Verify interception is active:**
```bash
LD_PRELOAD=/path/to/libouterlink_client.so \
OUTERLINK_SERVER=localhost:14833 \
python3 -c "import ctypes; ctypes.CDLL('libcuda.so').cuInit(0)"
```

If OuterLink is active, you should see connection log messages.

### cuModuleLoadData fails (wrong PTX target)

**Symptom:** CUDA error 209 (`CUDA_ERROR_NO_BINARY_FOR_GPU`) or similar.

**Cause:** The PTX or cubin embedded in the application targets a compute capability that the remote GPU does not support (e.g., PTX for sm_90 on a sm_86 GPU).

**Fix:** Recompile the CUDA code with a compatible target, or use JIT-compilable PTX (which CUDA will JIT-compile for the actual GPU).

### Server crashes on shutdown

**Symptom:** Server does not exit cleanly after Ctrl+C.

**Expected behavior:** The server catches SIGINT/SIGTERM, stops accepting connections, waits up to 5 seconds for in-flight requests to drain, then exits. If in-flight requests hang (e.g., long kernel execution), the server exits after the drain timeout.

**If it hangs past 5 seconds:** Send SIGKILL as a last resort: `kill -9 <pid>`.

### Performance is slower than expected

**Causes:**
1. **Network latency:** Every CUDA call is a round-trip. Minimize the number of small CUDA calls; batch operations where possible.
2. **Data transfer:** Large `cuMemcpyHtoD` / `cuMemcpyDtoH` operations transfer data over TCP. Network bandwidth is the bottleneck. This is expected for Phase 1 (host-staged). Phase 2 (UCX/RDMA) and Phase 5 (OpenDMA) will eliminate these bottlenecks.
3. **Debug logging:** `--verbose` mode adds overhead. Disable it for benchmarks.
4. **Stub mode:** If running against the stub backend, no real GPU work happens. Use `--real-gpu` for actual workloads.

---

## 7. What Happens Under the Hood

When you run `LD_PRELOAD=libouterlink_client.so python3 my_model.py`:

1. The dynamic linker loads `libouterlink_client.so` before any other library.
2. The `.so` exports its own `dlsym()`, which shadows the system `dlsym`.
3. When PyTorch's CUDA runtime calls `dlsym(handle, "cuInit")`, OuterLink's version runs first. It finds `"cuInit"` in the hook table and returns `hook_cuInit`.
4. PyTorch calls `hook_cuInit(0)`, which calls `ensure_init()` (connects to the server via TCP, performs handshake), then calls `ol_cuInit(0)` (Rust FFI).
5. `ol_cuInit` serializes `[flags: u32 = 0]`, builds an OLNK header (magic "OLNK", version 1, msg_type = Init), sends it over TCP.
6. The server receives the frame, dispatches to `handle_request_full()`, calls `backend.init()`, serializes the `CuResult`, sends the response.
7. The client receives the response, extracts the `CuResult`, and returns it to PyTorch as a `CUresult`.
8. This process repeats for every CUDA Driver API call the application makes.

Handle translation happens transparently: when `cuCtxCreate` returns a context handle, the server returns the real handle, and the client maps it to a synthetic handle (e.g., `0x0C00_0000_0000_0001`) before returning it to the application. When the application later uses that handle in `cuCtxSetCurrent`, the client translates it back to the real handle before sending the request.

---

## Related Documents

- [Installation Guide](01-installation.md)
- [System Architecture](../architecture/01-system-architecture.md)
- [Testing on Linux](02-testing-on-linux.md)
- [Project Vision](../architecture/00-project-vision.md)

## Open Questions

- [ ] What is the measured latency overhead per CUDA call over a local network (1Gbps, 10Gbps)?
- [ ] How does PyTorch's `torch.compile` with `reduce-overhead` mode interact with the CUDA Graph interception?
- [ ] Are there CUDA Runtime API calls that bypass the Driver API and would not be intercepted?
- [ ] What is the maximum tested model size / VRAM usage through OuterLink?
- [ ] Should there be a health-check endpoint or CLI command to verify server status?
