# First Linux Test Guide

**Created:** 2026-03-23
**Last Updated:** 2026-03-23
**Status:** Draft

## Purpose

Step-by-step instructions for verifying that OutterLink works end-to-end on real Linux hardware: building both components, running the server on the GPU machine, and running test workloads through the client.

---

## Prerequisites

- Both machines running Linux (Ubuntu 20.04+ recommended)
- GPU machine has NVIDIA driver and CUDA Toolkit installed (`nvidia-smi` works)
- Both machines have Rust stable installed
- Both machines can reach each other over TCP on port 14833
- You have followed the [Installation Guide](01-installation.md) and successfully built both binaries

---

## Step 1: Start the Server on the GPU Machine

SSH into the GPU machine:

```bash
ssh user@gpu-machine
cd ~/outerlink
```

Start the server with the real GPU backend:

```bash
./outerlink-server --listen 0.0.0.0:14833 --real-gpu --verbose
```

Expected output:

```
INFO outerlink_server: OuterLink Server starting on 0.0.0.0:14833
INFO outerlink_server: OpenDMA: non-proprietary GPU direct access
INFO outerlink_server: Loading real CUDA GPU backend...
INFO outerlink_server: GPU backend initialised (real GPU mode)
INFO outerlink_server: Listening on 0.0.0.0:14833
```

If you see `GPU backend init failed`, check:
- `nvidia-smi` works on the GPU machine
- The NVIDIA driver is loaded: `lsmod | grep nvidia`
- `libcuda.so` exists: `ldconfig -p | grep libcuda`

Leave this terminal open. The server runs in the foreground. Use a second terminal or `screen`/`tmux` for the next steps.

---

## Step 2: Verify Basic Connectivity

From the application machine, confirm the server is reachable:

```bash
nc -zv <gpu-machine-ip> 14833
# Expected: Connection to <ip> 14833 port [tcp/*] succeeded!
```

If this fails, check the GPU machine firewall (`sudo ufw status`) and that the server is actually running.

---

## Step 3: Run the Integration Test Suite Against a Real Server

On the application machine, you can run the stub-based integration tests first (no server needed) to confirm the build is correct:

```bash
cd ~/outerlink
cargo test --workspace -- --test-threads=1
```

All tests should pass. These use `StubGpuBackend` locally and do not require the remote server.

To run the real GPU tests (requires the server to be running with `--real-gpu`):

```bash
cargo test -p outerlink-server --features real-gpu-test --test real_gpu_test -- --nocapture
```

Expected: all `test_real_*` tests pass and print actual GPU information like:

```
[REAL GPU] Device count: 1
[REAL GPU] Driver version: 12040
[REAL GPU] Device name: NVIDIA GeForce RTX 3090
[REAL GPU] Total memory: 25769803776 bytes (~24 GB)
```

---

## Step 4: Run a Simple C Test Program

Create a minimal CUDA test program that performs a vector addition. Save it as `test_vector_add.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Simple vector add PTX kernel (for device sm_86 / Ampere)
static const char *PTX =
".version 7.0\n"
".target sm_86\n"
".address_size 64\n"
".visible .entry vector_add(.param .u64 a, .param .u64 b, .param .u64 c, .param .u32 n)\n"
"{\n"
"  .reg .u32 %tid, %n;\n"
"  .reg .u64 %a, %b, %c, %pa, %pb, %pc;\n"
"  .reg .f32 %va, %vb, %vc;\n"
"  ld.param.u64 %a, [a];\n"
"  ld.param.u64 %b, [b];\n"
"  ld.param.u64 %c, [c];\n"
"  ld.param.u32 %n, [n];\n"
"  mov.u32 %tid, %tid.x;\n"
"  setp.ge.u32 %n, %tid, %n;\n"
"  @%n bra end;\n"
"  mul.wide.u32 %pa, %tid, 4;\n"
"  add.u64 %pa, %a, %pa;\n"
"  mul.wide.u32 %pb, %tid, 4;\n"
"  add.u64 %pb, %b, %pb;\n"
"  mul.wide.u32 %pc, %tid, 4;\n"
"  add.u64 %pc, %c, %pc;\n"
"  ld.global.f32 %va, [%pa];\n"
"  ld.global.f32 %vb, [%pb];\n"
"  add.f32 %vc, %va, %vb;\n"
"  st.global.f32 [%pc], %vc;\n"
"  end:\n"
"  ret;\n"
"}\n";

int main(void) {
    CUresult res;
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUdeviceptr da, db, dc;

    const int N = 1024;
    float ha[N], hb[N], hc[N];
    for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = (float)(N - i); }

    res = cuInit(0);                      if (res) { printf("cuInit failed: %d\n", res); return 1; }
    res = cuDeviceGet(&dev, 0);           if (res) { printf("cuDeviceGet failed: %d\n", res); return 1; }
    res = cuCtxCreate(&ctx, 0, dev);      if (res) { printf("cuCtxCreate failed: %d\n", res); return 1; }
    res = cuModuleLoadData(&mod, PTX);    if (res) { printf("cuModuleLoadData failed: %d\n", res); return 1; }
    res = cuModuleGetFunction(&fn, mod, "vector_add");
    if (res) { printf("cuModuleGetFunction failed: %d\n", res); return 1; }

    res = cuMemAlloc(&da, N * sizeof(float)); if (res) { printf("cuMemAlloc A failed: %d\n", res); return 1; }
    res = cuMemAlloc(&db, N * sizeof(float)); if (res) { printf("cuMemAlloc B failed: %d\n", res); return 1; }
    res = cuMemAlloc(&dc, N * sizeof(float)); if (res) { printf("cuMemAlloc C failed: %d\n", res); return 1; }

    res = cuMemcpyHtoD(da, ha, N * sizeof(float)); if (res) { printf("HtoD A failed: %d\n", res); return 1; }
    res = cuMemcpyHtoD(db, hb, N * sizeof(float)); if (res) { printf("HtoD B failed: %d\n", res); return 1; }

    int n = N;
    void *args[] = { &da, &db, &dc, &n };
    res = cuLaunchKernel(fn, 4, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    if (res) { printf("cuLaunchKernel failed: %d\n", res); return 1; }

    res = cuCtxSynchronize(); if (res) { printf("cuCtxSynchronize failed: %d\n", res); return 1; }
    res = cuMemcpyDtoH(hc, dc, N * sizeof(float)); if (res) { printf("DtoH failed: %d\n", res); return 1; }

    // Verify: hc[i] should be N (ha[i] + hb[i] = i + (N-i) = N)
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (hc[i] != (float)N) {
            printf("ERROR at [%d]: got %f, expected %f\n", i, hc[i], (float)N);
            errors++;
        }
    }
    if (errors == 0) {
        printf("SUCCESS: vector_add passed (%d elements, each = %d)\n", N, N);
    } else {
        printf("FAILED: %d incorrect results\n", errors);
    }

    cuMemFree(da); cuMemFree(db); cuMemFree(dc);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return errors ? 1 : 0;
}
```

Compile and run:

```bash
# Compile against libcuda headers (adjust path if needed)
gcc test_vector_add.c -o test_vector_add \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcuda

# Run through OutterLink
export OUTERLINK_SERVER=<gpu-machine-ip>:14833
LD_PRELOAD=~/outerlink/libouterlink_client.so ./test_vector_add
```

Expected output:

```
SUCCESS: vector_add passed (1024 elements, each = 1024)
```

On the server terminal you should see connection and request logs (if `--verbose` was used).

---

## Step 5: Run PyTorch Through OutterLink

Install PyTorch on the application machine (CPU-only install is fine for the application machine since OutterLink provides the GPU remotely):

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

Run a simple test:

```bash
export OUTERLINK_SERVER=<gpu-machine-ip>:14833

LD_PRELOAD=~/outerlink/libouterlink_client.so python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
    print('Total memory:', torch.cuda.get_device_properties(0).total_memory // (1024**3), 'GB')
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y
    print('Matrix multiply result shape:', z.shape)
    print('SUCCESS')
"
```

Expected output (values depend on the actual GPU):

```
CUDA available: True
Device name: NVIDIA GeForce RTX 3090
Total memory: 24 GB
Matrix multiply result shape: torch.Size([1000, 1000])
SUCCESS
```

---

## Troubleshooting

### "CUDA available: False" in PyTorch

PyTorch uses both the Runtime API (`libcudart`) and the Driver API (`libcuda`). OutterLink only intercepts the Driver API. If PyTorch falls back to Runtime API paths before the Driver API intercept runs, it may not see the GPU. Try:

```bash
# Force PyTorch to use the driver API path
CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=~/outerlink/libouterlink_client.so python3 your_script.py
```

### Server Logs Show No Connections

Check that the correct IP and port are set:

```bash
echo $OUTERLINK_SERVER  # should print <gpu-ip>:14833
```

Check the firewall on the GPU machine:

```bash
sudo ufw status
sudo iptables -L INPUT -n | grep 14833
```

Try connecting with netcat from the application machine:

```bash
nc -zv <gpu-machine-ip> 14833
```

### "failed to connect to server: connection refused"

The server is not running or not listening on the expected interface. Verify on the GPU machine:

```bash
ss -tlnp | grep 14833
```

### "cuModuleLoadData failed: 200" (CUDA_ERROR_INVALID_CONTEXT)

This typically means the context was not created, or the wrong context is current. Make sure `cuInit(0)`, `cuDeviceGet`, and `cuCtxCreate` all succeed before loading modules.

### Kernel Launch Fails or Returns Wrong Results

Check the PTX target architecture. The example PTX above targets `sm_86` (Ampere). For different GPUs:
- Volta (V100): `sm_70`
- Turing (RTX 20xx): `sm_75`
- Ampere (RTX 30xx): `sm_86`
- Ada Lovelace (RTX 40xx): `sm_89`
- Blackwell (RTX 50xx): `sm_100`

### Application Machine Has No CUDA Headers

The test program above needs CUDA headers for compilation. Install the CUDA toolkit headers-only package:

```bash
# Ubuntu
sudo apt install nvidia-cuda-toolkit
# or download the CUDA toolkit runfile and use --toolkit --silent
```

Alternatively, you can download just the headers from the CUDA samples repository.

### Callback-based Applications Hang

If an application uses `cuStreamAddCallback` and appears to hang after `cuStreamSynchronize`, check that the callback channel is being established. Look for `CallbackChannelInit` in the server verbose logs. If missing, the callback connection may be blocked by a firewall rule that only allows the first connection from each source IP+port combination.

---

## Related Documents

- [Installation Guide](01-installation.md)
- [System Architecture](../architecture/01-system-architecture.md)
- [CUDA Function Coverage](../specs/cuda-function-coverage.md)

## Open Questions

- [ ] Does PyTorch's `torch.compile` with `reduce-overhead` (CUDA Graphs) work end-to-end?
- [ ] What CUDA driver version is on each target machine?
- [ ] Do cuDNN calls work through OutterLink (cuDNN uses the Driver API internally)?
- [ ] Is there a meaningful performance benchmark we can run on day one?
