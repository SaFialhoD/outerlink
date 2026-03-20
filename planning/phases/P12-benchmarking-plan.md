# P12: Benchmarking Plan

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - Parallel with P5 (PoC)

## Purpose

Define what performance metrics to measure, how to measure them, what baselines to compare against, and how to store and report results. Every performance claim about OutterLink must be backed by reproducible benchmark data.

## Goal

A benchmark suite that measures OutterLink's overhead per call type, memory transfer throughput, kernel launch latency, and end-to-end application performance. Results are stored in a machine-readable format for tracking regressions across versions.

## Milestone

- Microbenchmarks for every intercepted CUDA operation (latency in microseconds)
- Memory throughput benchmarks (GB/s) for various transfer sizes
- Baseline measurements: direct GPU (no OutterLink) vs OutterLink loopback vs OutterLink cross-PC
- Results stored as JSON with hardware/software metadata
- Automated benchmark runner script

## Prerequisites

- [ ] P5: PoC complete (basic interception working)
- [ ] P2: Development environment set up on at least one GPU machine
- [ ] At least one machine with GPU for baselines

---

## 1. Metrics to Measure

### 1.1 Per-Call Latency

Measure the round-trip time for each intercepted CUDA function call.

| Category | Calls to Measure | Expected Baseline (direct) | Expected OutterLink Overhead |
|----------|-----------------|---------------------------|------------------------------|
| **Init** | `cuInit` | ~100us (one-time) | +TCP connect time (~1ms first call) |
| **Device query** | `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceTotalMem` | <1us each | +1 RTT (~100us loopback, ~200us cross-PC) |
| **Context** | `cuCtxCreate`, `cuCtxDestroy` | ~10-50us | +1 RTT |
| **Memory alloc** | `cuMemAlloc` | ~5-20us | +1 RTT |
| **Memory free** | `cuMemFree` | ~5-10us | +1 RTT |
| **Memory copy HtoD** | `cuMemcpyHtoD` (various sizes) | Depends on size | +1 RTT + data transfer time |
| **Memory copy DtoH** | `cuMemcpyDtoH` (various sizes) | Depends on size | +1 RTT + data transfer time |
| **Module load** | `cuModuleLoadData` | ~100-500us | +1 RTT + module data transfer |
| **Function lookup** | `cuModuleGetFunction` | <5us | +1 RTT |
| **Kernel launch** | `cuLaunchKernel` + sync | Varies by kernel | +1 RTT + param serialization |

**Measurement method:** `std::time::Instant` on client side, measuring from before the interception function call to after the response is received. Also measure server-side CUDA execution time separately.

```
Total latency = client_serialization + network_send + server_deserialize +
                server_cuda_call + server_serialize + network_recv + client_deserialize
```

### 1.2 Memory Transfer Throughput

| Transfer Size | Metric | Purpose |
|--------------|--------|---------|
| 1 B | Latency only | Minimum overhead measurement |
| 64 B | Latency only | Typical scalar/pointer arg |
| 1 KB | Latency + throughput | Small buffer |
| 4 KB | Latency + throughput | Page size |
| 64 KB | Throughput | L2 cache size |
| 1 MB | Throughput | Typical tensor slice |
| 16 MB | Throughput | Larger tensor |
| 64 MB | Throughput | Large buffer |
| 256 MB | Throughput | Large model weight |
| 1 GB | Throughput | Full layer transfer |

**Throughput formula:** `bytes_transferred / wall_clock_time_seconds = GB/s`

### 1.3 Overhead Percentage

For each operation: `overhead_pct = ((outterlink_time - direct_time) / direct_time) * 100`

This is the key metric for understanding the cost of remoting. Target: device queries should be <10x overhead; memory transfers at large sizes should approach wire speed.

### 1.4 CPU Usage

Measure CPU utilization during bulk transfers:
- Client CPU during HtoD copy
- Server CPU during HtoD copy
- Client CPU during kernel launch
- Server CPU during kernel launch

**Tool:** `/proc/self/stat` sampling or `getrusage()` before/after. Also `perf stat` for detailed breakdown.

### 1.5 Kernel Launch Overhead Breakdown

For `cuLaunchKernel`, break down the overhead:

```
| Phase                  | Time   |
|------------------------|--------|
| Client param serialize | __us   |
| Network send           | __us   |
| Server param deserialize| __us  |
| Server cuLaunchKernel  | __us   |
| Server cuCtxSynchronize| __us   |
| Network recv           | __us   |
| Client response parse  | __us   |
| TOTAL                  | __us   |
```

### 1.6 Connection Metrics

| Metric | Description |
|--------|-------------|
| Handshake time | Time from TCP connect to ready for first CUDA call |
| Reconnection time | Time to re-establish after connection drop |
| Concurrent connections | Max simultaneous clients before degradation |

---

## 2. Benchmark Programs

### 2.1 Microbenchmark: Memory Copy Throughput

```cuda
// benchmarks/memcpy_bench.cu
// Measures HtoD and DtoH throughput for various sizes
// Compile: nvcc -o memcpy_bench memcpy_bench.cu -lcuda

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

typedef struct {
    size_t size;
    int iterations;
    double htod_us;       // avg microseconds
    double dtoh_us;       // avg microseconds
    double htod_gbps;     // GB/s
    double dtoh_gbps;     // GB/s
} BenchResult;

static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

void bench_memcpy(size_t size, int iterations, BenchResult *result) {
    CUdeviceptr d_buf;
    void *h_buf;

    CHECK(cuMemAlloc(&d_buf, size));
    h_buf = malloc(size);
    memset(h_buf, 0xAB, size);

    // Warmup
    for (int i = 0; i < 3; i++) {
        CHECK(cuMemcpyHtoD(d_buf, h_buf, size));
        CHECK(cuMemcpyDtoH(h_buf, d_buf, size));
    }

    // HtoD benchmark
    double start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuMemcpyHtoD(d_buf, h_buf, size));
    }
    double htod_total = get_time_us() - start;

    // DtoH benchmark
    start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuMemcpyDtoH(h_buf, d_buf, size));
    }
    double dtoh_total = get_time_us() - start;

    result->size = size;
    result->iterations = iterations;
    result->htod_us = htod_total / iterations;
    result->dtoh_us = dtoh_total / iterations;
    result->htod_gbps = (double)size / (result->htod_us * 1e3); // bytes/us -> GB/s
    result->dtoh_gbps = (double)size / (result->dtoh_us * 1e3);

    CHECK(cuMemFree(d_buf));
    free(h_buf);
}

int main() {
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    char name[256];
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    printf("{\"gpu\": \"%s\", \"results\": [\n", name);

    size_t sizes[] = {1, 64, 1024, 4096, 65536, 1048576, 16777216, 67108864, 268435456};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        // More iterations for smaller sizes
        int iters = sizes[i] < 65536 ? 10000 :
                    sizes[i] < 16777216 ? 1000 : 100;

        BenchResult r;
        bench_memcpy(sizes[i], iters, &r);

        printf("  {\"size\": %zu, \"iterations\": %d, "
               "\"htod_us\": %.2f, \"dtoh_us\": %.2f, "
               "\"htod_gbps\": %.4f, \"dtoh_gbps\": %.4f}%s\n",
               r.size, r.iterations,
               r.htod_us, r.dtoh_us,
               r.htod_gbps, r.dtoh_gbps,
               i < num_sizes - 1 ? "," : "");
    }

    printf("]}\n");

    CHECK(cuCtxDestroy(ctx));
    return 0;
}
```

### 2.2 Microbenchmark: Alloc/Free Throughput

```cuda
// benchmarks/alloc_bench.cu
// Measures cuMemAlloc + cuMemFree latency

#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    int iterations = 10000;
    size_t alloc_size = 4096;

    // Warmup
    for (int i = 0; i < 100; i++) {
        CUdeviceptr p;
        CHECK(cuMemAlloc(&p, alloc_size));
        CHECK(cuMemFree(p));
    }

    // Benchmark alloc
    double start = get_time_us();
    CUdeviceptr ptrs[10000];
    for (int i = 0; i < iterations; i++) {
        CHECK(cuMemAlloc(&ptrs[i], alloc_size));
    }
    double alloc_total = get_time_us() - start;

    // Benchmark free
    start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuMemFree(ptrs[i]));
    }
    double free_total = get_time_us() - start;

    printf("{\"alloc_size\": %zu, \"iterations\": %d, "
           "\"alloc_avg_us\": %.2f, \"free_avg_us\": %.2f, "
           "\"alloc_total_us\": %.0f, \"free_total_us\": %.0f}\n",
           alloc_size, iterations,
           alloc_total / iterations, free_total / iterations,
           alloc_total, free_total);

    CHECK(cuCtxDestroy(ctx));
    return 0;
}
```

### 2.3 Microbenchmark: Kernel Launch Latency

```cuda
// benchmarks/launch_bench.cu
// Measures cuLaunchKernel round-trip latency with a trivial kernel

#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Trivial kernel: writes threadIdx.x to output
static const char *noop_ptx =
    ".version 7.0\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry noop() { ret; }\n"
    ".visible .entry write_one(\n"
    "    .param .u64 out\n"
    ")\n"
    "{\n"
    "    .reg .b64 %rd<2>;\n"
    "    .reg .b32 %r<1>;\n"
    "    ld.param.u64 %rd0, [out];\n"
    "    mov.u32 %r0, 42;\n"
    "    st.global.u32 [%rd0], %r0;\n"
    "    ret;\n"
    "}\n";

static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, noop_ptx));

    CUfunction noop_func, write_func;
    CHECK(cuModuleGetFunction(&noop_func, mod, "noop"));
    CHECK(cuModuleGetFunction(&write_func, mod, "write_one"));

    CUdeviceptr d_out;
    CHECK(cuMemAlloc(&d_out, sizeof(int)));

    int iterations = 10000;

    // Warmup
    for (int i = 0; i < 100; i++) {
        CHECK(cuLaunchKernel(noop_func, 1,1,1, 1,1,1, 0, 0, NULL, NULL));
        CHECK(cuCtxSynchronize());
    }

    // Benchmark: noop kernel launch + sync
    double start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuLaunchKernel(noop_func, 1,1,1, 1,1,1, 0, 0, NULL, NULL));
        CHECK(cuCtxSynchronize());
    }
    double noop_total = get_time_us() - start;

    // Benchmark: kernel with one device pointer argument
    void *args[] = { &d_out };
    start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuLaunchKernel(write_func, 1,1,1, 1,1,1, 0, 0, args, NULL));
        CHECK(cuCtxSynchronize());
    }
    double write_total = get_time_us() - start;

    printf("{\"iterations\": %d, "
           "\"noop_launch_sync_avg_us\": %.2f, "
           "\"onearg_launch_sync_avg_us\": %.2f}\n",
           iterations,
           noop_total / iterations,
           write_total / iterations);

    CHECK(cuMemFree(d_out));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    return 0;
}
```

### 2.4 Microbenchmark: Device Query Latency

```cuda
// benchmarks/device_query_bench.cu
// Measures latency of device property queries

#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    CHECK(cuInit(0));

    int iterations = 100000;
    int count;
    CUdevice dev;
    char name[256];
    size_t mem;
    int attr;

    // Warmup
    CHECK(cuDeviceGetCount(&count));
    CHECK(cuDeviceGet(&dev, 0));

    // DeviceGetCount
    double start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuDeviceGetCount(&count));
    }
    double count_total = get_time_us() - start;

    // DeviceGetName
    start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuDeviceGetName(name, sizeof(name), dev));
    }
    double name_total = get_time_us() - start;

    // DeviceTotalMem
    start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuDeviceTotalMem(&mem, dev));
    }
    double mem_total = get_time_us() - start;

    // DeviceGetAttribute
    start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        CHECK(cuDeviceGetAttribute(&attr, 1, dev)); // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    }
    double attr_total = get_time_us() - start;

    printf("{\"iterations\": %d, "
           "\"device_get_count_avg_us\": %.4f, "
           "\"device_get_name_avg_us\": %.4f, "
           "\"device_total_mem_avg_us\": %.4f, "
           "\"device_get_attribute_avg_us\": %.4f}\n",
           iterations,
           count_total / iterations,
           name_total / iterations,
           mem_total / iterations,
           attr_total / iterations);

    return 0;
}
```

### 2.5 Application-Level Benchmark: Matrix Multiply

```cuda
// benchmarks/matmul_bench.cu
// Matrix multiplication benchmark - realistic workload
// Measures total time including data transfer and kernel execution

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static const char *matmul_ptx =
    ".version 7.0\n"
    ".target sm_52\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry matmul(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B,\n"
    "    .param .u64 C,\n"
    "    .param .u32 N\n"
    ")\n"
    "{\n"
    "    .reg .pred %p<2>;\n"
    "    .reg .f32 %f<4>;\n"
    "    .reg .b32 %r<10>;\n"
    "    .reg .b64 %rd<20>;\n"
    "\n"
    "    ld.param.u64 %rd1, [A];\n"
    "    ld.param.u64 %rd2, [B];\n"
    "    ld.param.u64 %rd3, [C];\n"
    "    ld.param.u32 %r1, [N];\n"
    "\n"
    "    // row = blockIdx.y * blockDim.y + threadIdx.y\n"
    "    mov.u32 %r2, %ctaid.y;\n"
    "    mov.u32 %r3, %ntid.y;\n"
    "    mov.u32 %r4, %tid.y;\n"
    "    mad.lo.s32 %r2, %r2, %r3, %r4;\n"
    "\n"
    "    // col = blockIdx.x * blockDim.x + threadIdx.x\n"
    "    mov.u32 %r5, %ctaid.x;\n"
    "    mov.u32 %r6, %ntid.x;\n"
    "    mov.u32 %r7, %tid.x;\n"
    "    mad.lo.s32 %r5, %r5, %r6, %r7;\n"
    "\n"
    "    setp.ge.s32 %p1, %r2, %r1;\n"
    "    @%p1 bra $done;\n"
    "    setp.ge.s32 %p1, %r5, %r1;\n"
    "    @%p1 bra $done;\n"
    "\n"
    "    // sum = 0\n"
    "    mov.f32 %f3, 0f00000000;\n"
    "\n"
    "    // for k = 0 to N-1\n"
    "    mov.u32 %r8, 0;\n"
    "$loop:\n"
    "    setp.ge.s32 %p1, %r8, %r1;\n"
    "    @%p1 bra $end_loop;\n"
    "\n"
    "    // A[row * N + k]\n"
    "    mad.lo.s32 %r9, %r2, %r1, %r8;\n"
    "    cvt.s64.s32 %rd4, %r9;\n"
    "    shl.b64 %rd5, %rd4, 2;\n"
    "    add.s64 %rd6, %rd1, %rd5;\n"
    "    ld.global.f32 %f1, [%rd6];\n"
    "\n"
    "    // B[k * N + col]\n"
    "    mad.lo.s32 %r9, %r8, %r1, %r5;\n"
    "    cvt.s64.s32 %rd7, %r9;\n"
    "    shl.b64 %rd8, %rd7, 2;\n"
    "    add.s64 %rd9, %rd2, %rd8;\n"
    "    ld.global.f32 %f2, [%rd9];\n"
    "\n"
    "    fma.rn.f32 %f3, %f1, %f2, %f3;\n"
    "    add.s32 %r8, %r8, 1;\n"
    "    bra $loop;\n"
    "$end_loop:\n"
    "\n"
    "    // C[row * N + col] = sum\n"
    "    mad.lo.s32 %r9, %r2, %r1, %r5;\n"
    "    cvt.s64.s32 %rd10, %r9;\n"
    "    shl.b64 %rd11, %rd10, 2;\n"
    "    add.s64 %rd12, %rd3, %rd11;\n"
    "    st.global.f32 [%rd12], %f3;\n"
    "\n"
    "$done:\n"
    "    ret;\n"
    "}\n";

static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char **argv) {
    int N = 512;
    if (argc > 1) N = atoi(argv[1]);

    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, matmul_ptx));
    CUfunction func;
    CHECK(cuModuleGetFunction(&func, mod, "matmul"));

    size_t mat_bytes = (size_t)N * N * sizeof(float);

    // Host data
    float *h_A = (float*)malloc(mat_bytes);
    float *h_B = (float*)malloc(mat_bytes);
    float *h_C = (float*)malloc(mat_bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // Device buffers
    CUdeviceptr d_A, d_B, d_C;
    CHECK(cuMemAlloc(&d_A, mat_bytes));
    CHECK(cuMemAlloc(&d_B, mat_bytes));
    CHECK(cuMemAlloc(&d_C, mat_bytes));

    int iterations = 5;
    double upload_us = 0, compute_us = 0, download_us = 0;

    for (int iter = 0; iter < iterations; iter++) {
        // Upload
        double t0 = get_time_us();
        CHECK(cuMemcpyHtoD(d_A, h_A, mat_bytes));
        CHECK(cuMemcpyHtoD(d_B, h_B, mat_bytes));
        double t1 = get_time_us();

        // Compute
        unsigned int block = 16;
        unsigned int grid = (N + block - 1) / block;
        void *args[] = { &d_A, &d_B, &d_C, &N };
        CHECK(cuLaunchKernel(func, grid, grid, 1, block, block, 1, 0, 0, args, NULL));
        CHECK(cuCtxSynchronize());
        double t2 = get_time_us();

        // Download
        CHECK(cuMemcpyDtoH(h_C, d_C, mat_bytes));
        double t3 = get_time_us();

        if (iter > 0) { // Skip first iteration (warmup)
            upload_us += (t1 - t0);
            compute_us += (t2 - t1);
            download_us += (t3 - t2);
        }
    }

    int real_iters = iterations - 1; // Exclude warmup
    printf("{\"N\": %d, \"mat_bytes\": %zu, \"iterations\": %d, "
           "\"upload_avg_us\": %.2f, \"compute_avg_us\": %.2f, \"download_avg_us\": %.2f, "
           "\"total_avg_us\": %.2f, "
           "\"upload_gbps\": %.4f, \"download_gbps\": %.4f}\n",
           N, mat_bytes, real_iters,
           upload_us / real_iters,
           compute_us / real_iters,
           download_us / real_iters,
           (upload_us + compute_us + download_us) / real_iters,
           (2.0 * mat_bytes) / (upload_us / real_iters * 1e3),
           mat_bytes / (download_us / real_iters * 1e3));

    CHECK(cuMemFree(d_A));
    CHECK(cuMemFree(d_B));
    CHECK(cuMemFree(d_C));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    free(h_A); free(h_B); free(h_C);
    return 0;
}
```

---

## 3. Baselines

### 3.1 Baseline Configurations

Every benchmark runs in three configurations:

| Configuration | Description | Purpose |
|--------------|-------------|---------|
| **Direct** | Application runs with real `libcuda.so`, no OutterLink | Establishes the "zero overhead" baseline |
| **Loopback** | Client and server on same machine, TCP `127.0.0.1` | Isolates OutterLink overhead from network effects |
| **Cross-PC** | Client on PC-A, server on PC-B, TCP over 100GbE | Measures real-world performance |

### 3.2 How to Collect Baselines

```bash
# 1. Direct baseline (no OutterLink)
./memcpy_bench > results/baseline_direct.json

# 2. Loopback baseline
#    Terminal 1: outterlink-server --port 9370
#    Terminal 2:
OUTTERLINK_SERVER=127.0.0.1:9370 \
LD_PRELOAD=target/release/liboutterlink_client.so \
./memcpy_bench > results/baseline_loopback.json

# 3. Cross-PC baseline
#    PC-B: outterlink-server --port 9370
#    PC-A:
OUTTERLINK_SERVER=192.168.100.2:9370 \
LD_PRELOAD=target/release/liboutterlink_client.so \
./memcpy_bench > results/baseline_crosspc.json
```

### 3.3 Network Baseline

Separate from CUDA benchmarks, measure raw network performance:

```bash
# TCP throughput
iperf3 -c 192.168.100.2 -t 10 -J > results/network_tcp.json

# TCP latency
ping -c 100 192.168.100.2 | tail -1  # min/avg/max/mdev

# Future (Phase 4): RDMA throughput
ib_write_bw -d mlx5_0 192.168.100.2 --output=json > results/network_rdma_write.json
ib_read_bw -d mlx5_0 192.168.100.2 --output=json > results/network_rdma_read.json

# Future (Phase 4): RDMA latency
ib_write_lat -d mlx5_0 192.168.100.2 --output=json > results/network_rdma_lat.json
```

### 3.4 Theoretical Maximums

| Path | Bandwidth | Latency |
|------|-----------|---------|
| PCIe 4.0 x16 (CPU to GPU) | 32 GB/s | ~1us |
| PCIe 5.0 x16 (CPU to GPU) | 64 GB/s | ~1us |
| NVLink (3090 Ti pair) | 112.5 GB/s | <1us |
| TCP loopback | ~40 GB/s | ~10us |
| TCP 100GbE | ~12 GB/s | ~100us |
| RDMA 100GbE (future) | ~12.5 GB/s | ~2us |

---

## 4. Tools

### 4.1 Rust Benchmarking (criterion)

For benchmarking the Rust components (serialization, handle lookup, protocol encode/decode):

```toml
# Cargo.toml (workspace)
[workspace.dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

```rust
// benches/protocol_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_encode_memcpy_htod(c: &mut Criterion) {
    let mut group = c.benchmark_group("protocol_encode");

    for size in [64, 1024, 65536, 1048576].iter() {
        let data = vec![0xABu8; *size];
        let req = Request::MemcpyHtoD {
            dptr: 0x7F0000001000,
            size: *size as u64,
            data: data.clone(),
        };

        group.bench_with_input(
            BenchmarkId::new("memcpy_htod", size),
            size,
            |b, _| b.iter(|| black_box(req.encode())),
        );
    }
    group.finish();
}

fn bench_handle_lookup(c: &mut Criterion) {
    let table = HandleTables::new();
    // Pre-populate with 10000 handles
    let handles: Vec<_> = (0..10000).map(|i| table.create_device_ptr(i)).collect();

    c.bench_function("handle_lookup_10k", |b| {
        b.iter(|| {
            for h in &handles {
                black_box(table.lookup_device_ptr(*h));
            }
        })
    });
}

criterion_group!(benches, bench_encode_memcpy_htod, bench_handle_lookup);
criterion_main!(benches);
```

### 4.2 CUDA Profiling Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `nvprof` | Legacy CUDA profiler | `nvprof ./benchmark` |
| `nsys` (Nsight Systems) | System-level timeline | `nsys profile ./benchmark` |
| `ncu` (Nsight Compute) | Kernel-level metrics | `ncu ./benchmark` |
| `nvidia-smi dmon` | Real-time GPU monitoring | `nvidia-smi dmon -d 1` (during benchmark) |

### 4.3 Network Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `iperf3` | TCP/UDP bandwidth | `iperf3 -c <ip> -t 10` |
| `perftest` (RDMA) | `ib_write_bw`, `ib_read_bw`, `ib_write_lat` | `ib_write_bw -d mlx5_0 <ip>` |
| `sockperf` | TCP/UDP latency | `sockperf ping-pong -i <ip>` |
| `ethtool` | NIC statistics | `ethtool -S enp1s0f0` |

### 4.4 System Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `perf stat` | CPU cycle/instruction counting | `perf stat ./benchmark` |
| `htop` | CPU utilization | Visual monitoring during benchmarks |
| `vmstat` | Memory/IO statistics | `vmstat 1` during benchmarks |
| `mpstat` | Per-CPU utilization | `mpstat -P ALL 1` during benchmarks |

---

## 5. Reporting Format

### 5.1 Result File Format

All benchmark results are stored as JSON with metadata headers.

```json
{
    "benchmark": "memcpy_throughput",
    "version": "0.1.0",
    "timestamp": "2026-03-19T14:30:00Z",
    "git_commit": "abc123def456",
    "configuration": "loopback",

    "hardware": {
        "client": {
            "hostname": "outterlink-pc1",
            "cpu": "Intel Core i9-14900K",
            "ram_gb": 256,
            "nic": "ConnectX-5 100GbE"
        },
        "server": {
            "hostname": "outterlink-pc2",
            "cpu": "AMD Threadripper 9960X",
            "ram_gb": 256,
            "gpu": "NVIDIA RTX 3090 Ti",
            "gpu_vram_gb": 24,
            "nic": "ConnectX-5 100GbE"
        }
    },

    "software": {
        "os": "Ubuntu 24.04",
        "kernel": "6.8.0-generic",
        "cuda_version": "12.6",
        "nvidia_driver": "560.35.03",
        "rust_version": "1.78.0",
        "outterlink_version": "0.1.0",
        "mlnx_ofed": "24.01"
    },

    "network": {
        "link_speed_gbps": 100,
        "mtu": 9000,
        "transport": "tcp"
    },

    "results": [
        {
            "size_bytes": 1024,
            "iterations": 10000,
            "htod_avg_us": 150.25,
            "htod_p50_us": 148.00,
            "htod_p99_us": 210.50,
            "htod_min_us": 140.00,
            "htod_max_us": 350.00,
            "htod_gbps": 0.0068,
            "dtoh_avg_us": 155.30,
            "dtoh_gbps": 0.0066
        }
    ]
}
```

### 5.2 Result Directory Structure

```
benchmarks/
├── src/                         # Benchmark source code
│   ├── memcpy_bench.cu
│   ├── alloc_bench.cu
│   ├── launch_bench.cu
│   ├── device_query_bench.cu
│   └── matmul_bench.cu
├── benches/                     # Rust criterion benchmarks
│   ├── protocol_bench.rs
│   └── handle_bench.rs
├── results/                     # Raw results (JSON)
│   ├── 2026-03-19/
│   │   ├── memcpy_direct_abc123.json
│   │   ├── memcpy_loopback_abc123.json
│   │   ├── memcpy_crosspc_abc123.json
│   │   ├── alloc_direct_abc123.json
│   │   ├── launch_direct_abc123.json
│   │   └── network_baseline_abc123.json
│   └── latest -> 2026-03-19/   # Symlink to latest results
├── scripts/
│   ├── run_all.sh              # Run all benchmarks in all configurations
│   ├── compare.py              # Compare two result sets
│   └── report.py               # Generate markdown report from results
├── reports/                     # Generated markdown reports
│   └── 2026-03-19-report.md
└── Makefile                    # Build benchmark binaries
```

### 5.3 Comparison Report Format

```markdown
# OutterLink Benchmark Report
**Date:** 2026-03-19
**Commit:** abc123
**GPU:** RTX 3090 Ti | **Network:** 100GbE TCP

## Memory Copy Throughput

| Size | Direct (GB/s) | Loopback (GB/s) | Cross-PC (GB/s) | Overhead (loopback) | Overhead (cross-PC) |
|------|--------------|-----------------|-----------------|--------------------|--------------------|
| 1 KB | 2.50 | 0.007 | 0.005 | 357x | 500x |
| 64 KB | 15.00 | 0.43 | 0.32 | 35x | 47x |
| 1 MB | 25.00 | 4.80 | 3.50 | 5.2x | 7.1x |
| 64 MB | 30.00 | 10.50 | 9.80 | 2.9x | 3.1x |
| 256 MB | 31.00 | 11.20 | 10.50 | 2.8x | 2.9x |

## Call Latency

| Operation | Direct (us) | Loopback (us) | Cross-PC (us) |
|-----------|------------|---------------|---------------|
| cuDeviceGetCount | 0.3 | 120 | 250 |
| cuMemAlloc | 15 | 135 | 265 |
| cuLaunchKernel+sync | 25 | 145 | 275 |

## Key Findings
- Large transfers (64MB+) approach wire speed
- Per-call overhead dominated by TCP round-trip
- Kernel launch overhead = ~120us (loopback), acceptable for compute-heavy kernels
```

### 5.4 Automated Benchmark Runner

```bash
#!/bin/bash
# benchmarks/scripts/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
DATE=$(date +%Y-%m-%d)
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
RESULT_DIR="$BENCH_DIR/results/$DATE"
PORT=9370

mkdir -p "$RESULT_DIR"

echo "=== OutterLink Benchmark Suite ==="
echo "Date: $DATE"
echo "Commit: $COMMIT"
echo "Results: $RESULT_DIR"
echo ""

# Build
echo "--- Building benchmarks ---"
cd "$BENCH_DIR/src"
for f in *.cu; do
    name="${f%.cu}"
    nvcc -o "$name" "$f" -lcuda 2>/dev/null && echo "  Built $name" || echo "  SKIP $name (nvcc not available)"
done
cd "$PROJECT_ROOT"
cargo build --release
echo ""

# Config: direct | loopback | crosspc
CONFIG="${1:-all}"
SERVER_IP="${2:-127.0.0.1}"

run_bench() {
    local bench_name="$1"
    local config="$2"
    local bench_binary="$BENCH_DIR/src/$bench_name"
    local output="$RESULT_DIR/${bench_name}_${config}_${COMMIT}.json"

    if [ ! -x "$bench_binary" ]; then
        echo "  SKIP $bench_name (not built)"
        return
    fi

    echo "  Running $bench_name ($config)..."
    case "$config" in
        direct)
            "$bench_binary" > "$output"
            ;;
        loopback|crosspc)
            local ip="$SERVER_IP"
            [ "$config" = "loopback" ] && ip="127.0.0.1"
            OUTTERLINK_SERVER="$ip:$PORT" \
            LD_PRELOAD="$PROJECT_ROOT/target/release/liboutterlink_client.so" \
            "$bench_binary" > "$output"
            ;;
    esac
    echo "    -> $output"
}

run_all_benches() {
    local config="$1"
    echo "--- Configuration: $config ---"
    run_bench "memcpy_bench" "$config"
    run_bench "alloc_bench" "$config"
    run_bench "launch_bench" "$config"
    run_bench "device_query_bench" "$config"
    run_bench "matmul_bench" "$config"
    echo ""
}

case "$CONFIG" in
    direct)
        run_all_benches "direct"
        ;;
    loopback)
        run_all_benches "loopback"
        ;;
    crosspc)
        run_all_benches "crosspc"
        ;;
    all)
        run_all_benches "direct"
        echo "--- Start server for loopback test ---"
        echo "    (Ensure outterlink-server is running on port $PORT)"
        run_all_benches "loopback"
        ;;
esac

# Update latest symlink
ln -sfn "$DATE" "$BENCH_DIR/results/latest"

echo "=== Benchmarks Complete ==="
echo "Results in: $RESULT_DIR"

# Run Rust criterion benchmarks
echo ""
echo "--- Rust Protocol Benchmarks ---"
cargo bench --bench protocol_bench 2>/dev/null || echo "  SKIP (criterion not configured yet)"
```

### 5.5 Regression Detection

Track benchmark results over time. Flag any regression greater than 10% from the previous run.

```python
#!/usr/bin/env python3
# benchmarks/scripts/compare.py
# Compare two benchmark result sets

import json
import sys
import os

def load_results(path):
    with open(path) as f:
        return json.load(f)

def compare(baseline_path, current_path, threshold_pct=10):
    baseline = load_results(baseline_path)
    current = load_results(current_path)

    print(f"Baseline: {baseline_path}")
    print(f"Current:  {current_path}")
    print(f"Threshold: {threshold_pct}%")
    print()

    regressions = []
    improvements = []

    for b_result in baseline.get("results", []):
        size = b_result.get("size_bytes") or b_result.get("size", 0)
        c_result = next(
            (r for r in current.get("results", [])
             if (r.get("size_bytes") or r.get("size", 0)) == size),
            None
        )
        if not c_result:
            continue

        for metric in ["htod_avg_us", "dtoh_avg_us", "htod_gbps", "dtoh_gbps",
                       "alloc_avg_us", "free_avg_us",
                       "noop_launch_sync_avg_us"]:
            b_val = b_result.get(metric)
            c_val = c_result.get(metric)
            if b_val is None or c_val is None:
                continue

            # For latency metrics (us), higher is worse
            # For throughput metrics (gbps), lower is worse
            is_latency = metric.endswith("_us")
            if is_latency:
                change_pct = ((c_val - b_val) / b_val) * 100
                is_regression = change_pct > threshold_pct
            else:
                change_pct = ((b_val - c_val) / b_val) * 100
                is_regression = change_pct > threshold_pct

            entry = {
                "size": size,
                "metric": metric,
                "baseline": b_val,
                "current": c_val,
                "change_pct": change_pct,
            }

            if is_regression:
                regressions.append(entry)
            elif abs(change_pct) > threshold_pct:
                improvements.append(entry)

    if regressions:
        print("REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  [{r['metric']}] size={r['size']}: "
                  f"{r['baseline']:.2f} -> {r['current']:.2f} "
                  f"({r['change_pct']:+.1f}%)")
        print()

    if improvements:
        print("IMPROVEMENTS:")
        for r in improvements:
            print(f"  [{r['metric']}] size={r['size']}: "
                  f"{r['baseline']:.2f} -> {r['current']:.2f} "
                  f"({r['change_pct']:+.1f}%)")
        print()

    if not regressions:
        print("No regressions detected.")
        return 0
    return 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <baseline.json> <current.json> [threshold_pct]")
        sys.exit(1)

    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    sys.exit(compare(sys.argv[1], sys.argv[2], threshold))
```

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Benchmark results vary between runs (noise) | MEDIUM | Run multiple iterations, report median/p50/p99, use `cpupower` to lock frequency |
| TCP performance varies with system load | LOW | Run benchmarks on idle system, disable unnecessary services |
| CUDA driver version affects baseline numbers | LOW | Record driver version in metadata; compare like-for-like |
| Benchmark binaries require CUDA toolkit to compile | MEDIUM | Pre-compile binaries; store in repo or CI artifacts |
| criterion HTML reports large | LOW | Only generate on demand; gitignore `target/criterion` |

## Related Documents

- [P5: PoC Plan](P5-poc-plan.md)
- [P11: Testing Strategy](P11-testing-strategy.md)
- [R4: ConnectX-5 + Transport Stack](../research/R4-connectx5-transport-stack.md)
- [Project Vision](../../docs/architecture/00-project-vision.md)

## Open Questions

- [ ] Should benchmarks run as part of CI or only manually? (CI is expensive for GPU benchmarks)
- [ ] What is the acceptable latency overhead target for PoC? (e.g., "kernel launch < 500us over loopback")
- [ ] Should we benchmark with CUDA pinned memory (`cuMemAllocHost`) in Phase 1 or defer to Phase 4?
- [ ] Do we need GPU power/temperature monitoring during benchmarks to detect thermal throttling?
- [ ] Should criterion benchmarks cover the C interception layer overhead or just the Rust protocol code?
