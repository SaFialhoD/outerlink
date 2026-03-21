/*
 * OuterLink CUDA Driver API Test Application -- Vector Addition
 *
 * A minimal C program that exercises the CUDA Driver API functions
 * intercepted by OuterLink's LD_PRELOAD layer. Uses ONLY the Driver API
 * (cuXxx functions), never the Runtime API (cudaXxx functions).
 *
 * Purpose: Verify that OuterLink correctly intercepts and forwards
 * every Driver API call needed for a real compute workload.
 *
 * API calls exercised (in order):
 *   cuInit, cuDeviceGetCount, cuDeviceGet, cuDeviceGetName,
 *   cuCtxCreate_v2, cuMemAlloc_v2 (x3), cuMemcpyHtoD_v2 (x2),
 *   cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel,
 *   cuCtxSynchronize, cuMemcpyDtoH_v2, cuMemFree_v2 (x3),
 *   cuModuleUnload, cuCtxDestroy_v2
 *
 * Build:  make native       (links against real CUDA driver)
 * Test:   make interpose    (runs under LD_PRELOAD with OuterLink)
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Error handling
 * ----------------------------------------------------------------------- */

#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        CUresult err = (call);                                                 \
        if (err != CUDA_SUCCESS) {                                             \
            const char *err_name = "UNKNOWN";                                  \
            const char *err_str  = "Unknown error";                            \
            cuGetErrorName(err, &err_name);                                    \
            cuGetErrorString(err, &err_str);                                   \
            fprintf(stderr, "CUDA ERROR at %s:%d\n"                            \
                            "  Call:  %s\n"                                    \
                            "  Code:  %d (%s)\n"                               \
                            "  Desc:  %s\n",                                   \
                    __FILE__, __LINE__, #call,                                  \
                    (int)err, err_name, err_str);                              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* -----------------------------------------------------------------------
 * Embedded PTX kernel: vector_add
 *
 * C[i] = A[i] + B[i]  for float32 vectors
 *
 * PTX ISA version 7.0, target sm_80 (Ampere).
 * Compatible with sm_80+ (RTX 3090, 3090 Ti, A100, etc.)
 *
 * The kernel takes three 64-bit device pointers (A, B, C) and one
 * 32-bit integer (N = element count). Uses 1D grid indexing:
 *   tid = blockIdx.x * blockDim.x + threadIdx.x
 *   if (tid < N) C[tid] = A[tid] + B[tid]
 * ----------------------------------------------------------------------- */

static const char ptx_source[] =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry vector_add(\n"
    "    .param .u64 param_A,\n"
    "    .param .u64 param_B,\n"
    "    .param .u64 param_C,\n"
    "    .param .u32 param_N\n"
    ")\n"
    "{\n"
    "    .reg .pred %p<2>;\n"
    "    .reg .f32  %f<4>;\n"
    "    .reg .b32  %r<6>;\n"
    "    .reg .b64  %rd<8>;\n"
    "\n"
    "    // tid = blockIdx.x * blockDim.x + threadIdx.x\n"
    "    mov.u32         %r0, %ctaid.x;\n"
    "    mov.u32         %r1, %ntid.x;\n"
    "    mov.u32         %r2, %tid.x;\n"
    "    mad.lo.s32      %r3, %r0, %r1, %r2;\n"
    "\n"
    "    // Bounds check: if (tid >= N) return\n"
    "    ld.param.u32    %r4, [param_N];\n"
    "    setp.ge.u32     %p0, %r3, %r4;\n"
    "    @%p0 bra        done;\n"
    "\n"
    "    // Convert tid to byte offset (float32 = 4 bytes)\n"
    "    mul.wide.u32    %rd0, %r3, 4;\n"
    "\n"
    "    // Load A[tid]\n"
    "    ld.param.u64    %rd1, [param_A];\n"
    "    add.u64         %rd2, %rd1, %rd0;\n"
    "    ld.global.f32   %f0, [%rd2];\n"
    "\n"
    "    // Load B[tid]\n"
    "    ld.param.u64    %rd3, [param_B];\n"
    "    add.u64         %rd4, %rd3, %rd0;\n"
    "    ld.global.f32   %f1, [%rd4];\n"
    "\n"
    "    // C[tid] = A[tid] + B[tid]\n"
    "    add.f32         %f2, %f0, %f1;\n"
    "    ld.param.u64    %rd5, [param_C];\n"
    "    add.u64         %rd6, %rd5, %rd0;\n"
    "    st.global.f32   [%rd6], %f2;\n"
    "\n"
    "done:\n"
    "    ret;\n"
    "}\n";

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */

#define N 1024          /* Number of elements in each vector */
#define BLOCK_SIZE 256  /* Threads per block */

int main(void) {
    CUdevice   device;
    CUcontext  ctx;
    CUmodule   module;
    CUfunction kernel;

    printf("=== OuterLink CUDA Driver API Test: Vector Add ===\n\n");

    /* ---- Initialize CUDA ---- */
    printf("[1/12] cuInit\n");
    CHECK_CUDA(cuInit(0));

    /* ---- Enumerate devices ---- */
    int device_count = 0;
    printf("[2/12] cuDeviceGetCount\n");
    CHECK_CUDA(cuDeviceGetCount(&device_count));
    printf("        Found %d device(s)\n", device_count);

    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA devices found.\n");
        return 1;
    }

    /* ---- Get device 0 ---- */
    printf("[3/12] cuDeviceGet (ordinal=0)\n");
    CHECK_CUDA(cuDeviceGet(&device, 0));

    char dev_name[256] = {0};
    cuDeviceGetName(dev_name, sizeof(dev_name), device);
    printf("        Device: %s\n", dev_name);

    /* ---- Create context ---- */
    printf("[4/12] cuCtxCreate_v2\n");
    CHECK_CUDA(cuCtxCreate_v2(&ctx, 0, device));

    /* ---- Allocate device memory ---- */
    size_t bytes = N * sizeof(float);
    CUdeviceptr d_A, d_B, d_C;

    printf("[5/12] cuMemAlloc_v2 (3 buffers, %zu bytes each)\n", bytes);
    CHECK_CUDA(cuMemAlloc_v2(&d_A, bytes));
    CHECK_CUDA(cuMemAlloc_v2(&d_B, bytes));
    CHECK_CUDA(cuMemAlloc_v2(&d_C, bytes));

    /* ---- Prepare host data ---- */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "ERROR: Host malloc failed.\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }
    memset(h_C, 0, bytes);

    /* ---- Copy host -> device ---- */
    printf("[6/12] cuMemcpyHtoD_v2 (A and B)\n");
    CHECK_CUDA(cuMemcpyHtoD_v2(d_A, h_A, bytes));
    CHECK_CUDA(cuMemcpyHtoD_v2(d_B, h_B, bytes));

    /* ---- Load PTX module ---- */
    printf("[7/12] cuModuleLoadData (embedded PTX, %zu bytes)\n",
           sizeof(ptx_source));
    CHECK_CUDA(cuModuleLoadData(&module, ptx_source));

    /* ---- Get kernel function ---- */
    printf("[8/12] cuModuleGetFunction (\"vector_add\")\n");
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "vector_add"));

    /* ---- Launch kernel ---- */
    unsigned int grid_dim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int n_val = N;

    /*
     * CUDA Driver API kernel parameter passing:
     * kernelParams is an array of pointers, where each pointer points
     * to the storage of the corresponding kernel argument.
     */
    void *params[] = {
        &d_A,       /* param_A: .param .u64 */
        &d_B,       /* param_B: .param .u64 */
        &d_C,       /* param_C: .param .u64 */
        &n_val      /* param_N: .param .u32 */
    };

    printf("[9/12] cuLaunchKernel (grid=%u, block=%d, params=4)\n",
           grid_dim, BLOCK_SIZE);
    CHECK_CUDA(cuLaunchKernel(kernel,
                               grid_dim, 1, 1,     /* grid  (x, y, z) */
                               BLOCK_SIZE, 1, 1,    /* block (x, y, z) */
                               0,                    /* shared mem bytes */
                               NULL,                 /* stream (default) */
                               params,               /* kernel params */
                               NULL));               /* extra (unused) */

    /* ---- Synchronize ---- */
    printf("[10/12] cuCtxSynchronize\n");
    CHECK_CUDA(cuCtxSynchronize());

    /* ---- Copy results device -> host ---- */
    printf("[11/12] cuMemcpyDtoH_v2 (C)\n");
    CHECK_CUDA(cuMemcpyDtoH_v2(h_C, d_C, bytes));

    /* ---- Verify results ---- */
    printf("\n--- Verification ---\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i * 2);
        if (h_C[i] != expected) {
            if (errors < 10) {
                fprintf(stderr, "  MISMATCH at [%d]: got %.1f, expected %.1f\n",
                        i, h_C[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  PASS: All %d elements correct.\n", N);
        printf("  Sample: C[0]=%.1f, C[1]=%.1f, C[%d]=%.1f\n",
               h_C[0], h_C[1], N - 1, h_C[N - 1]);
    } else {
        printf("  FAIL: %d / %d elements incorrect.\n", errors, N);
    }

    /* ---- Cleanup ---- */
    printf("\n[12/12] Cleanup\n");
    CHECK_CUDA(cuMemFree_v2(d_A));
    CHECK_CUDA(cuMemFree_v2(d_B));
    CHECK_CUDA(cuMemFree_v2(d_C));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy_v2(ctx));

    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n=== Test complete (exit code: %d) ===\n", errors ? 1 : 0);
    return errors ? 1 : 0;
}
