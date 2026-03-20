/*
 * OuterLink CUDA Driver API Interposition Library
 *
 * Loaded via LD_PRELOAD to intercept CUDA Driver API calls. This thin C layer
 * hooks two entry points that applications use to discover CUDA functions:
 *
 *   1. dlsym()            -- classic dynamic symbol resolution
 *   2. cuGetProcAddress() -- CUDA 11.3+ driver entry point API
 *
 * When either resolves a CUDA function we intercept, we return a pointer to
 * our hook function instead. Each hook simply forwards the call into the Rust
 * client library via FFI (the ol_* functions declared in interpose.h).
 *
 * Architecture:
 *   Application -> dlsym("cuMemAlloc_v2")
 *               -> returns hook_cuMemAlloc_v2 (this file)
 *               -> calls ol_cuMemAlloc_v2 (Rust FFI)
 *               -> serializes + sends to remote OuterLink server
 *
 * Linux/glibc only. Uses __libc_dlsym to get the real dlsym without
 * infinite recursion.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "interpose.h"

/* -----------------------------------------------------------------------
 * Real dlsym access
 *
 * We override dlsym(), so we need the glibc-internal __libc_dlsym to
 * call the real implementation without recursing into ourselves.
 * ----------------------------------------------------------------------- */

extern void *__libc_dlsym(void *handle, const char *name);

/* Cache the real dlsym handle for RTLD_NEXT lookups */
static void *(*real_dlsym)(void *handle, const char *name) = NULL;

/* -----------------------------------------------------------------------
 * One-time initialization
 *
 * Thread-safe via pthread_once. Called lazily on the first intercepted
 * CUDA call. Initializes the Rust client (which reads OUTERLINK_SERVER
 * from the environment and prepares the transport layer).
 * ----------------------------------------------------------------------- */

static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static int initialized = 0;

static void do_init(void) {
    /* Resolve the real dlsym if we haven't already */
    if (!real_dlsym) {
        real_dlsym = (void *(*)(void *, const char *))
            __libc_dlsym(RTLD_NEXT, "dlsym");
    }

    /* Initialize the Rust client (connects to server, sets up handle tables) */
    ol_client_init();
    initialized = 1;
}

static void ensure_init(void) {
    pthread_once(&init_once, do_init);
}

/* -----------------------------------------------------------------------
 * Hook table
 *
 * Maps CUDA function names to our hook function pointers. When dlsym or
 * cuGetProcAddress looks up one of these names, we return the hook instead
 * of the real CUDA function.
 * ----------------------------------------------------------------------- */

typedef struct {
    const char *name;
    void *hook_fn;
} hook_entry_t;

static const hook_entry_t hook_table[] = {
    /* Init */
    { "cuInit",                  (void *)hook_cuInit },
    { "cuDriverGetVersion",      (void *)hook_cuDriverGetVersion },

    /* Device */
    { "cuDeviceGet",             (void *)hook_cuDeviceGet },
    { "cuDeviceGetCount",        (void *)hook_cuDeviceGetCount },
    { "cuDeviceGetName",         (void *)hook_cuDeviceGetName },
    { "cuDeviceGetAttribute",    (void *)hook_cuDeviceGetAttribute },
    { "cuDeviceTotalMem_v2",     (void *)hook_cuDeviceTotalMem_v2 },
    { "cuDeviceGetUuid",         (void *)hook_cuDeviceGetUuid },

    /* Context */
    { "cuCtxCreate_v2",          (void *)hook_cuCtxCreate_v2 },
    { "cuCtxDestroy_v2",         (void *)hook_cuCtxDestroy_v2 },
    { "cuCtxSetCurrent",         (void *)hook_cuCtxSetCurrent },
    { "cuCtxGetCurrent",         (void *)hook_cuCtxGetCurrent },
    { "cuCtxGetDevice",          (void *)hook_cuCtxGetDevice },
    { "cuCtxSynchronize",        (void *)hook_cuCtxSynchronize },

    /* Memory */
    { "cuMemAlloc_v2",           (void *)hook_cuMemAlloc_v2 },
    { "cuMemFree_v2",            (void *)hook_cuMemFree_v2 },
    { "cuMemcpyHtoD_v2",        (void *)hook_cuMemcpyHtoD_v2 },
    { "cuMemcpyDtoH_v2",        (void *)hook_cuMemcpyDtoH_v2 },
    { "cuMemcpyDtoD",            (void *)hook_cuMemcpyDtoD },
    { "cuMemcpyDtoD_v2",        (void *)hook_cuMemcpyDtoD },
    { "cuMemAllocHost",          (void *)hook_cuMemAllocHost },
    { "cuMemAllocHost_v2",      (void *)hook_cuMemAllocHost },
    { "cuMemFreeHost",           (void *)hook_cuMemFreeHost },
    { "cuMemcpyHtoDAsync",       (void *)hook_cuMemcpyHtoDAsync_v2 },
    { "cuMemcpyHtoDAsync_v2",   (void *)hook_cuMemcpyHtoDAsync_v2 },
    { "cuMemcpyDtoHAsync",      (void *)hook_cuMemcpyDtoHAsync_v2 },
    { "cuMemcpyDtoHAsync_v2",   (void *)hook_cuMemcpyDtoHAsync_v2 },
    { "cuMemsetD8",              (void *)hook_cuMemsetD8 },
    { "cuMemsetD8_v2",          (void *)hook_cuMemsetD8 },
    { "cuMemsetD32",             (void *)hook_cuMemsetD32 },
    { "cuMemsetD32_v2",         (void *)hook_cuMemsetD32 },
    { "cuMemsetD8Async",         (void *)hook_cuMemsetD8Async },
    { "cuMemsetD32Async",        (void *)hook_cuMemsetD32Async },
    { "cuMemGetInfo_v2",         (void *)hook_cuMemGetInfo_v2 },

    /* Error */
    { "cuGetErrorName",          (void *)hook_cuGetErrorName },
    { "cuGetErrorString",        (void *)hook_cuGetErrorString },

    /* Module */
    { "cuModuleLoadData",        (void *)hook_cuModuleLoadData },
    { "cuModuleUnload",          (void *)hook_cuModuleUnload },
    { "cuModuleGetFunction",     (void *)hook_cuModuleGetFunction },
    { "cuModuleGetGlobal_v2",    (void *)hook_cuModuleGetGlobal },

    /* Stream */
    { "cuStreamCreate",          (void *)hook_cuStreamCreate },
    { "cuStreamDestroy",         (void *)hook_cuStreamDestroy },
    { "cuStreamDestroy_v2",      (void *)hook_cuStreamDestroy },
    { "cuStreamSynchronize",     (void *)hook_cuStreamSynchronize },
    { "cuStreamQuery",           (void *)hook_cuStreamQuery },
    { "cuStreamWaitEvent",       (void *)hook_cuStreamWaitEvent },

    /* Event */
    { "cuEventCreate",           (void *)hook_cuEventCreate },
    { "cuEventDestroy",          (void *)hook_cuEventDestroy },
    { "cuEventDestroy_v2",       (void *)hook_cuEventDestroy },
    { "cuEventRecord",           (void *)hook_cuEventRecord },
    { "cuEventSynchronize",      (void *)hook_cuEventSynchronize },
    { "cuEventElapsedTime",      (void *)hook_cuEventElapsedTime },
    { "cuEventQuery",            (void *)hook_cuEventQuery },

    /* Kernel launch */
    { "cuLaunchKernel",          (void *)hook_cuLaunchKernel },

    /* cuGetProcAddress itself -- we hook the hooking mechanism */
    { "cuGetProcAddress",        (void *)hook_cuGetProcAddress },
    { "cuGetProcAddress_v2",     (void *)hook_cuGetProcAddress_v2 },

    /* Sentinel */
    { NULL, NULL }
};

#define HOOK_TABLE_SIZE (sizeof(hook_table) / sizeof(hook_table[0]) - 1)

/*
 * Look up a function name in the hook table.
 * Returns the hook function pointer, or NULL if not intercepted.
 */
static void *find_hook(const char *name) {
    if (!name) return NULL;
    for (size_t i = 0; i < HOOK_TABLE_SIZE; i++) {
        if (strcmp(hook_table[i].name, name) == 0) {
            return hook_table[i].hook_fn;
        }
    }
    return NULL;
}

/* -----------------------------------------------------------------------
 * dlsym override
 *
 * This is the core of LD_PRELOAD interposition. When any code in the
 * process calls dlsym() to look up a symbol, our version runs first.
 * If the symbol is a CUDA function we intercept, we return our hook.
 * Otherwise we forward to the real dlsym.
 *
 * The __attribute__((visibility("default"))) ensures this symbol is
 * exported from the shared library even if -fvisibility=hidden is used.
 * ----------------------------------------------------------------------- */

__attribute__((visibility("default")))
void *dlsym(void *handle, const char *name) {
    /* Bootstrap: we need real_dlsym to forward non-CUDA lookups */
    if (!real_dlsym) {
        real_dlsym = (void *(*)(void *, const char *))
            __libc_dlsym(RTLD_NEXT, "dlsym");
    }

    /* Check if this is a CUDA symbol we intercept */
    void *hook = find_hook(name);
    if (hook) {
        ensure_init();
        return hook;
    }

    /* Not one of ours -- forward to real dlsym */
    return real_dlsym(handle, name);
}

/* -----------------------------------------------------------------------
 * Hook function implementations
 *
 * Each hook has the exact CUDA Driver API signature. It simply forwards
 * the call to the corresponding Rust FFI function (ol_* prefix).
 * ----------------------------------------------------------------------- */

/* -- Init -- */

CUresult hook_cuInit(unsigned int Flags) {
    ensure_init();
    return ol_cuInit(Flags);
}

CUresult hook_cuDriverGetVersion(int *driverVersion) {
    ensure_init();
    return ol_cuDriverGetVersion(driverVersion);
}

/* -- Device -- */

CUresult hook_cuDeviceGet(CUdevice *device, int ordinal) {
    ensure_init();
    return ol_cuDeviceGet(device, ordinal);
}

CUresult hook_cuDeviceGetCount(int *count) {
    ensure_init();
    return ol_cuDeviceGetCount(count);
}

CUresult hook_cuDeviceGetName(char *name, int len, CUdevice dev) {
    ensure_init();
    return ol_cuDeviceGetName(name, len, dev);
}

CUresult hook_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    ensure_init();
    return ol_cuDeviceGetAttribute(pi, (int)attrib, dev);
}

CUresult hook_cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    ensure_init();
    return ol_cuDeviceTotalMem_v2(bytes, dev);
}

CUresult hook_cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    ensure_init();
    return ol_cuDeviceGetUuid((unsigned char *)uuid->bytes, dev);
}

/* -- Context -- */

/*
 * Context handles: CUDA uses opaque pointers (CUcontext = struct CUctx_st *).
 * Rust FFI uses u64 (unsigned long long). On 64-bit Linux both are 8 bytes.
 * We cast between them at this boundary.
 */

CUresult hook_cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    ensure_init();
    unsigned long long ctx_u64 = 0;
    CUresult r = ol_cuCtxCreate_v2(&ctx_u64, flags, dev);
    if (r == CUDA_SUCCESS && pctx) {
        *pctx = (CUcontext)(uintptr_t)ctx_u64;
    }
    return r;
}

CUresult hook_cuCtxDestroy_v2(CUcontext ctx) {
    ensure_init();
    return ol_cuCtxDestroy_v2((unsigned long long)(uintptr_t)ctx);
}

CUresult hook_cuCtxSetCurrent(CUcontext ctx) {
    ensure_init();
    return ol_cuCtxSetCurrent((unsigned long long)(uintptr_t)ctx);
}

CUresult hook_cuCtxGetCurrent(CUcontext *pctx) {
    ensure_init();
    unsigned long long ctx_u64 = 0;
    CUresult r = ol_cuCtxGetCurrent(&ctx_u64);
    if (r == CUDA_SUCCESS && pctx) {
        *pctx = (CUcontext)(uintptr_t)ctx_u64;
    }
    return r;
}

CUresult hook_cuCtxGetDevice(CUdevice *dev) {
    ensure_init();
    return ol_cuCtxGetDevice(dev);
}

CUresult hook_cuCtxSynchronize(void) {
    ensure_init();
    return ol_cuCtxSynchronize();
}

/* -- Memory -- */

CUresult hook_cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    ensure_init();
    return ol_cuMemAlloc_v2(dptr, bytesize);
}

CUresult hook_cuMemFree_v2(CUdeviceptr dptr) {
    ensure_init();
    return ol_cuMemFree_v2(dptr);
}

CUresult hook_cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    ensure_init();
    return ol_cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

CUresult hook_cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    ensure_init();
    return ol_cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

CUresult hook_cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    ensure_init();
    return ol_cuMemcpyDtoD((unsigned long long)dst, (unsigned long long)src, ByteCount);
}

CUresult hook_cuMemAllocHost(void **pp, size_t bytesize) {
    ensure_init();
    return ol_cuMemAllocHost(pp, bytesize);
}

CUresult hook_cuMemFreeHost(void *p) {
    ensure_init();
    return ol_cuMemFreeHost(p);
}

CUresult hook_cuMemGetInfo_v2(size_t *free, size_t *total) {
    ensure_init();
    return ol_cuMemGetInfo_v2(free, total);
}

CUresult hook_cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    ensure_init();
    return ol_cuMemcpyHtoDAsync_v2((unsigned long long)dstDevice, srcHost, ByteCount,
                                     (unsigned long long)(uintptr_t)hStream);
}

CUresult hook_cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    ensure_init();
    return ol_cuMemcpyDtoHAsync_v2(dstHost, (unsigned long long)srcDevice, ByteCount,
                                     (unsigned long long)(uintptr_t)hStream);
}

CUresult hook_cuMemsetD8(CUdeviceptr dstDevice, unsigned char value, size_t count) {
    ensure_init();
    return ol_cuMemsetD8((unsigned long long)dstDevice, value, count);
}

CUresult hook_cuMemsetD32(CUdeviceptr dstDevice, unsigned int value, size_t count) {
    ensure_init();
    return ol_cuMemsetD32((unsigned long long)dstDevice, value, count);
}

CUresult hook_cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char value, size_t count, CUstream hStream) {
    ensure_init();
    return ol_cuMemsetD8Async((unsigned long long)dstDevice, value, count,
                               (unsigned long long)(uintptr_t)hStream);
}

CUresult hook_cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int value, size_t count, CUstream hStream) {
    ensure_init();
    return ol_cuMemsetD32Async((unsigned long long)dstDevice, value, count,
                                (unsigned long long)(uintptr_t)hStream);
}

/* -- Error -- */

CUresult hook_cuGetErrorName(CUresult error, const char **pStr) {
    ensure_init();
    return ol_cuGetErrorName((unsigned int)error, pStr);
}

CUresult hook_cuGetErrorString(CUresult error, const char **pStr) {
    ensure_init();
    return ol_cuGetErrorString((unsigned int)error, pStr);
}

/* -- Module -- */

CUresult hook_cuModuleLoadData(CUmodule *module, const void *image) {
    ensure_init();
    unsigned long long mod_u64 = 0;
    /* Note: we pass image pointer and a size of 0 (size is not available from
     * the CUDA API signature -- the server will need to parse the binary to
     * determine its size, or we calculate it from the PTX/cubin header). */
    CUresult r = ol_cuModuleLoadData(&mod_u64, image, 0);
    if (r == CUDA_SUCCESS && module) {
        *module = (CUmodule)(uintptr_t)mod_u64;
    }
    return r;
}

CUresult hook_cuModuleUnload(CUmodule hmod) {
    ensure_init();
    return ol_cuModuleUnload((unsigned long long)(uintptr_t)hmod);
}

CUresult hook_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    ensure_init();
    unsigned long long func_u64 = 0;
    CUresult r = ol_cuModuleGetFunction(&func_u64,
                                         (unsigned long long)(uintptr_t)hmod,
                                         name);
    if (r == CUDA_SUCCESS && hfunc) {
        *hfunc = (CUfunction)(uintptr_t)func_u64;
    }
    return r;
}

CUresult hook_cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    ensure_init();
    unsigned long long dptr_u64 = 0;
    size_t size_out = 0;
    size_t name_len = name ? strlen(name) : 0;
    CUresult r = ol_cuModuleGetGlobal(&dptr_u64, &size_out,
                                       (unsigned long long)(uintptr_t)hmod,
                                       (const unsigned char *)name, name_len);
    if (r == CUDA_SUCCESS) {
        if (dptr) *dptr = (CUdeviceptr)dptr_u64;
        if (bytes) *bytes = size_out;
    }
    return r;
}

/* -- Stream -- */

CUresult hook_cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    ensure_init();
    unsigned long long stream_u64 = 0;
    CUresult r = ol_cuStreamCreate(&stream_u64, Flags);
    if (r == CUDA_SUCCESS && phStream) {
        *phStream = (CUstream)(uintptr_t)stream_u64;
    }
    return r;
}

CUresult hook_cuStreamDestroy(CUstream hStream) {
    ensure_init();
    return ol_cuStreamDestroy((unsigned long long)(uintptr_t)hStream);
}

CUresult hook_cuStreamSynchronize(CUstream hStream) {
    ensure_init();
    return ol_cuStreamSynchronize((unsigned long long)(uintptr_t)hStream);
}

CUresult hook_cuStreamQuery(CUstream hStream) {
    ensure_init();
    return ol_cuStreamQuery((unsigned long long)(uintptr_t)hStream);
}

CUresult hook_cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    ensure_init();
    return ol_cuStreamWaitEvent((unsigned long long)(uintptr_t)hStream,
                                (unsigned long long)(uintptr_t)hEvent,
                                Flags);
}

/* -- Event -- */

CUresult hook_cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    ensure_init();
    unsigned long long event_u64 = 0;
    CUresult r = ol_cuEventCreate(&event_u64, Flags);
    if (r == CUDA_SUCCESS && phEvent) {
        *phEvent = (CUevent)(uintptr_t)event_u64;
    }
    return r;
}

CUresult hook_cuEventDestroy(CUevent hEvent) {
    ensure_init();
    return ol_cuEventDestroy((unsigned long long)(uintptr_t)hEvent);
}

CUresult hook_cuEventRecord(CUevent hEvent, CUstream hStream) {
    ensure_init();
    return ol_cuEventRecord((unsigned long long)(uintptr_t)hEvent,
                            (unsigned long long)(uintptr_t)hStream);
}

CUresult hook_cuEventSynchronize(CUevent hEvent) {
    ensure_init();
    return ol_cuEventSynchronize((unsigned long long)(uintptr_t)hEvent);
}

CUresult hook_cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    ensure_init();
    return ol_cuEventElapsedTime(pMilliseconds,
                                  (unsigned long long)(uintptr_t)hStart,
                                  (unsigned long long)(uintptr_t)hEnd);
}

CUresult hook_cuEventQuery(CUevent hEvent) {
    ensure_init();
    return ol_cuEventQuery((unsigned long long)(uintptr_t)hEvent);
}

/* -- Kernel launch -- */

/*
 * The real CUDA cuLaunchKernel uses void** kernelParams where each pointer
 * points to a kernel argument's storage. However, the sizes of those arguments
 * are NOT part of the API -- they are determined by the kernel's signature.
 *
 * At LD_PRELOAD interception time, we don't have access to the kernel's
 * parameter metadata. So we pass NULL/0 for params here. Applications that
 * need kernel params over the network must use the extended OuterLink API
 * (with explicit num_params and param_sizes).
 *
 * Phase 2: Introspect cubin/PTX module metadata to infer param sizes.
 */
CUresult hook_cuLaunchKernel(CUfunction f,
                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, CUstream hStream,
                              void **kernelParams, void **extra) {
    ensure_init();
    (void)kernelParams;  /* Cannot serialize without param sizes -- see comment above */
    (void)extra;         /* 'extra' parameter style not yet supported */
    return ol_cuLaunchKernel((unsigned long long)(uintptr_t)f,
                              gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ,
                              sharedMemBytes,
                              (unsigned long long)(uintptr_t)hStream,
                              NULL,  /* kernelParams -- requires extended API */
                              0,     /* numParams */
                              NULL); /* paramSizes */
}

/* -----------------------------------------------------------------------
 * cuGetProcAddress hooks
 *
 * CUDA 11.3+ introduced cuGetProcAddress as a way for the runtime to
 * resolve driver functions dynamically (bypassing dlsym). We must hook
 * this too, otherwise applications using the CUDA runtime would get
 * real function pointers and bypass our interception.
 *
 * cuGetProcAddress (3 args):
 *   CUresult cuGetProcAddress(const char *symbol, void **pfn,
 *                             int cudaVersion, cuuint64_t flags)
 *
 * cuGetProcAddress_v2 (5 args):
 *   CUresult cuGetProcAddress_v2(const char *symbol, void **pfn,
 *                                int cudaVersion, cuuint64_t flags,
 *                                void *symbolStatus)
 *
 * Strategy: call the real cuGetProcAddress first (to validate the symbol
 * and version), then replace the returned function pointer with our hook
 * if we intercept that function.
 * ----------------------------------------------------------------------- */

/* Typedef for real cuGetProcAddress functions */
typedef CUresult (*cuGetProcAddress_fn)(const char *, void **, int, cuuint64_t);
typedef CUresult (*cuGetProcAddress_v2_fn)(const char *, void **, int, cuuint64_t, void *);

/* Cached pointers to real cuGetProcAddress implementations */
static cuGetProcAddress_fn    real_cuGetProcAddress    = NULL;
static cuGetProcAddress_v2_fn real_cuGetProcAddress_v2 = NULL;

/*
 * Resolve real cuGetProcAddress from the actual CUDA driver.
 * Uses dlsym(RTLD_NEXT, ...) to skip past our interposed versions.
 */
static void resolve_real_cuGetProcAddress(void) {
    if (!real_dlsym) {
        real_dlsym = (void *(*)(void *, const char *))
            __libc_dlsym(RTLD_NEXT, "dlsym");
    }
    if (!real_cuGetProcAddress) {
        real_cuGetProcAddress = (cuGetProcAddress_fn)
            real_dlsym(RTLD_NEXT, "cuGetProcAddress");
    }
    if (!real_cuGetProcAddress_v2) {
        real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_fn)
            real_dlsym(RTLD_NEXT, "cuGetProcAddress_v2");
    }
}

CUresult hook_cuGetProcAddress(const char *symbol, void **pfn,
                               int cudaVersion, cuuint64_t flags) {
    ensure_init();

    if (!symbol || !pfn) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Check if we intercept this symbol */
    void *hook = find_hook(symbol);
    if (hook) {
        *pfn = hook;
        return CUDA_SUCCESS;
    }

    /* Not intercepted -- try to get the real function pointer.
     * If the real CUDA driver is not loaded (pure virtual GPU mode),
     * return NOT_FOUND for functions we don't handle. */
    resolve_real_cuGetProcAddress();
    if (real_cuGetProcAddress) {
        return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags);
    }

    /* No real CUDA driver available */
    *pfn = NULL;
    return CUDA_ERROR_NOT_FOUND;
}

CUresult hook_cuGetProcAddress_v2(const char *symbol, void **pfn,
                                  int cudaVersion, cuuint64_t flags,
                                  void *symbolStatus) {
    ensure_init();

    if (!symbol || !pfn) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Check if we intercept this symbol */
    void *hook = find_hook(symbol);
    if (hook) {
        *pfn = hook;
        if (symbolStatus) {
            /* TODO: Use CUdriverProcAddressQueryResult enum from real CUDA SDK
             * instead of hardcoded 0. CU_GET_PROC_ADDRESS_SUCCESS = 0. */
            *(int *)symbolStatus = 0; /* CU_GET_PROC_ADDRESS_SUCCESS */
        }
        return CUDA_SUCCESS;
    }

    /* Not intercepted -- forward to real driver */
    resolve_real_cuGetProcAddress();
    if (real_cuGetProcAddress_v2) {
        return real_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
    }

    /* No real CUDA driver available */
    *pfn = NULL;
    return CUDA_ERROR_NOT_FOUND;
}
