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

    /* Primary context */
    { "cuDevicePrimaryCtxRetain",       (void *)hook_cuDevicePrimaryCtxRetain },
    { "cuDevicePrimaryCtxRelease",      (void *)hook_cuDevicePrimaryCtxRelease_v2 },
    { "cuDevicePrimaryCtxRelease_v2",   (void *)hook_cuDevicePrimaryCtxRelease_v2 },
    { "cuDevicePrimaryCtxGetState",     (void *)hook_cuDevicePrimaryCtxGetState },
    { "cuDevicePrimaryCtxSetFlags",     (void *)hook_cuDevicePrimaryCtxSetFlags_v2 },
    { "cuDevicePrimaryCtxSetFlags_v2",  (void *)hook_cuDevicePrimaryCtxSetFlags_v2 },
    { "cuDevicePrimaryCtxReset",        (void *)hook_cuDevicePrimaryCtxReset_v2 },
    { "cuDevicePrimaryCtxReset_v2",     (void *)hook_cuDevicePrimaryCtxReset_v2 },

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
    { "cuModuleLoadDataEx",      (void *)hook_cuModuleLoadDataEx },
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

/* -- Primary context -- */

CUresult hook_cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    ensure_init();
    unsigned long long ctx_u64 = 0;
    CUresult r = ol_cuDevicePrimaryCtxRetain(&ctx_u64, dev);
    if (r == CUDA_SUCCESS && pctx)
        *pctx = (CUcontext)(uintptr_t)ctx_u64;
    return r;
}

CUresult hook_cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    ensure_init();
    return ol_cuDevicePrimaryCtxRelease(dev);
}

CUresult hook_cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    ensure_init();
    return ol_cuDevicePrimaryCtxGetState(dev, flags, active);
}

CUresult hook_cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    ensure_init();
    return ol_cuDevicePrimaryCtxSetFlags(dev, flags);
}

CUresult hook_cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    ensure_init();
    return ol_cuDevicePrimaryCtxReset(dev);
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
    /* Determine image size: PTX is a null-terminated string; cubin is ELF.
     * Detect format by inspecting first bytes. */
    size_t data_len = 0;
    if (image) {
        const unsigned char *p = (const unsigned char *)image;
        if (p[0] >= 0x20 && p[0] < 0x7F) {
            /* PTX text — null-terminated string */
            data_len = strlen((const char *)image) + 1;
        } else if (p[0] == 0x7F && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
            /* ELF cubin — compute a lower-bound estimate of the image size
             * from the section header table (e_shoff + e_shnum * e_shentsize).
             * This is NOT the exact file size but it covers all section headers,
             * which is sufficient for the server to load the module. */
            unsigned long long e_shoff = 0;
            unsigned short e_shnum = 0, e_shentsize = 0;
            /* ELF64 header is 64 bytes; reject anything smaller. */
            memcpy(&e_shoff, p + 0x28, 8);
            memcpy(&e_shentsize, p + 0x3A, 2);
            memcpy(&e_shnum, p + 0x3C, 2);
            /* Overflow check: ensure e_shoff + e_shnum * e_shentsize
             * doesn't wrap around. */
            unsigned long long sh_table_size = (unsigned long long)e_shnum * e_shentsize;
            if (e_shoff <= SIZE_MAX - sh_table_size) {
                data_len = (size_t)(e_shoff + sh_table_size);
            }
            /* If overflow detected, data_len stays 0 and we send
             * an empty payload — the server will reject the load. */
        }
    }
    CUresult r = ol_cuModuleLoadData(&mod_u64, image, data_len);
    if (r == CUDA_SUCCESS && module) {
        *module = (CUmodule)(uintptr_t)mod_u64;
    }
    return r;
}

CUresult hook_cuModuleLoadDataEx(CUmodule *module, const void *image,
                                  unsigned int numOptions, void *options,
                                  void **optionValues) {
    ensure_init();
    unsigned long long mod_u64 = 0;
    /* Determine image size — same PTX/ELF logic as hook_cuModuleLoadData. */
    size_t data_len = 0;
    if (image) {
        const unsigned char *p = (const unsigned char *)image;
        if (p[0] >= 0x20 && p[0] < 0x7F) {
            /* PTX text — null-terminated string */
            data_len = strlen((const char *)image) + 1;
        } else if (p[0] == 0x7F && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
            /* ELF cubin */
            unsigned long long e_shoff = 0;
            unsigned short e_shnum = 0, e_shentsize = 0;
            memcpy(&e_shoff, p + 0x28, 8);
            memcpy(&e_shentsize, p + 0x3A, 2);
            memcpy(&e_shnum, p + 0x3C, 2);
            unsigned long long sh_table_size = (unsigned long long)e_shnum * e_shentsize;
            if (e_shoff <= SIZE_MAX - sh_table_size) {
                data_len = (size_t)(e_shoff + sh_table_size);
            }
        }
    }

    /* Serialize CUjit_option enum values (int-sized) and their void* values
     * as (i32, u64) pairs for the Rust FFI. */
    int *opts_i32 = NULL;
    unsigned long long *vals_u64 = NULL;
    if (numOptions > 0 && options && optionValues) {
        opts_i32 = (int *)malloc(numOptions * sizeof(int));
        vals_u64 = (unsigned long long *)malloc(numOptions * sizeof(unsigned long long));
        if (opts_i32 && vals_u64) {
            const int *src_opts = (const int *)options;
            for (unsigned int i = 0; i < numOptions; i++) {
                opts_i32[i] = src_opts[i];
                vals_u64[i] = (unsigned long long)(uintptr_t)optionValues[i];
            }
        } else {
            /* Allocation failed — fall through with 0 options */
            free(opts_i32);
            free(vals_u64);
            opts_i32 = NULL;
            vals_u64 = NULL;
            numOptions = 0;
        }
    }

    CUresult r = ol_cuModuleLoadDataEx(&mod_u64, image, data_len,
                                        numOptions, opts_i32, vals_u64);
    free(opts_i32);
    free(vals_u64);

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
 * Kernel parameter forwarding
 *
 * cuLaunchKernel provides two ways to pass arguments:
 *
 *   1. `extra` array (CU_LAUNCH_PARAM_BUFFER_POINTER + _SIZE tags)
 *      A packed buffer with total size. Trivial to forward -- we treat it
 *      as a single "parameter" of the given size.
 *
 *   2. `kernelParams` (void** array of pointers to each argument)
 *      Requires knowing per-parameter sizes. We use cuFuncGetParamInfo
 *      (CUDA 12.3+) to introspect, with per-CUfunction caching.
 *      Falls back to NULL if cuFuncGetParamInfo is unavailable.
 */

/* ---- cuFuncGetParamInfo dynamic resolution ---- */

typedef CUresult (*cuFuncGetParamInfo_fn)(CUfunction func, size_t paramIndex,
                                          size_t *paramOffset, size_t *paramSize);

static cuFuncGetParamInfo_fn real_cuFuncGetParamInfo = NULL;
static int cuFuncGetParamInfo_resolved = 0; /* 0 = not tried, 1 = resolved, -1 = unavailable */

static void resolve_cuFuncGetParamInfo(void) {
    if (cuFuncGetParamInfo_resolved != 0)
        return;
    if (!real_dlsym) {
        real_dlsym = (void *(*)(void *, const char *))
            __libc_dlsym(RTLD_NEXT, "dlsym");
    }
    real_cuFuncGetParamInfo = (cuFuncGetParamInfo_fn)
        real_dlsym(RTLD_NEXT, "cuFuncGetParamInfo");
    cuFuncGetParamInfo_resolved = real_cuFuncGetParamInfo ? 1 : -1;
}

/* ---- Per-CUfunction param info cache ---- */

#define PARAM_CACHE_MAX_FUNCS   256
#define PARAM_CACHE_MAX_PARAMS  64   /* CUDA max is ~128 params, 4096 bytes total */

typedef struct {
    CUfunction func;
    unsigned int num_params;
    unsigned int param_sizes[PARAM_CACHE_MAX_PARAMS];
} param_cache_entry_t;

static param_cache_entry_t param_cache[PARAM_CACHE_MAX_FUNCS];
static unsigned int param_cache_count = 0;
static pthread_mutex_t param_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * Look up cached param info for a CUfunction.
 * Returns pointer to cache entry if found, NULL otherwise.
 */
static const param_cache_entry_t *param_cache_lookup(CUfunction func) {
    for (unsigned int i = 0; i < param_cache_count; i++) {
        if (param_cache[i].func == func) {
            return &param_cache[i];
        }
    }
    return NULL;
}

/*
 * Query cuFuncGetParamInfo for all params and store in cache.
 * Returns pointer to new cache entry, or NULL on failure.
 * Caller must hold param_cache_mutex.
 */
static const param_cache_entry_t *param_cache_populate(CUfunction func) {
    if (param_cache_count >= PARAM_CACHE_MAX_FUNCS) {
        /* Cache full -- evict oldest (slot 0) by shifting.
         * This is rare: most apps use far fewer than 256 kernels. */
        memmove(&param_cache[0], &param_cache[1],
                (PARAM_CACHE_MAX_FUNCS - 1) * sizeof(param_cache_entry_t));
        param_cache_count = PARAM_CACHE_MAX_FUNCS - 1;
    }

    param_cache_entry_t *entry = &param_cache[param_cache_count];
    entry->func = func;
    entry->num_params = 0;

    for (size_t i = 0; i < PARAM_CACHE_MAX_PARAMS; i++) {
        size_t offset = 0, size = 0;
        CUresult r = real_cuFuncGetParamInfo(func, i, &offset, &size);
        if (r != CUDA_SUCCESS) {
            /* CUDA_ERROR_INVALID_VALUE signals end of params */
            break;
        }
        entry->param_sizes[i] = (unsigned int)size;
        entry->num_params = (unsigned int)(i + 1);
    }

    param_cache_count++;
    return entry;
}

/*
 * Get param info for a CUfunction, using cache.
 * Returns cache entry or NULL if cuFuncGetParamInfo is unavailable.
 */
static const param_cache_entry_t *get_func_param_info(CUfunction func) {
    resolve_cuFuncGetParamInfo();
    if (cuFuncGetParamInfo_resolved != 1)
        return NULL;

    /* Hold the mutex for both read and write.  The cache is only 256 entries
     * and kernel launches aren't on the hot path relative to network overhead,
     * so there is no benefit to a lockless fast-path (which would be a data
     * race on param_cache_count / array contents under C11). */
    pthread_mutex_lock(&param_cache_mutex);
    const param_cache_entry_t *cached = param_cache_lookup(func);
    if (!cached) {
        cached = param_cache_populate(func);
    }
    pthread_mutex_unlock(&param_cache_mutex);
    return cached;
}

/* ---- hook_cuLaunchKernel ---- */

CUresult hook_cuLaunchKernel(CUfunction f,
                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, CUstream hStream,
                              void **kernelParams, void **extra) {
    ensure_init();

    /*
     * Path 1: `extra` array -- the caller provides a packed parameter buffer.
     *
     * The array contains tagged entries:
     *   CU_LAUNCH_PARAM_BUFFER_POINTER, <void* buffer>,
     *   CU_LAUNCH_PARAM_BUFFER_SIZE,    <size_t* pSize>,
     *   CU_LAUNCH_PARAM_END
     *
     * We extract the buffer and size, then pass the entire buffer as a
     * single "parameter" to the Rust FFI. The server replays it via the
     * same `extra` mechanism on the real GPU.
     */
    if (extra != NULL) {
        const unsigned char *buffer_ptr = NULL;
        unsigned int buffer_size = 0;

        for (int i = 0; i < 14 && extra[i] != CU_LAUNCH_PARAM_END; /* manual advance */) {
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
                buffer_ptr = (const unsigned char *)extra[i + 1];
                i += 2;
            } else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
                size_t *psize = (size_t *)extra[i + 1];
                buffer_size = (unsigned int)(*psize);
                i += 2;
            } else {
                /* Unknown tag -- skip pair */
                i += 2;
            }
        }

        if (buffer_ptr != NULL && buffer_size > 0) {
            /* Pass as a single param: the packed buffer */
            const unsigned char *params_array[1] = { buffer_ptr };
            unsigned int sizes_array[1] = { buffer_size };

            return ol_cuLaunchKernel(
                (unsigned long long)(uintptr_t)f,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes,
                (unsigned long long)(uintptr_t)hStream,
                (const unsigned char *const *)params_array,
                1,
                sizes_array);
        }

        /* extra was provided but had no buffer -- launch with no params */
        return ol_cuLaunchKernel(
            (unsigned long long)(uintptr_t)f,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes,
            (unsigned long long)(uintptr_t)hStream,
            NULL, 0, NULL);
    }

    /*
     * Path 2: `kernelParams` -- void** array of pointers to each argument.
     *
     * We need per-parameter sizes from cuFuncGetParamInfo (CUDA 12.3+).
     * If that API is unavailable, fall back to passing NULL (existing behavior).
     */
    if (kernelParams != NULL) {
        const param_cache_entry_t *info = get_func_param_info(f);
        if (info != NULL && info->num_params > 0) {
            /* Build the params array for the Rust FFI.
             * kernelParams[i] points to the storage for parameter i. */
            return ol_cuLaunchKernel(
                (unsigned long long)(uintptr_t)f,
                gridDimX, gridDimY, gridDimZ,
                blockDimX, blockDimY, blockDimZ,
                sharedMemBytes,
                (unsigned long long)(uintptr_t)hStream,
                (const unsigned char *const *)kernelParams,
                info->num_params,
                info->param_sizes);
        }
        /* cuFuncGetParamInfo unavailable or zero params -- fall through */
    }

    /*
     * Fallback: no params (kernelParams==NULL, or cuFuncGetParamInfo unavailable).
     */
    return ol_cuLaunchKernel(
        (unsigned long long)(uintptr_t)f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        (unsigned long long)(uintptr_t)hStream,
        NULL, 0, NULL);
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
