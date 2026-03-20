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

    /* Memory */
    { "cuMemAlloc_v2",           (void *)hook_cuMemAlloc_v2 },
    { "cuMemFree_v2",            (void *)hook_cuMemFree_v2 },
    { "cuMemcpyHtoD_v2",        (void *)hook_cuMemcpyHtoD_v2 },
    { "cuMemcpyDtoH_v2",        (void *)hook_cuMemcpyDtoH_v2 },
    { "cuMemGetInfo_v2",         (void *)hook_cuMemGetInfo_v2 },

    /* Error */
    { "cuGetErrorName",          (void *)hook_cuGetErrorName },
    { "cuGetErrorString",        (void *)hook_cuGetErrorString },

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

CUresult hook_cuMemGetInfo_v2(size_t *free, size_t *total) {
    ensure_init();
    return ol_cuMemGetInfo_v2(free, total);
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
