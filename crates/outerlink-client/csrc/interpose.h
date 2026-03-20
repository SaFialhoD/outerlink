/*
 * OuterLink CUDA Driver API Interposition Header
 *
 * Declares the hook functions and Rust FFI imports used by interpose.c.
 * The interposition library (loaded via LD_PRELOAD) redirects CUDA Driver
 * API calls through these hooks into the Rust client library.
 */

#ifndef OUTERLINK_INTERPOSE_H
#define OUTERLINK_INTERPOSE_H

#include "cuda.h"

/* -----------------------------------------------------------------------
 * Rust FFI imports
 *
 * These functions are implemented in Rust (src/ffi.rs) and exported as
 * extern "C" symbols from the outerlink-client cdylib. The C interposition
 * layer calls these to forward each intercepted CUDA call into Rust.
 *
 * IMPORTANT: The types here must match the Rust signatures exactly.
 * Opaque CUDA handles (CUcontext, etc.) are passed as u64/unsigned long long
 * on the Rust side, not as actual pointer types. The hook functions in
 * interpose.c handle the casting between CUDA pointer types and u64.
 * ----------------------------------------------------------------------- */

/* Initialization */
extern void     ol_client_init(void);
extern CUresult ol_cuInit(unsigned int flags);
extern CUresult ol_cuDriverGetVersion(int *driverVersion);

/* Device management -- CUdevice is int, so these match directly */
extern CUresult ol_cuDeviceGet(int *device, int ordinal);
extern CUresult ol_cuDeviceGetCount(int *count);
extern CUresult ol_cuDeviceGetName(char *name, int len, int dev);
extern CUresult ol_cuDeviceGetAttribute(int *pi, int attrib, int dev);
extern CUresult ol_cuDeviceTotalMem_v2(size_t *bytes, int dev);
extern CUresult ol_cuDeviceGetUuid(unsigned char *uuid, int dev);

/* Context management -- handles passed as u64 (unsigned long long) */
extern CUresult ol_cuCtxCreate_v2(unsigned long long *pctx, unsigned int flags, int dev);
extern CUresult ol_cuCtxDestroy_v2(unsigned long long ctx);
extern CUresult ol_cuCtxSetCurrent(unsigned long long ctx);
extern CUresult ol_cuCtxGetCurrent(unsigned long long *pctx);
extern CUresult ol_cuCtxGetDevice(int *dev);
extern CUresult ol_cuCtxSynchronize(void);

/* Memory management -- CUdeviceptr is unsigned long long */
extern CUresult ol_cuMemAlloc_v2(unsigned long long *dptr, size_t bytesize);
extern CUresult ol_cuMemFree_v2(unsigned long long dptr);
extern CUresult ol_cuMemcpyHtoD_v2(unsigned long long dstDevice, const void *srcHost, size_t ByteCount);
extern CUresult ol_cuMemcpyDtoH_v2(void *dstHost, unsigned long long srcDevice, size_t ByteCount);
extern CUresult ol_cuMemGetInfo_v2(size_t *free, size_t *total);

/* Error handling -- CUresult is an enum (int-sized) */
extern CUresult ol_cuGetErrorName(unsigned int error, const char **pStr);
extern CUresult ol_cuGetErrorString(unsigned int error, const char **pStr);

/* -----------------------------------------------------------------------
 * Hook function declarations
 *
 * These have the exact same signature as the corresponding CUDA Driver API
 * functions. The interposition layer returns pointers to these when an
 * application looks up a CUDA symbol via dlsym() or cuGetProcAddress().
 * ----------------------------------------------------------------------- */

/* Init */
CUresult hook_cuInit(unsigned int Flags);
CUresult hook_cuDriverGetVersion(int *driverVersion);

/* Device */
CUresult hook_cuDeviceGet(CUdevice *device, int ordinal);
CUresult hook_cuDeviceGetCount(int *count);
CUresult hook_cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult hook_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult hook_cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult hook_cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);

/* Context */
CUresult hook_cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult hook_cuCtxDestroy_v2(CUcontext ctx);
CUresult hook_cuCtxSetCurrent(CUcontext ctx);
CUresult hook_cuCtxGetCurrent(CUcontext *pctx);
CUresult hook_cuCtxGetDevice(CUdevice *dev);
CUresult hook_cuCtxSynchronize(void);

/* Memory */
CUresult hook_cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult hook_cuMemFree_v2(CUdeviceptr dptr);
CUresult hook_cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult hook_cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult hook_cuMemGetInfo_v2(size_t *free, size_t *total);

/* Error */
CUresult hook_cuGetErrorName(CUresult error, const char **pStr);
CUresult hook_cuGetErrorString(CUresult error, const char **pStr);

/* cuGetProcAddress hooks */
CUresult hook_cuGetProcAddress(const char *symbol, void **pfn,
                               int cudaVersion, cuuint64_t flags);
CUresult hook_cuGetProcAddress_v2(const char *symbol, void **pfn,
                                  int cudaVersion, cuuint64_t flags,
                                  void *symbolStatus);

#endif /* OUTERLINK_INTERPOSE_H */
