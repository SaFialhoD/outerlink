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
extern CUresult ol_cuCtxPushCurrent(unsigned long long ctx);
extern CUresult ol_cuCtxPopCurrent(unsigned long long *pctx);
extern CUresult ol_cuCtxGetDevice(int *dev);
extern CUresult ol_cuCtxSynchronize(void);

/* Function attributes */
extern CUresult ol_cuFuncGetAttribute(int *pi, int attrib, unsigned long long func);

/* Memory management -- CUdeviceptr is unsigned long long */
extern CUresult ol_cuMemAlloc_v2(unsigned long long *dptr, size_t bytesize);
extern CUresult ol_cuMemFree_v2(unsigned long long dptr);
extern CUresult ol_cuMemcpyHtoD_v2(unsigned long long dstDevice, const void *srcHost, size_t ByteCount);
extern CUresult ol_cuMemcpyDtoH_v2(void *dstHost, unsigned long long srcDevice, size_t ByteCount);
extern CUresult ol_cuMemcpyDtoD(unsigned long long dst, unsigned long long src, size_t ByteCount);
extern CUresult ol_cuMemAllocHost(void **pp, size_t bytesize);
extern CUresult ol_cuMemFreeHost(void *p);
extern CUresult ol_cuMemGetInfo_v2(size_t *free, size_t *total);

/* Error handling -- CUresult is an enum (int-sized) */
extern CUresult ol_cuGetErrorName(unsigned int error, const char **pStr);
extern CUresult ol_cuGetErrorString(unsigned int error, const char **pStr);

/* Module management -- handles passed as u64 (unsigned long long) */
extern CUresult ol_cuModuleLoadData(unsigned long long *module, const void *data, size_t data_len);
extern CUresult ol_cuModuleUnload(unsigned long long module);
extern CUresult ol_cuModuleGetFunction(unsigned long long *func, unsigned long long module, const char *name);
extern CUresult ol_cuModuleGetGlobal(unsigned long long *dptr, size_t *size, unsigned long long module, const unsigned char *name, size_t name_len);

/* Stream management */
extern CUresult ol_cuStreamCreate(unsigned long long *stream, unsigned int flags);
extern CUresult ol_cuStreamDestroy(unsigned long long stream);
extern CUresult ol_cuStreamSynchronize(unsigned long long stream);
extern CUresult ol_cuStreamQuery(unsigned long long stream);
extern CUresult ol_cuStreamWaitEvent(unsigned long long stream, unsigned long long event, unsigned int flags);

/* Event management */
extern CUresult ol_cuEventCreate(unsigned long long *event, unsigned int flags);
extern CUresult ol_cuEventDestroy(unsigned long long event);
extern CUresult ol_cuEventRecord(unsigned long long event, unsigned long long stream);
extern CUresult ol_cuEventSynchronize(unsigned long long event);
extern CUresult ol_cuEventElapsedTime(float *ms, unsigned long long start, unsigned long long end);
extern CUresult ol_cuEventQuery(unsigned long long event);

/* Kernel launch
 *
 * Uses the extended OuterLink signature with explicit param count and sizes.
 * The real CUDA API uses void** kernelParams with implicit sizes, but at
 * interception time we cannot infer parameter sizes without kernel metadata.
 * The C hook passes NULL/0 for params; real param passing requires the
 * extended API (num_params + param_sizes).
 *
 * Phase 2 will add kernel signature introspection from module metadata.
 */
extern CUresult ol_cuLaunchKernel(unsigned long long func,
                                   unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, unsigned long long hStream,
                                   const unsigned char *const *kernelParams,
                                   unsigned int numParams,
                                   const unsigned int *paramSizes);

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
CUresult hook_cuCtxPushCurrent_v2(CUcontext ctx);
CUresult hook_cuCtxPopCurrent_v2(CUcontext *pctx);
CUresult hook_cuCtxGetDevice(CUdevice *dev);
CUresult hook_cuCtxSynchronize(void);

/* Function attributes */
CUresult hook_cuFuncGetAttribute(int *pi, CUdevice_attribute attrib, CUfunction hfunc);

/* Memory */
CUresult hook_cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult hook_cuMemFree_v2(CUdeviceptr dptr);
CUresult hook_cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult hook_cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult hook_cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult hook_cuMemAllocHost(void **pp, size_t bytesize);
CUresult hook_cuMemFreeHost(void *p);
CUresult hook_cuMemGetInfo_v2(size_t *free, size_t *total);

/* Error */
CUresult hook_cuGetErrorName(CUresult error, const char **pStr);
CUresult hook_cuGetErrorString(CUresult error, const char **pStr);

/* Module */
CUresult hook_cuModuleLoadData(CUmodule *module, const void *image);
CUresult hook_cuModuleUnload(CUmodule hmod);
CUresult hook_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult hook_cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);

/* Stream */
CUresult hook_cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult hook_cuStreamDestroy(CUstream hStream);
CUresult hook_cuStreamSynchronize(CUstream hStream);
CUresult hook_cuStreamQuery(CUstream hStream);
CUresult hook_cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);

/* Event */
CUresult hook_cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult hook_cuEventDestroy(CUevent hEvent);
CUresult hook_cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult hook_cuEventSynchronize(CUevent hEvent);
CUresult hook_cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult hook_cuEventQuery(CUevent hEvent);

/* Kernel launch */
CUresult hook_cuLaunchKernel(CUfunction f,
                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, CUstream hStream,
                              void **kernelParams, void **extra);

/* cuGetProcAddress hooks */
CUresult hook_cuGetProcAddress(const char *symbol, void **pfn,
                               int cudaVersion, cuuint64_t flags);
CUresult hook_cuGetProcAddress_v2(const char *symbol, void **pfn,
                                  int cudaVersion, cuuint64_t flags,
                                  void *symbolStatus);

#endif /* OUTERLINK_INTERPOSE_H */
