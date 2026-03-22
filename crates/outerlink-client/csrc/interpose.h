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
extern CUresult ol_cuCtxPushCurrent_v2(unsigned long long ctx);
extern CUresult ol_cuCtxPopCurrent_v2(unsigned long long *pctx);
extern CUresult ol_cuCtxGetApiVersion(unsigned long long ctx, unsigned int *version);
extern CUresult ol_cuCtxGetLimit(unsigned long long *pvalue, unsigned int limit);
extern CUresult ol_cuCtxSetLimit(unsigned int limit, unsigned long long value);
extern CUresult ol_cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
extern CUresult ol_cuCtxGetFlags(unsigned int *flags);

/* Primary context management */
extern CUresult ol_cuDevicePrimaryCtxRetain(unsigned long long *pctx, int dev);
extern CUresult ol_cuDevicePrimaryCtxRelease(int dev);
extern CUresult ol_cuDevicePrimaryCtxGetState(int dev, unsigned int *flags, int *active);
extern CUresult ol_cuDevicePrimaryCtxSetFlags(int dev, unsigned int flags);
extern CUresult ol_cuDevicePrimaryCtxReset(int dev);

/* Peer access */
extern CUresult ol_cuDeviceCanAccessPeer(int *canAccessPeer, int dev, int peerDev);
extern CUresult ol_cuDeviceGetP2PAttribute(int *value, int attrib, int srcDevice, int dstDevice);
extern CUresult ol_cuCtxEnablePeerAccess(unsigned long long peerContext, unsigned int flags);
extern CUresult ol_cuCtxDisablePeerAccess(unsigned long long peerContext);

/* Memory management -- CUdeviceptr is unsigned long long */
extern CUresult ol_cuMemAlloc_v2(unsigned long long *dptr, size_t bytesize);
extern CUresult ol_cuMemFree_v2(unsigned long long dptr);
extern CUresult ol_cuMemcpyHtoD_v2(unsigned long long dstDevice, const void *srcHost, size_t ByteCount);
extern CUresult ol_cuMemcpyDtoH_v2(void *dstHost, unsigned long long srcDevice, size_t ByteCount);
extern CUresult ol_cuMemcpyDtoD(unsigned long long dst, unsigned long long src, size_t ByteCount);
extern CUresult ol_cuMemAllocHost(void **pp, size_t bytesize);
extern CUresult ol_cuMemFreeHost(void *p);
extern CUresult ol_cuMemcpyHtoDAsync_v2(unsigned long long dstDevice, const void *srcHost, size_t ByteCount, unsigned long long hStream);
extern CUresult ol_cuMemcpyDtoHAsync_v2(void *dstHost, unsigned long long srcDevice, size_t ByteCount, unsigned long long hStream);
extern CUresult ol_cuMemsetD8(unsigned long long dstDevice, unsigned char value, size_t count);
extern CUresult ol_cuMemsetD32(unsigned long long dstDevice, unsigned int value, size_t count);
extern CUresult ol_cuMemsetD8Async(unsigned long long dstDevice, unsigned char value, size_t count, unsigned long long hStream);
extern CUresult ol_cuMemsetD32Async(unsigned long long dstDevice, unsigned int value, size_t count, unsigned long long hStream);
extern CUresult ol_cuMemGetInfo_v2(size_t *free, size_t *total);

/* Error handling -- CUresult is an enum (int-sized) */
extern CUresult ol_cuGetErrorName(unsigned int error, const char **pStr);
extern CUresult ol_cuGetErrorString(unsigned int error, const char **pStr);

/* Module management -- handles passed as u64 (unsigned long long) */
extern CUresult ol_cuModuleLoadData(unsigned long long *module, const void *data, size_t data_len);
extern CUresult ol_cuModuleLoadDataEx(unsigned long long *module, const void *data, size_t data_len,
                                       unsigned int numOptions, const int *options, const unsigned long long *optionValues);
extern CUresult ol_cuModuleUnload(unsigned long long module);
extern CUresult ol_cuModuleGetFunction(unsigned long long *func, unsigned long long module, const char *name);
extern CUresult ol_cuModuleGetGlobal(unsigned long long *dptr, size_t *size, unsigned long long module, const unsigned char *name, size_t name_len);
extern CUresult ol_cuFuncGetAttribute(int *pi, int attrib, unsigned long long func);
extern CUresult ol_cuFuncSetAttribute(unsigned long long func, int attrib, int value);
extern CUresult ol_cuMemGetAddressRange_v2(unsigned long long *pbase, size_t *psize, unsigned long long dptr);

/* Pointer attributes */
extern CUresult ol_cuPointerGetAttribute(unsigned char *data, int attribute, unsigned long long devPtr);
extern CUresult ol_cuPointerGetAttributes(unsigned int numAttributes, const int *attributes, unsigned char **data, unsigned long long ptr);

/* Occupancy */
extern CUresult ol_cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, unsigned long long func, int blockSize, unsigned long long dynamicSMemSize);
extern CUresult ol_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, unsigned long long func, int blockSize, unsigned long long dynamicSMemSize, unsigned int flags);
extern CUresult ol_cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, unsigned long long func, const void *callback, unsigned long long dynamicSMemSize, int blockSizeLimit);
extern CUresult ol_cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, unsigned long long func, const void *callback, unsigned long long dynamicSMemSize, int blockSizeLimit, unsigned int flags);

/* Stream management */
extern CUresult ol_cuStreamCreate(unsigned long long *stream, unsigned int flags);
extern CUresult ol_cuStreamCreateWithPriority(unsigned long long *stream, unsigned int flags, int priority);
extern CUresult ol_cuStreamDestroy(unsigned long long stream);
extern CUresult ol_cuStreamSynchronize(unsigned long long stream);
extern CUresult ol_cuStreamQuery(unsigned long long stream);
extern CUresult ol_cuStreamGetPriority(unsigned long long stream, int *priority);
extern CUresult ol_cuStreamGetFlags(unsigned long long stream, unsigned int *flags);
extern CUresult ol_cuStreamGetCtx(unsigned long long stream, unsigned long long *pctx);
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
 * The C hook (hook_cuLaunchKernel) handles two CUDA calling conventions:
 *
 *   1. `extra` path: Parses CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE tags to
 *      extract the packed buffer. Passed as a single param to Rust FFI.
 *
 *   2. `kernelParams` path: Uses cuFuncGetParamInfo (CUDA 12.3+, resolved
 *      via dlsym) to introspect per-parameter sizes. Results are cached
 *      per CUfunction. Falls back to NULL if unavailable (old drivers).
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
CUresult hook_cuCtxGetDevice(CUdevice *dev);
CUresult hook_cuCtxSynchronize(void);
CUresult hook_cuCtxPushCurrent_v2(CUcontext ctx);
CUresult hook_cuCtxPopCurrent_v2(CUcontext *pctx);
CUresult hook_cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
CUresult hook_cuCtxGetLimit(size_t *pvalue, int limit);
CUresult hook_cuCtxSetLimit(int limit, size_t value);
CUresult hook_cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
CUresult hook_cuCtxGetFlags(unsigned int *flags);

/* Primary context */
CUresult hook_cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult hook_cuDevicePrimaryCtxRelease_v2(CUdevice dev);
CUresult hook_cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
CUresult hook_cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags);
CUresult hook_cuDevicePrimaryCtxReset_v2(CUdevice dev);

/* Peer access */
CUresult hook_cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev);
CUresult hook_cuDeviceGetP2PAttribute(int *value, int attrib, CUdevice srcDevice, CUdevice dstDevice);
CUresult hook_cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags);
CUresult hook_cuCtxDisablePeerAccess(CUcontext peerContext);

/* Memory */
CUresult hook_cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult hook_cuMemFree_v2(CUdeviceptr dptr);
CUresult hook_cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult hook_cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult hook_cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult hook_cuMemAllocHost(void **pp, size_t bytesize);
CUresult hook_cuMemFreeHost(void *p);
CUresult hook_cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult hook_cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult hook_cuMemsetD8(CUdeviceptr dstDevice, unsigned char value, size_t count);
CUresult hook_cuMemsetD32(CUdeviceptr dstDevice, unsigned int value, size_t count);
CUresult hook_cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char value, size_t count, CUstream hStream);
CUresult hook_cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int value, size_t count, CUstream hStream);
CUresult hook_cuMemGetInfo_v2(size_t *free, size_t *total);

/* Error */
CUresult hook_cuGetErrorName(CUresult error, const char **pStr);
CUresult hook_cuGetErrorString(CUresult error, const char **pStr);

/* Module */
CUresult hook_cuModuleLoadData(CUmodule *module, const void *image);
CUresult hook_cuModuleLoadDataEx(CUmodule *module, const void *image,
                                  unsigned int numOptions, void *options, void **optionValues);
CUresult hook_cuModuleUnload(CUmodule hmod);
CUresult hook_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult hook_cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult hook_cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);

/* Pointer attributes */
CUresult hook_cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr devPtr);
CUresult hook_cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);

/* Occupancy */
CUresult hook_cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
CUresult hook_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
CUresult hook_cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
CUresult hook_cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);

/* Stream */
CUresult hook_cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult hook_cuStreamCreateWithPriority(CUstream *phStream, unsigned int Flags, int priority);
CUresult hook_cuStreamDestroy(CUstream hStream);
CUresult hook_cuStreamSynchronize(CUstream hStream);
CUresult hook_cuStreamQuery(CUstream hStream);
CUresult hook_cuStreamGetPriority(CUstream hStream, int *priority);
CUresult hook_cuStreamGetFlags(CUstream hStream, unsigned int *flags);
CUresult hook_cuStreamGetCtx(CUstream hStream, CUcontext *pctx);
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
