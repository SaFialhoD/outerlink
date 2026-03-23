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
extern CUresult ol_cuCtxGetCacheConfig(unsigned int *pconfig);
extern CUresult ol_cuCtxSetCacheConfig(unsigned int config);
extern CUresult ol_cuCtxGetSharedMemConfig(unsigned int *pConfig);
extern CUresult ol_cuCtxSetSharedMemConfig(unsigned int config);

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
extern CUresult ol_cuMemHostGetDevicePointer(unsigned long long *pdptr, void *p, unsigned int Flags);
extern CUresult ol_cuMemHostGetFlags(unsigned int *pFlags, void *p);
extern CUresult ol_cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
extern CUresult ol_cuMemHostUnregister(void *p);
extern CUresult ol_cuMemcpyHtoDAsync_v2(unsigned long long dstDevice, const void *srcHost, size_t ByteCount, unsigned long long hStream);
extern CUresult ol_cuMemcpyDtoHAsync_v2(void *dstHost, unsigned long long srcDevice, size_t ByteCount, unsigned long long hStream);
extern CUresult ol_cuMemsetD8(unsigned long long dstDevice, unsigned char value, size_t count);
extern CUresult ol_cuMemsetD32(unsigned long long dstDevice, unsigned int value, size_t count);
extern CUresult ol_cuMemsetD16(unsigned long long dstDevice, unsigned short value, size_t count);
extern CUresult ol_cuMemsetD8Async(unsigned long long dstDevice, unsigned char value, size_t count, unsigned long long hStream);
extern CUresult ol_cuMemsetD32Async(unsigned long long dstDevice, unsigned int value, size_t count, unsigned long long hStream);
extern CUresult ol_cuMemsetD16Async(unsigned long long dstDevice, unsigned short value, size_t count, unsigned long long hStream);
extern CUresult ol_cuMemcpy(unsigned long long dst, unsigned long long src, size_t ByteCount);
extern CUresult ol_cuMemcpyAsync(unsigned long long dst, unsigned long long src, size_t ByteCount, unsigned long long hStream);
extern CUresult ol_cuMemGetInfo_v2(size_t *free, size_t *total);

/* D-to-D async, host alloc, pitch alloc */
extern CUresult ol_cuMemcpyDtoDAsync_v2(unsigned long long dst, unsigned long long src, size_t ByteCount, unsigned long long hStream);
extern CUresult ol_cuMemHostAlloc(unsigned char **pp, size_t bytesize, unsigned int flags);
extern CUresult ol_cuMemAllocPitch_v2(unsigned long long *dptr, size_t *pitch, size_t widthInBytes, size_t height, unsigned int elementSize);

/* Module load (file path), fat binary */
extern CUresult ol_cuModuleLoad(unsigned long long *module, const unsigned char *fname);
extern CUresult ol_cuModuleLoadFatBinary(unsigned long long *module, const unsigned char *fatCubin);

/* Device mem pool get/set, allocation granularity */
extern CUresult ol_cuDeviceGetMemPool(unsigned long long *pool, int dev);
extern CUresult ol_cuDeviceSetMemPool(int dev, unsigned long long pool);
extern CUresult ol_cuMemGetAllocationGranularity(size_t *granularity, const unsigned char *prop, int option);

/* Graph stubs */
extern CUresult ol_cuStreamBeginCapture_v2(unsigned long long stream, int mode);
extern CUresult ol_cuStreamEndCapture(unsigned long long stream, unsigned long long *graph);
extern CUresult ol_cuStreamIsCapturing(unsigned long long stream, int *captureStatus);
extern CUresult ol_cuStreamGetCaptureInfo_v2(unsigned long long stream, int *captureStatus,
                                              unsigned long long *id, unsigned long long *graph,
                                              const unsigned long long **deps, size_t *numDeps);
extern CUresult ol_cuGraphCreate(unsigned long long *graph, unsigned int flags);
extern CUresult ol_cuGraphInstantiate_v2(unsigned long long *graphExec, unsigned long long graph,
                                          unsigned long long *errNode, unsigned char *logBuffer, size_t bufferSize);
extern CUresult ol_cuGraphInstantiate(unsigned long long *graphExec, unsigned long long graph,
                                       unsigned long long *errNode, unsigned char *logBuffer, size_t bufferSize);
extern CUresult ol_cuGraphInstantiateWithFlags(unsigned long long *graphExec, unsigned long long graph, unsigned long long flags);
extern CUresult ol_cuGraphLaunch(unsigned long long graphExec, unsigned long long stream);
extern CUresult ol_cuGraphExecDestroy(unsigned long long graphExec);
extern CUresult ol_cuGraphDestroy(unsigned long long graph);

/* Managed / unified memory (CUDA 6.0+) */
extern CUresult ol_cuMemAllocManaged(unsigned long long *dptr, size_t bytesize, unsigned int flags);
extern CUresult ol_cuMemPrefetchAsync(unsigned long long dptr, size_t count, int dstDevice, unsigned long long hStream);
extern CUresult ol_cuMemAdvise(unsigned long long dptr, size_t count, int advice, int device);
extern CUresult ol_cuMemRangeGetAttribute(unsigned char *data, size_t dataSize, int attribute, unsigned long long devPtr, size_t count);
extern CUresult ol_cuMemRangeGetAttributes(unsigned char **data, const size_t *dataSizes, const int *attributes, size_t numAttributes, unsigned long long devPtr, size_t count);

/* Stream-ordered memory / pool (CUDA 11.2+) */
extern CUresult ol_cuMemAllocAsync(unsigned long long *dptr, size_t bytesize, unsigned long long hStream);
extern CUresult ol_cuMemFreeAsync(unsigned long long dptr, unsigned long long hStream);
extern CUresult ol_cuDeviceGetDefaultMemPool(unsigned long long *pool, int dev);
extern CUresult ol_cuMemPoolCreate(unsigned long long *pool, int alloc_type, int loc_type, int loc_id);
extern CUresult ol_cuMemPoolDestroy(unsigned long long pool);
extern CUresult ol_cuMemPoolGetAttribute(unsigned long long pool, int attr, unsigned long long *value);
extern CUresult ol_cuMemPoolSetAttribute(unsigned long long pool, int attr, unsigned long long value);
extern CUresult ol_cuMemPoolTrimTo(unsigned long long pool, unsigned long long minBytesToKeep);
extern CUresult ol_cuMemAllocFromPoolAsync(unsigned long long *dptr, size_t bytesize, unsigned long long pool, unsigned long long hStream);

/* JIT Linker */
extern CUresult ol_cuLinkCreate_v2(unsigned int numOptions, const int *options,
                                    const unsigned long long *optionValues,
                                    unsigned long long *stateOut);
extern CUresult ol_cuLinkCreate(unsigned int numOptions, const int *options,
                                 const unsigned long long *optionValues,
                                 unsigned long long *stateOut);
extern CUresult ol_cuLinkAddData_v2(unsigned long long state, int type,
                                     const unsigned char *data, size_t size,
                                     const char *name, unsigned int numOptions,
                                     const int *options,
                                     const unsigned long long *optionValues);
extern CUresult ol_cuLinkAddData(unsigned long long state, int type,
                                  const unsigned char *data, size_t size,
                                  const char *name, unsigned int numOptions,
                                  const int *options,
                                  const unsigned long long *optionValues);
extern CUresult ol_cuLinkAddFile_v2(unsigned long long state, int type,
                                     const char *path, unsigned int numOptions,
                                     const int *options,
                                     const unsigned long long *optionValues);
extern CUresult ol_cuLinkAddFile(unsigned long long state, int type,
                                  const char *path, unsigned int numOptions,
                                  const int *options,
                                  const unsigned long long *optionValues);
extern CUresult ol_cuLinkComplete(unsigned long long state,
                                   const unsigned char **cubinOut,
                                   size_t *sizeOut);
extern CUresult ol_cuLinkDestroy(unsigned long long state);

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
extern CUresult ol_cuFuncGetParamInfo(unsigned long long func, unsigned long long param_index,
                                       unsigned long long *param_offset, unsigned long long *param_size);
extern CUresult ol_cuFuncSetCacheConfig(unsigned long long func, unsigned int config);
extern CUresult ol_cuFuncSetSharedMemConfig(unsigned long long func, unsigned int config);
extern CUresult ol_cuMemGetAddressRange_v2(unsigned long long *pbase, size_t *psize, unsigned long long dptr);

/* Library API (CUDA 12+) -- handles passed as u64 (unsigned long long) */
extern CUresult ol_cuLibraryLoadData(unsigned long long *library, const void *data, size_t data_len,
                                      unsigned int numJitOptions, const int *jitOptions, const unsigned long long *jitOptionValues,
                                      unsigned int numLibOptions, const int *libOptions, const unsigned long long *libOptionValues);
extern CUresult ol_cuLibraryUnload(unsigned long long library);
extern CUresult ol_cuLibraryGetKernel(unsigned long long *kernel, unsigned long long library, const char *name);
extern CUresult ol_cuLibraryGetModule(unsigned long long *module, unsigned long long library);
extern CUresult ol_cuKernelGetFunction(unsigned long long *func, unsigned long long kernel);

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
extern CUresult ol_cuStreamAddCallback(unsigned long long stream, unsigned long long callback, unsigned long long userData, unsigned int flags);
extern CUresult ol_cuLaunchHostFunc(unsigned long long stream, unsigned long long fn_ptr, unsigned long long userData);

/* Event management */
extern CUresult ol_cuEventCreate(unsigned long long *event, unsigned int flags);
extern CUresult ol_cuEventDestroy(unsigned long long event);
extern CUresult ol_cuEventRecord(unsigned long long event, unsigned long long stream);
extern CUresult ol_cuEventRecordWithFlags(unsigned long long event, unsigned long long stream, unsigned int flags);
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
 *   2. `kernelParams` path: Uses ol_cuFuncGetParamInfo (routed through the
 *      Rust FFI to the server) to introspect per-parameter sizes. Results
 *      are cached per CUfunction. This avoids calling the real driver's
 *      cuFuncGetParamInfo with synthetic handles (which would SEGFAULT).
 */
extern CUresult ol_cuLaunchCooperativeKernel(unsigned long long func,
                                   unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, unsigned long long hStream,
                                   const unsigned char *const *kernelParams,
                                   unsigned int numParams,
                                   const unsigned int *paramSizes);
extern CUresult ol_cuDeviceGetPCIBusId(unsigned char *pciBusId, int len, int dev);
extern CUresult ol_cuDeviceGetByPCIBusId(int *dev, const unsigned char *pciBusId);

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
CUresult hook_cuCtxGetCacheConfig(int *pconfig);
CUresult hook_cuCtxSetCacheConfig(int config);
CUresult hook_cuCtxGetSharedMemConfig(int *pConfig);
CUresult hook_cuCtxSetSharedMemConfig(int config);

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
CUresult hook_cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags);
CUresult hook_cuMemHostGetFlags(unsigned int *pFlags, void *p);
CUresult hook_cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
CUresult hook_cuMemHostUnregister(void *p);
CUresult hook_cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult hook_cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult hook_cuMemsetD8(CUdeviceptr dstDevice, unsigned char value, size_t count);
CUresult hook_cuMemsetD32(CUdeviceptr dstDevice, unsigned int value, size_t count);
CUresult hook_cuMemsetD16(CUdeviceptr dstDevice, unsigned short value, size_t count);
CUresult hook_cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char value, size_t count, CUstream hStream);
CUresult hook_cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int value, size_t count, CUstream hStream);
CUresult hook_cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short value, size_t count, CUstream hStream);
CUresult hook_cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult hook_cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
CUresult hook_cuMemGetInfo_v2(size_t *free, size_t *total);

/* Managed / unified memory */
CUresult hook_cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
CUresult hook_cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
CUresult hook_cuMemAdvise(CUdeviceptr devPtr, size_t count, int advice, CUdevice device);
CUresult hook_cuMemRangeGetAttribute(void *data, size_t dataSize, int attribute, CUdeviceptr devPtr, size_t count);
CUresult hook_cuMemRangeGetAttributes(void **data, size_t *dataSizes, int *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);

/* Memory pool (CUDA 11.2+) */
CUresult hook_cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
CUresult hook_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
CUresult hook_cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev);
CUresult hook_cuMemPoolCreate(CUmemoryPool *pool, const void *poolProps);
CUresult hook_cuMemPoolDestroy(CUmemoryPool pool);
CUresult hook_cuMemPoolGetAttribute(CUmemoryPool pool, int attr, void *value);
CUresult hook_cuMemPoolSetAttribute(CUmemoryPool pool, int attr, void *value);
CUresult hook_cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
CUresult hook_cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);

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
CUresult hook_cuFuncSetCacheConfig(CUfunction hfunc, int config);
CUresult hook_cuFuncSetSharedMemConfig(CUfunction hfunc, int config);

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
CUresult hook_cuStreamAddCallback(CUstream hStream, void *callback, void *userData, unsigned int flags);
CUresult hook_cuLaunchHostFunc(CUstream hStream, void *fn_ptr, void *userData);

/* Event */
CUresult hook_cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult hook_cuEventDestroy(CUevent hEvent);
CUresult hook_cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult hook_cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags);
CUresult hook_cuEventSynchronize(CUevent hEvent);
CUresult hook_cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult hook_cuEventQuery(CUevent hEvent);

/* Device PCI ID */
CUresult hook_cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
CUresult hook_cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId);

/* Kernel launch */
CUresult hook_cuLaunchKernel(CUfunction f,
                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, CUstream hStream,
                              void **kernelParams, void **extra);
CUresult hook_cuLaunchCooperativeKernel(CUfunction f,
                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, CUstream hStream,
                              void **kernelParams);

/* cuGetExportTable -- passthrough to real libcuda.so */
CUresult hook_cuGetExportTable(const void **ppExportTable, const void *pExportTableId);

/* Batch 2: D-to-D async, host alloc, pitch alloc, module load, fat binary,
 * device mem pool get/set, allocation granularity */
CUresult hook_cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
CUresult hook_cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
CUresult hook_cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
CUresult hook_cuModuleLoad(CUmodule *module, const char *fname);
CUresult hook_cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
CUresult hook_cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev);
CUresult hook_cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);
CUresult hook_cuMemGetAllocationGranularity(size_t *granularity, const void *prop, int option);

/* Graph stubs */
CUresult hook_cuStreamBeginCapture_v2(CUstream hStream, int mode);
CUresult hook_cuStreamEndCapture(CUstream hStream, CUgraph *phGraph);
CUresult hook_cuStreamIsCapturing(CUstream hStream, int *captureStatus);
CUresult hook_cuStreamGetCaptureInfo_v2(CUstream hStream, int *captureStatus,
                                         unsigned long long *id, CUgraph *graph,
                                         const void **deps, size_t *numDeps);
CUresult hook_cuGraphCreate(CUgraph *phGraph, unsigned int flags);
CUresult hook_cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph,
                                     void *phErrorNode, char *logBuffer, size_t bufferSize);
CUresult hook_cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                  void *phErrorNode, char *logBuffer, size_t bufferSize);
CUresult hook_cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags);
CUresult hook_cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
CUresult hook_cuGraphExecDestroy(CUgraphExec hGraphExec);
CUresult hook_cuGraphDestroy(CUgraph hGraph);

/* cuLaunchKernelEx (CUDA 12+) */
CUresult hook_cuLaunchKernelEx(const void *config, CUfunction f,
                                void **kernelParams, void **extra);

/* Library API (CUDA 12+) */
CUresult hook_cuLibraryLoadData(CUlibrary *library, const void *code,
                                 void *jitOptions, void **jitOptionsValues, unsigned int numJitOptions,
                                 void *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
CUresult hook_cuLibraryUnload(CUlibrary library);
CUresult hook_cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name);
CUresult hook_cuLibraryGetModule(CUmodule *pMod, CUlibrary library);
CUresult hook_cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel);

/* cuGetProcAddress hooks */
CUresult hook_cuGetProcAddress(const char *symbol, void **pfn,
                               int cudaVersion, cuuint64_t flags);
CUresult hook_cuGetProcAddress_v2(const char *symbol, void **pfn,
                                  int cudaVersion, cuuint64_t flags,
                                  void *symbolStatus);

#endif /* OUTERLINK_INTERPOSE_H */
