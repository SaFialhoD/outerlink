/*
 * CUDA Driver API Stub Headers for OuterLink
 * These allow compilation without a real CUDA installation.
 * Only type definitions and function declarations - no implementations.
 */

#ifndef __CUDA_H__
#define __CUDA_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Basic types */
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef unsigned long long CUdeviceptr;
typedef uint64_t cuuint64_t;

/* UUID */
typedef struct CUuuid_st {
    char bytes[16];
} CUuuid;

/* Result codes */
typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_UNKNOWN = 999,
} CUresult;

/* Device attributes */
typedef enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
} CUdevice_attribute;

/* Memory copy direction */
typedef enum {
    CU_MEMORYTYPE_HOST = 0x01,
    CU_MEMORYTYPE_DEVICE = 0x02,
    CU_MEMORYTYPE_ARRAY = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04,
} CUmemorytype;

/* Context flags */
typedef enum {
    CU_CTX_SCHED_AUTO = 0x00,
    CU_CTX_SCHED_SPIN = 0x01,
    CU_CTX_SCHED_YIELD = 0x02,
    CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
    CU_CTX_MAP_HOST = 0x08,
} CUctx_flags;

/* Stream flags */
typedef enum {
    CU_STREAM_DEFAULT = 0x0,
    CU_STREAM_NON_BLOCKING = 0x1,
} CUstream_flags;

/* Event flags */
typedef enum {
    CU_EVENT_DEFAULT = 0x0,
    CU_EVENT_BLOCKING_SYNC = 0x1,
    CU_EVENT_DISABLE_TIMING = 0x2,
} CUevent_flags;

/* Kernel launch params */
#define CU_LAUNCH_PARAM_END            ((void*)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)

/* Function declarations - Init */
CUresult cuInit(unsigned int Flags);
CUresult cuDriverGetVersion(int *driverVersion);

/* Function declarations - Device */
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

/* Function declarations - Context */
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuCtxSynchronize(void);

/* Function declarations - Memory */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize);
CUresult cuMemFreeHost(void *p);
CUresult cuMemGetInfo_v2(size_t *free, size_t *total);

/* Function declarations - Module */
CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, void *options, void **optionValues);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);

/* Function declarations - Execution */
CUresult cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra);

/* Function declarations - Stream */
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult cuStreamDestroy_v2(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
CUresult cuStreamQuery(CUstream hStream);

/* Function declarations - Event */
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult cuEventDestroy_v2(CUevent hEvent);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult cuEventQuery(CUevent hEvent);

/* Function declarations - Entry Point Access (CUDA 11.3+) */
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, void *symbolStatus);

#ifdef __cplusplus
}
#endif

#endif /* __CUDA_H__ */
