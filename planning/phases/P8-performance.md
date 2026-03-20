# P8: Performance Optimization

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - Phase 4 Implementation

## Goal

Minimize OutterLink's overhead on CUDA applications through four strategies: io_uring zero-copy networking, CUDA call batching, UCX transport with automatic RDMA negotiation, and host-staged RDMA pipelining. After this phase, OutterLink should add less than 50us of overhead per CUDA call and achieve over 10 GB/s sustained transfer throughput on 100GbE.

## Milestone

- io_uring zero-copy send/recv integrated with CUDA pinned memory buffers
- Call batching reduces per-call network round trips by 5-10x for batchable sequences
- UCX transport backend passes all existing tests and auto-selects RDMA when available
- Host-staged double-buffer pipeline overlaps compute with transfer
- Tracing infrastructure measures per-call overhead with microsecond precision
- Benchmarks demonstrate measurable improvement over Phase 2/3 baseline

## Prerequisites

- [ ] P6: Core Transport complete (memory transfers + kernel launch working)
- [ ] P7: CUDA Completeness complete (streams, events, multi-GPU working)
- [ ] Both PCs have ConnectX-5 with RDMA verified (P2)
- [ ] Kernel 6.15+ available for io_uring zero-copy recv (or 6.0+ for send-only ZC)

---

## 1. io_uring Integration

### Background

io_uring eliminates kernel-to-userspace copies for network I/O. When combined with CUDA pinned memory, the data path becomes: GPU VRAM -> pinned host buffer -> NIC DMA (no CPU copy in the network layer).

### Kernel Requirements

| Feature | Minimum Kernel | Notes |
|---------|---------------|-------|
| io_uring basic | 5.1 | Submission/completion ring |
| `IORING_OP_SEND_ZC` | 6.0 | Zero-copy send |
| `IORING_OP_RECV_ZC` | 6.15 | Zero-copy recv (Ubuntu 24.04 needs HWE kernel) |
| `IORING_REGISTER_BUFFERS` | 5.1 | Register fixed buffers with kernel |

### Design: io_uring + CUDA Pinned Memory

The key insight is that `cudaHostAlloc` returns pinned (non-pageable) memory, and io_uring's `IORING_REGISTER_BUFFERS` requires non-pageable memory. These are the same requirement -- pinned CUDA buffers can be registered directly with io_uring.

```rust
// crates/outterlink-common/src/transport/iouring_transport.rs

use io_uring::{IoUring, opcode, types};
use std::os::fd::AsRawFd;

/// Number of io_uring submission queue entries
const RING_SIZE: u32 = 256;

/// Size of each registered buffer (aligned to page size)
const BUF_SIZE: usize = 2 * 1024 * 1024; // 2 MiB

/// Number of registered buffers for double-buffering
const NUM_BUFS: usize = 2;

pub struct IoUringTransport {
    ring: IoUring,
    socket_fd: i32,
    /// Pinned buffers registered with both CUDA and io_uring
    send_bufs: [PinnedBuffer; NUM_BUFS],
    recv_bufs: [PinnedBuffer; NUM_BUFS],
    /// Which buffer is currently in use (for double-buffering)
    send_idx: usize,
    recv_idx: usize,
}

/// A buffer that is pinned by CUDA and registered with io_uring
struct PinnedBuffer {
    ptr: *mut u8,
    len: usize,
}

impl IoUringTransport {
    pub fn new(socket_fd: i32) -> Result<Self, TransportError> {
        let mut ring = IoUring::new(RING_SIZE)?;

        // Allocate CUDA pinned memory for send/recv buffers
        let send_bufs = std::array::from_fn(|_| {
            let ptr = cuda_host_alloc(BUF_SIZE);
            PinnedBuffer { ptr, len: BUF_SIZE }
        });
        let recv_bufs = std::array::from_fn(|_| {
            let ptr = cuda_host_alloc(BUF_SIZE);
            PinnedBuffer { ptr, len: BUF_SIZE }
        });

        // Register all buffers with io_uring for zero-copy
        let iovecs: Vec<libc::iovec> = send_bufs.iter()
            .chain(recv_bufs.iter())
            .map(|buf| libc::iovec {
                iov_base: buf.ptr as *mut _,
                iov_len: buf.len,
            })
            .collect();

        // SAFETY: buffers are pinned and will outlive the ring
        unsafe {
            ring.submitter().register_buffers(&iovecs)?;
        }

        Ok(Self {
            ring,
            socket_fd,
            send_bufs,
            recv_bufs,
            send_idx: 0,
            recv_idx: 0,
        })
    }

    /// Zero-copy send from a registered pinned buffer.
    /// Data must already be in send_bufs[self.send_idx].
    pub fn submit_send_zc(&mut self, len: usize) -> Result<(), TransportError> {
        let buf_idx = self.send_idx as u16;

        let send_op = opcode::SendZc::new(
            types::Fd(self.socket_fd),
            self.send_bufs[self.send_idx].ptr,
            len as u32,
        )
        .buf_index(Some(buf_idx))
        .build()
        .user_data(0x01); // tag for completion identification

        unsafe {
            self.ring.submission().push(&send_op)?;
        }
        self.ring.submit()?;

        // Flip to other buffer so caller can fill it while send is in-flight
        self.send_idx = (self.send_idx + 1) % NUM_BUFS;
        Ok(())
    }

    /// Zero-copy recv into a registered pinned buffer.
    pub fn submit_recv_zc(&mut self) -> Result<(), TransportError> {
        let buf_idx = (NUM_BUFS + self.recv_idx) as u16; // offset past send bufs

        let recv_op = opcode::RecvZc::new(
            types::Fd(self.socket_fd),
            self.recv_bufs[self.recv_idx].ptr,
            self.recv_bufs[self.recv_idx].len as u32,
        )
        .buf_index(Some(buf_idx))
        .build()
        .user_data(0x02);

        unsafe {
            self.ring.submission().push(&recv_op)?;
        }
        self.ring.submit()?;

        self.recv_idx = (self.recv_idx + 1) % NUM_BUFS;
        Ok(())
    }

    /// Wait for completion of submitted operations
    pub fn wait_completion(&mut self) -> Result<i32, TransportError> {
        let cqe = self.ring.completion().next()
            .ok_or(TransportError::NoCompletion)?;
        let result = cqe.result();
        if result < 0 {
            return Err(TransportError::Io(std::io::Error::from_raw_os_error(-result)));
        }
        Ok(result)
    }
}

/// Allocate CUDA pinned host memory
fn cuda_host_alloc(size: usize) -> *mut u8 {
    let mut ptr: *mut u8 = std::ptr::null_mut();
    // cuMemAllocHost or cudaHostAlloc with cudaHostAllocPortable
    unsafe {
        let result = cuda_ffi::cuMemAllocHost(
            &mut ptr as *mut *mut u8 as *mut *mut std::ffi::c_void,
            size,
        );
        assert_eq!(result, 0, "cuMemAllocHost failed");
    }
    ptr
}
```

### Fallback When io_uring Is Unavailable

```rust
// crates/outterlink-common/src/transport/mod.rs

pub fn create_transport(config: &TransportConfig) -> Box<dyn Transport> {
    // Try io_uring first
    if iouring_available() {
        match IoUringTransport::new(config.socket_fd) {
            Ok(t) => return Box::new(t),
            Err(e) => {
                tracing::warn!("io_uring init failed, falling back to tokio TCP: {}", e);
            }
        }
    }

    // Fallback: standard tokio TCP (always works)
    Box::new(TokioTcpTransport::new(config))
}

fn iouring_available() -> bool {
    // Check kernel version >= 6.0 for send_zc
    let uname = nix::sys::utsname::uname().ok();
    match uname {
        Some(u) => {
            let release = u.release().to_string_lossy();
            // Parse major.minor from release string
            let parts: Vec<&str> = release.split('.').collect();
            if parts.len() >= 2 {
                let major: u32 = parts[0].parse().unwrap_or(0);
                let minor: u32 = parts[1].parse().unwrap_or(0);
                major > 6 || (major == 6 && minor >= 0)
            } else {
                false
            }
        }
        None => false,
    }
}
```

### Expected Performance Gains

| Metric | tokio TCP | io_uring ZC send | io_uring ZC send+recv |
|--------|----------|-----------------|----------------------|
| Single-flow throughput (100GbE) | ~8-9 GB/s | ~10-11 GB/s | ~11-12 GB/s |
| CPU usage per GB transferred | ~15% | ~5% | ~3% |
| Latency per 1MB transfer | ~120us | ~100us | ~90us |

The primary win is CPU offload. The NIC's DMA engine reads directly from pinned memory without kernel socket buffer copies.

---

## 2. Call Batching (Lazy Updates)

### Problem

Every CUDA Driver API call currently produces a network round trip. Many calls have no visible side effect until a synchronization point. Batching groups these calls into a single network message.

### Batchable vs Flush-Forcing Calls

#### Batchable Calls (No Immediate Side Effect)

These calls modify server-side state but the client does not need the result immediately:

| Call | Why Batchable |
|------|--------------|
| `cuMemAlloc` | Returns a device pointer, but the client can predict it (monotonic allocator) or defer usage |
| `cuMemFree` | Fire and forget |
| `cuModuleLoadData` | Returns a handle, can use predicted handle |
| `cuModuleGetFunction` | Returns a handle, can use predicted handle |
| `cuMemcpyHtoD` | Client already has the data, does not need confirmation |
| `cuStreamCreate` | Returns a handle, can use predicted handle |
| `cuEventCreate` | Returns a handle, can use predicted handle |
| `cuEventRecord` | Fire and forget |
| `cuLaunchKernel` | Fire and forget (async by nature) |
| `cuCtxSetCurrent` | State change, no return value needed |

#### Flush-Forcing Calls (Need Immediate Result)

These calls require the server to process all pending operations and return a result:

| Call | Why Flush Required |
|------|-------------------|
| `cuMemcpyDtoH` | Client needs the data -- must wait for GPU to finish |
| `cuStreamSynchronize` | Client is explicitly waiting for completion |
| `cuCtxSynchronize` | Client is explicitly waiting for all streams |
| `cuEventSynchronize` | Client is waiting for a specific event |
| `cuEventQuery` | Client needs the current event status |
| `cuStreamQuery` | Client needs the current stream status |
| `cuDeviceGetAttribute` | Client needs the attribute value |
| `cuDeviceGetName` | Client needs the name string |
| `cuDeviceTotalMem` | Client needs the memory size |
| `cuMemGetInfo` | Client needs free/total memory |

### Batch Buffer Design

```rust
// crates/outterlink-client/src/batch.rs

use std::time::{Duration, Instant};

/// Maximum number of calls to accumulate before auto-flushing
const MAX_BATCH_SIZE: usize = 64;

/// Maximum time to hold calls before auto-flushing (prevents stalls)
const MAX_BATCH_AGE: Duration = Duration::from_micros(100);

/// A pending CUDA call in the batch
pub struct PendingCall {
    pub opcode: u16,
    pub payload: Vec<u8>,
    /// If this call returns a handle, the predicted handle value
    pub predicted_result: Option<u64>,
}

pub struct BatchBuffer {
    calls: Vec<PendingCall>,
    first_insert: Option<Instant>,
    transport: Arc<dyn Transport>,
    /// Monotonic counter for predicting server-side handles
    next_predicted_handle: u64,
}

impl BatchBuffer {
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self {
            calls: Vec::with_capacity(MAX_BATCH_SIZE),
            first_insert: None,
            transport,
            next_predicted_handle: 0x1000, // start above NULL
        }
    }

    /// Add a batchable call. Returns a predicted handle if applicable.
    pub fn push(&mut self, opcode: u16, payload: Vec<u8>, needs_handle: bool) -> Option<u64> {
        if self.first_insert.is_none() {
            self.first_insert = Some(Instant::now());
        }

        let predicted = if needs_handle {
            let h = self.next_predicted_handle;
            self.next_predicted_handle += 1;
            Some(h)
        } else {
            None
        };

        self.calls.push(PendingCall {
            opcode,
            payload,
            predicted_result: predicted,
        });

        // Auto-flush if batch is full
        if self.calls.len() >= MAX_BATCH_SIZE {
            self.flush().ok(); // errors handled by caller on next sync
        }

        predicted
    }

    /// Check if batch should be flushed due to age
    pub fn should_flush(&self) -> bool {
        if let Some(first) = self.first_insert {
            first.elapsed() >= MAX_BATCH_AGE
        } else {
            false
        }
    }

    /// Flush all pending calls to server as a single batch message.
    /// Returns the server's responses for handle verification.
    pub fn flush(&mut self) -> Result<Vec<ServerResponse>, TransportError> {
        if self.calls.is_empty() {
            return Ok(vec![]);
        }

        let batch = std::mem::take(&mut self.calls);
        self.first_insert = None;

        // Serialize entire batch as one network message
        let msg = BatchMessage {
            call_count: batch.len() as u32,
            calls: batch,
        };

        let response_bytes = self.transport.send_and_recv(&msg.serialize())?;
        let responses = BatchResponse::deserialize(&response_bytes)?;

        // Verify predicted handles match server-assigned handles
        // If mismatch: update local handle map (rare, only on server-side contention)
        Ok(responses.responses)
    }
}
```

### Handle Prediction

For batchable calls that return handles (`cuMemAlloc`, `cuModuleLoadData`, etc.), the client predicts what handle the server will assign. This avoids a round trip to get the handle.

The server uses a monotonic allocator for handles in the same sequence. If prediction and server diverge (should not happen in single-client mode), the flush response includes correction data.

```
Client prediction:  cuMemAlloc -> predicted handle 0x1000
Client prediction:  cuMemAlloc -> predicted handle 0x1001
Client calls:       cuMemcpyHtoD(0x1000, data, size) -- uses predicted handle
Client calls:       cuLaunchKernel(... 0x1000 ...) -- uses predicted handle
Client calls:       cuMemcpyDtoH(...) -- FLUSH POINT

Flush sends all 5 calls at once. Server processes sequentially:
  cuMemAlloc -> actual handle 0x1000 (matches prediction)
  cuMemAlloc -> actual handle 0x1001 (matches prediction)
  cuMemcpyHtoD(0x1000, data, size) -> OK
  cuLaunchKernel(... 0x1000 ...) -> OK
  cuMemcpyDtoH(...) -> returns data to client
```

### Expected Latency Improvement

| Scenario | Without Batching | With Batching | Improvement |
|----------|-----------------|--------------|-------------|
| 10 allocations + 1 sync | 11 round trips (~550us at 50us/RT) | 1 round trip (~80us) | ~7x |
| Module load + get function + launch + sync | 4 round trips (~200us) | 1 round trip (~70us) | ~3x |
| Typical inference step (50 calls) | 50 round trips (~2500us) | 2-3 round trips (~200us) | ~12x |

### Background Flush Thread

A background task periodically checks `should_flush()` and triggers a flush if the batch has been accumulating for more than `MAX_BATCH_AGE`:

```rust
// Spawned once at client initialization
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_micros(50));
    loop {
        interval.tick().await;
        let mut batch = batch_buffer.lock().await;
        if batch.should_flush() {
            if let Err(e) = batch.flush() {
                tracing::error!("Background batch flush failed: {}", e);
            }
        }
    }
});
```

---

## 3. UCX Transport Backend

### What UCX Provides

UCX (Unified Communication X) is a communication framework that auto-negotiates the best available transport:

| Available Hardware | UCX Chooses |
|-------------------|------------|
| Both sides have ConnectX with RDMA | InfiniBand verbs or RoCE (RDMA) |
| One side has ConnectX, other does not | TCP |
| Neither has RDMA hardware | TCP with optimizations |
| Both have ConnectX + CUDA GPUs (data center) | GPUDirect RDMA (not us, but shows the capability) |

### UCX Rust Integration

```rust
// crates/outterlink-common/src/transport/ucx_transport.rs

/// UCX transport backend
/// Links against libucx via ucx-sys FFI bindings

use ucx_sys::*;
use std::ptr;
use std::ffi::CString;

pub struct UcxTransport {
    context: ucp_context_h,
    worker: ucp_worker_h,
    endpoint: ucp_ep_h,
    /// Pinned buffers registered with UCX for zero-copy
    send_buf: UcxMemHandle,
    recv_buf: UcxMemHandle,
}

struct UcxMemHandle {
    ptr: *mut u8,
    len: usize,
    mem_h: ucp_mem_h,
}

impl UcxTransport {
    pub fn new(remote_addr: &str) -> Result<Self, TransportError> {
        // 1. Create UCX context with features we need
        let mut params: ucp_params_t = unsafe { std::mem::zeroed() };
        params.field_mask = UCP_PARAM_FIELD_FEATURES as u64;
        params.features = (UCP_FEATURE_TAG | UCP_FEATURE_RMA) as u64;

        let mut context: ucp_context_h = ptr::null_mut();
        let status = unsafe { ucp_init(&params, ptr::null(), &mut context) };
        if status != UCS_OK {
            return Err(TransportError::UcxInit(status));
        }

        // 2. Create worker (event loop handle)
        let mut worker_params: ucp_worker_params_t = unsafe { std::mem::zeroed() };
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE as u64;
        worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

        let mut worker: ucp_worker_h = ptr::null_mut();
        let status = unsafe {
            ucp_worker_create(context, &worker_params, &mut worker)
        };
        if status != UCS_OK {
            return Err(TransportError::UcxWorkerCreate(status));
        }

        // 3. Register pinned memory with UCX
        let send_buf = Self::register_pinned_buffer(context, 2 * 1024 * 1024)?;
        let recv_buf = Self::register_pinned_buffer(context, 2 * 1024 * 1024)?;

        // 4. Connect to remote endpoint
        let endpoint = Self::connect_to(worker, remote_addr)?;

        Ok(Self {
            context,
            worker,
            endpoint,
            send_buf,
            recv_buf,
        })
    }

    fn register_pinned_buffer(
        context: ucp_context_h,
        size: usize,
    ) -> Result<UcxMemHandle, TransportError> {
        // Allocate CUDA pinned memory
        let ptr = cuda_host_alloc(size);

        // Register with UCX so it can RDMA directly from/to this buffer
        let mut mem_params: ucp_mem_map_params_t = unsafe { std::mem::zeroed() };
        mem_params.field_mask = (UCP_MEM_MAP_PARAM_FIELD_ADDRESS
            | UCP_MEM_MAP_PARAM_FIELD_LENGTH
            | UCP_MEM_MAP_PARAM_FIELD_FLAGS) as u64;
        mem_params.address = ptr as *mut _;
        mem_params.length = size;
        mem_params.flags = UCP_MEM_MAP_FIXED as u32;

        let mut mem_h: ucp_mem_h = ptr::null_mut();
        let status = unsafe {
            ucp_mem_map(context, &mem_params, &mut mem_h)
        };
        if status != UCS_OK {
            return Err(TransportError::UcxMemMap(status));
        }

        Ok(UcxMemHandle { ptr, len: size, mem_h })
    }

    fn connect_to(
        worker: ucp_worker_h,
        remote_addr: &str,
    ) -> Result<ucp_ep_h, TransportError> {
        // Exchange worker addresses via a TCP control channel (out-of-band)
        // Then create UCX endpoint using the remote worker address
        // ... address exchange logic ...
        todo!("Address exchange via control channel")
    }

    /// Send data using UCX tag-matching (auto-selects RDMA or TCP)
    pub fn send(&self, data: &[u8], tag: u64) -> Result<(), TransportError> {
        // Copy data to pinned send buffer
        let len = data.len().min(self.send_buf.len);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.send_buf.ptr, len);
        }

        let mut params: ucp_request_param_t = unsafe { std::mem::zeroed() };
        params.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE as u32;
        params.memory_type = UCS_MEMORY_TYPE_HOST;

        let request = unsafe {
            ucp_tag_send_nbx(
                self.endpoint,
                self.send_buf.ptr as *const _,
                len,
                tag,
                &params,
            )
        };

        self.wait_request(request)
    }

    /// Receive data using UCX tag-matching
    pub fn recv(&self, tag: u64) -> Result<&[u8], TransportError> {
        let mut params: ucp_request_param_t = unsafe { std::mem::zeroed() };
        params.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE as u32;
        params.memory_type = UCS_MEMORY_TYPE_HOST;

        let request = unsafe {
            ucp_tag_recv_nbx(
                self.worker,
                self.recv_buf.ptr as *mut _,
                self.recv_buf.len,
                tag,
                u64::MAX, // match any tag mask
                &params,
            )
        };

        self.wait_request(request)?;

        // Return reference to received data in pinned buffer
        Ok(unsafe {
            std::slice::from_raw_parts(self.recv_buf.ptr, self.recv_buf.len)
        })
    }

    fn wait_request(&self, request: ucs_status_ptr_t) -> Result<(), TransportError> {
        if request.is_null() {
            return Ok(()); // completed inline
        }
        // ... poll worker until request completes ...
        loop {
            unsafe { ucp_worker_progress(self.worker); }
            // check request status
            // break when done
        }
    }
}

impl Drop for UcxTransport {
    fn drop(&mut self) {
        unsafe {
            ucp_mem_unmap(self.context, self.send_buf.mem_h);
            ucp_mem_unmap(self.context, self.recv_buf.mem_h);
            // ... cleanup endpoint, worker, context ...
        }
    }
}
```

### UCX CUDA-Aware Transfers

When UCX detects CUDA memory (via `UCS_MEMORY_TYPE_CUDA`), it automatically stages through pinned host memory. On data center GPUs with GPUDirect, it would RDMA directly. On our GeForce GPUs, it does host-staging transparently:

```rust
// Future: when transferring GPU buffers directly through UCX
params.memory_type = UCS_MEMORY_TYPE_CUDA;
// UCX internally: cuMemcpy to pinned host -> RDMA/TCP -> cuMemcpy from pinned host
```

This gives us automatic optimization without changing our code when hardware changes.

### Migration Path from TCP to UCX

| Step | Change | Risk |
|------|--------|------|
| 1. Add `ucx-sys` dependency behind `ucx` feature flag | `Cargo.toml` | None -- feature-gated |
| 2. Implement `Transport` trait for `UcxTransport` | New file | None -- additive |
| 3. Add UCX to transport selection in `create_transport()` | Config change | Low -- fallback to TCP |
| 4. Test: run existing test suite with UCX backend | CI + hardware | Medium -- new code path |
| 5. Benchmark: compare UCX vs TCP on same hardware | Manual | None |
| 6. If UCX wins: make it default when detected | Config change | Low |

### UCX Build Requirements

UCX must be installed on the system. Add to P2 environment setup:

```bash
# Build UCX from source (for latest RDMA features)
git clone https://github.com/openucx/ucx.git
cd ucx
./autogen.sh
./configure --prefix=/usr/local --with-cuda=/usr/local/cuda --with-mlnx-ofed
make -j$(nproc)
sudo make install
```

In CI: UCX is only needed for the `ucx` feature flag. The default CI build (no features) skips UCX entirely.

---

## 4. RDMA Host-Staged Optimization

### The Pipeline

Even without GPUDirect, we can overlap the three stages of a GPU-to-GPU transfer:

```
Stage 1: cudaMemcpyAsync(pinned_buf_A, gpu_src, size, stream_copy)
Stage 2: RDMA WRITE(remote_pinned_buf, pinned_buf_A, size)
Stage 3: cudaMemcpyAsync(gpu_dst, remote_pinned_buf, size, stream_copy)
```

Without pipelining, these are sequential: total time = S1 + S2 + S3.
With double-buffering, we overlap stages across chunks.

### Double-Buffer Pipeline Design

Split a large transfer into chunks. While chunk N is being sent over RDMA, chunk N+1 is being copied from GPU to pinned memory:

```
Time -->

Chunk 0: [GPU->Host_A] [RDMA Host_A->Remote] [Remote->GPU]
Chunk 1:               [GPU->Host_B] [RDMA Host_B->Remote] [Remote->GPU]
Chunk 2:                             [GPU->Host_A] [RDMA Host_A->Remote] [Remote->GPU]
```

After the pipeline fills, the effective throughput is limited by the slowest stage, not the sum of all stages.

```rust
// crates/outterlink-common/src/transport/rdma_pipeline.rs

/// Chunk size for pipeline. Tuned to balance latency and throughput.
/// Too small: RDMA setup overhead dominates.
/// Too large: pipeline does not overlap well.
const CHUNK_SIZE: usize = 256 * 1024; // 256 KiB

/// Double-buffered pinned memory for pipelined transfer
pub struct RdmaPipeline {
    /// Two pinned host buffers for send-side double buffering
    send_bufs: [PinnedBuffer; 2],
    /// Two pinned host buffers for recv-side double buffering
    recv_bufs: [PinnedBuffer; 2],
    /// CUDA streams for async memcpy (separate from app streams)
    copy_stream_a: CUstream,
    copy_stream_b: CUstream,
    /// RDMA connection handle
    rdma: RdmaConnection,
}

impl RdmaPipeline {
    /// Transfer `size` bytes from local GPU to remote GPU using pipelined
    /// double-buffered host-staged RDMA.
    pub async fn gpu_to_gpu(
        &self,
        local_gpu_ptr: CUdeviceptr,
        remote_gpu_ptr: CUdeviceptr,
        size: usize,
    ) -> Result<(), TransportError> {
        let num_chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut buf_idx = 0usize;

        for chunk in 0..num_chunks {
            let offset = chunk * CHUNK_SIZE;
            let chunk_len = (size - offset).min(CHUNK_SIZE);
            let buf = &self.send_bufs[buf_idx];
            let stream = if buf_idx == 0 {
                self.copy_stream_a
            } else {
                self.copy_stream_b
            };

            // Stage 1: Async GPU -> pinned host (on dedicated stream)
            unsafe {
                cuMemcpyDtoHAsync_v2(
                    buf.ptr as *mut _,
                    local_gpu_ptr + offset as u64,
                    chunk_len,
                    stream,
                );
            }

            // Wait for GPU->host copy to complete before RDMA
            unsafe { cuStreamSynchronize(stream); }

            // Stage 2: RDMA WRITE pinned host -> remote pinned host
            // This is non-blocking -- the NIC DMA engine handles it
            self.rdma.write(
                buf.ptr,
                chunk_len,
                // Remote side's buffer at corresponding index
                self.rdma.remote_buf_addr(buf_idx),
                self.rdma.remote_buf_rkey(buf_idx),
            ).await?;

            // Stage 3: Signal remote side to copy from pinned host -> GPU
            // The remote side runs its own cudaMemcpyAsync
            self.rdma.signal_remote_copy(
                remote_gpu_ptr + offset as u64,
                buf_idx,
                chunk_len,
            ).await?;

            buf_idx = (buf_idx + 1) % 2;
        }

        Ok(())
    }
}
```

### Expected Throughput

| Transfer Size | Without Pipeline | With Pipeline | Speedup |
|--------------|-----------------|--------------|---------|
| 1 MB | 180us (60+60+60) | 120us (fill + 2 chunks) | 1.5x |
| 10 MB | 1800us | 700us | 2.6x |
| 100 MB | 18ms | 7ms | 2.6x |

The theoretical maximum speedup is 3x (three stages fully overlapped), but in practice PCIe and network bandwidth share the same root complex, so ~2.5x is realistic.

---

## 5. Profiling and Measurement

### Tracing Infrastructure

Every CUDA call intercepted by OutterLink is instrumented with `tracing` spans:

```rust
// crates/outterlink-client/src/intercept.rs

use tracing::{instrument, info_span, Span};

#[instrument(skip(data), fields(size = data.len()))]
pub fn handle_cuMemcpyHtoD(
    dst: CUdeviceptr,
    data: &[u8],
) -> CUresult {
    let _serialize = info_span!("serialize").entered();
    let msg = serialize_memcpy_h2d(dst, data);
    drop(_serialize);

    let _network = info_span!("network_roundtrip").entered();
    let response = transport.send_and_recv(&msg)?;
    drop(_network);

    let _deserialize = info_span!("deserialize").entered();
    let result = deserialize_result(&response);
    drop(_deserialize);

    result
}
```

### Where Bottlenecks Will Be

Based on the architecture and transport research, expected bottleneck ranking:

| Rank | Bottleneck | Location | Measurement |
|------|-----------|----------|-------------|
| 1 | Network round-trip latency | Per-call overhead | `tracing` span duration on `network_roundtrip` |
| 2 | Serialization/deserialization | CPU overhead per call | `tracing` span duration on `serialize`/`deserialize` |
| 3 | cudaMemcpy to/from pinned memory | PCIe bandwidth | CUDA event timing around cuMemcpyAsync |
| 4 | Handle translation lookups | HashMap overhead | `tracing` span duration, flamegraph analysis |
| 5 | Lock contention (multi-stream) | Thread synchronization | `tokio::sync` metrics, contention counters |

### Overhead Measurement Tool

```rust
// crates/outterlink-cli/src/commands/benchmark.rs

/// Measure per-call overhead for common CUDA operations
pub async fn measure_overhead(config: &Config) -> BenchmarkResults {
    let iterations = 10_000;

    // Baseline: local CUDA call time
    let local_time = measure_local_cuda_calls(iterations);

    // OutterLink: same calls through interception
    let remote_time = measure_remote_cuda_calls(iterations);

    // Overhead = remote - local
    let overhead_per_call = (remote_time - local_time) / iterations;

    // Breakdown by category
    BenchmarkResults {
        device_query_overhead: measure_device_query(iterations),
        mem_alloc_overhead: measure_mem_alloc(iterations),
        memcpy_h2d_overhead: measure_memcpy_h2d(iterations, &[1024, 1_048_576, 16_777_216]),
        memcpy_d2h_overhead: measure_memcpy_d2h(iterations, &[1024, 1_048_576, 16_777_216]),
        kernel_launch_overhead: measure_kernel_launch(iterations),
        batch_vs_individual: measure_batching_benefit(iterations),
    }
}
```

### Profiling Checklist

| What to Measure | Tool | When |
|-----------------|------|------|
| Per-call overhead breakdown | `tracing` + `tracing-chrome` (generates Chrome trace JSON) | Every benchmark run |
| Network throughput | `iperf3` baseline, then OutterLink sustained throughput | After transport changes |
| CPU utilization during transfer | `perf stat`, `htop` | After io_uring or UCX changes |
| Memory bandwidth utilization | `pcm-memory` (Intel) or `perf mem` | After pipeline changes |
| Flamegraph (hot functions) | `perf record` + `inferno` | When optimizing specific calls |
| End-to-end ML workload | PyTorch benchmark with `torch.profiler` | After each phase milestone |

### Tracing Output Example

Using `tracing-chrome` subscriber, OutterLink generates a Chrome trace file (viewable in `chrome://tracing`):

```
outterlink-trace.json:

cuMemcpyHtoD (total: 85us)
  |-- serialize: 3us
  |-- batch_check: 1us (batchable, added to batch)

cuLaunchKernel (total: 2us)
  |-- serialize: 1us
  |-- batch_check: 1us (batchable, added to batch)

cuStreamSynchronize (total: 120us)
  |-- batch_flush: 80us
  |     |-- serialize_batch: 5us
  |     |-- network_roundtrip: 70us
  |     |-- deserialize_batch: 5us
  |-- wait_gpu: 40us
```

This visualization makes it immediately clear where time is spent.

---

## 6. Implementation Phases

### Phase 8a: Tracing Infrastructure (Do First)

**Files to create/modify:**
- `crates/outterlink-common/Cargo.toml` -- add `tracing`, `tracing-subscriber`, `tracing-chrome`
- `crates/outterlink-client/src/tracing_setup.rs` -- initialize tracing
- Instrument all existing intercepted calls with spans

**Acceptance criteria:**
- [ ] Chrome trace JSON generated for a test run
- [ ] Per-call overhead visible in trace

### Phase 8b: Call Batching

**Files to create/modify:**
- `crates/outterlink-client/src/batch.rs` -- batch buffer logic
- `crates/outterlink-common/src/protocol/batch.rs` -- batch message format
- `crates/outterlink-server/src/batch_handler.rs` -- process batch messages
- Modify all intercepted calls to go through batch buffer

**Acceptance criteria:**
- [ ] Batchable calls accumulate without network round trips
- [ ] Flush-forcing calls flush the batch first
- [ ] Benchmark shows >3x latency improvement for batchable sequences
- [ ] All existing tests still pass

### Phase 8c: io_uring Transport

**Files to create/modify:**
- `crates/outterlink-common/src/transport/iouring_transport.rs` -- new transport
- `crates/outterlink-common/src/transport/mod.rs` -- add iouring to transport selection
- `crates/outterlink-common/Cargo.toml` -- add `io-uring` crate dependency

**Acceptance criteria:**
- [ ] io_uring transport passes all existing tests
- [ ] Falls back to tokio TCP on kernels < 6.0
- [ ] Benchmark shows throughput improvement and CPU reduction

### Phase 8d: UCX Transport

**Files to create/modify:**
- `crates/outterlink-common/src/transport/ucx_transport.rs` -- new transport
- `crates/outterlink-common/Cargo.toml` -- add `ucx-sys` behind `ucx` feature

**Acceptance criteria:**
- [ ] UCX transport passes all existing tests
- [ ] Auto-negotiates RDMA when both sides have ConnectX
- [ ] Falls back gracefully when UCX is not installed

### Phase 8e: RDMA Pipeline

**Files to create/modify:**
- `crates/outterlink-common/src/transport/rdma_pipeline.rs` -- pipelined transfer
- Integrate with UCX or direct libibverbs RDMA backend

**Acceptance criteria:**
- [ ] Double-buffered pipeline overlaps stages
- [ ] Benchmark shows >2x throughput improvement for large transfers (>1MB)

---

## Test Plan

| Test | Expected |
|------|----------|
| Batch buffer accumulates batchable calls | No network traffic until flush |
| Flush-forcing call sends accumulated batch | All calls processed in order |
| io_uring transport: send 100MB | Throughput > 9 GB/s on 100GbE |
| io_uring fallback on old kernel | Tokio TCP used, no crash |
| UCX with RDMA hardware | RDMA transport selected |
| UCX without RDMA hardware | TCP transport selected |
| Pipeline 100MB GPU-to-GPU | Throughput > 6 GB/s (vs ~4 GB/s without pipeline) |
| Chrome trace generated | Valid JSON, viewable in chrome://tracing |
| All Phase 7 tests still pass | No regressions |

## Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| io_uring API changes between kernel versions | Build failure | Pin minimum kernel version, use `io-uring` crate (abstracts) |
| Handle prediction mismatch in batching | Silent corruption | Verification in flush response, assertion on mismatch |
| UCX build complexity | Hard to install | Feature-gated, optional, document build steps |
| Double-buffer pipeline adds latency for small transfers | Regression | Only use pipeline for transfers > threshold (e.g., 64KB) |
| `tracing` overhead in hot path | Performance impact | Use `tracing` with `release_max_level_info` to compile out debug spans |

## Estimated Scope

| Component | New Files | Modified Files | Complexity |
|-----------|-----------|----------------|-----------|
| Tracing infrastructure | 2 | 5-10 (instrument existing) | Low |
| Call batching | 3 | 5-10 (modify intercept paths) | High |
| io_uring transport | 1 | 2 | High |
| UCX transport | 1 | 2 | High |
| RDMA pipeline | 1 | 2 | High |
| Benchmarking tool | 1 | 1 | Medium |

## Related Documents

- [R4: ConnectX-5 + Transport Stack](../research/R4-connectx5-transport-stack.md)
- [Research Consolidation](../research/CONSOLIDATION-all-research.md)
- [P2: Development Environment](P2-dev-environment.md) (kernel version, UCX install)
- [Hardware Inventory](../pre-planning/01-hardware-inventory.md)
