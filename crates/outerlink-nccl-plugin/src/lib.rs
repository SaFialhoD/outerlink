//! OuterLink NCCL Net Plugin (v8)
//!
//! This crate produces a `cdylib` that NCCL discovers via the `ncclNet_v8`
//! symbol. It implements the 19-function NCCL Net Plugin API, using TCP
//! transport in Phase 1 with a path toward RDMA/OpenDMA in later phases.
//!
//! # Architecture
//!
//! - `ffi_types` -- C-compatible struct and constant definitions
//! - `handles`   -- Safe handle table mapping opaque pointers to Rust state
//! - `plugin`    -- The actual function implementations

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod ffi_types;
pub mod handles;
pub mod plugin;

/// The symbol NCCL looks for when loading a net plugin.
///
/// This must be a `#[no_mangle]` static with exactly the name `ncclNet_v8`.
/// NCCL loads the shared library, does `dlsym("ncclNet_v8")`, and casts
/// the result to its internal `ncclNet_v8_t*`.
#[no_mangle]
pub static ncclNet_v8: ffi_types::NcclNet_v8 = ffi_types::NcclNet_v8 {
    name: b"OuterLink\0" as *const u8 as *const i8,
    init: plugin::net_init,
    devices: plugin::net_devices,
    getProperties: plugin::net_get_properties,
    listen: plugin::net_listen,
    connect: plugin::net_connect,
    accept: plugin::net_accept,
    regMr: plugin::net_reg_mr,
    regMrDmaBuf: plugin::net_reg_mr_dma_buf,
    deregMr: plugin::net_dereg_mr,
    isend: plugin::net_isend,
    irecv: plugin::net_irecv,
    iflush: plugin::net_iflush,
    test: plugin::net_test,
    closeSend: plugin::net_close_send,
    closeRecv: plugin::net_close_recv,
    closeListen: plugin::net_close_listen,
    getDeviceMr: None,
    irecvConsumed: None,
};
