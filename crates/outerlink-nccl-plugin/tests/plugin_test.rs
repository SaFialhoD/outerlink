//! Integration tests for the OuterLink NCCL Net Plugin.
//!
//! These tests exercise the plugin through the same FFI interface that NCCL
//! would use, verifying correctness of the handle lifecycle and data transfer.

use outerlink_nccl_plugin::ffi_types::*;
use outerlink_nccl_plugin::ncclNet_v8;

use std::ffi::{c_int, c_void, CStr};
use std::ptr;

// ---------------------------------------------------------------------------
// Symbol and initialization tests
// ---------------------------------------------------------------------------

#[test]
fn test_nccl_net_symbol_exists() {
    // Verify the symbol is accessible and the name field is correct
    let name_ptr = ncclNet_v8.name;
    assert!(!name_ptr.is_null());
    let name = unsafe { CStr::from_ptr(name_ptr) };
    assert_eq!(name.to_str().unwrap(), "OuterLink");
}

#[test]
fn test_init_returns_success() {
    let result = unsafe { (ncclNet_v8.init)(ptr::null_mut()) };
    assert_eq!(result, NCCL_SUCCESS);
}

#[test]
fn test_devices_returns_one() {
    let mut ndev: c_int = 0;
    let result = unsafe { (ncclNet_v8.devices)(&mut ndev) };
    assert_eq!(result, NCCL_SUCCESS);
    assert_eq!(ndev, 1);
}

#[test]
fn test_devices_null_pointer() {
    let result = unsafe { (ncclNet_v8.devices)(ptr::null_mut()) };
    assert_eq!(result, NCCL_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// Properties tests
// ---------------------------------------------------------------------------

#[test]
fn test_get_properties() {
    let mut props = unsafe { std::mem::zeroed::<NcclNetProperties_v8>() };
    let result = unsafe { (ncclNet_v8.getProperties)(0, &mut props) };
    assert_eq!(result, NCCL_SUCCESS);

    // Verify key properties
    let name = unsafe { CStr::from_ptr(props.name) };
    assert_eq!(name.to_str().unwrap(), "OuterLink-TCP");
    assert_eq!(props.speed, 100_000); // 100 Gbps
    assert_eq!(props.ptrSupport, NCCL_PTR_HOST);
    assert_eq!(props.maxComms, 65536);
    assert_eq!(props.maxRecvs, 8);
    assert_eq!(props.netDeviceType, 0); // NCCL_NET_DEVICE_HOST
}

#[test]
fn test_get_properties_invalid_device() {
    let mut props = unsafe { std::mem::zeroed::<NcclNetProperties_v8>() };
    let result = unsafe { (ncclNet_v8.getProperties)(1, &mut props) };
    assert_eq!(result, NCCL_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// Listen / Connect / Accept lifecycle
// ---------------------------------------------------------------------------

#[test]
fn test_listen_connect_accept_close() {
    // Step 1: Listen
    let mut handle_buf = [0u8; HANDLE_SIZE];
    let mut listen_comm: *mut c_void = ptr::null_mut();
    let result = unsafe {
        (ncclNet_v8.listen)(
            0,
            handle_buf.as_mut_ptr() as *mut c_void,
            &mut listen_comm,
        )
    };
    assert_eq!(result, NCCL_SUCCESS);
    assert!(!listen_comm.is_null());

    // Step 2: Connect (from "remote" side)
    // Connect must happen concurrently with accept (accept blocks).
    // We use usize to cross thread boundary since *mut c_void is not Send.
    let handle_copy = handle_buf;
    let connect_thread = std::thread::spawn(move || {
        let mut sc: *mut c_void = ptr::null_mut();
        let mut sdc: *mut c_void = ptr::null_mut();
        let r = unsafe {
            (ncclNet_v8.connect)(
                0,
                handle_copy.as_ptr() as *const c_void,
                &mut sc,
                &mut sdc,
            )
        };
        (r, sc as usize)
    });

    // Step 3: Accept
    let mut recv_comm: *mut c_void = ptr::null_mut();
    let mut recv_dev_comm: *mut c_void = ptr::null_mut();
    let result = unsafe {
        (ncclNet_v8.accept)(listen_comm, &mut recv_comm, &mut recv_dev_comm)
    };
    assert_eq!(result, NCCL_SUCCESS);
    assert!(!recv_comm.is_null());

    let (connect_result, send_comm_usize) = connect_thread.join().unwrap();
    assert_eq!(connect_result, NCCL_SUCCESS);
    assert_ne!(send_comm_usize, 0);
    let send_comm = send_comm_usize as *mut c_void;

    // Step 4: Close everything
    assert_eq!(unsafe { (ncclNet_v8.closeSend)(send_comm) }, NCCL_SUCCESS);
    assert_eq!(unsafe { (ncclNet_v8.closeRecv)(recv_comm) }, NCCL_SUCCESS);
    assert_eq!(
        unsafe { (ncclNet_v8.closeListen)(listen_comm) },
        NCCL_SUCCESS
    );
}

// ---------------------------------------------------------------------------
// Memory registration
// ---------------------------------------------------------------------------

#[test]
fn test_reg_dereg_mr() {
    // We need a comm handle for regMr (even though TCP ignores it)
    let fake_comm = 1usize as *mut c_void;

    let mut buffer = vec![0u8; 1024];
    let mut mr_handle: *mut c_void = ptr::null_mut();

    let result = unsafe {
        (ncclNet_v8.regMr)(
            fake_comm,
            buffer.as_mut_ptr() as *mut c_void,
            buffer.len(),
            NCCL_PTR_HOST,
            &mut mr_handle,
        )
    };
    assert_eq!(result, NCCL_SUCCESS);
    assert!(!mr_handle.is_null());

    // Deregister
    let result = unsafe { (ncclNet_v8.deregMr)(fake_comm, mr_handle) };
    assert_eq!(result, NCCL_SUCCESS);
}

#[test]
fn test_reg_mr_dma_buf() {
    let fake_comm = 1usize as *mut c_void;
    let mut buffer = vec![0u8; 512];
    let mut mr_handle: *mut c_void = ptr::null_mut();

    let result = unsafe {
        (ncclNet_v8.regMrDmaBuf)(
            fake_comm,
            buffer.as_mut_ptr() as *mut c_void,
            buffer.len(),
            NCCL_PTR_HOST,
            0,  // offset
            -1, // fd (not used for TCP)
            &mut mr_handle,
        )
    };
    assert_eq!(result, NCCL_SUCCESS);
    assert!(!mr_handle.is_null());

    let result = unsafe { (ncclNet_v8.deregMr)(fake_comm, mr_handle) };
    assert_eq!(result, NCCL_SUCCESS);
}

// ---------------------------------------------------------------------------
// Send / Recv / Test roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_send_recv_roundtrip() {
    // Set up a connection
    let mut handle_buf = [0u8; HANDLE_SIZE];
    let mut listen_comm: *mut c_void = ptr::null_mut();
    let result = unsafe {
        (ncclNet_v8.listen)(
            0,
            handle_buf.as_mut_ptr() as *mut c_void,
            &mut listen_comm,
        )
    };
    assert_eq!(result, NCCL_SUCCESS);

    let handle_copy = handle_buf;
    let connect_thread = std::thread::spawn(move || {
        let mut sc: *mut c_void = ptr::null_mut();
        let mut sdc: *mut c_void = ptr::null_mut();
        let r = unsafe {
            (ncclNet_v8.connect)(
                0,
                handle_copy.as_ptr() as *const c_void,
                &mut sc,
                &mut sdc,
            )
        };
        assert_eq!(r, NCCL_SUCCESS);
        sc as usize
    });

    let mut recv_comm: *mut c_void = ptr::null_mut();
    let mut recv_dev_comm: *mut c_void = ptr::null_mut();
    let result = unsafe {
        (ncclNet_v8.accept)(listen_comm, &mut recv_comm, &mut recv_dev_comm)
    };
    assert_eq!(result, NCCL_SUCCESS);

    let send_comm = connect_thread.join().unwrap() as *mut c_void;

    // Prepare data
    let send_data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
    let mut recv_data = vec![0u8; 256];

    // Register memory regions
    let fake_comm = 1usize as *mut c_void;
    let mut send_mr: *mut c_void = ptr::null_mut();
    let mut recv_mr: *mut c_void = ptr::null_mut();
    unsafe {
        (ncclNet_v8.regMr)(
            fake_comm,
            send_data.as_ptr() as *mut c_void,
            send_data.len(),
            NCCL_PTR_HOST,
            &mut send_mr,
        );
        (ncclNet_v8.regMr)(
            fake_comm,
            recv_data.as_mut_ptr() as *mut c_void,
            recv_data.len(),
            NCCL_PTR_HOST,
            &mut recv_mr,
        );
    }

    // Send
    let mut send_request: *mut c_void = ptr::null_mut();
    let result = unsafe {
        (ncclNet_v8.isend)(
            send_comm,
            send_data.as_ptr() as *mut c_void,
            send_data.len() as c_int,
            0, // tag
            send_mr,
            &mut send_request,
        )
    };
    assert_eq!(result, NCCL_SUCCESS);

    // Test send completion
    let mut done: c_int = 0;
    let mut size: c_int = 0;
    let result = unsafe { (ncclNet_v8.test)(send_request, &mut done, &mut size) };
    assert_eq!(result, NCCL_SUCCESS);
    assert_eq!(done, 1);
    assert_eq!(size, 256);

    // Recv
    let mut recv_request: *mut c_void = ptr::null_mut();
    let mut data_ptr = recv_data.as_mut_ptr() as *mut c_void;
    let mut recv_size: c_int = 256;
    let mut tag: c_int = 0;
    let result = unsafe {
        (ncclNet_v8.irecv)(
            recv_comm,
            1,
            &mut data_ptr,
            &mut recv_size,
            &mut tag,
            &mut recv_mr,
            &mut recv_request,
        )
    };
    assert_eq!(result, NCCL_SUCCESS);

    // Test recv completion
    done = 0;
    size = 0;
    let result = unsafe { (ncclNet_v8.test)(recv_request, &mut done, &mut size) };
    assert_eq!(result, NCCL_SUCCESS);
    assert_eq!(done, 1);
    assert_eq!(size, 256);

    // Verify data matches
    assert_eq!(send_data, recv_data);

    // Cleanup
    unsafe {
        (ncclNet_v8.deregMr)(fake_comm, send_mr);
        (ncclNet_v8.deregMr)(fake_comm, recv_mr);
        (ncclNet_v8.closeSend)(send_comm);
        (ncclNet_v8.closeRecv)(recv_comm);
        (ncclNet_v8.closeListen)(listen_comm);
    }
}

// ---------------------------------------------------------------------------
// Flush (no-op) test
// ---------------------------------------------------------------------------

#[test]
fn test_iflush_returns_success() {
    let mut request: *mut c_void = ptr::null_mut();
    let result = unsafe {
        (ncclNet_v8.iflush)(
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut request,
        )
    };
    assert_eq!(result, NCCL_SUCCESS);
    assert!(!request.is_null());

    // The flush request should be immediately done
    let mut done: c_int = 0;
    let mut size: c_int = 0;
    let result = unsafe { (ncclNet_v8.test)(request, &mut done, &mut size) };
    assert_eq!(result, NCCL_SUCCESS);
    assert_eq!(done, 1);
}

// ---------------------------------------------------------------------------
// Handle table unit tests
// ---------------------------------------------------------------------------

#[test]
fn test_handle_table_insert_get_remove() {
    use outerlink_nccl_plugin::handles::HandleTable;

    let table: HandleTable<String> = HandleTable::new();

    let id1 = table.insert("hello".to_string());
    let id2 = table.insert("world".to_string());
    assert_ne!(id1, id2);

    assert_eq!(*table.get(id1).unwrap(), "hello");
    assert_eq!(*table.get(id2).unwrap(), "world");

    let removed = table.remove(id1);
    assert_eq!(removed, Some("hello".to_string()));
    assert!(table.get(id1).is_none());

    // id2 still accessible
    assert_eq!(*table.get(id2).unwrap(), "world");
}

#[test]
fn test_handle_table_ids_start_at_one() {
    use outerlink_nccl_plugin::handles::HandleTable;

    let table: HandleTable<i32> = HandleTable::new();
    let id = table.insert(42);
    // ID should be >= 1 (0 is reserved as null)
    assert!(id >= 1);
}
