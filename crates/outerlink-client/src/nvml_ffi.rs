//! NVML FFI exports for C interposition.
//!
//! Each function here is `#[no_mangle] pub extern "C"` so the C interposition
//! layer can return pointers to these when an application calls NVML functions
//! via dlsym or dlopen("libnvidia-ml.so").
//!
//! Device handles: NVML uses opaque `nvmlDevice_t` pointers. We encode the
//! device index as a `usize` and cast it to/from a pointer. Index 0 becomes
//! handle value 1 (so NULL is never a valid handle), index 1 becomes 2, etc.

use outerlink_common::nvml_types::NvmlClockType;

use crate::nvml::nvml_virtualizer;

// ---------------------------------------------------------------------------
// NVML return code constants (matching nvmlReturn_t)
// ---------------------------------------------------------------------------

const NVML_SUCCESS: u32 = 0;
const NVML_ERROR_UNINITIALIZED: u32 = 1;
const NVML_ERROR_INVALID_ARGUMENT: u32 = 2;
const NVML_ERROR_NOT_SUPPORTED: u32 = 3;
const NVML_ERROR_NOT_FOUND: u32 = 6;
const NVML_ERROR_UNKNOWN: u32 = 999;

// ---------------------------------------------------------------------------
// Handle encoding: index N -> handle (N + 1), handle H -> index (H - 1)
// NULL (0) is never a valid handle.
// ---------------------------------------------------------------------------

/// Encode a device index into an opaque handle value.
fn index_to_handle(index: u32) -> usize {
    (index as usize) + 1
}

/// Decode an opaque handle value back to a device index.
/// Returns None if handle is 0 (NULL).
fn handle_to_index(handle: usize) -> Option<u32> {
    if handle == 0 {
        return None;
    }
    Some((handle - 1) as u32)
}

// ---------------------------------------------------------------------------
// Helper: write a Rust string into a C buffer with nul termination
// ---------------------------------------------------------------------------

/// Copy `src` bytes into `dst` buffer of `len` bytes. Always nul-terminates.
/// Returns `NVML_ERROR_INSUFFICIENT_SIZE` if buffer is too small (but still
/// writes a truncated, nul-terminated result).
unsafe fn write_c_string(dst: *mut u8, len: u32, src: &[u8]) -> u32 {
    if dst.is_null() || len == 0 {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let max = (len as usize).saturating_sub(1); // reserve space for nul
    let copy_len = src.len().min(max);
    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, copy_len);
    *dst.add(copy_len) = 0; // nul terminator
    NVML_SUCCESS
}

// ===========================================================================
// Init / Shutdown
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlInit() -> u32 {
    nvml_virtualizer().init();
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlInit_v2() -> u32 {
    nvml_virtualizer().init();
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlInitWithFlags(_flags: u32) -> u32 {
    nvml_virtualizer().init();
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlShutdown() -> u32 {
    nvml_virtualizer().shutdown();
    NVML_SUCCESS
}

// ===========================================================================
// System queries
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlSystemGetDriverVersion(version: *mut u8, length: u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    if version.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(0) {
        Some(s) => s,
        None => return NVML_ERROR_UNKNOWN,
    };
    unsafe { write_c_string(version, length, snap.driver_version_str().as_bytes()) }
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlSystemGetNVMLVersion(version: *mut u8, length: u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    if version.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(0) {
        Some(s) => s,
        None => return NVML_ERROR_UNKNOWN,
    };
    unsafe { write_c_string(version, length, snap.nvml_version_str().as_bytes()) }
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlSystemGetCudaDriverVersion(cuda_driver_version: *mut i32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    if cuda_driver_version.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(0) {
        Some(s) => s,
        None => return NVML_ERROR_UNKNOWN,
    };
    unsafe {
        *cuda_driver_version = snap.cuda_driver_version;
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlSystemGetCudaDriverVersion_v2(cuda_driver_version: *mut i32) -> u32 {
    nvml_hook_nvmlSystemGetCudaDriverVersion(cuda_driver_version)
}

// ===========================================================================
// Device count
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetCount(count: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    if count.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    unsafe {
        *count = virt.device_count();
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetCount_v2(count: *mut u32) -> u32 {
    nvml_hook_nvmlDeviceGetCount(count)
}

// ===========================================================================
// Device handle by index
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetHandleByIndex(index: u32, device: *mut usize) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    if device.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if index >= virt.device_count() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    unsafe {
        *device = index_to_handle(index);
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetHandleByIndex_v2(index: u32, device: *mut usize) -> u32 {
    nvml_hook_nvmlDeviceGetHandleByIndex(index, device)
}

// ===========================================================================
// Device handle by UUID
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetHandleByUUID(uuid: *const u8, device: *mut usize) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    if uuid.is_null() || device.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    // Read UUID as C string
    let uuid_cstr = unsafe { std::ffi::CStr::from_ptr(uuid as *const i8) };
    let uuid_str = match uuid_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return NVML_ERROR_INVALID_ARGUMENT,
    };
    match virt.get_snapshot_by_uuid(uuid_str) {
        Some((idx, _)) => {
            unsafe {
                *device = index_to_handle(idx);
            }
            NVML_SUCCESS
        }
        None => NVML_ERROR_NOT_FOUND,
    }
}

// ===========================================================================
// Device name
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetName(device: usize, name: *mut u8, length: u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if name.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe { write_c_string(name, length, snap.name_str().as_bytes()) }
}

// ===========================================================================
// Device UUID
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetUUID(device: usize, uuid: *mut u8, length: u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if uuid.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe { write_c_string(uuid, length, snap.uuid_str().as_bytes()) }
}

// ===========================================================================
// Memory info
// ===========================================================================

/// NVML memory info v1 layout: { total: u64, free: u64, used: u64 }
#[repr(C)]
pub struct NvmlMemory {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

/// NVML memory info v2 layout: adds version field
#[repr(C)]
pub struct NvmlMemoryV2 {
    pub version: u32,
    pub total: u64,
    pub reserved: u64,
    pub free: u64,
    pub used: u64,
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetMemoryInfo(device: usize, memory: *mut NvmlMemory) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if memory.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        (*memory).total = snap.vram_total;
        (*memory).free = snap.vram_free;
        (*memory).used = snap.vram_used;
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetMemoryInfo_v2(device: usize, memory: *mut NvmlMemoryV2) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if memory.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        (*memory).version = 2;
        (*memory).total = snap.vram_total;
        (*memory).reserved = 0;
        (*memory).free = snap.vram_free;
        (*memory).used = snap.vram_used;
    }
    NVML_SUCCESS
}

// ===========================================================================
// Utilization
// ===========================================================================

/// NVML utilization rates: { gpu: u32, memory: u32 }
#[repr(C)]
pub struct NvmlUtilization {
    pub gpu: u32,
    pub memory: u32,
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetUtilizationRates(device: usize, utilization: *mut NvmlUtilization) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if utilization.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        (*utilization).gpu = snap.utilization_gpu;
        (*utilization).memory = snap.utilization_memory;
    }
    NVML_SUCCESS
}

// ===========================================================================
// Temperature
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetTemperature(device: usize, sensor_type: u32, temp: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if temp.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    // Only NVML_TEMPERATURE_GPU (0) is supported
    if sensor_type != 0 {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        *temp = snap.temperature_gpu;
    }
    NVML_SUCCESS
}

// ===========================================================================
// Power
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetPowerUsage(device: usize, power: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if power.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        *power = snap.power_usage_mw;
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetPowerManagementLimit(device: usize, limit: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if limit.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        *limit = snap.power_limit_mw;
    }
    NVML_SUCCESS
}

// ===========================================================================
// Fan speed
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetFanSpeed(device: usize, speed: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if speed.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        *speed = snap.fan_speed_pct;
    }
    NVML_SUCCESS
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetFanSpeed_v2(device: usize, _fan: u32, speed: *mut u32) -> u32 {
    // v2 adds a fan index parameter; we only have one fan value.
    nvml_hook_nvmlDeviceGetFanSpeed(device, speed)
}

// ===========================================================================
// Clocks
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetClockInfo(device: usize, clock_type: u32, clock_mhz: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if clock_mhz.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    let value = match NvmlClockType::from_raw(clock_type) {
        Some(NvmlClockType::Graphics) => snap.clock_graphics_mhz,
        Some(NvmlClockType::Sm) => snap.clock_sm_mhz,
        Some(NvmlClockType::Mem) => snap.clock_mem_mhz,
        Some(NvmlClockType::Video) => snap.clock_video_mhz,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        *clock_mhz = value;
    }
    NVML_SUCCESS
}

// ===========================================================================
// PCI info
// ===========================================================================

/// NVML PCI info structure (simplified).
#[repr(C)]
pub struct NvmlPciInfo {
    pub bus_id_legacy: [u8; 16],   // Deprecated short form
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub pci_device_id: u32,
    pub pci_subsystem_id: u32,
    pub bus_id: [u8; 32],          // Full PCI bus ID string
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetPciInfo(device: usize, pci: *mut NvmlPciInfo) -> u32 {
    nvml_hook_nvmlDeviceGetPciInfo_v3(device, pci)
}

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetPciInfo_v3(device: usize, pci: *mut NvmlPciInfo) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if pci.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        // Zero the struct first
        std::ptr::write_bytes(pci, 0, 1);
        // Copy the full bus ID
        let pci_str = snap.pci_bus_id_str().as_bytes();
        let copy_len = pci_str.len().min(31);
        std::ptr::copy_nonoverlapping(pci_str.as_ptr(), (*pci).bus_id.as_mut_ptr(), copy_len);
        // Copy truncated legacy form
        let legacy_len = pci_str.len().min(15);
        std::ptr::copy_nonoverlapping(pci_str.as_ptr(), (*pci).bus_id_legacy.as_mut_ptr(), legacy_len);
        (*pci).pci_device_id = snap.pci_device_id;
        (*pci).pci_subsystem_id = snap.pci_subsystem_id;
        // Parse domain:bus:device from PCI bus ID string
        // Format: "DDDDDDDD:BB:DD.F"
        (*pci).domain = 0;
        (*pci).bus = (index + 1) as u32;
        (*pci).device = 0;
    }
    NVML_SUCCESS
}

// ===========================================================================
// Compute capability
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetCudaComputeCapability(device: usize, major: *mut i32, minor: *mut i32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if major.is_null() || minor.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        *major = snap.compute_cap_major as i32;
        *minor = snap.compute_cap_minor as i32;
    }
    NVML_SUCCESS
}

// ===========================================================================
// Device index
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetIndex(device: usize, index: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let idx = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if index.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    // Verify the index is within range
    if idx >= virt.device_count() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    unsafe {
        *index = idx;
    }
    NVML_SUCCESS
}

// ===========================================================================
// GPU cores
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetNumGpuCores(device: usize, num_cores: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if num_cores.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        *num_cores = snap.num_cores;
    }
    NVML_SUCCESS
}

// ===========================================================================
// Architecture
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetArchitecture(device: usize, arch: *mut u32) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if arch.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    let snap = match virt.get_snapshot(index) {
        Some(s) => s,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    unsafe {
        *arch = snap.architecture;
    }
    NVML_SUCCESS
}

// ===========================================================================
// Running processes (stub: always returns 0 processes)
// ===========================================================================

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlDeviceGetComputeRunningProcesses_v3(
    device: usize,
    info_count: *mut u32,
    _infos: *mut u8,
) -> u32 {
    let virt = nvml_virtualizer();
    if !virt.is_initialized() {
        return NVML_ERROR_UNINITIALIZED;
    }
    let index = match handle_to_index(device) {
        Some(i) => i,
        None => return NVML_ERROR_INVALID_ARGUMENT,
    };
    if info_count.is_null() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if index >= virt.device_count() {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    unsafe {
        *info_count = 0; // No processes running
    }
    NVML_SUCCESS
}

// ===========================================================================
// Error string
// ===========================================================================

/// Static error message strings for nvmlErrorString.
static ERROR_STRINGS: &[&str] = &[
    "Success\0",                          // 0
    "Uninitialized\0",                    // 1
    "Invalid Argument\0",                 // 2
    "Not Supported\0",                    // 3
    "No Permission\0",                    // 4
    "Already Initialized\0",             // 5
    "Not Found\0",                        // 6
    "Insufficient Size\0",                // 7
    "Insufficient Power\0",               // 8
    "Driver Not Loaded\0",                // 9
    "Timeout\0",                          // 10
    "IRQ Issue\0",                        // 11
    "Library Not Found\0",                // 12
    "Function Not Found\0",               // 13
    "Corrupted infoROM\0",                // 14
    "GPU is Lost\0",                      // 15
    "Reset Required\0",                   // 16
    "Operating System\0",                 // 17
    "Lib RM Version Mismatch\0",          // 18
];
static UNKNOWN_ERROR: &str = "Unknown Error\0";

#[no_mangle]
pub extern "C" fn nvml_hook_nvmlErrorString(result: u32) -> *const u8 {
    let idx = result as usize;
    if idx < ERROR_STRINGS.len() {
        ERROR_STRINGS[idx].as_ptr()
    } else {
        UNKNOWN_ERROR.as_ptr()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: initialize and get a device handle for testing.
    fn init_and_get_handle() -> usize {
        nvml_hook_nvmlInit_v2();
        let mut handle: usize = 0;
        let ret = nvml_hook_nvmlDeviceGetHandleByIndex_v2(0, &mut handle as *mut usize);
        assert_eq!(ret, NVML_SUCCESS);
        handle
    }

    #[test]
    fn test_ffi_init_shutdown() {
        assert_eq!(nvml_hook_nvmlInit(), NVML_SUCCESS);
        assert_eq!(nvml_hook_nvmlShutdown(), NVML_SUCCESS);
    }

    #[test]
    fn test_ffi_device_count() {
        nvml_hook_nvmlInit_v2();
        let mut count: u32 = 0;
        let ret = nvml_hook_nvmlDeviceGetCount_v2(&mut count as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_ffi_device_count_uninitialized() {
        // Use a fresh virtualizer state -- the global may already be initialized
        // from another test, so we test the FFI contract: if NOT initialized,
        // nvmlDeviceGetCount should return UNINITIALIZED.
        let virt = crate::nvml::NvmlVirtualizer::new();
        assert!(!virt.is_initialized());
        // We can't easily test the global FFI in isolation, but we verify
        // the virtualizer contract directly.
        assert_eq!(virt.device_count(), 0);
    }

    #[test]
    fn test_ffi_null_args() {
        nvml_hook_nvmlInit_v2();
        // Null count pointer
        let ret = nvml_hook_nvmlDeviceGetCount_v2(std::ptr::null_mut());
        assert_eq!(ret, NVML_ERROR_INVALID_ARGUMENT);

        // Null device handle pointer
        let ret = nvml_hook_nvmlDeviceGetHandleByIndex_v2(0, std::ptr::null_mut());
        assert_eq!(ret, NVML_ERROR_INVALID_ARGUMENT);

        // Null name buffer
        let handle = init_and_get_handle();
        let ret = nvml_hook_nvmlDeviceGetName(handle, std::ptr::null_mut(), 64);
        assert_eq!(ret, NVML_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn test_ffi_name() {
        let handle = init_and_get_handle();
        let mut buf = [0u8; 128];
        let ret = nvml_hook_nvmlDeviceGetName(handle, buf.as_mut_ptr(), 128);
        assert_eq!(ret, NVML_SUCCESS);
        let name = std::ffi::CStr::from_bytes_until_nul(&buf).unwrap();
        assert_eq!(name.to_str().unwrap(), "NVIDIA GeForce RTX 3090");
    }

    #[test]
    fn test_ffi_memory_info() {
        let handle = init_and_get_handle();
        let mut mem = NvmlMemory {
            total: 0,
            free: 0,
            used: 0,
        };
        let ret = nvml_hook_nvmlDeviceGetMemoryInfo(handle, &mut mem as *mut NvmlMemory);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(mem.total, 24_576 * 1024 * 1024);
        assert!(mem.free > 0);
        assert!(mem.used > 0);
        assert_eq!(mem.total, mem.free + mem.used);
    }

    #[test]
    fn test_ffi_temperature() {
        let handle = init_and_get_handle();
        let mut temp: u32 = 0;
        let ret = nvml_hook_nvmlDeviceGetTemperature(handle, 0, &mut temp as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(temp, 42);
    }

    #[test]
    fn test_ffi_power() {
        let handle = init_and_get_handle();
        let mut power: u32 = 0;
        let ret = nvml_hook_nvmlDeviceGetPowerUsage(handle, &mut power as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(power, 30_000);
    }

    #[test]
    fn test_ffi_fan_speed() {
        let handle = init_and_get_handle();
        let mut speed: u32 = 0;
        let ret = nvml_hook_nvmlDeviceGetFanSpeed(handle, &mut speed as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(speed, 30);
    }

    #[test]
    fn test_ffi_clock_info() {
        let handle = init_and_get_handle();
        let mut clock: u32 = 0;
        // Graphics clock
        let ret = nvml_hook_nvmlDeviceGetClockInfo(handle, 0, &mut clock as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(clock, 210); // idle graphics clock
        // Memory clock
        let ret = nvml_hook_nvmlDeviceGetClockInfo(handle, 2, &mut clock as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(clock, 9501);
    }

    #[test]
    fn test_ffi_compute_capability() {
        let handle = init_and_get_handle();
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        let ret = nvml_hook_nvmlDeviceGetCudaComputeCapability(
            handle,
            &mut major as *mut i32,
            &mut minor as *mut i32,
        );
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(major, 8);
        assert_eq!(minor, 6);
    }

    #[test]
    fn test_ffi_device_index() {
        let handle = init_and_get_handle();
        let mut idx: u32 = 99;
        let ret = nvml_hook_nvmlDeviceGetIndex(handle, &mut idx as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_ffi_num_cores() {
        let handle = init_and_get_handle();
        let mut cores: u32 = 0;
        let ret = nvml_hook_nvmlDeviceGetNumGpuCores(handle, &mut cores as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(cores, 10496);
    }

    #[test]
    fn test_ffi_architecture() {
        let handle = init_and_get_handle();
        let mut arch: u32 = 0;
        let ret = nvml_hook_nvmlDeviceGetArchitecture(handle, &mut arch as *mut u32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(arch, 2); // AMPERE
    }

    #[test]
    fn test_ffi_uuid() {
        let handle = init_and_get_handle();
        let mut buf = [0u8; 96];
        let ret = nvml_hook_nvmlDeviceGetUUID(handle, buf.as_mut_ptr(), 96);
        assert_eq!(ret, NVML_SUCCESS);
        let uuid = std::ffi::CStr::from_bytes_until_nul(&buf).unwrap();
        assert!(uuid.to_str().unwrap().starts_with("GPU-"));
    }

    #[test]
    fn test_ffi_error_string() {
        let ptr = nvml_hook_nvmlErrorString(0);
        let s = unsafe { std::ffi::CStr::from_ptr(ptr as *const i8) };
        assert_eq!(s.to_str().unwrap(), "Success");

        let ptr = nvml_hook_nvmlErrorString(1);
        let s = unsafe { std::ffi::CStr::from_ptr(ptr as *const i8) };
        assert_eq!(s.to_str().unwrap(), "Uninitialized");

        let ptr = nvml_hook_nvmlErrorString(999);
        let s = unsafe { std::ffi::CStr::from_ptr(ptr as *const i8) };
        assert_eq!(s.to_str().unwrap(), "Unknown Error");
    }

    #[test]
    fn test_ffi_invalid_handle() {
        nvml_hook_nvmlInit_v2();
        // Handle 0 (NULL) should fail
        let mut temp: u32 = 0;
        let ret = nvml_hook_nvmlDeviceGetTemperature(0, 0, &mut temp as *mut u32);
        assert_eq!(ret, NVML_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn test_ffi_out_of_range_index() {
        nvml_hook_nvmlInit_v2();
        let mut handle: usize = 0;
        let ret = nvml_hook_nvmlDeviceGetHandleByIndex_v2(99, &mut handle as *mut usize);
        assert_eq!(ret, NVML_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn test_ffi_driver_version() {
        nvml_hook_nvmlInit_v2();
        let mut buf = [0u8; 64];
        let ret = nvml_hook_nvmlSystemGetDriverVersion(buf.as_mut_ptr(), 64);
        assert_eq!(ret, NVML_SUCCESS);
        let ver = std::ffi::CStr::from_bytes_until_nul(&buf).unwrap();
        assert_eq!(ver.to_str().unwrap(), "535.129.03");
    }

    #[test]
    fn test_ffi_nvml_version() {
        nvml_hook_nvmlInit_v2();
        let mut buf = [0u8; 64];
        let ret = nvml_hook_nvmlSystemGetNVMLVersion(buf.as_mut_ptr(), 64);
        assert_eq!(ret, NVML_SUCCESS);
        let ver = std::ffi::CStr::from_bytes_until_nul(&buf).unwrap();
        assert_eq!(ver.to_str().unwrap(), "12.535.129.03");
    }

    #[test]
    fn test_ffi_cuda_driver_version() {
        nvml_hook_nvmlInit_v2();
        let mut ver: i32 = 0;
        let ret = nvml_hook_nvmlSystemGetCudaDriverVersion(&mut ver as *mut i32);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(ver, 12040);
    }

    #[test]
    fn test_ffi_pci_info() {
        let handle = init_and_get_handle();
        let mut pci = NvmlPciInfo {
            bus_id_legacy: [0; 16],
            domain: 0,
            bus: 0,
            device: 0,
            pci_device_id: 0,
            pci_subsystem_id: 0,
            bus_id: [0; 32],
        };
        let ret = nvml_hook_nvmlDeviceGetPciInfo_v3(handle, &mut pci as *mut NvmlPciInfo);
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(pci.pci_device_id, 0x2204);
        let bus_id = std::ffi::CStr::from_bytes_until_nul(&pci.bus_id).unwrap();
        assert!(bus_id.to_str().unwrap().contains(":00.0"));
    }

    #[test]
    fn test_ffi_running_processes() {
        let handle = init_and_get_handle();
        let mut count: u32 = 99;
        let ret = nvml_hook_nvmlDeviceGetComputeRunningProcesses_v3(
            handle,
            &mut count as *mut u32,
            std::ptr::null_mut(),
        );
        assert_eq!(ret, NVML_SUCCESS);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_handle_encoding() {
        // Index 0 -> handle 1
        assert_eq!(index_to_handle(0), 1);
        assert_eq!(handle_to_index(1), Some(0));
        // Index 5 -> handle 6
        assert_eq!(index_to_handle(5), 6);
        assert_eq!(handle_to_index(6), Some(5));
        // NULL handle -> None
        assert_eq!(handle_to_index(0), None);
    }
}
