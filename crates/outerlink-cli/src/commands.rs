//! CLI command implementations.
//!
//! Each command connects to the OuterLink server via TCP, sends the
//! appropriate protocol messages, and returns structured results. The
//! caller (main.rs) decides whether to format as a table or JSON.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;

use outerlink_common::cuda_types::CuResult;
use outerlink_common::protocol::{MessageHeader, MessageType};
use outerlink_common::tcp_transport::TcpTransportConnection;
use outerlink_common::transport::TransportConnection;

// ---------------------------------------------------------------------------
// Data types returned by commands
// ---------------------------------------------------------------------------

/// Information about a single GPU, returned by `cmd_list`.
#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub id: i32,
    pub name: String,
    pub total_mem_bytes: u64,
    pub compute_major: i32,
    pub compute_minor: i32,
}

/// Per-GPU memory status, returned as part of `StatusInfo`.
#[derive(Debug, Clone, Serialize)]
pub struct GpuStatus {
    pub id: i32,
    pub name: String,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

/// Server status information, returned by `cmd_status`.
#[derive(Debug, Clone, Serialize)]
pub struct StatusInfo {
    pub driver_version: i32,
    pub latency_ms: f64,
    pub gpus: Vec<GpuStatus>,
}

/// A single transfer measurement.
#[derive(Debug, Clone, Serialize)]
pub struct TransferEntry {
    pub size_bytes: usize,
    pub duration_ms: f64,
    pub throughput_mb_s: f64,
}

/// Benchmark results, returned by `cmd_bench`.
#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub htod: Vec<TransferEntry>,
    pub dtoh: Vec<TransferEntry>,
    pub latency_avg_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_samples: usize,
}

// ---------------------------------------------------------------------------
// Wire-protocol helpers
// ---------------------------------------------------------------------------

/// Connect to the server and return a transport connection.
async fn connect(addr: &str) -> Result<TcpTransportConnection> {
    let stream = tokio::net::TcpStream::connect(addr)
        .await
        .with_context(|| format!("failed to connect to {addr}"))?;
    let conn = TcpTransportConnection::new(stream)
        .with_context(|| "failed to initialise transport connection")?;
    Ok(conn)
}

/// Send a request and receive its response. Returns `(response_header, payload)`.
/// Verifies the response type and request_id match.
async fn roundtrip(
    conn: &TcpTransportConnection,
    msg_type: MessageType,
    payload: &[u8],
    request_id: u64,
) -> Result<Vec<u8>> {
    let header = MessageHeader::new_request(request_id, msg_type, payload.len() as u32);
    conn.send_message(&header, payload)
        .await
        .context("failed to send request")?;
    let (resp_hdr, resp_payload) = conn
        .recv_message()
        .await
        .context("failed to receive response")?;
    anyhow::ensure!(
        resp_hdr.msg_type == MessageType::Response,
        "unexpected response type: {:?}",
        resp_hdr.msg_type
    );
    anyhow::ensure!(
        resp_hdr.request_id == request_id,
        "request_id mismatch: sent {request_id}, got {}",
        resp_hdr.request_id
    );
    Ok(resp_payload)
}

/// Extract the CuResult from the first 4 bytes and return the data portion.
/// Returns an error if the CUDA call failed.
fn check_success(payload: &[u8]) -> Result<&[u8]> {
    anyhow::ensure!(payload.len() >= 4, "response payload too short");
    let result = CuResult::from_raw(u32::from_le_bytes(payload[..4].try_into().unwrap()));
    anyhow::ensure!(
        result == CuResult::Success,
        "CUDA operation failed: {:?}",
        result
    );
    Ok(&payload[4..])
}

// ---------------------------------------------------------------------------
// Command: list
// ---------------------------------------------------------------------------

/// Query the server for GPU inventory and return structured info.
pub async fn cmd_list(addr: &str) -> Result<Vec<GpuInfo>> {
    let conn = connect(addr).await?;
    let mut rid: u64 = 1;

    // 1. Handshake
    let payload = roundtrip(&conn, MessageType::Handshake, &[], rid).await?;
    check_success(&payload)?;
    rid += 1;

    // 2. DeviceGetCount
    let payload = roundtrip(&conn, MessageType::DeviceGetCount, &[], rid).await?;
    let data = check_success(&payload)?;
    let count = i32::from_le_bytes(data[..4].try_into().unwrap());
    rid += 1;

    let mut gpus = Vec::with_capacity(count as usize);

    for device in 0..count {
        let dev_bytes = device.to_le_bytes();

        // DeviceGetName
        let payload = roundtrip(&conn, MessageType::DeviceGetName, &dev_bytes, rid).await?;
        let data = check_success(&payload)?;
        let name_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
        let name = std::str::from_utf8(&data[4..4 + name_len])
            .context("device name is not valid UTF-8")?
            .to_string();
        rid += 1;

        // DeviceTotalMem
        let payload = roundtrip(&conn, MessageType::DeviceTotalMem, &dev_bytes, rid).await?;
        let data = check_success(&payload)?;
        let total_mem_bytes = u64::from_le_bytes(data[..8].try_into().unwrap());
        rid += 1;

        // DeviceGetAttribute: ComputeCapabilityMajor (attrib=75)
        let mut attr_payload = 75_i32.to_le_bytes().to_vec();
        attr_payload.extend_from_slice(&dev_bytes);
        let payload = roundtrip(&conn, MessageType::DeviceGetAttribute, &attr_payload, rid).await?;
        let data = check_success(&payload)?;
        let compute_major = i32::from_le_bytes(data[..4].try_into().unwrap());
        rid += 1;

        // DeviceGetAttribute: ComputeCapabilityMinor (attrib=76)
        let mut attr_payload = 76_i32.to_le_bytes().to_vec();
        attr_payload.extend_from_slice(&dev_bytes);
        let payload = roundtrip(&conn, MessageType::DeviceGetAttribute, &attr_payload, rid).await?;
        let data = check_success(&payload)?;
        let compute_minor = i32::from_le_bytes(data[..4].try_into().unwrap());
        rid += 1;

        gpus.push(GpuInfo {
            id: device,
            name,
            total_mem_bytes,
            compute_major,
            compute_minor,
        });
    }

    Ok(gpus)
}

// ---------------------------------------------------------------------------
// Command: status
// ---------------------------------------------------------------------------

/// Query the server for health status including VRAM usage per GPU.
pub async fn cmd_status(addr: &str) -> Result<StatusInfo> {
    let start = Instant::now();
    let conn = connect(addr).await?;
    let mut rid: u64 = 1;

    // 1. Handshake (measure latency including connect)
    let payload = roundtrip(&conn, MessageType::Handshake, &[], rid).await?;
    check_success(&payload)?;
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    rid += 1;

    // 2. Init
    let init_payload = 0_u32.to_le_bytes();
    let payload = roundtrip(&conn, MessageType::Init, &init_payload, rid).await?;
    check_success(&payload)?;
    rid += 1;

    // 3. DriverGetVersion
    let payload = roundtrip(&conn, MessageType::DriverGetVersion, &[], rid).await?;
    let data = check_success(&payload)?;
    let driver_version = i32::from_le_bytes(data[..4].try_into().unwrap());
    rid += 1;

    // 4. DeviceGetCount
    let payload = roundtrip(&conn, MessageType::DeviceGetCount, &[], rid).await?;
    let data = check_success(&payload)?;
    let count = i32::from_le_bytes(data[..4].try_into().unwrap());
    rid += 1;

    let mut gpus = Vec::with_capacity(count as usize);

    for device in 0..count {
        let dev_bytes = device.to_le_bytes();

        // DeviceGetName
        let payload = roundtrip(&conn, MessageType::DeviceGetName, &dev_bytes, rid).await?;
        let data = check_success(&payload)?;
        let name_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
        let name = std::str::from_utf8(&data[4..4 + name_len])
            .context("device name is not valid UTF-8")?
            .to_string();
        rid += 1;

        // CtxCreate for this device (needed for MemGetInfo)
        let mut ctx_payload = 0_u32.to_le_bytes().to_vec();
        ctx_payload.extend_from_slice(&dev_bytes);
        let payload = roundtrip(&conn, MessageType::CtxCreate, &ctx_payload, rid).await?;
        let data = check_success(&payload)?;
        let ctx_handle = u64::from_le_bytes(data[..8].try_into().unwrap());
        rid += 1;

        // MemGetInfo
        let payload = roundtrip(&conn, MessageType::MemGetInfo, &[], rid).await?;
        let data = check_success(&payload)?;
        let free_bytes = u64::from_le_bytes(data[..8].try_into().unwrap());
        let total_bytes = u64::from_le_bytes(data[8..16].try_into().unwrap());
        rid += 1;

        // CtxDestroy
        let ctx_destroy_payload = ctx_handle.to_le_bytes();
        let payload = roundtrip(&conn, MessageType::CtxDestroy, &ctx_destroy_payload, rid).await?;
        check_success(&payload)?;
        rid += 1;

        gpus.push(GpuStatus {
            id: device,
            name,
            free_bytes,
            total_bytes,
        });
    }

    Ok(StatusInfo {
        driver_version,
        latency_ms,
        gpus,
    })
}

// ---------------------------------------------------------------------------
// Command: bench
// ---------------------------------------------------------------------------

/// Run transfer benchmarks. If `size_mb` is Some, benchmark that single size.
/// Otherwise, benchmark a standard set of sizes.
pub async fn cmd_bench(addr: &str, custom_size: Option<usize>) -> Result<BenchResult> {
    let conn = connect(addr).await?;
    let mut rid: u64 = 1;

    // 1. Handshake
    let payload = roundtrip(&conn, MessageType::Handshake, &[], rid).await?;
    check_success(&payload)?;
    rid += 1;

    // 2. Determine sizes to benchmark
    let sizes: Vec<usize> = if let Some(s) = custom_size {
        vec![s]
    } else {
        vec![
            1024,             // 1 KiB
            64 * 1024,        // 64 KiB
            1024 * 1024,      // 1 MiB
            16 * 1024 * 1024, // 16 MiB
            64 * 1024 * 1024, // 64 MiB
        ]
    };

    let mut htod_entries = Vec::new();
    let mut dtoh_entries = Vec::new();

    for &size in &sizes {
        // Allocate
        let alloc_payload = (size as u64).to_le_bytes();
        let payload = roundtrip(&conn, MessageType::MemAlloc, &alloc_payload, rid).await?;
        let data = check_success(&payload)?;
        let device_ptr = u64::from_le_bytes(data[..8].try_into().unwrap());
        rid += 1;

        // Prepare test data
        let test_data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();

        // Time MemcpyHtoD
        let mut htod_payload = device_ptr.to_le_bytes().to_vec();
        htod_payload.extend_from_slice(&test_data);

        let htod_start = Instant::now();
        let payload = roundtrip(&conn, MessageType::MemcpyHtoD, &htod_payload, rid).await?;
        let htod_elapsed = htod_start.elapsed();
        check_success(&payload)?;
        rid += 1;

        // Time MemcpyDtoH
        let mut dtoh_payload = device_ptr.to_le_bytes().to_vec();
        dtoh_payload.extend_from_slice(&(size as u64).to_le_bytes());

        let dtoh_start = Instant::now();
        let payload = roundtrip(&conn, MessageType::MemcpyDtoH, &dtoh_payload, rid).await?;
        let dtoh_elapsed = dtoh_start.elapsed();
        check_success(&payload)?;
        rid += 1;

        // Free
        let free_payload = device_ptr.to_le_bytes();
        let payload = roundtrip(&conn, MessageType::MemFree, &free_payload, rid).await?;
        check_success(&payload)?;
        rid += 1;

        htod_entries.push(TransferEntry {
            size_bytes: size,
            duration_ms: htod_elapsed.as_secs_f64() * 1000.0,
            throughput_mb_s: throughput_mb_s(size, htod_elapsed),
        });

        dtoh_entries.push(TransferEntry {
            size_bytes: size,
            duration_ms: dtoh_elapsed.as_secs_f64() * 1000.0,
            throughput_mb_s: throughput_mb_s(size, dtoh_elapsed),
        });
    }

    // 3. Small-message latency (DeviceGetCount x 100)
    const LATENCY_SAMPLES: usize = 100;
    let mut latencies: Vec<f64> = Vec::with_capacity(LATENCY_SAMPLES);

    for _ in 0..LATENCY_SAMPLES {
        let start = Instant::now();
        let payload = roundtrip(&conn, MessageType::DeviceGetCount, &[], rid).await?;
        let elapsed = start.elapsed();
        check_success(&payload)?;
        rid += 1;
        latencies.push(elapsed.as_secs_f64() * 1000.0);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let latency_avg_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99_idx = ((LATENCY_SAMPLES as f64) * 0.99).ceil() as usize - 1;
    let latency_p99_ms = latencies[p99_idx.min(LATENCY_SAMPLES - 1)];

    Ok(BenchResult {
        htod: htod_entries,
        dtoh: dtoh_entries,
        latency_avg_ms,
        latency_p99_ms,
        latency_samples: LATENCY_SAMPLES,
    })
}

// ---------------------------------------------------------------------------
// Formatting helpers (used by main.rs for table output)
// ---------------------------------------------------------------------------

/// Format bytes as a human-readable size string (MiB).
pub fn format_mem_mib(bytes: u64) -> String {
    format!("{} MiB", bytes / (1024 * 1024))
}

/// Format a transfer size for display.
pub fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{} MiB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KiB", bytes / 1024)
    } else {
        format!("{} B", bytes)
    }
}

/// Format throughput as MB/s or GB/s.
pub fn format_throughput(mb_s: f64) -> String {
    if mb_s >= 1000.0 {
        format!("{:.1} GB/s", mb_s / 1024.0)
    } else {
        format!("{:.1} MB/s", mb_s)
    }
}

/// Print GPU list as a formatted ASCII table.
pub fn print_list_table(server: &str, gpus: &[GpuInfo]) {
    println!("Server: {server}");

    // Calculate column widths
    let id_width = 4;
    let name_width = gpus
        .iter()
        .map(|g| g.name.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let vram_width = 12;
    let cc_width = 11;

    let total_width = id_width + name_width + vram_width + cc_width + 5 * 3 + 2; // separators
    let _ = total_width; // suppress unused warning; widths used below

    // Header
    println!(
        "{0}{1}{2}{3}{4}{5}{6}{7}{8}",
        corner_tl(),
        horiz(id_width + 2),
        tee_top(),
        horiz(name_width + 2),
        tee_top(),
        horiz(vram_width + 2),
        tee_top(),
        horiz(cc_width + 2),
        corner_tr(),
    );
    println!(
        "{0} {1:<id_w$} {0} {2:<name_w$} {0} {3:<vram_w$} {0} {4:<cc_w$} {0}",
        vert(),
        "ID",
        "Name",
        "VRAM",
        "Compute Cap",
        id_w = id_width,
        name_w = name_width,
        vram_w = vram_width,
        cc_w = cc_width,
    );
    println!(
        "{0}{1}{2}{3}{4}{5}{6}{7}{8}",
        tee_left(),
        horiz(id_width + 2),
        cross(),
        horiz(name_width + 2),
        cross(),
        horiz(vram_width + 2),
        cross(),
        horiz(cc_width + 2),
        tee_right(),
    );

    // Rows
    for gpu in gpus {
        let vram_str = format_mem_mib(gpu.total_mem_bytes);
        let cc_str = format!("{}.{}", gpu.compute_major, gpu.compute_minor);
        println!(
            "{0} {1:>id_w$} {0} {2:<name_w$} {0} {3:<vram_w$} {0} {4:<cc_w$} {0}",
            vert(),
            gpu.id,
            gpu.name,
            vram_str,
            cc_str,
            id_w = id_width,
            name_w = name_width,
            vram_w = vram_width,
            cc_w = cc_width,
        );
    }

    // Footer
    println!(
        "{0}{1}{2}{3}{4}{5}{6}{7}{8}",
        corner_bl(),
        horiz(id_width + 2),
        tee_bottom(),
        horiz(name_width + 2),
        tee_bottom(),
        horiz(vram_width + 2),
        tee_bottom(),
        horiz(cc_width + 2),
        corner_br(),
    );
}

/// Print server status.
pub fn print_status(server: &str, status: &StatusInfo) {
    let major = status.driver_version / 1000;
    let minor = (status.driver_version % 1000) / 10;
    println!("Server: {server}");
    println!(
        "Driver: CUDA {major}.{minor} ({})",
        status.driver_version
    );
    println!("Latency: {:.2}ms", status.latency_ms);
    println!();

    for gpu in &status.gpus {
        let free_mib = gpu.free_bytes / (1024 * 1024);
        let total_mib = gpu.total_bytes / (1024 * 1024);
        let pct = if gpu.total_bytes > 0 {
            (gpu.free_bytes as f64 / gpu.total_bytes as f64 * 100.0) as u64
        } else {
            0
        };
        println!(
            "GPU {}: {} -- {}/{} MiB free ({}%)",
            gpu.id, gpu.name, free_mib, total_mib, pct
        );
    }
}

/// Print benchmark results.
pub fn print_bench(server: &str, result: &BenchResult) {
    println!("Transfer Benchmarks ({server})");
    println!();

    println!("Host -> Device:");
    for entry in &result.htod {
        println!(
            "  {:>8}:  {:>7.2}ms  ({:>10})",
            format_size(entry.size_bytes),
            entry.duration_ms,
            format_throughput(entry.throughput_mb_s),
        );
    }
    println!();

    println!("Device -> Host:");
    for entry in &result.dtoh {
        println!(
            "  {:>8}:  {:>7.2}ms  ({:>10})",
            format_size(entry.size_bytes),
            entry.duration_ms,
            format_throughput(entry.throughput_mb_s),
        );
    }
    println!();

    println!(
        "Latency: avg {:.2}ms, p99 {:.2}ms ({} samples)",
        result.latency_avg_ms, result.latency_p99_ms, result.latency_samples
    );
}

// ---------------------------------------------------------------------------
// Box-drawing characters
// ---------------------------------------------------------------------------

fn corner_tl() -> &'static str { "\u{250c}" } // top-left
fn corner_tr() -> &'static str { "\u{2510}" } // top-right
fn corner_bl() -> &'static str { "\u{2514}" } // bottom-left
fn corner_br() -> &'static str { "\u{2518}" } // bottom-right
fn tee_top() -> &'static str { "\u{252c}" }   // T down
fn tee_bottom() -> &'static str { "\u{2534}" } // T up
fn tee_left() -> &'static str { "\u{251c}" }  // T right
fn tee_right() -> &'static str { "\u{2524}" } // T left
fn cross() -> &'static str { "\u{253c}" }     // +
fn vert() -> &'static str { "\u{2502}" }      // |
fn horiz(n: usize) -> String { "\u{2500}".repeat(n) } // ---

/// Compute throughput in MB/s.
fn throughput_mb_s(bytes: usize, duration: Duration) -> f64 {
    let secs = duration.as_secs_f64();
    if secs == 0.0 {
        return f64::INFINITY;
    }
    (bytes as f64) / (1024.0 * 1024.0) / secs
}
