//! API Recording and Replay types for OuterLink.
//!
//! Captures full CUDA API call traces for debugging, benchmarking,
//! regression testing, and cost estimation. Pure data structures with
//! no I/O or CUDA dependencies.

use std::collections::HashMap;

/// Represents a single CUDA function argument.
#[derive(Debug, Clone, PartialEq)]
pub enum CallArg {
    /// Signed integer argument.
    Int(i64),
    /// Unsigned integer argument.
    Uint(u64),
    /// Floating-point argument.
    Float(f64),
    /// Pointer value (raw address or scrubbed ID).
    Ptr(u64),
    /// Size/length argument.
    Size(usize),
    /// String argument.
    String(String),
    /// Raw byte buffer.
    Bytes(Vec<u8>),
    /// Null / no value.
    Null,
}

/// A single captured CUDA API call.
#[derive(Debug, Clone)]
pub struct RecordedCall {
    /// Monotonically increasing sequence number within a session.
    pub sequence_id: u64,
    /// Wall-clock timestamp in nanoseconds (e.g. from `Instant`).
    pub timestamp_ns: u64,
    /// Name of the CUDA driver API function (e.g. `"cuMemAlloc"`).
    pub function_name: String,
    /// Arguments passed to the function.
    pub args: Vec<CallArg>,
    /// CUresult return value (0 = CUDA_SUCCESS).
    pub return_value: i32,
    /// Time spent inside the real CUDA call, in nanoseconds.
    pub duration_ns: u64,
    /// CUDA context handle active at call time.
    pub context_id: u64,
    /// OS thread ID that made the call.
    pub thread_id: u64,
}

/// Metadata captured once at the start of a recording session.
#[derive(Debug, Clone)]
pub struct RecordingMetadata {
    /// OuterLink version string.
    pub outerlink_version: String,
    /// CUDA version as returned by `cuDriverGetVersion` (e.g. 12040).
    pub cuda_version: u32,
    /// GPU device name.
    pub gpu_name: String,
    /// GPU UUID string.
    pub gpu_uuid: String,
    /// NVIDIA driver version string.
    pub driver_version: String,
    /// Hostname of the machine where the recording was made.
    pub hostname: String,
}

/// A complete recording session containing a sequence of captured calls.
#[derive(Debug, Clone)]
pub struct RecordingSession {
    /// Unique session identifier.
    pub session_id: String,
    /// GPU index this session is recording on.
    pub gpu_index: u32,
    /// Session start timestamp in nanoseconds.
    pub start_time_ns: u64,
    /// Ordered list of recorded calls.
    pub calls: Vec<RecordedCall>,
    /// Session metadata.
    pub metadata: RecordingMetadata,
}

impl RecordingSession {
    /// Create a new empty recording session.
    pub fn new(
        session_id: String,
        gpu_index: u32,
        start_time_ns: u64,
        metadata: RecordingMetadata,
    ) -> Self {
        Self {
            session_id,
            gpu_index,
            start_time_ns,
            calls: Vec::new(),
            metadata,
        }
    }

    /// Append a recorded call to the session.
    pub fn push(&mut self, call: RecordedCall) {
        self.calls.push(call);
    }

    /// Number of calls recorded so far.
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    /// Whether no calls have been recorded.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Total wall-clock duration of the session in nanoseconds.
    ///
    /// Computed as `(last_call.timestamp_ns + last_call.duration_ns) - first_call.timestamp_ns`.
    /// Returns 0 for empty sessions.
    pub fn duration_ns(&self) -> u64 {
        if self.calls.is_empty() {
            return 0;
        }
        let first = &self.calls[0];
        let last = &self.calls[self.calls.len() - 1];
        (last.timestamp_ns + last.duration_ns).saturating_sub(first.timestamp_ns)
    }

    /// Compute a histogram of call counts per function name.
    pub fn call_histogram(&self) -> HashMap<String, u64> {
        let mut hist = HashMap::new();
        for call in &self.calls {
            *hist.entry(call.function_name.clone()).or_insert(0) += 1;
        }
        hist
    }
}

/// Configuration controlling what gets recorded.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Whether recording is active.
    pub enabled: bool,
    /// Maximum number of calls to record before stopping.
    pub max_calls: u64,
    /// Maximum total byte budget for the recording buffer.
    pub max_bytes: u64,
    /// Whether to capture function arguments.
    pub record_args: bool,
    /// Whether to capture per-call timing.
    pub record_timing: bool,
    /// Replace real pointer values with sequential IDs for privacy.
    pub scrub_pointers: bool,
    /// If set, only record calls to these function names.
    pub filter_functions: Option<Vec<String>>,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_calls: 1_000_000,
            max_bytes: 1_073_741_824, // 1 GB
            record_args: true,
            record_timing: true,
            scrub_pointers: true,
            filter_functions: None,
        }
    }
}

/// Configuration controlling how a recorded session is replayed.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Playback speed multiplier (1.0 = real-time).
    pub speed_multiplier: f64,
    /// If true, continue replaying after a call returns an error.
    pub skip_errors: bool,
    /// If true, log calls without executing them.
    pub dry_run: bool,
    /// If true, stop replay when actual results diverge from recorded.
    pub stop_on_divergence: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            speed_multiplier: 1.0,
            skip_errors: false,
            dry_run: false,
            stop_on_divergence: true,
        }
    }
}

/// A single point where replay diverged from the recorded trace.
#[derive(Debug, Clone)]
pub struct ReplayDivergence {
    /// Sequence ID of the divergent call.
    pub sequence_id: u64,
    /// Function that diverged.
    pub function_name: String,
    /// CUresult that was recorded.
    pub expected_result: i32,
    /// CUresult that replay produced.
    pub actual_result: i32,
    /// Human-readable description of the divergence.
    pub description: String,
}

/// Summary of a replay run.
#[derive(Debug, Clone)]
pub struct ReplayResult {
    /// Number of calls successfully replayed.
    pub calls_replayed: u64,
    /// Number of calls that returned errors during replay.
    pub calls_failed: u64,
    /// Points where replay diverged from the recording.
    pub divergences: Vec<ReplayDivergence>,
    /// Total wall-clock time of the replay in nanoseconds.
    pub total_duration_ns: u64,
}

/// Replace a real pointer value with a stable sequential ID.
///
/// Null pointers (0) are always mapped to 0. Non-null pointers are assigned
/// IDs starting at 1 in the order they are first seen. The same pointer
/// always maps to the same ID within a given `map`.
pub fn scrub_pointer(ptr: u64, map: &mut HashMap<u64, u64>) -> u64 {
    if ptr == 0 {
        return 0;
    }
    let next_id = map.len() as u64 + 1;
    *map.entry(ptr).or_insert(next_id)
}
