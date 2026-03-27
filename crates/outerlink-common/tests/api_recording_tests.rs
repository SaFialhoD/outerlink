//! Tests for API Recording/Replay types (G6).

use outerlink_common::api_recording::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CallArg
// ---------------------------------------------------------------------------

#[test]
fn call_arg_int_variant() {
    let arg = CallArg::Int(-42);
    assert!(matches!(arg, CallArg::Int(-42)));
}

#[test]
fn call_arg_uint_variant() {
    let arg = CallArg::Uint(999);
    assert!(matches!(arg, CallArg::Uint(999)));
}

#[test]
fn call_arg_float_variant() {
    let arg = CallArg::Float(3.14);
    if let CallArg::Float(v) = arg {
        assert!((v - 3.14).abs() < f64::EPSILON);
    } else {
        panic!("expected Float");
    }
}

#[test]
fn call_arg_ptr_variant() {
    let arg = CallArg::Ptr(0xDEAD_BEEF);
    assert!(matches!(arg, CallArg::Ptr(0xDEAD_BEEF)));
}

#[test]
fn call_arg_size_variant() {
    let arg = CallArg::Size(1024);
    assert!(matches!(arg, CallArg::Size(1024)));
}

#[test]
fn call_arg_string_variant() {
    let arg = CallArg::String("cuMemAlloc".into());
    assert!(matches!(arg, CallArg::String(ref s) if s == "cuMemAlloc"));
}

#[test]
fn call_arg_bytes_variant() {
    let arg = CallArg::Bytes(vec![0xCA, 0xFE]);
    assert!(matches!(arg, CallArg::Bytes(ref b) if b == &[0xCA, 0xFE]));
}

#[test]
fn call_arg_null_variant() {
    let arg = CallArg::Null;
    assert!(matches!(arg, CallArg::Null));
}

#[test]
fn call_arg_clone() {
    let arg = CallArg::String("hello".into());
    let cloned = arg.clone();
    assert!(matches!(cloned, CallArg::String(ref s) if s == "hello"));
}

// ---------------------------------------------------------------------------
// RecordedCall
// ---------------------------------------------------------------------------

fn sample_call(seq: u64) -> RecordedCall {
    RecordedCall {
        sequence_id: seq,
        timestamp_ns: 1_000_000 + seq * 100,
        function_name: format!("cuFunc_{}", seq),
        args: vec![CallArg::Int(seq as i64)],
        return_value: 0,
        duration_ns: 500,
        context_id: 1,
        thread_id: 42,
    }
}

#[test]
fn recorded_call_construction() {
    let call = sample_call(1);
    assert_eq!(call.sequence_id, 1);
    assert_eq!(call.function_name, "cuFunc_1");
    assert_eq!(call.return_value, 0);
    assert_eq!(call.thread_id, 42);
}

#[test]
fn recorded_call_with_multiple_args() {
    let call = RecordedCall {
        sequence_id: 0,
        timestamp_ns: 0,
        function_name: "cuMemcpyDtoH".into(),
        args: vec![
            CallArg::Ptr(0x1000),
            CallArg::Ptr(0x2000),
            CallArg::Size(4096),
        ],
        return_value: 0,
        duration_ns: 1200,
        context_id: 5,
        thread_id: 7,
    };
    assert_eq!(call.args.len(), 3);
}

// ---------------------------------------------------------------------------
// RecordingMetadata
// ---------------------------------------------------------------------------

#[test]
fn recording_metadata_construction() {
    let meta = RecordingMetadata {
        outerlink_version: "0.1.0".into(),
        cuda_version: 12040,
        gpu_name: "NVIDIA GeForce RTX 3090".into(),
        gpu_uuid: "GPU-abc-123".into(),
        driver_version: "550.54.14".into(),
        hostname: "builder-pc".into(),
    };
    assert_eq!(meta.cuda_version, 12040);
    assert_eq!(meta.hostname, "builder-pc");
}

// ---------------------------------------------------------------------------
// RecordingSession
// ---------------------------------------------------------------------------

fn sample_session() -> RecordingSession {
    let meta = RecordingMetadata {
        outerlink_version: "0.1.0".into(),
        cuda_version: 12040,
        gpu_name: "RTX 3090".into(),
        gpu_uuid: "GPU-xxx".into(),
        driver_version: "550.0".into(),
        hostname: "test".into(),
    };
    RecordingSession::new("sess-001".into(), 0, 1_000_000, meta)
}

#[test]
fn recording_session_new_is_empty() {
    let session = sample_session();
    assert_eq!(session.len(), 0);
    assert_eq!(session.session_id, "sess-001");
    assert_eq!(session.gpu_index, 0);
}

#[test]
fn recording_session_push_increments_len() {
    let mut session = sample_session();
    session.push(sample_call(0));
    session.push(sample_call(1));
    assert_eq!(session.len(), 2);
}

#[test]
fn recording_session_duration_ns() {
    let mut session = sample_session();
    // Push calls at timestamps 1_000_000 and 1_000_200 with duration 500
    let mut c1 = sample_call(0);
    c1.timestamp_ns = 1_000_000;
    c1.duration_ns = 100;
    let mut c2 = sample_call(1);
    c2.timestamp_ns = 1_000_200;
    c2.duration_ns = 300;
    session.push(c1);
    session.push(c2);
    // Duration = last call end - first call start = (1_000_200 + 300) - 1_000_000 = 500
    assert_eq!(session.duration_ns(), 500);
}

#[test]
fn recording_session_duration_ns_empty() {
    let session = sample_session();
    assert_eq!(session.duration_ns(), 0);
}

#[test]
fn recording_session_call_histogram() {
    let mut session = sample_session();
    let mut c1 = sample_call(0);
    c1.function_name = "cuMemAlloc".into();
    let mut c2 = sample_call(1);
    c2.function_name = "cuMemAlloc".into();
    let mut c3 = sample_call(2);
    c3.function_name = "cuLaunchKernel".into();
    session.push(c1);
    session.push(c2);
    session.push(c3);

    let hist = session.call_histogram();
    assert_eq!(hist.get("cuMemAlloc"), Some(&2));
    assert_eq!(hist.get("cuLaunchKernel"), Some(&1));
    assert_eq!(hist.len(), 2);
}

// ---------------------------------------------------------------------------
// RecordingConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn recording_config_defaults() {
    let cfg = RecordingConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.max_calls, 1_000_000);
    assert_eq!(cfg.max_bytes, 1_073_741_824); // 1 GB
    assert!(cfg.record_args);
    assert!(cfg.record_timing);
    assert!(cfg.scrub_pointers);
    assert!(cfg.filter_functions.is_none());
}

// ---------------------------------------------------------------------------
// ReplayConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn replay_config_defaults() {
    let cfg = ReplayConfig::default();
    assert!((cfg.speed_multiplier - 1.0).abs() < f64::EPSILON);
    assert!(!cfg.skip_errors);
    assert!(!cfg.dry_run);
    assert!(cfg.stop_on_divergence);
}

// ---------------------------------------------------------------------------
// ReplayDivergence
// ---------------------------------------------------------------------------

#[test]
fn replay_divergence_construction() {
    let div = ReplayDivergence {
        sequence_id: 42,
        function_name: "cuMemFree".into(),
        expected_result: 0,
        actual_result: 2,
        description: "CUDA_ERROR_OUT_OF_MEMORY".into(),
    };
    assert_eq!(div.sequence_id, 42);
    assert_eq!(div.actual_result, 2);
}

// ---------------------------------------------------------------------------
// ReplayResult
// ---------------------------------------------------------------------------

#[test]
fn replay_result_construction() {
    let result = ReplayResult {
        calls_replayed: 100,
        calls_failed: 2,
        divergences: vec![ReplayDivergence {
            sequence_id: 50,
            function_name: "cuCtxCreate".into(),
            expected_result: 0,
            actual_result: 999,
            description: "unknown error".into(),
        }],
        total_duration_ns: 5_000_000,
    };
    assert_eq!(result.calls_replayed, 100);
    assert_eq!(result.calls_failed, 2);
    assert_eq!(result.divergences.len(), 1);
}

// ---------------------------------------------------------------------------
// scrub_pointer
// ---------------------------------------------------------------------------

#[test]
fn scrub_pointer_assigns_sequential_ids() {
    let mut map = HashMap::new();
    let id1 = scrub_pointer(0xAAAA_0000, &mut map);
    let id2 = scrub_pointer(0xBBBB_0000, &mut map);
    assert_eq!(id1, 1);
    assert_eq!(id2, 2);
}

#[test]
fn scrub_pointer_is_idempotent() {
    let mut map = HashMap::new();
    let first = scrub_pointer(0xDEAD, &mut map);
    let second = scrub_pointer(0xDEAD, &mut map);
    assert_eq!(first, second);
}

#[test]
fn scrub_pointer_null_returns_zero() {
    let mut map = HashMap::new();
    assert_eq!(scrub_pointer(0, &mut map), 0);
}

#[test]
fn scrub_pointer_different_ptrs_get_different_ids() {
    let mut map = HashMap::new();
    let a = scrub_pointer(100, &mut map);
    let b = scrub_pointer(200, &mut map);
    let c = scrub_pointer(300, &mut map);
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_ne!(a, c);
}

// ---------------------------------------------------------------------------
// RecordingConfig filter_functions behavior
// ---------------------------------------------------------------------------

#[test]
fn recording_config_filter_functions_some() {
    let cfg = RecordingConfig {
        filter_functions: Some(vec!["cuMemAlloc".into(), "cuMemFree".into()]),
        ..Default::default()
    };
    let filters = cfg.filter_functions.as_ref().unwrap();
    assert_eq!(filters.len(), 2);
    assert!(filters.contains(&"cuMemAlloc".to_string()));
}

#[test]
fn recording_config_filter_functions_none_means_record_all() {
    let cfg = RecordingConfig::default();
    // None means no filtering = record everything
    assert!(cfg.filter_functions.is_none());
}

// ---------------------------------------------------------------------------
// Debug derives
// ---------------------------------------------------------------------------

#[test]
fn types_implement_debug() {
    // Compile-time check that Debug is derived; runtime format check
    let call = sample_call(0);
    let dbg = format!("{:?}", call);
    assert!(dbg.contains("cuFunc_0"));

    let cfg = RecordingConfig::default();
    let dbg = format!("{:?}", cfg);
    assert!(dbg.contains("enabled"));
}

#[test]
fn types_implement_clone() {
    let call = sample_call(0);
    let cloned = call.clone();
    assert_eq!(cloned.sequence_id, call.sequence_id);
    assert_eq!(cloned.function_name, call.function_name);

    let cfg = RecordingConfig::default();
    let _ = cfg.clone();

    let rcfg = ReplayConfig::default();
    let _ = rcfg.clone();
}
