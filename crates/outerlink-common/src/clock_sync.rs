//! PTP (Precision Time Protocol) clock synchronization for coordinated GPU operations.
//!
//! Provides sub-microsecond time synchronization across networked nodes using
//! IEEE 1588 PTP with hardware timestamping on ConnectX-5 NICs. Integrates with
//! GPU globaltimer for coordinated kernel launches across distributed GPUs.
//!
//! # Architecture
//!
//! The clock sync subsystem manages three layers of time:
//!
//! 1. **PTP grandmaster**: A single node's PHC (PTP Hardware Clock) is the
//!    authoritative time source for the cluster.
//! 2. **System clock**: Each node's `CLOCK_REALTIME` is synced to the PHC via
//!    `phc2sys`, achieving <1us accuracy.
//! 3. **GPU globaltimer**: Calibrated against the PTP-synced system clock using
//!    bracketed measurements, achieving <5us accuracy.
//!
//! # Hardware Requirements
//!
//! - ConnectX-5 or later with hardware timestamping support
//! - Linux with ptp4l and phc2sys (linuxptp package)
//! - `CAP_NET_RAW`, `CAP_SYS_TIME`, `CAP_NET_ADMIN` on ptp4l binary

use dashmap::DashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Core clock types
// ---------------------------------------------------------------------------

/// PTP time with nanosecond precision.
///
/// Represents a point in time synchronized across all cluster nodes via PTP.
/// The time base is TAI or UTC depending on ptp4l configuration (OuterLink
/// uses UTC by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtpTime {
    /// Seconds since epoch.
    pub seconds: i64,
    /// Nanoseconds within the second `[0, 999_999_999]`.
    pub nanoseconds: u32,
}

impl PtpTime {
    /// Create a PTP time from seconds and nanoseconds.
    pub fn new(seconds: i64, nanoseconds: u32) -> Self {
        Self {
            seconds,
            nanoseconds: nanoseconds.min(999_999_999),
        }
    }

    /// Get current PTP-synced time from system clock.
    ///
    /// In production, this reads `CLOCK_REALTIME` which is synced to the PTP
    /// hardware clock via phc2sys. In tests, this returns a monotonic
    /// approximation.
    pub fn now() -> Self {
        // NOTE: In production on Linux, this would use:
        //   libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts)
        // For portability (Windows dev, tests), we use std::time.
        // TODO: requires Linux with phc2sys running for real PTP accuracy
        let since_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            seconds: since_epoch.as_secs() as i64,
            nanoseconds: since_epoch.subsec_nanos(),
        }
    }

    /// Convert to total nanoseconds since epoch.
    pub fn as_nanos(&self) -> i128 {
        (self.seconds as i128) * 1_000_000_000 + self.nanoseconds as i128
    }

    /// Create from total nanoseconds since epoch.
    ///
    /// Handles negative nanos correctly by normalizing the remainder
    /// so that `nanoseconds` is always in `[0, 999_999_999]`.
    pub fn from_nanos(nanos: i128) -> Self {
        let mut seconds = (nanos / 1_000_000_000) as i64;
        let remainder = (nanos % 1_000_000_000) as i64;
        let nanoseconds = if remainder < 0 {
            seconds -= 1;
            (remainder + 1_000_000_000) as u32
        } else {
            remainder as u32
        };
        Self {
            seconds,
            nanoseconds,
        }
    }

    /// Create a future time offset from a given base time.
    pub fn offset_from(base: Self, offset: Duration) -> Self {
        let total_ns = base.as_nanos() + offset.as_nanos() as i128;
        Self::from_nanos(total_ns)
    }

    /// Compute the signed difference in nanoseconds: `self - other`.
    pub fn diff_ns(&self, other: &PtpTime) -> i128 {
        self.as_nanos() - other.as_nanos()
    }
}

impl std::fmt::Display for PtpTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{:09}", self.seconds, self.nanoseconds)
    }
}

// ---------------------------------------------------------------------------
// Sync state machine
// ---------------------------------------------------------------------------

/// PTP synchronization state (maps to ptp4l servo states).
///
/// Transitions follow the ptp4l servo state machine:
/// `Unavailable -> Unlocked (s0) -> Stepping (s1) -> Locked (s2) -> LockedStable`
#[derive(Clone, Debug, PartialEq)]
pub enum PtpSyncState {
    /// s0: Collecting initial samples, not yet synced.
    Unlocked,
    /// s1: Large offset detected, stepping clock.
    Stepping,
    /// s2: Frequency steering active, offset converging.
    Locked { offset_ns: i64 },
    /// s2 with offset consistently below stability threshold.
    LockedStable { offset_ns: i64, stable_since: Instant },
    /// PTP lost -- no sync messages from grandmaster.
    Holdover {
        since: Instant,
        estimated_drift_ns: i64,
    },
    /// PTP daemon not running or not configured.
    Unavailable,
}

impl PtpSyncState {
    /// Returns true if PTP is locked (either Locked or LockedStable).
    pub fn is_locked(&self) -> bool {
        matches!(self, Self::Locked { .. } | Self::LockedStable { .. })
    }

    /// Returns true if PTP is in a usable state (locked or holdover).
    pub fn is_usable(&self) -> bool {
        matches!(
            self,
            Self::Locked { .. } | Self::LockedStable { .. } | Self::Holdover { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// PTP port state
// ---------------------------------------------------------------------------

/// PTP port state as reported by the BMCA (Best Master Clock Algorithm).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtpPortState {
    Initializing,
    FaultyPort,
    Disabled,
    Listening,
    PreMaster,
    Master,
    Passive,
    Uncalibrated,
    Slave,
}

impl PtpPortState {
    /// Parse from ptp4l port state string.
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "INITIALIZING" => Some(Self::Initializing),
            "FAULTY" => Some(Self::FaultyPort),
            "DISABLED" => Some(Self::Disabled),
            "LISTENING" => Some(Self::Listening),
            "PRE_MASTER" | "PRE-MASTER" => Some(Self::PreMaster),
            "MASTER" => Some(Self::Master),
            "PASSIVE" => Some(Self::Passive),
            "UNCALIBRATED" => Some(Self::Uncalibrated),
            "SLAVE" => Some(Self::Slave),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Health report
// ---------------------------------------------------------------------------

/// PTP health report collected from ptp4l.
///
/// Updated by the health monitor thread which parses ptp4l stdout.
#[derive(Clone, Debug)]
pub struct PtpHealthReport {
    /// Current sync state.
    pub state: PtpSyncState,
    /// Current offset from grandmaster (nanoseconds).
    pub master_offset_ns: i64,
    /// Current path delay to grandmaster (nanoseconds).
    pub path_delay_ns: i64,
    /// Frequency adjustment being applied (parts per billion).
    pub freq_ppb: i64,
    /// Identity of the current grandmaster (8 bytes).
    pub grandmaster_id: [u8; 8],
    /// Port state (MASTER, SLAVE, LISTENING, etc.).
    pub port_state: PtpPortState,
    /// When this report was last updated.
    pub reported_at: Instant,
}

impl Default for PtpHealthReport {
    fn default() -> Self {
        Self {
            state: PtpSyncState::Unavailable,
            master_offset_ns: 0,
            path_delay_ns: 0,
            freq_ppb: 0,
            grandmaster_id: [0; 8],
            port_state: PtpPortState::Initializing,
            reported_at: Instant::now(),
        }
    }
}

impl PtpHealthReport {
    /// Returns true if PTP is in a healthy locked state.
    pub fn is_healthy(&self) -> bool {
        self.state.is_locked()
    }
}

// ---------------------------------------------------------------------------
// GPU clock calibration
// ---------------------------------------------------------------------------

/// GPU clock calibration state.
///
/// Maps the GPU's `%globaltimer` register to PTP time using a linear model:
/// `ptp_time = gpu_globaltimer + offset_ns + drift_rate * elapsed`
#[derive(Clone, Debug)]
pub struct GpuClockCalibration {
    /// GPU device index.
    pub gpu_id: u32,
    /// Offset: `ptp_time = gpu_globaltimer + offset_ns`.
    pub offset_ns: i64,
    /// Measured drift rate (nanoseconds per second).
    pub drift_rate_ns_per_sec: f64,
    /// When this calibration was taken.
    pub calibrated_at: PtpTime,
    /// PCIe round-trip time of the calibration measurement.
    pub measurement_rtt_ns: u64,
    /// Number of samples averaged for this calibration.
    pub sample_count: u32,
}

impl GpuClockCalibration {
    /// Estimate the current GPU-to-PTP offset accounting for drift.
    ///
    /// Uses linear extrapolation from the last calibration point.
    pub fn current_offset_ns(&self, now: &PtpTime) -> i64 {
        let elapsed_ns = now.diff_ns(&self.calibrated_at) as f64;
        let drift_correction =
            (self.drift_rate_ns_per_sec * elapsed_ns / 1_000_000_000.0) as i64;
        self.offset_ns + drift_correction
    }

    /// Returns true if the calibration is considered stale.
    ///
    /// A calibration older than `max_age` should be refreshed.
    pub fn is_stale(&self, now: &PtpTime, max_age: Duration) -> bool {
        let age_ns = now.diff_ns(&self.calibrated_at).unsigned_abs();
        age_ns > max_age.as_nanos()
    }
}

// ---------------------------------------------------------------------------
// PTP configuration
// ---------------------------------------------------------------------------

/// PTP transport protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtpTransport {
    /// IEEE 802.3 raw Ethernet (lowest jitter).
    L2,
    /// UDP/IPv4 (works through routers, slightly higher jitter).
    UdpIpv4,
}

/// PTP profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtpProfile {
    /// IEEE 1588 default profile.
    Default1588,
}

/// PTP configuration for a specific node.
///
/// Used to generate ptp4l.conf and phc2sys command-line arguments.
#[derive(Clone, Debug)]
pub struct PtpConfig {
    /// Network interface for PTP (ConnectX-5 RDMA interface).
    pub interface: String,
    /// BMCA Priority1 (lower = more likely grandmaster). Coordinator=100, secondary=110, worker=128.
    pub priority1: u8,
    /// BMCA Priority2 (tiebreaker).
    pub priority2: u8,
    /// PTP domain number (all cluster nodes must match).
    pub domain: u8,
    /// Transport: L2 (raw Ethernet) for best accuracy.
    pub transport: PtpTransport,
    /// Sync interval exponent: 2^N seconds (0 = 1 second).
    pub log_sync_interval: i8,
    /// Delay request interval (should match sync interval).
    pub log_delay_req_interval: i8,
    /// PTP profile.
    pub profile: PtpProfile,
    /// GPU calibration interval.
    pub calibration_interval: Duration,
    /// Stability threshold: offset below this (in ns) is considered stable.
    pub stability_threshold_ns: i64,
}

impl Default for PtpConfig {
    fn default() -> Self {
        Self {
            interface: "eth0".to_string(),
            priority1: 128,
            priority2: 128,
            domain: 0,
            transport: PtpTransport::L2,
            log_sync_interval: 0,
            log_delay_req_interval: 0,
            profile: PtpProfile::Default1588,
            calibration_interval: Duration::from_secs(5),
            stability_threshold_ns: 100,
        }
    }
}

impl PtpConfig {
    /// Create a config for the grandmaster (coordinator) node.
    pub fn grandmaster(interface: &str) -> Self {
        Self {
            interface: interface.to_string(),
            priority1: 100,
            ..Default::default()
        }
    }

    /// Create a config for a secondary (backup grandmaster) node.
    pub fn secondary(interface: &str) -> Self {
        Self {
            interface: interface.to_string(),
            priority1: 110,
            ..Default::default()
        }
    }

    /// Create a config for a worker (slave-only) node.
    pub fn worker(interface: &str) -> Self {
        Self {
            interface: interface.to_string(),
            priority1: 128,
            ..Default::default()
        }
    }

    /// Generate ptp4l.conf content from this config.
    pub fn to_ptp4l_conf(&self) -> String {
        let transport = match self.transport {
            PtpTransport::L2 => "L2",
            PtpTransport::UdpIpv4 => "UDPv4",
        };

        format!(
            "[global]\n\
             time_stamping           hardware\n\
             network_transport       {transport}\n\
             logSyncInterval         {sync}\n\
             logAnnounceInterval     1\n\
             logMinDelayReqInterval  {delay}\n\
             pi_proportional_const   0.0\n\
             pi_integral_const       0.0\n\
             step_threshold          1.0\n\
             first_step_threshold    0.00002\n\
             clockClass              248\n\
             priority1               {p1}\n\
             priority2               {p2}\n\
             domainNumber            {domain}\n\
             verbose                 1\n\
             summary_interval        1\n\
             \n\
             [{iface}]\n",
            sync = self.log_sync_interval,
            delay = self.log_delay_req_interval,
            p1 = self.priority1,
            p2 = self.priority2,
            domain = self.domain,
            iface = self.interface,
        )
    }

    /// Generate phc2sys command-line arguments.
    pub fn phc2sys_args(&self) -> Vec<String> {
        vec![
            "-a".into(),
            "-r".into(),
            "-r".into(),
            "-O".into(),
            "0".into(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Coordinated launch types
// ---------------------------------------------------------------------------

/// A coordinated action to execute at a specific PTP time.
///
/// Used for synchronized kernel launches across distributed GPUs.
#[derive(Debug, Clone)]
pub struct CoordinatedLaunch {
    /// Target PTP time for the action.
    pub target_time: PtpTime,
    /// Which nodes participate (by node ID).
    pub participating_nodes: Vec<u8>,
    /// Action identifier (opaque to the clock system).
    pub action_id: u64,
    /// Safety margin before target_time to begin spin-wait.
    pub spin_margin: Duration,
}

/// Result of a coordinated launch attempt.
#[derive(Debug, Clone)]
pub struct LaunchResult {
    /// Actual PTP time when the action was triggered.
    pub actual_time: PtpTime,
    /// Difference from target (positive = late, negative = early).
    pub jitter_ns: i64,
    /// Whether GPU spin-wait was used.
    pub gpu_spin_used: bool,
}

/// GPU-side launch parameters passed to the coordinated kernel wrapper.
///
/// The wrapper kernel reads `%globaltimer` and spin-waits until it reaches
/// `target_globaltimer_ns`, achieving sub-2us alignment across GPUs.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuLaunchParams {
    /// Target GPU globaltimer value (`ptp_target - gpu_calibration_offset`).
    pub target_globaltimer_ns: u64,
    /// Maximum spin iterations before giving up (prevents infinite spin).
    pub max_spin_cycles: u64,
}

impl Default for GpuLaunchParams {
    fn default() -> Self {
        Self {
            target_globaltimer_ns: 0,
            max_spin_cycles: 100_000, // ~100us max spin at ~1ns per iteration
        }
    }
}

// ---------------------------------------------------------------------------
// ClockSync trait
// ---------------------------------------------------------------------------

/// Central clock synchronization service for OuterLink.
///
/// Provides PTP time, sync health, GPU calibration, and coordinated scheduling.
pub trait ClockSync: Send + Sync {
    /// Get current PTP-synced time.
    fn now_ptp(&self) -> PtpTime;

    /// Check if PTP is synchronized and healthy.
    fn is_synced(&self) -> bool;

    /// Get current sync state and health report.
    fn health(&self) -> PtpHealthReport;

    /// Get current offset from grandmaster (nanoseconds).
    fn offset_ns(&self) -> i64;

    /// Get GPU clock calibration for a specific GPU.
    fn gpu_calibration(&self, gpu_id: u32) -> Option<GpuClockCalibration>;

    /// Convert PTP time to GPU globaltimer value (for GPU-side spin-wait).
    fn ptp_to_gpu_time(&self, ptp_time: PtpTime, gpu_id: u32) -> Result<u64, ClockSyncError>;

    /// Schedule an action to execute at a specific PTP time.
    ///
    /// Strategy: coarse sleep until ~1ms before target, then spin-wait on
    /// `PtpTime::now()` until the target is reached, then invoke the action.
    /// Returns `TargetTimeInPast` if the target time has already passed.
    fn schedule_at(
        &self,
        target: PtpTime,
        action: Box<dyn FnOnce() + Send>,
    ) -> Result<(), ClockSyncError>;
}

/// Errors specific to the clock sync subsystem.
#[derive(Debug, thiserror::Error)]
pub enum ClockSyncError {
    #[error("PTP is not synchronized")]
    PtpNotSynced,

    #[error("GPU {0} has not been calibrated")]
    GpuNotCalibrated(u32),

    #[error("GPU calibration failed: all samples exceeded RTT threshold")]
    CalibrationFailed,

    #[error("PTP daemon startup failed: {0}")]
    DaemonStartFailed(String),

    #[error("Scheduling error: target time is in the past")]
    TargetTimeInPast,

    #[error("Configuration error: {0}")]
    Config(String),
}

// ---------------------------------------------------------------------------
// PTP health monitor (parses ptp4l output)
// ---------------------------------------------------------------------------

/// Monitors ptp4l output and updates health state.
///
/// Parses ptp4l's summary lines which have the format:
/// `ptp4l[XXX.XXX]: master offset   23 s2 freq +1234 path delay   456`
pub struct PtpHealthMonitor {
    /// Shared health state updated by the monitor.
    health: std::sync::Arc<RwLock<PtpHealthReport>>,
    /// Stability threshold in nanoseconds.
    stability_threshold_ns: i64,
}

impl PtpHealthMonitor {
    /// Create a new health monitor with the given shared health state.
    pub fn new(
        health: std::sync::Arc<RwLock<PtpHealthReport>>,
        stability_threshold_ns: i64,
    ) -> Self {
        Self {
            health,
            stability_threshold_ns,
        }
    }

    /// Parse a single line from ptp4l stdout and update health state.
    ///
    /// Expected format:
    /// `ptp4l[XXX.XXX]: master offset   <offset> s<N> freq <freq> path delay   <delay>`
    pub fn parse_line(&self, line: &str) {
        // Find the "master offset" pattern and extract fields.
        // We use manual parsing instead of regex to avoid the regex dependency.
        let Some(offset_start) = line.find("master offset") else {
            return;
        };

        let after_offset = &line[offset_start + "master offset".len()..];
        let tokens: Vec<&str> = after_offset.split_whitespace().collect();

        // Expected tokens: [offset_value, servo_state, "freq", freq_value, "path", "delay", delay_value]
        if tokens.len() < 7 {
            return;
        }

        let Ok(offset) = tokens[0].parse::<i64>() else {
            return;
        };
        let servo_state = tokens[1];
        let freq = tokens[3].parse::<i64>().unwrap_or(0);
        let path_delay = tokens[6].parse::<i64>().unwrap_or(0);

        // Preserve stable_since when already in LockedStable to avoid resetting
        // the stability timer on every ptp4l output line.
        let existing_stable_since = if let Ok(h) = self.health.read() {
            if let PtpSyncState::LockedStable { stable_since, .. } = h.state {
                Some(stable_since)
            } else {
                None
            }
        } else {
            None
        };

        let state = match servo_state {
            "s0" => PtpSyncState::Unlocked,
            "s1" => PtpSyncState::Stepping,
            "s2" => {
                if offset.abs() < self.stability_threshold_ns {
                    PtpSyncState::LockedStable {
                        offset_ns: offset,
                        stable_since: existing_stable_since.unwrap_or_else(Instant::now),
                    }
                } else {
                    PtpSyncState::Locked { offset_ns: offset }
                }
            }
            _ => PtpSyncState::Unavailable,
        };

        if let Ok(mut health) = self.health.write() {
            health.state = state;
            health.master_offset_ns = offset;
            health.path_delay_ns = path_delay;
            health.freq_ppb = freq;
            health.reported_at = Instant::now();
        }
    }
}

// ---------------------------------------------------------------------------
// LinuxPtpClockSync implementation
// ---------------------------------------------------------------------------

/// Linux-based PTP clock synchronization service.
///
/// Manages ptp4l and phc2sys lifecycle, monitors health, calibrates GPU clocks,
/// and provides the `ClockSync` trait implementation.
///
/// In production, this spawns actual ptp4l/phc2sys processes. For testing,
/// the health state can be manually set.
pub struct LinuxPtpClockSync {
    /// Current PTP health (updated by monitor thread or manually in tests).
    health: std::sync::Arc<RwLock<PtpHealthReport>>,
    /// Per-GPU calibration state.
    gpu_calibrations: DashMap<u32, GpuClockCalibration>,
    /// Configuration.
    config: PtpConfig,
    /// Health monitor for parsing ptp4l output.
    monitor: PtpHealthMonitor,
}

impl LinuxPtpClockSync {
    /// Create a new clock sync service with the given configuration.
    ///
    /// Does NOT start ptp4l/phc2sys processes. Call `start()` for that.
    /// This constructor is useful for testing where processes are not needed.
    pub fn new(config: PtpConfig) -> Self {
        let health = std::sync::Arc::new(RwLock::new(PtpHealthReport::default()));
        let stability_threshold_ns = config.stability_threshold_ns;
        let monitor = PtpHealthMonitor::new(health.clone(), stability_threshold_ns);

        Self {
            health,
            gpu_calibrations: DashMap::new(),
            config,
            monitor,
        }
    }

    /// Start the PTP subsystem.
    ///
    /// Production sequence:
    /// 1. Generate ptp4l.conf from config
    /// 2. Spawn ptp4l process
    /// 3. Wait for s2 (locked) state
    /// 4. Spawn phc2sys process
    /// 5. Start monitor thread (parses ptp4l output)
    /// 6. Start GPU calibration thread
    ///
    /// TODO: requires Linux with ptp4l/phc2sys installed
    pub fn start(&self) -> Result<(), ClockSyncError> {
        // In production, this would:
        // 1. Write config to /tmp/outerlink-ptp4l.conf
        // 2. Spawn ptp4l -f /tmp/outerlink-ptp4l.conf -m
        // 3. Pipe stdout to monitor.parse_line()
        // 4. Wait for s2 state (timeout 120s)
        // 5. Spawn phc2sys -a -r -r -O 0
        // 6. Start calibration loop

        // For now, mark as unavailable until hardware is present
        tracing::info!(
            "PTP clock sync start requested for interface {}",
            self.config.interface
        );
        Ok(())
    }

    /// Stop the PTP subsystem gracefully.
    ///
    /// TODO: requires actual process handles
    pub fn stop(&self) -> Result<(), ClockSyncError> {
        tracing::info!("PTP clock sync stop requested");
        Ok(())
    }

    /// Feed a line from ptp4l stdout to the health monitor.
    ///
    /// This is the integration point for the monitor thread. In production,
    /// the thread reads from ptp4l's stdout pipe and calls this for each line.
    pub fn feed_ptp4l_line(&self, line: &str) {
        self.monitor.parse_line(line);
    }

    /// Manually set the health state (useful for testing).
    pub fn set_health(&self, report: PtpHealthReport) {
        if let Ok(mut health) = self.health.write() {
            *health = report;
        }
    }

    /// Store a GPU calibration result.
    pub fn set_gpu_calibration(&self, calibration: GpuClockCalibration) {
        self.gpu_calibrations.insert(calibration.gpu_id, calibration);
    }

    /// Perform GPU-to-PTP calibration using the bracketing technique.
    ///
    /// Takes multiple samples and picks the one with the lowest PCIe RTT.
    /// Each sample brackets a GPU globaltimer read between two host clock reads.
    ///
    /// Returns `Err(CalibrationFailed)` if all samples exceed the RTT threshold.
    ///
    /// TODO: requires CUDA runtime for actual GPU timer reads
    pub fn calibrate_gpu_from_samples(
        &self,
        gpu_id: u32,
        samples: &[(i64, i64, i64)], // (t1_host, t_gpu, t2_host) triples
        max_rtt_ns: u64,
    ) -> Result<GpuClockCalibration, ClockSyncError> {
        const DEFAULT_SAMPLE_COUNT: u32 = 11;

        let mut best_sample: Option<(i64, u64)> = None; // (offset, rtt)

        for &(t1_host, t_gpu, t2_host) in samples {
            let rtt = (t2_host - t1_host) as u64;
            if rtt > max_rtt_ns {
                continue; // PCIe was delayed, skip this sample
            }

            let host_midpoint = (t1_host + t2_host) / 2;
            let offset = host_midpoint - t_gpu;

            match &best_sample {
                None => best_sample = Some((offset, rtt)),
                Some((_, best_rtt)) if rtt < *best_rtt => {
                    best_sample = Some((offset, rtt));
                }
                _ => {}
            }
        }

        let (offset, rtt) = best_sample.ok_or(ClockSyncError::CalibrationFailed)?;

        // Compute drift rate from previous calibration (if exists)
        let drift_rate = if let Some(prev) = self.gpu_calibrations.get(&gpu_id) {
            let now_ptp = PtpTime::now();
            let dt_ns = now_ptp.diff_ns(&prev.calibrated_at);
            if dt_ns > 0 {
                let offset_change = offset - prev.offset_ns;
                (offset_change as f64 / dt_ns as f64) * 1_000_000_000.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        let calibration = GpuClockCalibration {
            gpu_id,
            offset_ns: offset,
            drift_rate_ns_per_sec: drift_rate,
            calibrated_at: PtpTime::now(),
            measurement_rtt_ns: rtt,
            sample_count: samples.len().min(DEFAULT_SAMPLE_COUNT as usize) as u32,
        };

        self.gpu_calibrations.insert(gpu_id, calibration.clone());
        Ok(calibration)
    }

    /// Get the PTP configuration.
    pub fn config(&self) -> &PtpConfig {
        &self.config
    }

    /// Get the health monitor reference (for feeding lines in tests).
    pub fn monitor(&self) -> &PtpHealthMonitor {
        &self.monitor
    }

    /// Update BMCA priority (called when R17 topology changes).
    ///
    /// In production, this would send a UDS command to ptp4l via pmc.
    /// TODO: requires pmc subprocess
    pub fn update_priority(&self, new_priority1: u8) -> Result<(), ClockSyncError> {
        tracing::info!("Updating PTP priority1 to {}", new_priority1);
        // In production: pmc -u -b 0 "SET GRANDMASTER_SETTINGS_NP ..."
        Ok(())
    }
}

impl ClockSync for LinuxPtpClockSync {
    fn now_ptp(&self) -> PtpTime {
        PtpTime::now()
    }

    fn is_synced(&self) -> bool {
        self.health
            .read()
            .map(|h| h.state.is_locked())
            .unwrap_or(false)
    }

    fn health(&self) -> PtpHealthReport {
        self.health
            .read()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    fn offset_ns(&self) -> i64 {
        self.health
            .read()
            .map(|h| h.master_offset_ns)
            .unwrap_or(0)
    }

    fn gpu_calibration(&self, gpu_id: u32) -> Option<GpuClockCalibration> {
        self.gpu_calibrations.get(&gpu_id).map(|c| c.clone())
    }

    fn ptp_to_gpu_time(
        &self,
        ptp_time: PtpTime,
        gpu_id: u32,
    ) -> Result<u64, ClockSyncError> {
        let cal = self
            .gpu_calibrations
            .get(&gpu_id)
            .ok_or(ClockSyncError::GpuNotCalibrated(gpu_id))?;

        let ptp_ns = ptp_time.as_nanos() as i64;
        let elapsed_since_cal = ptp_ns - cal.calibrated_at.as_nanos() as i64;
        let drift_correction = (cal.drift_rate_ns_per_sec * elapsed_since_cal as f64
            / 1_000_000_000.0) as i64;

        // gpu_time = ptp_time - offset - drift_correction
        let gpu_time = ptp_ns - cal.offset_ns - drift_correction;
        if gpu_time < 0 {
            return Err(ClockSyncError::TargetTimeInPast);
        }
        Ok(gpu_time as u64)
    }

    fn schedule_at(
        &self,
        target: PtpTime,
        action: Box<dyn FnOnce() + Send>,
    ) -> Result<(), ClockSyncError> {
        if !self.is_synced() {
            return Err(ClockSyncError::PtpNotSynced);
        }

        let now = PtpTime::now();
        if target.as_nanos() <= now.as_nanos() {
            return Err(ClockSyncError::TargetTimeInPast);
        }

        // Coarse sleep: sleep until ~1ms before target to avoid burning CPU.
        let target_ns = target.as_nanos();
        let now_ns = now.as_nanos();
        let delta_ns = target_ns - now_ns;
        let coarse_sleep_ns = delta_ns - 1_000_000; // wake 1ms early
        if coarse_sleep_ns > 0 {
            std::thread::sleep(Duration::from_nanos(coarse_sleep_ns as u64));
        }

        // Spin-wait: busy-wait on PtpTime::now() until target is reached.
        // This gives sub-microsecond precision on the action invocation.
        loop {
            let current = PtpTime::now();
            if current.as_nanos() >= target_ns {
                break;
            }
            std::hint::spin_loop();
        }

        action();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PtpTime tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ptp_time_as_nanos() {
        let t = PtpTime::new(1, 500_000_000);
        assert_eq!(t.as_nanos(), 1_500_000_000);
    }

    #[test]
    fn test_ptp_time_from_nanos() {
        let t = PtpTime::from_nanos(2_750_000_000);
        assert_eq!(t.seconds, 2);
        assert_eq!(t.nanoseconds, 750_000_000);
    }

    #[test]
    fn test_ptp_time_roundtrip() {
        let original = PtpTime::new(123456, 789_012_345);
        let nanos = original.as_nanos();
        let restored = PtpTime::from_nanos(nanos);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_ptp_time_offset() {
        let base = PtpTime::new(10, 0);
        let future = PtpTime::offset_from(base, Duration::from_millis(500));
        assert_eq!(future.seconds, 10);
        assert_eq!(future.nanoseconds, 500_000_000);
    }

    #[test]
    fn test_ptp_time_offset_wraps_seconds() {
        let base = PtpTime::new(10, 800_000_000);
        let future = PtpTime::offset_from(base, Duration::from_millis(300));
        assert_eq!(future.seconds, 11);
        assert_eq!(future.nanoseconds, 100_000_000);
    }

    #[test]
    fn test_ptp_time_diff() {
        let t1 = PtpTime::new(10, 500_000_000);
        let t2 = PtpTime::new(10, 200_000_000);
        assert_eq!(t1.diff_ns(&t2), 300_000_000);
        assert_eq!(t2.diff_ns(&t1), -300_000_000);
    }

    #[test]
    fn test_ptp_time_ordering() {
        let t1 = PtpTime::new(10, 0);
        let t2 = PtpTime::new(10, 1);
        let t3 = PtpTime::new(11, 0);
        assert!(t1 < t2);
        assert!(t2 < t3);
    }

    #[test]
    fn test_ptp_time_clamps_nanoseconds() {
        let t = PtpTime::new(1, 2_000_000_000);
        assert_eq!(t.nanoseconds, 999_999_999);
    }

    #[test]
    fn test_ptp_time_display() {
        let t = PtpTime::new(100, 42);
        assert_eq!(format!("{}", t), "100.000000042");
    }

    // -----------------------------------------------------------------------
    // PTP sync state tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sync_state_is_locked() {
        assert!(!PtpSyncState::Unlocked.is_locked());
        assert!(!PtpSyncState::Stepping.is_locked());
        assert!(PtpSyncState::Locked { offset_ns: 50 }.is_locked());
        assert!(PtpSyncState::LockedStable {
            offset_ns: 10,
            stable_since: Instant::now()
        }
        .is_locked());
        assert!(!PtpSyncState::Unavailable.is_locked());
    }

    #[test]
    fn test_sync_state_is_usable() {
        assert!(!PtpSyncState::Unlocked.is_usable());
        assert!(PtpSyncState::Locked { offset_ns: 50 }.is_usable());
        assert!(PtpSyncState::Holdover {
            since: Instant::now(),
            estimated_drift_ns: 100
        }
        .is_usable());
    }

    // -----------------------------------------------------------------------
    // Port state parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_port_state_from_str() {
        assert_eq!(PtpPortState::from_str_name("MASTER"), Some(PtpPortState::Master));
        assert_eq!(PtpPortState::from_str_name("SLAVE"), Some(PtpPortState::Slave));
        assert_eq!(PtpPortState::from_str_name("slave"), Some(PtpPortState::Slave));
        assert_eq!(PtpPortState::from_str_name("LISTENING"), Some(PtpPortState::Listening));
        assert_eq!(PtpPortState::from_str_name("UNKNOWN"), None);
    }

    // -----------------------------------------------------------------------
    // PTP config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_grandmaster() {
        let config = PtpConfig::grandmaster("mlx5_0");
        assert_eq!(config.priority1, 100);
        assert_eq!(config.interface, "mlx5_0");
    }

    #[test]
    fn test_config_secondary() {
        let config = PtpConfig::secondary("mlx5_0");
        assert_eq!(config.priority1, 110);
    }

    #[test]
    fn test_config_worker() {
        let config = PtpConfig::worker("mlx5_0");
        assert_eq!(config.priority1, 128);
    }

    #[test]
    fn test_config_to_ptp4l_conf() {
        let config = PtpConfig {
            interface: "eth1".to_string(),
            priority1: 100,
            priority2: 200,
            domain: 5,
            transport: PtpTransport::L2,
            log_sync_interval: -1,
            log_delay_req_interval: -1,
            ..Default::default()
        };

        let conf = config.to_ptp4l_conf();
        assert!(conf.contains("network_transport       L2"));
        assert!(conf.contains("priority1               100"));
        assert!(conf.contains("priority2               200"));
        assert!(conf.contains("domainNumber            5"));
        assert!(conf.contains("logSyncInterval         -1"));
        assert!(conf.contains("[eth1]"));
    }

    #[test]
    fn test_config_phc2sys_args() {
        let config = PtpConfig::default();
        let args = config.phc2sys_args();
        assert!(args.contains(&"-a".to_string()));
        assert!(args.contains(&"-r".to_string()));
    }

    #[test]
    fn test_config_udp_transport() {
        let config = PtpConfig {
            transport: PtpTransport::UdpIpv4,
            ..Default::default()
        };
        let conf = config.to_ptp4l_conf();
        assert!(conf.contains("network_transport       UDPv4"));
    }

    // -----------------------------------------------------------------------
    // Health monitor parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_monitor_parse_s2_locked() {
        let health = std::sync::Arc::new(RwLock::new(PtpHealthReport::default()));
        let monitor = PtpHealthMonitor::new(health.clone(), 100);

        monitor.parse_line(
            "ptp4l[123.456]: master offset     23 s2 freq  +1234 path delay     456",
        );

        let h = health.read().unwrap();
        assert_eq!(h.master_offset_ns, 23);
        assert!(matches!(h.state, PtpSyncState::LockedStable { offset_ns: 23, .. }));
        assert_eq!(h.freq_ppb, 1234);
        assert_eq!(h.path_delay_ns, 456);
    }

    #[test]
    fn test_monitor_parse_s2_locked_large_offset() {
        let health = std::sync::Arc::new(RwLock::new(PtpHealthReport::default()));
        let monitor = PtpHealthMonitor::new(health.clone(), 100);

        monitor.parse_line(
            "ptp4l[0.000]: master offset    500 s2 freq   -100 path delay    200",
        );

        let h = health.read().unwrap();
        assert_eq!(h.master_offset_ns, 500);
        assert!(matches!(h.state, PtpSyncState::Locked { offset_ns: 500 }));
    }

    #[test]
    fn test_monitor_parse_s0_unlocked() {
        let health = std::sync::Arc::new(RwLock::new(PtpHealthReport::default()));
        let monitor = PtpHealthMonitor::new(health.clone(), 100);

        monitor.parse_line(
            "ptp4l[0.000]: master offset  99999 s0 freq      0 path delay      0",
        );

        let h = health.read().unwrap();
        assert!(matches!(h.state, PtpSyncState::Unlocked));
    }

    #[test]
    fn test_monitor_parse_s1_stepping() {
        let health = std::sync::Arc::new(RwLock::new(PtpHealthReport::default()));
        let monitor = PtpHealthMonitor::new(health.clone(), 100);

        monitor.parse_line(
            "ptp4l[0.000]: master offset  50000 s1 freq   +500 path delay   1000",
        );

        let h = health.read().unwrap();
        assert!(matches!(h.state, PtpSyncState::Stepping));
    }

    #[test]
    fn test_monitor_parse_negative_offset() {
        let health = std::sync::Arc::new(RwLock::new(PtpHealthReport::default()));
        let monitor = PtpHealthMonitor::new(health.clone(), 100);

        monitor.parse_line(
            "ptp4l[0.000]: master offset    -42 s2 freq   -300 path delay     99",
        );

        let h = health.read().unwrap();
        assert_eq!(h.master_offset_ns, -42);
        assert_eq!(h.freq_ppb, -300);
        // abs(-42) < 100, so should be LockedStable
        assert!(matches!(h.state, PtpSyncState::LockedStable { offset_ns: -42, .. }));
    }

    #[test]
    fn test_monitor_ignores_non_matching_lines() {
        let health = std::sync::Arc::new(RwLock::new(PtpHealthReport::default()));
        let monitor = PtpHealthMonitor::new(health.clone(), 100);

        monitor.parse_line("ptp4l[0.000]: port 1 (eth0): UNCALIBRATED to SLAVE");

        let h = health.read().unwrap();
        // State should remain unchanged (Unavailable from default)
        assert!(matches!(h.state, PtpSyncState::Unavailable));
    }

    // -----------------------------------------------------------------------
    // GPU calibration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_calibration_from_samples() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        // Simulate 5 bracketed samples: (t1_host, t_gpu, t2_host)
        // Best sample is the one with lowest RTT
        let samples = vec![
            (1_000_000_000, 999_999_000, 1_000_003_000), // RTT=3000ns, offset=500
            (1_000_010_000, 1_000_008_000, 1_000_012_000), // RTT=2000ns, offset=3000
            (1_000_020_000, 1_000_019_000, 1_000_021_000), // RTT=1000ns, offset=500 -- best
            (1_000_030_000, 1_000_028_000, 1_000_036_000), // RTT=6000ns, exceeds threshold
            (1_000_040_000, 1_000_038_000, 1_000_042_500), // RTT=2500ns, offset=2250
        ];

        let cal = sync
            .calibrate_gpu_from_samples(0, &samples, 5000)
            .expect("calibration should succeed");

        assert_eq!(cal.gpu_id, 0);
        assert_eq!(cal.measurement_rtt_ns, 1000); // lowest RTT sample
        // midpoint = (1_000_020_000 + 1_000_021_000) / 2 = 1_000_020_500
        // offset = 1_000_020_500 - 1_000_019_000 = 1500
        assert_eq!(cal.offset_ns, 1500);
    }

    #[test]
    fn test_gpu_calibration_all_samples_rejected() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        // All samples exceed RTT threshold
        let samples = vec![
            (1000, 500, 7000), // RTT=6000
            (2000, 1500, 9000), // RTT=7000
        ];

        let result = sync.calibrate_gpu_from_samples(0, &samples, 5000);
        assert!(matches!(result, Err(ClockSyncError::CalibrationFailed)));
    }

    #[test]
    fn test_gpu_calibration_stored_and_retrievable() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        let cal = GpuClockCalibration {
            gpu_id: 1,
            offset_ns: 1000,
            drift_rate_ns_per_sec: 0.5,
            calibrated_at: PtpTime::new(100, 0),
            measurement_rtt_ns: 800,
            sample_count: 11,
        };

        sync.set_gpu_calibration(cal.clone());

        let retrieved = sync.gpu_calibration(1).expect("should find GPU 1");
        assert_eq!(retrieved.gpu_id, 1);
        assert_eq!(retrieved.offset_ns, 1000);
    }

    #[test]
    fn test_gpu_calibration_staleness() {
        let cal = GpuClockCalibration {
            gpu_id: 0,
            offset_ns: 0,
            drift_rate_ns_per_sec: 0.0,
            calibrated_at: PtpTime::new(100, 0),
            measurement_rtt_ns: 500,
            sample_count: 11,
        };

        // 10 seconds later
        let now_fresh = PtpTime::new(105, 0);
        assert!(!cal.is_stale(&now_fresh, Duration::from_secs(10)));

        let now_stale = PtpTime::new(115, 0);
        assert!(cal.is_stale(&now_stale, Duration::from_secs(10)));
    }

    // -----------------------------------------------------------------------
    // ClockSync trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_synced_when_locked() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        assert!(!sync.is_synced());

        sync.set_health(PtpHealthReport {
            state: PtpSyncState::Locked { offset_ns: 50 },
            ..Default::default()
        });

        assert!(sync.is_synced());
    }

    #[test]
    fn test_is_synced_when_locked_stable() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        sync.set_health(PtpHealthReport {
            state: PtpSyncState::LockedStable {
                offset_ns: 10,
                stable_since: Instant::now(),
            },
            ..Default::default()
        });

        assert!(sync.is_synced());
    }

    #[test]
    fn test_not_synced_when_holdover() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        sync.set_health(PtpHealthReport {
            state: PtpSyncState::Holdover {
                since: Instant::now(),
                estimated_drift_ns: 100,
            },
            ..Default::default()
        });

        // Holdover is NOT locked, so is_synced returns false
        assert!(!sync.is_synced());
    }

    #[test]
    fn test_offset_ns() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        sync.set_health(PtpHealthReport {
            master_offset_ns: -42,
            ..Default::default()
        });

        assert_eq!(sync.offset_ns(), -42);
    }

    #[test]
    fn test_ptp_to_gpu_time_no_calibration() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        let result = sync.ptp_to_gpu_time(PtpTime::new(100, 0), 0);
        assert!(matches!(result, Err(ClockSyncError::GpuNotCalibrated(0))));
    }

    #[test]
    fn test_ptp_to_gpu_time_with_calibration() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        let cal = GpuClockCalibration {
            gpu_id: 0,
            offset_ns: 1000, // PTP is 1000ns ahead of GPU
            drift_rate_ns_per_sec: 0.0, // no drift
            calibrated_at: PtpTime::new(100, 0),
            measurement_rtt_ns: 500,
            sample_count: 11,
        };
        sync.set_gpu_calibration(cal);

        let ptp_time = PtpTime::new(100, 5000); // 5us after calibration
        let gpu_time = sync.ptp_to_gpu_time(ptp_time, 0).expect("should succeed");

        // gpu_time = ptp_ns - offset - drift_correction
        // ptp_ns = 100 * 1e9 + 5000 = 100_000_005_000
        // offset = 1000
        // drift = 0
        // gpu_time = 100_000_005_000 - 1000 = 100_000_004_000
        assert_eq!(gpu_time, 100_000_004_000);
    }

    #[test]
    fn test_ptp_to_gpu_time_with_drift() {
        let sync = LinuxPtpClockSync::new(PtpConfig::default());

        let cal = GpuClockCalibration {
            gpu_id: 0,
            offset_ns: 1000,
            drift_rate_ns_per_sec: 10.0, // GPU drifts 10ns per second behind PTP
            calibrated_at: PtpTime::new(100, 0),
            measurement_rtt_ns: 500,
            sample_count: 11,
        };
        sync.set_gpu_calibration(cal);

        // 5 seconds after calibration
        let ptp_time = PtpTime::new(105, 0);
        let gpu_time = sync.ptp_to_gpu_time(ptp_time, 0).expect("should succeed");

        // elapsed = 5s = 5_000_000_000ns
        // drift_correction = 10.0 * 5_000_000_000 / 1_000_000_000 = 50ns
        // gpu_time = 105_000_000_000 - 1000 - 50 = 104_999_998_950
        assert_eq!(gpu_time, 104_999_998_950);
    }

    // -----------------------------------------------------------------------
    // Coordinated launch types tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_coordinated_launch_creation() {
        let launch = CoordinatedLaunch {
            target_time: PtpTime::new(200, 0),
            participating_nodes: vec![0, 1, 2, 3],
            action_id: 42,
            spin_margin: Duration::from_millis(1),
        };

        assert_eq!(launch.participating_nodes.len(), 4);
        assert_eq!(launch.action_id, 42);
    }

    #[test]
    fn test_gpu_launch_params_default() {
        let params = GpuLaunchParams::default();
        assert_eq!(params.target_globaltimer_ns, 0);
        assert_eq!(params.max_spin_cycles, 100_000);
    }

    // -----------------------------------------------------------------------
    // GpuClockCalibration drift model test
    // -----------------------------------------------------------------------

    #[test]
    fn test_calibration_current_offset_with_drift() {
        let cal = GpuClockCalibration {
            gpu_id: 0,
            offset_ns: 1000,
            drift_rate_ns_per_sec: 5.0, // 5ns per second drift
            calibrated_at: PtpTime::new(100, 0),
            measurement_rtt_ns: 500,
            sample_count: 11,
        };

        // 10 seconds later
        let now = PtpTime::new(110, 0);
        let current_offset = cal.current_offset_ns(&now);

        // drift_correction = 5.0 * 10e9 / 1e9 = 50ns
        // current_offset = 1000 + 50 = 1050
        assert_eq!(current_offset, 1050);
    }

    #[test]
    fn test_health_report_is_healthy() {
        let report = PtpHealthReport {
            state: PtpSyncState::Locked { offset_ns: 50 },
            ..Default::default()
        };
        assert!(report.is_healthy());

        let report = PtpHealthReport {
            state: PtpSyncState::Unlocked,
            ..Default::default()
        };
        assert!(!report.is_healthy());
    }
}
