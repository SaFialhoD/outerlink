# R26: Hardware Clock Sync via PTP -- Pre-Plan v2 (Refined)

**Created:** 2026-03-25
**Last Updated:** 2026-03-25
**Status:** Draft
**Phase:** 9 -- Hardening
**Priority:** HIGH
**Supersedes:** preplan.md (v1)

## Purpose

Second-round refinement of R26's PTP clock synchronization design. This document resolves open questions from v1, provides exact Rust structs and trait implementations, defines concrete protocols for PTP management and coordinated kernel launches, and specifies precise integration points with R17 (topology), R11 (prefetch), R19 (coherency), R25 (cooperative kernel splitting), and R30 (persistent kernels).

---

## 1. Resolved Open Questions (from v1)

### Q1: NTP conflict
**Resolved:** Disable NTP/chronyd on all OuterLink nodes entirely. Replace with PTP as the sole time source. Rationale: OuterLink clusters are LAN-only, wall-clock accuracy from internet time sources is unnecessary, and NTP/PTP conflicts cause oscillating adjustments that destroy precision. The setup guide documents this requirement.

If wall-clock accuracy matters (logging, etc.), configure chrony to use the PTP-synced PHC as its reference clock (`refclock PHC /dev/ptpN poll 3 dpoll -2`). This gives chrony PTP-level accuracy without conflicting with phc2sys.

### Q2: Multiple ConnectX-5 ports
**Resolved:** Run PTP on the primary RDMA interface only (the one carrying OuterLink traffic). The second port (if present) is a failover path -- it gets its own PTP session only if the primary fails. Do NOT run PTP on bonded interfaces; bond-level timestamping introduces jitter from the bonding driver.

### Q3: ptp4l permissions
**Resolved:** ptp4l requires either root or the following capabilities:
- `CAP_NET_RAW` (for L2 transport raw sockets)
- `CAP_SYS_TIME` (for clock adjustment)
- `CAP_NET_ADMIN` (for enabling hardware timestamping)

OuterLink's setup script grants these via `setcap` on the ptp4l binary, avoiding running as root.

### Q4: UDS protocol format
**Resolved:** The pmc UDS protocol is a TLV-based binary protocol on a Unix domain socket at `/var/run/ptp4l`. Rather than implementing the binary protocol directly, we use pmc as a subprocess and parse its text output. This is simpler and handles all TLV formats correctly. For high-frequency monitoring (>1Hz), we parse ptp4l's stdout `summary_interval` output directly via a pipe.

### Q5: statime evaluation
**Resolved:** Defer statime evaluation to Phase 2. The linuxptp approach is proven and introduces zero risk. Re-evaluate statime when: (a) we need tighter servo integration (custom PI constants per-node), or (b) we want to eliminate external process dependencies for containerized deployments.

### Q6: CUDA graph launch jitter
**Resolved:** CUDA graphs reduce launch jitter from ~5-20us to ~2-5us for pre-recorded launch sequences. This matters for R25 (cooperative kernel splitting). The GPU-side spin-wait (Strategy C) reduces effective jitter to <2us regardless of host-side jitter. Combined approach (host PTP schedule + GPU spin) is the production path.

### Q7: Holdover accuracy
**Resolved:** ConnectX-5's TCXO crystal holds <10us accuracy for approximately 10-60 seconds after losing PTP sync (depends on temperature stability). After 60 seconds, drift exceeds 100us. This means: PTP loss detection must trigger within 10 seconds for coordinated launches to remain valid. R15's failure detection (phi accrual at 300-600ms) easily meets this requirement.

### Q8: Temperature sensitivity
**Resolved:** TCXO drift is approximately 0.5-2 ppm per degree C change. At 2 ppm/C, a 5C ambient change causes 10 ppm drift = 10us/second. The PI servo in ptp4l handles this with a ~30 second tracking lag. For GPU clock calibration, calibrate every 5 seconds (default) which keeps GPU-to-PTP error under 5us even during temperature transients.

---

## 2. Rust Structs and Types

### 2.1 Core Clock Types

```rust
/// PTP time with nanosecond precision
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtpTime {
    /// Seconds since epoch (TAI or UTC depending on ptp4l config)
    pub seconds: i64,
    /// Nanoseconds within the second [0, 999_999_999]
    pub nanoseconds: u32,
}

impl PtpTime {
    /// Get current PTP-synced time from system clock.
    /// Requires phc2sys to be running and synced.
    pub fn now() -> Self {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) };
        Self {
            seconds: ts.tv_sec,
            nanoseconds: ts.tv_nsec as u32,
        }
    }

    /// Convert to nanoseconds since epoch
    pub fn as_nanos(&self) -> i128 {
        (self.seconds as i128) * 1_000_000_000 + self.nanoseconds as i128
    }

    /// Create a future time offset from now
    pub fn from_now_plus(offset: Duration) -> Self {
        let now = Self::now();
        let total_ns = now.as_nanos() + offset.as_nanos() as i128;
        Self {
            seconds: (total_ns / 1_000_000_000) as i64,
            nanoseconds: (total_ns % 1_000_000_000) as u32,
        }
    }
}

/// PTP synchronization state (maps to ptp4l servo states)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PtpSyncState {
    /// s0: Collecting initial samples, not yet synced
    Unlocked,
    /// s1: Large offset detected, stepping clock
    Stepping,
    /// s2: Frequency steering active, offset converging
    Locked { offset_ns: i64 },
    /// s2 with offset consistently below threshold
    LockedStable { offset_ns: i64, stable_since: Instant },
    /// PTP lost -- no sync messages from grandmaster
    Holdover { since: Instant, estimated_drift_ns: i64 },
    /// PTP daemon not running or not configured
    Unavailable,
}

/// PTP health report collected from ptp4l
#[derive(Clone, Debug)]
pub struct PtpHealthReport {
    /// Current sync state
    pub state: PtpSyncState,
    /// Current offset from grandmaster (nanoseconds)
    pub master_offset_ns: i64,
    /// Current path delay to grandmaster (nanoseconds)
    pub path_delay_ns: i64,
    /// Frequency adjustment being applied (parts per billion)
    pub freq_ppb: i64,
    /// Identity of the current grandmaster
    pub grandmaster_id: [u8; 8],
    /// Port state (MASTER, SLAVE, LISTENING, etc.)
    pub port_state: PtpPortState,
    /// Timestamp of this report
    pub reported_at: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq)]
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

/// GPU clock calibration state
#[derive(Clone, Debug)]
pub struct GpuClockCalibration {
    /// GPU device index
    pub gpu_id: u32,
    /// Offset: ptp_time = gpu_globaltimer + offset_ns
    pub offset_ns: i64,
    /// Measured drift rate (nanoseconds per second)
    pub drift_rate_ns_per_sec: f64,
    /// When this calibration was taken
    pub calibrated_at: PtpTime,
    /// PCIe round-trip time of the calibration measurement
    pub measurement_rtt_ns: u64,
    /// Number of samples averaged for this calibration
    pub sample_count: u32,
}
```

### 2.2 PTP Configuration Types

```rust
/// PTP configuration generated for a specific node
#[derive(Clone, Debug)]
pub struct PtpConfig {
    /// Network interface for PTP (ConnectX-5 RDMA interface)
    pub interface: String,
    /// BMCA Priority1 (lower = more likely grandmaster)
    pub priority1: u8,
    /// BMCA Priority2 (tiebreaker)
    pub priority2: u8,
    /// PTP domain number (all cluster nodes must match)
    pub domain: u8,
    /// Transport: L2 (raw Ethernet) for best accuracy
    pub transport: PtpTransport,
    /// Sync interval exponent: 2^N seconds (0 = 1 second)
    pub log_sync_interval: i8,
    /// Delay request interval (match sync interval)
    pub log_delay_req_interval: i8,
    /// PTP profile
    pub profile: PtpProfile,
}

#[derive(Clone, Copy, Debug)]
pub enum PtpTransport {
    /// IEEE 802.3 raw Ethernet (lowest jitter)
    L2,
    /// UDP/IPv4 (works through routers, slightly higher jitter)
    UdpIpv4,
}

#[derive(Clone, Copy, Debug)]
pub enum PtpProfile {
    /// IEEE 1588 default profile
    Default1588,
}

impl PtpConfig {
    /// Generate ptp4l.conf content from this config
    pub fn to_ptp4l_conf(&self) -> String {
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
            transport = match self.transport {
                PtpTransport::L2 => "L2",
                PtpTransport::UdpIpv4 => "UDPv4",
            },
            sync = self.log_sync_interval,
            delay = self.log_delay_req_interval,
            p1 = self.priority1,
            p2 = self.priority2,
            domain = self.domain,
            iface = self.interface,
        )
    }

    /// Generate phc2sys command-line arguments
    pub fn phc2sys_args(&self) -> Vec<String> {
        vec![
            "-a".into(),        // Automatic mode
            "-r".into(),        // Sync system clock
            "-r".into(),        // And other clocks
            "-O".into(), "0".into(),  // TAI-UTC offset (0 if ptp4l uses UTC)
        ]
    }
}
```

### 2.3 Coordinated Launch Types

```rust
/// A coordinated action to execute at a specific PTP time
#[derive(Debug)]
pub struct CoordinatedLaunch {
    /// Target PTP time for the action
    pub target_time: PtpTime,
    /// Which nodes participate
    pub participating_nodes: Vec<NodeId>,
    /// Action to perform (opaque to the clock system)
    pub action_id: u64,
    /// Safety margin before target_time to begin spin-wait
    pub spin_margin: Duration,
}

/// Result of a coordinated launch attempt
#[derive(Debug)]
pub struct LaunchResult {
    /// Actual PTP time when the action was triggered
    pub actual_time: PtpTime,
    /// Difference from target (positive = late, negative = early)
    pub jitter_ns: i64,
    /// Whether GPU spin-wait was used
    pub gpu_spin_used: bool,
}

/// GPU-side launch parameters passed to the coordinated kernel wrapper
#[repr(C)]
pub struct GpuLaunchParams {
    /// Target GPU globaltimer value (ptp_target - gpu_calibration_offset)
    pub target_globaltimer_ns: u64,
    /// Maximum spin iterations before giving up (prevents infinite spin)
    pub max_spin_cycles: u64,
}
```

---

## 3. Trait Implementations

### 3.1 ClockSync Service

```rust
/// Central clock synchronization service for OuterLink
pub trait ClockSync: Send + Sync {
    /// Get current PTP-synced time
    fn now_ptp(&self) -> PtpTime;

    /// Check if PTP is synchronized and healthy
    fn is_synced(&self) -> bool;

    /// Get current sync state and health
    fn health(&self) -> PtpHealthReport;

    /// Get current offset from grandmaster (nanoseconds)
    fn offset_ns(&self) -> i64;

    /// Schedule an action at a specific PTP time (host-side spin-wait)
    fn schedule_at<F: FnOnce() + Send + 'static>(
        &self,
        time: PtpTime,
        action: F,
    ) -> Result<()>;

    /// Get GPU clock calibration for a specific GPU
    fn gpu_calibration(&self, gpu_id: u32) -> Option<GpuClockCalibration>;

    /// Convert PTP time to GPU globaltimer value (for GPU-side spin-wait)
    fn ptp_to_gpu_time(&self, ptp_time: PtpTime, gpu_id: u32) -> Result<u64>;
}
```

**Implementation: `LinuxPtpClockSync`**

```rust
pub struct LinuxPtpClockSync {
    /// ptp4l process handle
    ptp4l_process: Mutex<Option<Child>>,
    /// phc2sys process handle
    phc2sys_process: Mutex<Option<Child>>,
    /// Current PTP health (updated by monitor thread)
    health: RwLock<PtpHealthReport>,
    /// ptp4l stdout parser thread
    monitor_thread: Option<JoinHandle<()>>,
    /// Per-GPU calibration state
    gpu_calibrations: DashMap<u32, GpuClockCalibration>,
    /// Calibration worker thread (recalibrates every N seconds)
    calibration_thread: Option<JoinHandle<()>>,
    /// Configuration
    config: PtpConfig,
    /// Calibration interval (default 5 seconds)
    calibration_interval: Duration,
}

impl LinuxPtpClockSync {
    /// Start the PTP subsystem:
    /// 1. Generate ptp4l.conf from config
    /// 2. Spawn ptp4l process
    /// 3. Wait for s2 (locked) state
    /// 4. Spawn phc2sys process
    /// 5. Start monitor thread (parses ptp4l output)
    /// 6. Start GPU calibration thread
    pub fn start(config: PtpConfig) -> Result<Self> {
        // ... implementation
        todo!()
    }

    /// Stop PTP subsystem gracefully
    pub fn stop(&mut self) -> Result<()> {
        // Kill phc2sys first, then ptp4l
        // Stop monitor and calibration threads
        todo!()
    }
}

impl ClockSync for LinuxPtpClockSync {
    fn now_ptp(&self) -> PtpTime {
        PtpTime::now()
    }

    fn is_synced(&self) -> bool {
        matches!(
            self.health.read().state,
            PtpSyncState::Locked { .. } | PtpSyncState::LockedStable { .. }
        )
    }

    fn health(&self) -> PtpHealthReport {
        self.health.read().clone()
    }

    fn offset_ns(&self) -> i64 {
        self.health.read().master_offset_ns
    }

    fn schedule_at<F: FnOnce() + Send + 'static>(
        &self,
        time: PtpTime,
        action: F,
    ) -> Result<()> {
        // 1. Verify PTP is synced
        if !self.is_synced() {
            return Err(Error::PtpNotSynced);
        }

        // 2. Spawn a high-priority thread that:
        //    a. Sleeps until ~1ms before target time
        //    b. Busy-spins on clock_gettime until target time
        //    c. Executes the action
        std::thread::Builder::new()
            .name("ptp-scheduler".into())
            .spawn(move || {
                let target_ns = time.as_nanos();

                // Coarse sleep until close to target
                loop {
                    let now = PtpTime::now().as_nanos();
                    let remaining = target_ns - now;
                    if remaining <= 1_000_000 {
                        break; // Within 1ms, switch to spin
                    }
                    std::thread::sleep(Duration::from_micros(
                        (remaining / 2000) as u64
                    ));
                }

                // Fine spin-wait
                loop {
                    let now = PtpTime::now().as_nanos();
                    if now >= target_ns {
                        break;
                    }
                    std::hint::spin_loop();
                }

                action();
            })?;
        Ok(())
    }

    fn gpu_calibration(&self, gpu_id: u32) -> Option<GpuClockCalibration> {
        self.gpu_calibrations.get(&gpu_id).map(|c| c.clone())
    }

    fn ptp_to_gpu_time(&self, ptp_time: PtpTime, gpu_id: u32) -> Result<u64> {
        let cal = self.gpu_calibrations.get(&gpu_id)
            .ok_or(Error::GpuNotCalibrated)?;

        let ptp_ns = ptp_time.as_nanos() as i64;
        let elapsed_since_cal = ptp_ns - cal.calibrated_at.as_nanos() as i64;
        let drift_correction = (cal.drift_rate_ns_per_sec * elapsed_since_cal as f64
            / 1_000_000_000.0) as i64;

        // gpu_time = ptp_time - offset - drift_correction
        let gpu_time = ptp_ns - cal.offset_ns - drift_correction;
        Ok(gpu_time as u64)
    }
}
```

### 3.2 GPU Clock Calibration

```rust
impl LinuxPtpClockSync {
    /// Perform GPU-to-PTP calibration using bracketing technique.
    /// Takes multiple samples and picks the one with lowest PCIe RTT.
    fn calibrate_gpu(&self, gpu_id: u32) -> Result<GpuClockCalibration> {
        const NUM_SAMPLES: usize = 11;
        const MAX_RTT_NS: u64 = 5_000; // Reject samples with RTT > 5us

        let mut best_sample: Option<(i64, u64, u64)> = None; // (offset, rtt, gpu_time)

        for _ in 0..NUM_SAMPLES {
            // Step 1: Read host clock (PTP-synced)
            let t1_host = self.read_clock_ns();

            // Step 2: Read GPU globaltimer via a tiny CUDA kernel
            // or via CUPTI cuptiDeviceGetTimestamp
            let t_gpu = self.read_gpu_globaltimer(gpu_id)?;

            // Step 3: Read host clock again
            let t2_host = self.read_clock_ns();

            let rtt = (t2_host - t1_host) as u64;
            if rtt > MAX_RTT_NS {
                continue; // PCIe was delayed, skip this sample
            }

            let host_midpoint = (t1_host + t2_host) / 2;
            let offset = host_midpoint - t_gpu;

            match &best_sample {
                None => best_sample = Some((offset, rtt, t_gpu as u64)),
                Some((_, best_rtt, _)) if rtt < *best_rtt => {
                    best_sample = Some((offset, rtt, t_gpu as u64));
                }
                _ => {}
            }
        }

        let (offset, rtt, _) = best_sample.ok_or(Error::CalibrationFailed)?;

        // Compute drift rate from previous calibration (if exists)
        let drift_rate = if let Some(prev) = self.gpu_calibrations.get(&gpu_id) {
            let dt_ns = self.read_clock_ns() - prev.calibrated_at.as_nanos() as i64;
            if dt_ns > 0 {
                let offset_change = offset - prev.offset_ns;
                (offset_change as f64 / dt_ns as f64) * 1_000_000_000.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(GpuClockCalibration {
            gpu_id,
            offset_ns: offset,
            drift_rate_ns_per_sec: drift_rate,
            calibrated_at: PtpTime::now(),
            measurement_rtt_ns: rtt,
            sample_count: NUM_SAMPLES as u32,
        })
    }

    fn read_clock_ns(&self) -> i64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) };
        ts.tv_sec as i64 * 1_000_000_000 + ts.tv_nsec as i64
    }

    fn read_gpu_globaltimer(&self, gpu_id: u32) -> Result<i64> {
        // Option A: Launch a minimal CUDA kernel that reads %globaltimer
        //           and writes it to a mapped pinned buffer
        // Option B: Use CUPTI cuptiDeviceGetTimestamp()
        //
        // We use Option B for lower overhead (no kernel launch needed).
        // CUPTI timestamp is from the same clock domain as %globaltimer.
        todo!()
    }
}
```

### 3.3 PTP Health Monitor

```rust
/// Monitors ptp4l output and updates health state
struct PtpHealthMonitor {
    health: Arc<RwLock<PtpHealthReport>>,
    /// Regex for parsing ptp4l summary line
    /// Format: "ptp4l[XXX.XXX]: master offset   23 s2 freq +1234 path delay   456"
    offset_regex: Regex,
}

impl PtpHealthMonitor {
    fn new(health: Arc<RwLock<PtpHealthReport>>) -> Self {
        Self {
            health,
            offset_regex: Regex::new(
                r"master offset\s+(-?\d+)\s+(s\d)\s+freq\s+([+-]?\d+)\s+path delay\s+(\d+)"
            ).unwrap(),
        }
    }

    /// Parse a single line from ptp4l stdout
    fn parse_line(&self, line: &str) {
        if let Some(caps) = self.offset_regex.captures(line) {
            let offset: i64 = caps[1].parse().unwrap_or(0);
            let servo_state = &caps[2];
            let freq: i64 = caps[3].parse().unwrap_or(0);
            let path_delay: i64 = caps[4].parse().unwrap_or(0);

            let state = match servo_state {
                "s0" => PtpSyncState::Unlocked,
                "s1" => PtpSyncState::Stepping,
                "s2" => {
                    if offset.abs() < 100 {
                        PtpSyncState::LockedStable {
                            offset_ns: offset,
                            stable_since: Instant::now(),
                        }
                    } else {
                        PtpSyncState::Locked { offset_ns: offset }
                    }
                }
                _ => PtpSyncState::Unavailable,
            };

            let mut health = self.health.write();
            health.state = state;
            health.master_offset_ns = offset;
            health.path_delay_ns = path_delay;
            health.freq_ppb = freq;
            health.reported_at = Instant::now();
        }
    }

    /// Run the monitor loop, reading from ptp4l's stdout pipe
    fn run(&self, stdout: impl BufRead) {
        for line in stdout.lines() {
            match line {
                Ok(l) => self.parse_line(&l),
                Err(_) => {
                    let mut health = self.health.write();
                    health.state = PtpSyncState::Unavailable;
                    break;
                }
            }
        }
    }
}
```

---

## 4. Algorithms and Protocols

### 4.1 PTP Daemon Lifecycle Management

**Startup sequence:**

1. Verify ConnectX-5 supports hardware timestamping: `ethtool -T <iface>` must report hardware TX/RX timestamps.
2. Identify PTP hardware clock device: parse `ethtool -T` output for PHC index, map to `/dev/ptpN`.
3. Disable NTP/chrony on this node (if running): `systemctl stop chronyd ntp`.
4. Generate `ptp4l.conf` from `PtpConfig` (including priority1 from R17 topology manager).
5. Spawn ptp4l: `ptp4l -f /tmp/outerlink-ptp4l.conf -m` with stdout piped to monitor.
6. Wait for servo state s2 (locked): monitor parses ptp4l output, timeout after 120 seconds.
7. Spawn phc2sys: `phc2sys -a -r -r` to sync system clock to PHC.
8. Wait for phc2sys to report stable sync (offset < 1us consistently for 5 seconds).
9. Begin GPU calibration loop.
10. Report PTP_READY to cluster coordinator.

**Shutdown sequence:**

1. Stop GPU calibration thread.
2. Send SIGTERM to phc2sys, wait up to 2 seconds.
3. Send SIGTERM to ptp4l, wait up to 2 seconds.
4. Re-enable chrony/NTP if configured as fallback.

**Failure recovery:**

- If ptp4l crashes: restart it. PHC retains its last frequency adjustment, so re-convergence is fast (~10 seconds to s2).
- If phc2sys crashes: restart it. System clock drifts at <10 ppm until phc2sys re-locks.
- If grandmaster node fails: BMCA automatically promotes backup (priority1=110) within one announce interval (~2 seconds). All slaves re-sync with ~10us transient offset spike.

### 4.2 Grandmaster Selection Protocol (R17 Integration)

**Step-by-step:**

1. R17 topology manager determines node roles at cluster formation.
2. The **coordinator node** gets `priority1 = 100` (highest priority grandmaster).
3. The **secondary node** (if designated) gets `priority1 = 110`.
4. All other nodes get `priority1 = 128` (default, slave-only).
5. When topology changes (node join/leave/fail):
   a. R17 calls `clock_sync.update_priority(node_id, new_priority1)`.
   b. If the affected node is running ptp4l, update is applied via pmc UDS command:
      `pmc -u -b 0 "SET GRANDMASTER_SETTINGS_NP clockClass 248 clockAccuracy 0xfe offsetScaledLogVariance 0xffff currentUtcOffset 37 leap61 0 leap59 0 currentUtcOffsetValid 0 ptpTimescale 1 timeTraceable 0 frequencyTraceable 0 timeSource 0xa0"`
   c. BMCA re-evaluates and triggers grandmaster change if needed.

**R17 v2 integration:** `NodeInfo` in R17 v2 includes `ptp_offset_ns: i64`. This field is populated from `ClockSync::offset_ns()` and included in heartbeat payloads. R17 uses it to:
- Monitor clock sync health across the cluster
- Trigger alerts if any node's PTP offset exceeds 1us
- Include one-way delay measurements in `LinkInfo.ptp_one_way_delay` (requires PTP timestamps on RDMA completions)

### 4.3 Coordinated Kernel Launch Protocol

**Host-side PTP scheduling (Strategy A -- default):**

1. Coordinator decides: "All nodes launch kernel K at PTP time T" where T = now + safety_margin (default 10ms).
2. Coordinator sends `CoordinatedLaunch { target_time: T, nodes: [...], action_id: 42 }` via RDMA multicast (UD) to all participating nodes.
3. Each node receives the message, verifies its PTP is synced, and calls:
   ```rust
   clock_sync.schedule_at(T, || {
       cuda_launch_kernel(K, args);
   });
   ```
4. The scheduler thread sleeps until ~1ms before T, then spin-waits on `clock_gettime`.
5. At time T (within ~50-100ns PTP accuracy + ~20ns clock_gettime overhead), `cudaLaunchKernel` is called.
6. **Effective jitter: 5-20us** (dominated by CUDA launch pipeline, not PTP).

**Hybrid host + GPU spin (Strategy D -- for R25/R30):**

1. Same as Strategy A for steps 1-4.
2. Instead of launching the actual kernel at time T, launch a **wrapper kernel** that includes a spin preamble:
   ```
   __global__ void coordinated_wrapper(GpuLaunchParams params, ...) {
       // Read current GPU globaltimer
       uint64_t now;
       asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));

       // Spin until target time (calibrated GPU time)
       while (now < params.target_globaltimer_ns) {
           asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
           if (--max_spin > 0) continue; else break; // safety exit
       }

       // All GPUs aligned to within ~1-2us
       actual_kernel_body(...);
   }
   ```
3. The host launches the wrapper kernel ~10us early (within jitter window).
4. The GPU spin-wait aligns all GPUs to the calibrated target time.
5. **Effective jitter: 1-2us** (GPU globaltimer resolution + calibration error).

**Converting PTP target to GPU globaltimer value:**
```rust
let gpu_target = clock_sync.ptp_to_gpu_time(target_ptp_time, gpu_id)?;
let params = GpuLaunchParams {
    target_globaltimer_ns: gpu_target,
    max_spin_cycles: 100_000, // ~100us max spin at ~1ns per iteration
};
```

### 4.4 GPU Clock Calibration Protocol

**Runs every `calibration_interval` (default 5 seconds) per GPU:**

1. Set current thread to SCHED_FIFO priority (minimize scheduling jitter).
2. Take 11 bracketed samples (see `calibrate_gpu()` in Section 3.2).
3. Filter: discard samples with PCIe RTT > 5us (these had contention).
4. Select the sample with the lowest RTT (most accurate midpoint estimate).
5. Compute offset: `offset = host_midpoint - gpu_time`.
6. Compute drift rate from previous calibration (if available).
7. Store new `GpuClockCalibration` in the calibration map.
8. If drift rate exceeds 50 ppm: log warning (unusual, may indicate thermal issue).

**Linear drift model between calibrations:**
```
corrected_gpu_time(t) = raw_gpu_time(t) + offset + drift_rate * (t - calibration_time)
```
This model is accurate to ~500ns for 5-second calibration intervals with typical crystal drift (<10 ppm).

---

## 5. Integration Points (Exact Function Calls)

### 5.1 R17 (Topology-Aware Scheduling) Integration

| R26 uses from R17 | Purpose |
|--------------------|---------|
| `topology.get_node_role(node_id)` | Determine priority1 for BMCA (coordinator = 100, secondary = 110, worker = 128) |
| `topology.on_role_change(callback)` | Get notified when node roles change to update PTP priority |
| `topology.heartbeat_payload` | R17 v2 heartbeat includes `ptp_offset_ns` field populated by R26 |

| R26 provides to R17 | Purpose |
|----------------------|---------|
| `clock_sync.offset_ns()` | Fills `NodeInfo.ptp_offset_ns` in R17's topology data |
| `clock_sync.health().path_delay_ns` | Fills `LinkInfo.ptp_one_way_delay` for accurate latency measurement |
| `clock_sync.is_synced()` | R17 can flag nodes with broken PTP as degraded |

### 5.2 R11 (Prefetch) Integration

| R26 provides to R11 | Purpose |
|----------------------|---------|
| `clock_sync.now_ptp()` | PTP timestamps on prefetch prediction deadlines. R11's prefetch engine predicts when data will be needed; PTP time makes these predictions meaningful across nodes. |
| Accuracy guarantee: <1us | R11 can set prefetch deadlines with microsecond precision: "data must arrive at GPU by PTP time T" |

### 5.3 R19 (Coordinated Kernel Launch for Coherency) Integration

| R26 provides to R19 | Purpose |
|----------------------|---------|
| `clock_sync.schedule_at(time, action)` | R19 can use PTP-coordinated timing for barrier-based coherency points. When multiple nodes need to observe a consistent snapshot, they can synchronize to a PTP timestamp. |

### 5.4 R25 (Cooperative Kernel Splitting) Integration

| R26 provides to R25 | Purpose |
|----------------------|---------|
| `clock_sync.ptp_to_gpu_time(ptp_time, gpu_id)` | Convert coordinated launch time to GPU globaltimer for spin-wait |
| `GpuLaunchParams` struct | Passed to split kernel wrapper for GPU-side synchronization |
| Accuracy guarantee: <5us with hybrid approach | R25 needs sub-5us coordinated launch; GPU spin achieves ~1-2us |

**R25 calling pattern:**
```rust
// R25's kernel splitting coordinator:
let target = PtpTime::from_now_plus(Duration::from_millis(10));
for (node, gpu) in participating_gpus {
    let gpu_target = clock_sync.ptp_to_gpu_time(target, gpu)?;
    let params = GpuLaunchParams {
        target_globaltimer_ns: gpu_target,
        max_spin_cycles: 100_000,
    };
    // Send params to node, which launches the split kernel with spin preamble
    transport.send_launch_command(node, kernel_id, params)?;
}
```

### 5.5 R30 (Persistent Kernels) Integration

| R26 provides to R30 | Purpose |
|----------------------|---------|
| `clock_sync.ptp_to_gpu_time()` | Persistent kernels use "doorbell" timing for network data handoff. PTP time ensures the producer (network receive) and consumer (persistent kernel) agree on when data is ready. |
| GPU calibration data | R30's persistent kernel reads `%globaltimer` to know when to check for new data. The calibration offset ensures this aligns with PTP time of the network transfer completion. |

---

## 6. Refined Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|-------------|--------------|
| **R26-1: PTP Infrastructure** | 2 weeks | ConnectX-5 RDMA working, R17 topology manager | ptp4l/phc2sys lifecycle management, config generation, grandmaster selection, health monitoring, verification tests |
| **R26-2: Clock API & Profiling** | 1.5 weeks | R26-1 complete | `LinuxPtpClockSync` service, `PtpTime` API, `now_ptp()`, `is_synced()`, `offset_ns()`, distributed event timestamps |
| **R26-3: GPU Calibration** | 1 week | R26-2 + CUDA runtime available | `GpuClockCalibration`, bracketing technique, drift model, periodic recalibration loop |
| **R26-4: Coordinated Launches** | 1.5 weeks | R26-2 + R26-3 | `schedule_at()` API, host-side spin-wait, GPU-side spin-wait wrapper, `ptp_to_gpu_time()`, jitter benchmarks |
| **R26-5: Hardening** | 1 week | All above | Grandmaster failover test, holdover behavior test, 24h drift verification, performance impact measurement |

**Total: 7 weeks** (previously 5-8; tightened with clearer specs, same ballpark)

**Parallelism:** R26-1 is sequential prerequisite. R26-2 and R26-3 can partially overlap (calibration needs basic clock API). R26-4 needs both R26-2 and R26-3. R26-5 is final integration.

---

## 7. Success Criteria (Updated)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Cross-node clock offset | **<1 microsecond** sustained | ptp4l master offset log, 24h run |
| Typical cross-node offset | **<100 nanoseconds** | ptp4l log median over 1 hour |
| Clock offset after grandmaster failover | **<10 microseconds** within 5 seconds | Kill grandmaster, measure recovery |
| Coordinated kernel launch jitter (host-side) | **<20 microseconds** | Measure cudaLaunchKernel timing spread across 4 nodes |
| Coordinated kernel launch jitter (GPU spin) | **<2 microseconds** | Measure %globaltimer alignment at kernel start |
| GPU-to-PTP calibration accuracy | **<1 microsecond** | Compare GPU timestamps to PTP-synced system clock (best sample) |
| PTP convergence time (cold start) | **<2 minutes** to <1us offset | Time from ptp4l start to stable s2 state |
| Calibration overhead | **<0.1%** CPU | perf stat during calibration loop |
| Zero impact on transport throughput | **<1% bandwidth reduction** | perftest with and without PTP running |
| **NEW:** Holdover accuracy at 30s | **<10 microseconds** | Stop grandmaster, measure slave drift |
| **NEW:** Profiling timeline accuracy | **<1 microsecond** across nodes | Cross-correlate RDMA completions (known events) |

---

## Related Documents

- [preplan.md](preplan.md) -- v1 pre-plan (superseded by this document)
- [research/01-ptp-protocol-and-hardware.md](research/01-ptp-protocol-and-hardware.md)
- [research/02-linux-ptp-stack.md](research/02-linux-ptp-stack.md)
- [research/03-gpu-clock-integration.md](research/03-gpu-clock-integration.md)
- R17: Topology-Aware Scheduling (NodeInfo.ptp_offset_ns, LinkInfo.ptp_one_way_delay, grandmaster selection)
- R11: Prefetch Scheduling (PTP-timestamped deadlines)
- R19: Network Page Faults (coordinated coherency barriers)
- R25: Cooperative Kernel Splitting (sub-5us coordinated GPU launches)
- R30: Persistent Kernels (doorbell timing coordination)
- R15: Fault Tolerance (failure detection triggers PTP holdover awareness)
