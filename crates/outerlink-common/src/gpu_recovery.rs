//! GPU Xid recovery types and classification logic (R48).
//!
//! Extends [`crate::health`] with recovery-specific types: recovery tiers,
//! per-Xid recovery actions, ECC counters, thermal actions, and a recovery
//! event log. All types are pure data -- no GPU or NVML calls.

use std::time::{Duration, Instant};

use crate::health::{ThermalThresholds, XidSeverity};

// ---------------------------------------------------------------------------
// RecoveryTier
// ---------------------------------------------------------------------------

/// The three recovery tiers from R48, ordered by severity.
///
/// - `ContextRecreate`: destroy and recreate the CUDA context.
/// - `GpuReset`: reset the GPU via NVML / nvidia-smi (all contexts lost).
/// - `PoolEviction`: remove the GPU from the OutterLink pool entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RecoveryTier {
    ContextRecreate,
    GpuReset,
    PoolEviction,
}

// ---------------------------------------------------------------------------
// XidRecoveryAction
// ---------------------------------------------------------------------------

/// Maps a single Xid code to its recovery action.
#[derive(Debug, Clone)]
pub struct XidRecoveryAction {
    /// The Xid error code.
    pub xid_code: u32,
    /// Which recovery tier applies.
    pub tier: RecoveryTier,
    /// Human-readable description of the action.
    pub description: &'static str,
    /// Whether all work on this GPU must be drained before recovery.
    pub requires_drain: bool,
    /// Whether all contexts on the GPU are affected (not just the faulting one).
    pub affects_all_contexts: bool,
}

impl XidRecoveryAction {
    /// Return the [`XidSeverity`] classification from `crate::health` for
    /// this Xid code. Bridges the health-monitoring severity with the
    /// recovery-tier classification.
    /// Derive severity from recovery tier, handling Info-level Xids correctly.
    /// Does not delegate to classify_xid (which has coverage gaps for R48 codes).
    pub fn severity(&self) -> XidSeverity {
        match self.tier {
            RecoveryTier::PoolEviction => XidSeverity::Critical,
            RecoveryTier::GpuReset => XidSeverity::Warning,
            RecoveryTier::ContextRecreate => {
                // Info-level Xids that need no real recovery
                match self.xid_code {
                    45 | 68 | 94 => XidSeverity::Info,
                    _ => XidSeverity::Warning,
                }
            }
        }
    }
}

/// Map an NVIDIA Xid error code to its recovery action.
///
/// Covers all 16 Xid codes cataloged in R48. Unknown codes default to
/// GPU reset (conservative).
pub fn recovery_action(xid: u32) -> XidRecoveryAction {
    match xid {
        // -- Tier 1: Context Recovery --
        13 => XidRecoveryAction {
            xid_code: 13,
            tier: RecoveryTier::ContextRecreate,
            description: "Graphics engine exception; recreate context and re-submit",
            requires_drain: false,
            affects_all_contexts: false,
        },
        32 => XidRecoveryAction {
            xid_code: 32,
            tier: RecoveryTier::ContextRecreate,
            description: "Invalid/corrupted push buffer; recreate context",
            requires_drain: false,
            affects_all_contexts: false,
        },
        45 => XidRecoveryAction {
            xid_code: 45,
            tier: RecoveryTier::ContextRecreate,
            description: "Preemptive cleanup; informational, no action needed",
            requires_drain: false,
            affects_all_contexts: false,
        },
        63 => XidRecoveryAction {
            xid_code: 63,
            tier: RecoveryTier::ContextRecreate,
            description: "ECC page retirement / row remap; track count",
            requires_drain: false,
            affects_all_contexts: false,
        },
        68 => XidRecoveryAction {
            xid_code: 68,
            tier: RecoveryTier::ContextRecreate,
            description: "Video processor exception; irrelevant for compute",
            requires_drain: false,
            affects_all_contexts: false,
        },
        69 => XidRecoveryAction {
            xid_code: 69,
            tier: RecoveryTier::ContextRecreate,
            description: "Graphics engine class error; context dead, recreate",
            requires_drain: false,
            affects_all_contexts: false,
        },
        94 => XidRecoveryAction {
            xid_code: 94,
            tier: RecoveryTier::ContextRecreate,
            description: "Contained ECC error (Ampere+); self-corrected, track frequency",
            requires_drain: false,
            affects_all_contexts: false,
        },

        // -- Tier 2: GPU Reset --
        31 => XidRecoveryAction {
            xid_code: 31,
            tier: RecoveryTier::GpuReset,
            description: "GPU memory page fault; check ECC, GPU reset if recurring",
            requires_drain: true,
            affects_all_contexts: true,
        },
        38 => XidRecoveryAction {
            xid_code: 38,
            tier: RecoveryTier::GpuReset,
            description: "Driver firmware error; GPU reset required",
            requires_drain: true,
            affects_all_contexts: true,
        },
        61 => XidRecoveryAction {
            xid_code: 61,
            tier: RecoveryTier::GpuReset,
            description: "Internal firmware error; GPU reset required",
            requires_drain: true,
            affects_all_contexts: true,
        },
        62 => XidRecoveryAction {
            xid_code: 62,
            tier: RecoveryTier::GpuReset,
            description: "Internal firmware error; GPU reset required",
            requires_drain: true,
            affects_all_contexts: true,
        },
        95 => XidRecoveryAction {
            xid_code: 95,
            tier: RecoveryTier::GpuReset,
            description: "Uncontained ECC error; all contexts affected, GPU reset",
            requires_drain: true,
            affects_all_contexts: true,
        },

        // -- Tier 3: Pool Eviction --
        43 => XidRecoveryAction {
            xid_code: 43,
            tier: RecoveryTier::PoolEviction,
            description: "GPU stopped processing; remove from pool",
            requires_drain: true,
            affects_all_contexts: true,
        },
        48 => XidRecoveryAction {
            xid_code: 48,
            tier: RecoveryTier::PoolEviction,
            description: "Double-bit ECC error; retire GPU from pool",
            requires_drain: true,
            affects_all_contexts: true,
        },
        64 => XidRecoveryAction {
            xid_code: 64,
            tier: RecoveryTier::PoolEviction,
            description: "ECC row remap failure; VRAM degrading, evict GPU",
            requires_drain: true,
            affects_all_contexts: true,
        },
        79 => XidRecoveryAction {
            xid_code: 79,
            tier: RecoveryTier::PoolEviction,
            description: "GPU fallen off PCIe bus; node-level recovery needed",
            requires_drain: true,
            affects_all_contexts: true,
        },

        // -- Unknown: conservative default --
        _ => XidRecoveryAction {
            xid_code: xid,
            tier: RecoveryTier::GpuReset,
            description: "Unknown Xid; defaulting to GPU reset",
            requires_drain: true,
            affects_all_contexts: true,
        },
    }
}

// ---------------------------------------------------------------------------
// GpuRecoveryState
// ---------------------------------------------------------------------------

/// The current recovery state of a GPU in the pool.
#[derive(Debug, Clone)]
pub enum GpuRecoveryState {
    /// GPU is operating normally.
    Normal,
    /// GPU context is being recovered after an Xid error.
    ContextRecovery { xid: u32 },
    /// GPU is undergoing a full reset.
    Resetting,
    /// GPU is draining work before a reset or eviction.
    Draining { reason: String },
    /// GPU has been evicted from the pool.
    Evicted { xid: u32 },
}

// ---------------------------------------------------------------------------
// EccCounters
// ---------------------------------------------------------------------------

/// ECC error counters from NVML, used to decide eviction/investigation.
#[derive(Debug, Clone)]
pub struct EccCounters {
    /// Single-bit (correctable) ECC errors.
    pub correctable_single: u64,
    /// Double-bit (uncorrectable) ECC errors.
    pub uncorrectable_double: u64,
    /// Number of retired VRAM pages.
    pub retired_pages: u32,
    /// Whether there are pages pending retirement (need reboot).
    pub pending_retirement: bool,
}

/// Threshold for retired pages that triggers investigation.
/// R48 Q5: "> 5 retired pages = warning", so 6+ pages triggers.
const RETIRED_PAGE_INVESTIGATION_THRESHOLD: u32 = 6;

impl EccCounters {
    /// Any double-bit error means data may be corrupted -- evict immediately.
    pub fn needs_eviction(&self) -> bool {
        self.uncorrectable_double > 0
    }

    /// High retired page count or pending retirement warrants investigation.
    pub fn needs_investigation(&self) -> bool {
        self.retired_pages >= RETIRED_PAGE_INVESTIGATION_THRESHOLD || self.pending_retirement
    }
}

// ---------------------------------------------------------------------------
// ThermalAction
// ---------------------------------------------------------------------------

/// Thermal response action based on GPU temperature.
///
/// Uses thresholds from [`ThermalThresholds`] to determine the action.
/// Default thresholds are 80C warn, 85C stop, 90C migrate, 95C emergency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThermalAction {
    /// Temperature is nominal; no action.
    None,
    /// Temperature >= throttle_warn; reduce scheduling pressure.
    ReduceLoad,
    /// Temperature >= throttle_stop; stop scheduling new work.
    StopScheduling,
    /// Temperature >= migrate; migrate in-flight work away.
    MigrateWorkloads,
    /// Temperature >= emergency; emergency shutdown of GPU workloads.
    EmergencyShutdown,
}

impl ThermalAction {
    /// Classify a GPU temperature into the appropriate thermal action.
    ///
    /// Uses thresholds from [`ThermalThresholds`] (the single source of truth
    /// for thermal boundaries). See [`ThermalThresholds::default()`] for values.
    pub fn from_temperature(temp_c: f64, thresholds: &ThermalThresholds) -> Self {
        if temp_c >= thresholds.emergency {
            Self::EmergencyShutdown
        } else if temp_c >= thresholds.migrate {
            Self::MigrateWorkloads
        } else if temp_c >= thresholds.throttle_stop {
            Self::StopScheduling
        } else if temp_c >= thresholds.throttle_warn {
            Self::ReduceLoad
        } else {
            Self::None
        }
    }
}

// ---------------------------------------------------------------------------
// GpuRecoveryLog
// ---------------------------------------------------------------------------

/// A single recovery event.
#[derive(Debug, Clone)]
pub struct RecoveryEvent {
    /// When the recovery was attempted.
    pub timestamp: Instant,
    /// The Xid that triggered recovery.
    pub xid: u32,
    /// Which tier of recovery was attempted.
    pub tier: RecoveryTier,
    /// Whether the recovery succeeded.
    pub success: bool,
}

/// Log of GPU recovery events, used for failure rate tracking and diagnostics.
#[derive(Debug, Clone)]
pub struct GpuRecoveryLog {
    events: Vec<RecoveryEvent>,
}

impl GpuRecoveryLog {
    /// Create an empty recovery log.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }

    /// Record a recovery event.
    pub fn push(&mut self, event: RecoveryEvent) {
        self.events.push(event);
    }

    /// Return events whose timestamp is within `window` of now.
    pub fn recent_xids(&self, window: Duration) -> Vec<&RecoveryEvent> {
        let now = Instant::now();
        self.events
            .iter()
            .filter(|e| now.duration_since(e.timestamp) <= window)
            .collect()
    }

    /// Fraction of events that failed (0.0 if empty).
    pub fn failure_rate(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        let failed = self.events.iter().filter(|e| !e.success).count();
        failed as f64 / self.events.len() as f64
    }

    /// Total number of events recorded.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for GpuRecoveryLog {
    fn default() -> Self {
        Self::new()
    }
}
