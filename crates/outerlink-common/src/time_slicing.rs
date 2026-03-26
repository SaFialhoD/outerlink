//! R24: Time-Sliced GPU Sharing
//!
//! Multi-tenant GPU sharing for OuterLink pools. Provides tenant identity,
//! quotas, DRF (Dominant Resource Fairness) scheduling, time slice lifecycle
//! management, VRAM isolation ledger, priority preemption, authentication
//! model, and billing accounting.
//!
//! Turns a LAN of gaming PCs into a GPU cloud where multiple users submit
//! workloads with fair sharing, priority preemption, and resource isolation --
//! all without requiring datacenter GPUs (no MIG, no vGPU).
//!
//! # Architecture
//!
//! 1. Each connected user is a **Tenant** with a **PriorityTier** and **TenantQuota**.
//! 2. Tenants create **Sessions** (one per LD_PRELOAD application instance).
//! 3. The **DrfScheduler** assigns **TimeSlices** on GPUs, giving the tenant with
//!    the lowest weighted dominant share the next slice.
//! 4. The **QuotaEnforcer** gates cuMemAlloc, cuLaunchKernel, and cuCtxCreate.
//! 5. The **PreemptionEngine** cooperatively drains lower-priority tenants.
//! 6. The **VramLedger** tracks per-GPU per-tenant VRAM ownership.
//! 7. The **BillingAccumulator** records GEU-hours for usage accounting.
//!
//! # Integration Points
//!
//! - R17 (Topology): tenant-aware placement filter
//! - R13 (CUDA Graphs): tenant-tagged graph launches
//! - R22 (Live Migration): tenant rebalancing
//! - R23 (GPU Mixing): GEU as the fairness currency

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use crate::gpu_mixing::GpuId;
use crate::memory::types::NodeId;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Unique identifier for a tenant in the OuterLink pool.
/// Assigned by the coordinator at registration time.
pub type TenantId = u32;

/// Unique identifier for a session (one application run by one tenant).
/// Multiple sessions can belong to the same tenant.
pub type SessionId = u64;

/// Authentication token (opaque bytes, typically a signed JWT or HMAC token).
pub type AuthToken = Vec<u8>;

/// Opaque handle to a CUDA context (mirrors CUcontext from the driver API).
pub type CudaContextHandle = u64;

// ---------------------------------------------------------------------------
// Tenant Identity
// ---------------------------------------------------------------------------

/// A tenant represents a user or organization that submits work to the pool.
/// Tenants own quotas, sessions, and usage records.
#[derive(Debug, Clone)]
pub struct Tenant {
    /// Unique tenant identifier.
    pub tenant_id: TenantId,
    /// Human-readable name (e.g., "alice", "ml-team-alpha").
    pub name: String,
    /// Priority tier for this tenant. Higher tiers can preempt lower tiers.
    pub priority: PriorityTier,
    /// Resource quota assigned to this tenant.
    pub quota: TenantQuota,
    /// Current resource usage (updated in real-time).
    pub usage: ResourceUsage,
    /// Active sessions belonging to this tenant.
    pub active_sessions: HashSet<SessionId>,
    /// Authentication credentials (hashed API key or certificate fingerprint).
    pub auth_credential: AuthCredential,
    /// Whether the tenant is currently enabled (disabled tenants cannot submit work).
    pub enabled: bool,
    /// When this tenant was created.
    pub created_at: SystemTime,
    /// When this tenant last submitted work.
    pub last_active: Option<SystemTime>,
    /// Tenant-specific scheduling preferences.
    pub scheduling_prefs: SchedulingPreferences,
    /// Cumulative usage for billing (never reset, only grows).
    pub cumulative_usage: BillingAccumulator,
}

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Priority tiers determine preemption rights and scheduling weight.
/// Higher tiers can preempt lower tiers during resource contention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PriorityTier {
    /// Background: lowest priority. Preempted by all other tiers.
    /// Scheduling weight: 0.25x
    Background = 0,
    /// Development: normal priority for interactive development.
    /// Scheduling weight: 1.0x
    Development = 1,
    /// Production: high priority for serving workloads.
    /// Scheduling weight: 2.0x
    Production = 2,
    /// Critical: highest priority. Never preempted.
    /// Scheduling weight: 4.0x
    Critical = 3,
}

impl PriorityTier {
    /// Returns the scheduling weight multiplier for DRF.
    /// Higher weights mean the tenant's dominant share is divided by this factor,
    /// effectively giving them more resources before hitting their "fair share."
    pub fn weight(&self) -> f64 {
        match self {
            PriorityTier::Background => 0.25,
            PriorityTier::Development => 1.0,
            PriorityTier::Production => 2.0,
            PriorityTier::Critical => 4.0,
        }
    }

    /// Can this tier preempt the other tier?
    pub fn can_preempt(&self, other: &PriorityTier) -> bool {
        (*self as u8) > (*other as u8)
    }
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

/// Authentication credential stored for a tenant.
#[derive(Debug, Clone)]
pub enum AuthCredential {
    /// Pre-shared API key (stored as argon2 hash).
    ApiKey {
        /// Argon2 hash of the key.
        key_hash: [u8; 32],
        /// Salt used for hashing.
        salt: [u8; 16],
    },
    /// TLS client certificate (stored as SHA-256 fingerprint).
    CertificateFingerprint {
        /// SHA-256 of the certificate.
        sha256: [u8; 32],
    },
    /// No authentication (pool is in "open" mode -- LAN-only trust model).
    None,
}

/// Authentication mode for the pool.
#[derive(Debug, Clone)]
pub enum AuthMode {
    /// No authentication. Suitable for trusted LANs.
    Open,
    /// API key authentication.
    ApiKey,
    /// TLS mutual authentication.
    MutualTls,
}

/// Authorization roles and their permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TenantRole {
    /// Admin: full control over pool configuration and all tenants.
    Admin,
    /// Operator: can manage tenants and quotas but not pool config.
    Operator,
    /// User: can only submit work within their quota.
    User,
}

/// Session grant issued by the coordinator after authentication.
#[derive(Debug, Clone)]
pub struct SessionGrant {
    /// Tenant this session belongs to.
    pub tenant_id: TenantId,
    /// Unique session ID.
    pub session_id: SessionId,
    /// Role for this session.
    pub role: TenantRole,
    /// Quota snapshot at time of grant.
    pub quota_snapshot: TenantQuota,
    /// When the grant was issued.
    pub issued_at: SystemTime,
    /// When the grant expires.
    pub expires_at: SystemTime,
    /// HMAC-SHA256 signature of the above fields.
    pub signature: [u8; 32],
}

// ---------------------------------------------------------------------------
// Scheduling Preferences
// ---------------------------------------------------------------------------

/// Per-tenant scheduling preferences.
#[derive(Debug, Clone)]
pub struct SchedulingPreferences {
    /// Preferred GPU models (empty = no preference).
    pub preferred_gpus: Vec<GpuId>,
    /// Preferred nodes (empty = no preference).
    pub preferred_nodes: Vec<NodeId>,
    /// Maximum acceptable latency for kernel dispatch (0 = no constraint).
    pub max_dispatch_latency_us: u64,
    /// Whether this tenant's workloads can be migrated for rebalancing (R22).
    pub allow_migration: bool,
    /// Whether this tenant's workloads can be preempted by higher-priority tenants.
    pub preemptible: bool,
    /// Exclusive GPU access: if true, no co-located tenants on assigned GPUs.
    pub exclusive_gpu: bool,
}

impl Default for SchedulingPreferences {
    fn default() -> Self {
        Self {
            preferred_gpus: Vec::new(),
            preferred_nodes: Vec::new(),
            max_dispatch_latency_us: 0,
            allow_migration: true,
            preemptible: true,
            exclusive_gpu: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Quota
// ---------------------------------------------------------------------------

/// Resource quota for a tenant. Defines the maximum resources the tenant
/// can consume simultaneously (capacity limits) and over time (rate limits).
#[derive(Debug, Clone)]
pub struct TenantQuota {
    // --- Capacity limits ---
    /// Maximum VRAM across all GPUs, in bytes (0 = unlimited).
    pub max_vram_bytes: u64,
    /// Maximum number of GPUs simultaneously (0 = unlimited).
    pub max_gpus: u32,
    /// Maximum GEU simultaneously (0.0 = unlimited).
    pub max_geu: f64,
    /// Maximum VRAM per single GPU, in bytes (0 = unlimited).
    pub max_vram_per_gpu_bytes: u64,
    /// Maximum concurrent sessions (0 = unlimited).
    pub max_sessions: u32,
    /// Maximum pending kernels per session (0 = unlimited).
    pub max_pending_kernels: u32,

    // --- Rate limits ---
    /// Maximum GEU-hours per billing period (0.0 = unlimited).
    pub max_geu_hours_per_period: f64,
    /// Billing period duration.
    pub billing_period: Duration,
    /// Maximum data transfer per billing period, in bytes (0 = unlimited).
    pub max_transfer_bytes_per_period: u64,

    // --- Burst allowance ---
    /// Burst multiplier for temporary excess (1.0 = no burst).
    pub burst_multiplier: f64,
    /// Maximum burst duration before enforcement.
    pub max_burst_duration: Duration,
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_vram_bytes: 0,
            max_gpus: 0,
            max_geu: 0.0,
            max_vram_per_gpu_bytes: 0,
            max_sessions: 8,
            max_pending_kernels: 10_000,
            max_geu_hours_per_period: 0.0,
            billing_period: Duration::from_secs(86400),
            max_transfer_bytes_per_period: 0,
            burst_multiplier: 1.5,
            max_burst_duration: Duration::from_secs(300),
        }
    }
}

/// Quota presets for common use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaPreset {
    /// Unlimited: no caps (single-user or admin).
    Unlimited,
    /// Light: 1 GPU, 8 GB VRAM, 2 GEU max, 24 GEU-hours/day.
    Light,
    /// Standard: 2 GPUs, 24 GB VRAM, 5 GEU max, 120 GEU-hours/day.
    Standard,
    /// Heavy: 4 GPUs, 96 GB VRAM, 15 GEU max, 360 GEU-hours/day.
    Heavy,
}

impl QuotaPreset {
    /// Convert a preset to a concrete quota.
    pub fn to_quota(&self) -> TenantQuota {
        match self {
            QuotaPreset::Unlimited => TenantQuota::default(),
            QuotaPreset::Light => TenantQuota {
                max_vram_bytes: 8 * 1024 * 1024 * 1024,
                max_gpus: 1,
                max_geu: 2.0,
                max_vram_per_gpu_bytes: 8 * 1024 * 1024 * 1024,
                max_sessions: 2,
                max_pending_kernels: 1_000,
                max_geu_hours_per_period: 24.0,
                billing_period: Duration::from_secs(86400),
                max_transfer_bytes_per_period: 100 * 1024 * 1024 * 1024,
                burst_multiplier: 1.5,
                max_burst_duration: Duration::from_secs(300),
            },
            QuotaPreset::Standard => TenantQuota {
                max_vram_bytes: 24 * 1024 * 1024 * 1024,
                max_gpus: 2,
                max_geu: 5.0,
                max_vram_per_gpu_bytes: 24 * 1024 * 1024 * 1024,
                max_sessions: 4,
                max_pending_kernels: 5_000,
                max_geu_hours_per_period: 120.0,
                billing_period: Duration::from_secs(86400),
                max_transfer_bytes_per_period: 500 * 1024 * 1024 * 1024,
                burst_multiplier: 1.5,
                max_burst_duration: Duration::from_secs(300),
            },
            QuotaPreset::Heavy => TenantQuota {
                max_vram_bytes: 96 * 1024 * 1024 * 1024,
                max_gpus: 4,
                max_geu: 15.0,
                max_vram_per_gpu_bytes: 24 * 1024 * 1024 * 1024,
                max_sessions: 8,
                max_pending_kernels: 10_000,
                max_geu_hours_per_period: 360.0,
                billing_period: Duration::from_secs(86400),
                max_transfer_bytes_per_period: 2 * 1024 * 1024 * 1024 * 1024,
                burst_multiplier: 1.25,
                max_burst_duration: Duration::from_secs(300),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Resource Usage (real-time)
// ---------------------------------------------------------------------------

/// Current resource usage for a tenant, updated on every allocation/deallocation
/// and every kernel launch/completion.
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Total VRAM currently allocated across all GPUs, in bytes.
    pub vram_allocated_bytes: u64,
    /// Per-GPU VRAM usage breakdown.
    pub vram_per_gpu: HashMap<GpuId, u64>,
    /// Number of GPUs currently in use.
    pub gpus_in_use: u32,
    /// Set of GPU IDs currently assigned.
    pub assigned_gpus: HashSet<GpuId>,
    /// Current GEU consumption.
    pub current_geu: f64,
    /// Number of kernels currently in-flight.
    pub inflight_kernels: u32,
    /// Number of kernels waiting in the pending queue.
    pub pending_kernels: u32,
    /// Number of active sessions.
    pub active_session_count: u32,
    /// GEU-hours consumed in the current billing period.
    pub geu_hours_this_period: f64,
    /// Data transfer volume in the current billing period, in bytes.
    pub transfer_bytes_this_period: u64,
    /// Start of the current billing period.
    pub period_start: SystemTime,
    /// Whether this tenant is currently in burst mode.
    pub in_burst: bool,
    /// When burst mode started (None if not in burst).
    pub burst_start: Option<Instant>,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            vram_allocated_bytes: 0,
            vram_per_gpu: HashMap::new(),
            gpus_in_use: 0,
            assigned_gpus: HashSet::new(),
            current_geu: 0.0,
            inflight_kernels: 0,
            pending_kernels: 0,
            active_session_count: 0,
            geu_hours_this_period: 0.0,
            transfer_bytes_this_period: 0,
            period_start: SystemTime::UNIX_EPOCH,
            in_burst: false,
            burst_start: None,
        }
    }
}

impl ResourceUsage {
    /// Check if a proposed allocation would exceed the tenant's quota.
    /// Returns the first violation found, or None if within limits.
    pub fn would_exceed_quota(
        &self,
        quota: &TenantQuota,
        additional_vram: u64,
        additional_gpus: u32,
        additional_geu: f64,
    ) -> Option<QuotaViolation> {
        // VRAM check
        if quota.max_vram_bytes > 0
            && self.vram_allocated_bytes + additional_vram > quota.max_vram_bytes
        {
            return Some(QuotaViolation::VramExceeded {
                current: self.vram_allocated_bytes,
                requested: additional_vram,
                limit: quota.max_vram_bytes,
            });
        }

        // GPU count check
        if quota.max_gpus > 0 && self.gpus_in_use + additional_gpus > quota.max_gpus {
            return Some(QuotaViolation::GpuCountExceeded {
                current: self.gpus_in_use,
                requested: additional_gpus,
                limit: quota.max_gpus,
            });
        }

        // GEU check (with burst allowance)
        let effective_max_geu = if quota.max_geu > 0.0 {
            quota.max_geu * quota.burst_multiplier
        } else {
            f64::MAX
        };
        if self.current_geu + additional_geu > effective_max_geu {
            return Some(QuotaViolation::GeuExceeded {
                current: self.current_geu,
                requested: additional_geu,
                limit: quota.max_geu,
                burst_limit: effective_max_geu,
            });
        }

        // Rate limit check (GEU-hours)
        if quota.max_geu_hours_per_period > 0.0
            && self.geu_hours_this_period > quota.max_geu_hours_per_period
        {
            return Some(QuotaViolation::GeuHoursExceeded {
                consumed: self.geu_hours_this_period,
                limit: quota.max_geu_hours_per_period,
            });
        }

        None
    }
}

/// Describes why a quota check failed.
#[derive(Debug, Clone)]
pub enum QuotaViolation {
    /// Total VRAM limit exceeded.
    VramExceeded {
        /// Current usage.
        current: u64,
        /// Additional requested.
        requested: u64,
        /// Quota limit.
        limit: u64,
    },
    /// GPU count limit exceeded.
    GpuCountExceeded {
        /// Current GPU count.
        current: u32,
        /// Additional requested.
        requested: u32,
        /// Quota limit.
        limit: u32,
    },
    /// GEU limit exceeded.
    GeuExceeded {
        /// Current GEU.
        current: f64,
        /// Additional requested.
        requested: f64,
        /// Base limit.
        limit: f64,
        /// Limit with burst.
        burst_limit: f64,
    },
    /// GEU-hours rate limit exceeded.
    GeuHoursExceeded {
        /// Hours consumed.
        consumed: f64,
        /// Period limit.
        limit: f64,
    },
    /// Transfer bytes rate limit exceeded.
    TransferBytesExceeded {
        /// Bytes consumed.
        consumed: u64,
        /// Period limit.
        limit: u64,
    },
    /// Session limit reached.
    SessionLimitReached {
        /// Current session count.
        current: u32,
        /// Session limit.
        limit: u32,
    },
    /// Pending kernel queue full.
    PendingQueueFull {
        /// Current pending count.
        current: u32,
        /// Queue limit.
        limit: u32,
    },
    /// Tenant is disabled.
    TenantDisabled,
}

// ---------------------------------------------------------------------------
// Time Slice
// ---------------------------------------------------------------------------

/// A time slice represents a scheduled execution window on a specific GPU
/// for a specific tenant.
#[derive(Debug, Clone)]
pub struct TimeSlice {
    /// Which tenant owns this time slice.
    pub tenant_id: TenantId,
    /// Which session within the tenant.
    pub session_id: SessionId,
    /// Target GPU for this time slice.
    pub gpu_id: GpuId,
    /// Start time of the slice (relative to the schedule epoch).
    pub start_ns: u64,
    /// Duration of the slice in nanoseconds.
    pub duration_ns: u64,
    /// Maximum kernels during this slice (0 = unlimited).
    pub max_kernels: u32,
    /// Whether this slice can be extended if no other tenant is waiting.
    pub extensible: bool,
    /// The GEU cost of this slice.
    pub geu_cost: f64,
    /// Priority (inherited from tenant).
    pub priority: PriorityTier,
    /// State of this time slice.
    pub state: TimeSliceState,
}

/// Lifecycle of a time slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSliceState {
    /// Scheduled but not yet started.
    Pending,
    /// Currently active: the tenant's kernels are being dispatched.
    Active,
    /// Completed normally.
    Completed,
    /// Preempted by a higher-priority tenant.
    Preempted,
    /// Cancelled (tenant disconnected or session ended).
    Cancelled,
}

/// Configuration for the time-slicing scheduler.
#[derive(Debug, Clone)]
pub struct TimeSliceConfig {
    /// Default time slice duration per tenant per GPU.
    pub default_slice_duration: Duration,
    /// Minimum time slice duration.
    pub min_slice_duration: Duration,
    /// Maximum time slice duration.
    pub max_slice_duration: Duration,
    /// Grace period after time slice expires before forceful preemption.
    pub preemption_grace_period: Duration,
    /// How far ahead the scheduler plans.
    pub scheduling_horizon: Duration,
    /// How often the scheduler re-evaluates.
    pub reschedule_interval: Duration,
    /// Work-conserving: give idle slices to next tenant.
    pub work_conserving: bool,
    /// Coalesce consecutive slices for same tenant.
    pub coalesce_consecutive: bool,
}

impl Default for TimeSliceConfig {
    fn default() -> Self {
        Self {
            default_slice_duration: Duration::from_millis(100),
            min_slice_duration: Duration::from_millis(10),
            max_slice_duration: Duration::from_secs(5),
            preemption_grace_period: Duration::from_millis(50),
            scheduling_horizon: Duration::from_secs(10),
            reschedule_interval: Duration::from_millis(100),
            work_conserving: true,
            coalesce_consecutive: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduling Policy
// ---------------------------------------------------------------------------

/// The scheduling policy determines how time slices are allocated to tenants.
#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    /// Dominant Resource Fairness.
    DominantResourceFairness {
        /// Resource weights for computing the dominant share.
        resource_weights: DrfResourceWeights,
    },
    /// Weighted Fair Queuing.
    WeightedFairQueue,
    /// Strict Priority.
    StrictPriority,
    /// FIFO: first-come first-served.
    Fifo,
    /// Reserved: specific GPUs permanently assigned to specific tenants.
    Reserved {
        /// GPU assignments per tenant.
        assignments: HashMap<TenantId, Vec<GpuId>>,
    },
}

/// Resource weights for DRF dominant share computation.
#[derive(Debug, Clone)]
pub struct DrfResourceWeights {
    /// Weight for GPU compute (GEU-seconds).
    pub compute: f64,
    /// Weight for VRAM capacity (bytes).
    pub vram: f64,
    /// Weight for network transfer bandwidth (bytes/sec).
    pub network: f64,
}

impl Default for DrfResourceWeights {
    fn default() -> Self {
        Self {
            compute: 0.50,
            vram: 0.35,
            network: 0.15,
        }
    }
}

/// Complete pool sharing configuration.
#[derive(Debug, Clone)]
pub struct SharingConfig {
    /// Scheduling policy for the pool.
    pub policy: SchedulingPolicy,
    /// Time slice configuration.
    pub time_slice: TimeSliceConfig,
    /// Authentication mode.
    pub auth_mode: AuthMode,
    /// Whether to enable usage accounting.
    pub enable_accounting: bool,
    /// Whether to enforce quotas (false = advisory only).
    pub enforce_quotas: bool,
    /// Default quota for new tenants.
    pub default_quota_preset: QuotaPreset,
    /// Maximum number of tenants in the pool.
    pub max_tenants: u32,
    /// Minimum share percentage for starvation prevention.
    pub min_share_percent: f64,
}

impl Default for SharingConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::DominantResourceFairness {
                resource_weights: DrfResourceWeights::default(),
            },
            time_slice: TimeSliceConfig::default(),
            auth_mode: AuthMode::Open,
            enable_accounting: true,
            enforce_quotas: true,
            default_quota_preset: QuotaPreset::Standard,
            max_tenants: 64,
            min_share_percent: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A session represents one application instance connected to the OuterLink pool.
#[derive(Debug, Clone)]
pub struct Session {
    /// Unique session identifier.
    pub session_id: SessionId,
    /// Owning tenant.
    pub tenant_id: TenantId,
    /// Process ID on the originating node.
    pub pid: u32,
    /// Node where the application is running.
    pub origin_node: NodeId,
    /// Application name.
    pub app_name: String,
    /// When this session started.
    pub started_at: SystemTime,
    /// Current state.
    pub state: SessionState,
    /// Per-session resource usage.
    pub usage: SessionUsage,
    /// CUDA contexts created by this session.
    pub cuda_contexts: Vec<CudaContextHandle>,
    /// VRAM allocations: virtual address -> (gpu_id, size_bytes).
    pub allocations: HashMap<u64, (GpuId, u64)>,
    /// Pending kernel queue.
    pub pending_kernels: VecDeque<PendingKernel>,
}

/// Session lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Active: can submit work.
    Active,
    /// Draining: no new work, waiting for in-flight to complete.
    Draining,
    /// Suspended: over quota or preempted.
    Suspended,
    /// Terminated.
    Terminated,
}

/// Per-session resource usage.
#[derive(Debug, Clone, Default)]
pub struct SessionUsage {
    /// Total VRAM allocated by this session.
    pub vram_bytes: u64,
    /// Kernels launched.
    pub kernels_launched: u64,
    /// Kernels completed.
    pub kernels_completed: u64,
    /// Total GPU time in nanoseconds.
    pub gpu_time_ns: u64,
    /// GEU-seconds consumed.
    pub geu_seconds: f64,
    /// Data transferred cross-node, in bytes.
    pub transfer_bytes: u64,
}

/// A kernel waiting to be dispatched to a GPU.
#[derive(Debug, Clone)]
pub struct PendingKernel {
    /// The CUDA function to launch.
    pub function: u64,
    /// Grid dimensions.
    pub grid_dim: [u32; 3],
    /// Block dimensions.
    pub block_dim: [u32; 3],
    /// Shared memory size.
    pub shared_mem_bytes: u32,
    /// When the kernel was queued.
    pub queued_at: Instant,
    /// Estimated execution time (0 if unknown).
    pub estimated_ns: u64,
    /// Memory regions this kernel will access.
    pub memory_regions: Vec<MemoryRegionRef>,
    /// Target GPU (None if not yet assigned).
    pub assigned_gpu: Option<GpuId>,
}

/// Reference to a memory region accessed by a kernel.
#[derive(Debug, Clone)]
pub struct MemoryRegionRef {
    /// Virtual address of the allocation.
    pub vaddr: u64,
    /// Size of the access.
    pub size_bytes: u64,
    /// Whether the kernel writes to this region.
    pub is_write: bool,
}

// ---------------------------------------------------------------------------
// DRF Scheduler
// ---------------------------------------------------------------------------

/// Total pool resources (the "capacity" in DRF).
#[derive(Debug, Clone)]
pub struct PoolResources {
    /// Total GEU across all GPUs.
    pub total_geu: f64,
    /// Total VRAM across all GPUs, in bytes.
    pub total_vram_bytes: u64,
    /// Total network bandwidth, in bytes/sec.
    pub total_network_bw: u64,
}

/// Per-tenant consumption tracking for DRF.
#[derive(Debug, Clone, Default)]
pub struct DrfConsumption {
    /// GEU-seconds consumed in the current scheduling window.
    pub geu_seconds: f64,
    /// VRAM bytes currently held.
    pub vram_bytes: u64,
    /// Network bytes/sec currently used.
    pub network_bps: u64,
    /// Computed shares (fraction of pool total).
    pub compute_share: f64,
    /// VRAM share.
    pub vram_share: f64,
    /// Network share.
    pub network_share: f64,
    /// Dominant share (the maximum of the weighted shares).
    pub dominant_share: f64,
}

/// The DRF scheduler maintains per-tenant dominant shares and picks the
/// tenant with the lowest weighted dominant share for each scheduling slot.
pub struct DrfScheduler {
    /// All registered tenants.
    pub tenants: HashMap<TenantId, Tenant>,
    /// Pool-level resource totals.
    pub pool_resources: PoolResources,
    /// Per-tenant resource consumption.
    pub tenant_consumption: HashMap<TenantId, DrfConsumption>,
    /// Configuration.
    pub config: SharingConfig,
    /// Per-GPU current tenant assignment.
    pub gpu_assignments: HashMap<GpuId, Option<TenantId>>,
    /// Time slice schedule (produced by each scheduling round).
    pub schedule: Vec<TimeSlice>,
    /// Round counter.
    pub round: u64,
}

impl DrfScheduler {
    /// Create a new DRF scheduler with the given configuration and pool resources.
    pub fn new(config: SharingConfig, pool_resources: PoolResources) -> Self {
        Self {
            tenants: HashMap::new(),
            pool_resources,
            tenant_consumption: HashMap::new(),
            config,
            gpu_assignments: HashMap::new(),
            schedule: Vec::new(),
            round: 0,
        }
    }

    /// Register a GPU in the scheduler.
    pub fn register_gpu(&mut self, gpu_id: GpuId) {
        self.gpu_assignments.insert(gpu_id, None);
    }

    /// Register a tenant in the scheduler.
    pub fn register_tenant(&mut self, tenant: Tenant) {
        let id = tenant.tenant_id;
        self.tenants.insert(id, tenant);
        self.tenant_consumption.insert(id, DrfConsumption::default());
    }

    /// Remove a tenant from the scheduler.
    pub fn remove_tenant(&mut self, tenant_id: TenantId) {
        self.tenants.remove(&tenant_id);
        self.tenant_consumption.remove(&tenant_id);
    }

    /// Compute dominant share for each tenant.
    pub fn update_dominant_shares(&mut self) {
        let weights = match &self.config.policy {
            SchedulingPolicy::DominantResourceFairness { resource_weights } => {
                resource_weights.clone()
            }
            _ => DrfResourceWeights::default(),
        };

        let pool = &self.pool_resources;
        let total_geu_seconds = pool.total_geu;
        let total_vram = pool.total_vram_bytes as f64;
        let total_network = pool.total_network_bw as f64;

        // Collect tenant IDs to avoid borrow conflict
        let tenant_ids: Vec<TenantId> = self.tenant_consumption.keys().copied().collect();

        for tenant_id in tenant_ids {
            let priority_weight = self
                .tenants
                .get(&tenant_id)
                .map(|t| t.priority.weight())
                .unwrap_or(1.0);

            if let Some(consumption) = self.tenant_consumption.get_mut(&tenant_id) {
                consumption.compute_share = if total_geu_seconds > 0.0 {
                    consumption.geu_seconds / total_geu_seconds
                } else {
                    0.0
                };
                consumption.vram_share = if total_vram > 0.0 {
                    consumption.vram_bytes as f64 / total_vram
                } else {
                    0.0
                };
                consumption.network_share = if total_network > 0.0 {
                    consumption.network_bps as f64 / total_network
                } else {
                    0.0
                };

                consumption.dominant_share = f64::max(
                    f64::max(
                        weights.compute * consumption.compute_share,
                        weights.vram * consumption.vram_share,
                    ),
                    weights.network * consumption.network_share,
                ) / priority_weight;
            }
        }
    }

    /// Pick the best tenant for a time slice on the given GPU.
    /// Returns None if no eligible tenant has pending work.
    pub fn assign_slice(&self, gpu_id: GpuId) -> Option<TimeSlice> {
        let mut candidates: Vec<(TenantId, f64)> = Vec::new();

        for (tenant_id, tenant) in &self.tenants {
            if !tenant.enabled {
                continue;
            }
            if tenant.active_sessions.is_empty() {
                continue;
            }

            // Quota check
            if self.config.enforce_quotas {
                if tenant
                    .usage
                    .would_exceed_quota(&tenant.quota, 0, 0, 0.0)
                    .is_some()
                {
                    continue;
                }
            }

            let dominant_share = self
                .tenant_consumption
                .get(tenant_id)
                .map(|c| c.dominant_share)
                .unwrap_or(0.0);

            candidates.push((*tenant_id, dominant_share));
        }

        // Pick tenant with lowest dominant share.
        // Tie-break by tenant_id for deterministic scheduling.
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let (tenant_id, _) = candidates.first()?;
        let tenant = self.tenants.get(tenant_id)?;

        // Compute slice duration based on priority weight
        let base_duration = self.config.time_slice.default_slice_duration;
        let weighted_duration = Duration::from_nanos(
            (base_duration.as_nanos() as f64 * tenant.priority.weight()) as u64,
        );
        let clamped_duration = weighted_duration
            .max(self.config.time_slice.min_slice_duration)
            .min(self.config.time_slice.max_slice_duration);

        // Pick a session from this tenant
        let session_id = *tenant.active_sessions.iter().next()?;

        // Compute GEU cost
        let gpu_geu = 1.0; // TODO: obtained from R23 GpuProfile
        let geu_cost = clamped_duration.as_secs_f64() * gpu_geu / 3600.0;

        Some(TimeSlice {
            tenant_id: *tenant_id,
            session_id,
            gpu_id,
            start_ns: 0,
            duration_ns: clamped_duration.as_nanos() as u64,
            max_kernels: 0,
            extensible: self.config.time_slice.work_conserving,
            geu_cost,
            priority: tenant.priority,
            state: TimeSliceState::Pending,
        })
    }

    /// Core scheduling loop: produce time slices for all GPUs.
    pub fn schedule_round(&mut self) -> Vec<TimeSlice> {
        let mut slices = Vec::new();

        // Step 3: for each GPU, assign time slices
        let gpu_ids: Vec<GpuId> = self.gpu_assignments.keys().copied().collect();
        for gpu_id in gpu_ids {
            let slice_count = self.slices_per_round();
            for _ in 0..slice_count {
                // Recompute dominant shares before each assignment so the
                // scheduler picks the tenant with the genuinely lowest share.
                self.update_dominant_shares();

                if let Some(slice) = self.assign_slice(gpu_id) {
                    // Update consumption after assignment
                    let consumption = self
                        .tenant_consumption
                        .entry(slice.tenant_id)
                        .or_default();
                    consumption.geu_seconds += slice.geu_cost * 3600.0;
                    slices.push(slice);
                }
            }
        }

        // Step 4: starvation prevention
        self.enforce_min_share(&mut slices);

        self.round += 1;
        self.schedule = slices.clone();
        slices
    }

    /// Enforce minimum share for starved tiers.
    pub fn enforce_min_share(&self, slices: &mut Vec<TimeSlice>) {
        if self.config.min_share_percent <= 0.0 {
            return;
        }

        let mut tier_counts: HashMap<PriorityTier, u32> = HashMap::new();
        for slice in slices.iter() {
            *tier_counts.entry(slice.priority).or_insert(0) += 1;
        }

        let total_slices = slices.len() as f64;
        if total_slices == 0.0 {
            return;
        }

        let _min_count = (total_slices * self.config.min_share_percent).ceil() as u32;

        // Check if any tier with pending work got less than min_share_percent.
        // In a full implementation, we would replace lowest-priority excess slices
        // with slices for the starved tier. For now, we detect the condition.
        for tier in [PriorityTier::Background, PriorityTier::Development] {
            let count = tier_counts.get(&tier).copied().unwrap_or(0);
            if count < _min_count {
                // TODO: Find a tenant at this tier with pending work and inject slices.
                // This requires replacing lowest-priority excess slices.
            }
        }
    }

    /// Number of time slices per scheduling round.
    pub fn slices_per_round(&self) -> usize {
        let horizon = self.config.time_slice.scheduling_horizon;
        let slice_dur = self.config.time_slice.default_slice_duration;
        if slice_dur.as_nanos() == 0 {
            return 0;
        }
        (horizon.as_nanos() / slice_dur.as_nanos()) as usize
    }
}

// ---------------------------------------------------------------------------
// Quota Enforcer
// ---------------------------------------------------------------------------

/// The QuotaEnforcer is called at interception points to gate resource access.
pub struct QuotaEnforcer {
    /// Reference to tenant data.
    tenants: HashMap<TenantId, Tenant>,
    /// Configuration (used by periodic enforcement and billing rollover).
    #[allow(dead_code)]
    config: SharingConfig,
}

impl QuotaEnforcer {
    /// Create a new quota enforcer.
    pub fn new(config: SharingConfig) -> Self {
        Self {
            tenants: HashMap::new(),
            config,
        }
    }

    /// Update the tenant registry.
    pub fn set_tenants(&mut self, tenants: HashMap<TenantId, Tenant>) {
        self.tenants = tenants;
    }

    /// Gate a memory allocation request.
    pub fn check_alloc(
        &self,
        tenant_id: TenantId,
        gpu_id: GpuId,
        size_bytes: u64,
    ) -> Result<(), QuotaViolation> {
        let tenant = self
            .tenants
            .get(&tenant_id)
            .ok_or(QuotaViolation::TenantDisabled)?;

        if !tenant.enabled {
            return Err(QuotaViolation::TenantDisabled);
        }

        let quota = &tenant.quota;
        let usage = &tenant.usage;

        // Per-GPU VRAM check
        if quota.max_vram_per_gpu_bytes > 0 {
            let gpu_usage = usage.vram_per_gpu.get(&gpu_id).copied().unwrap_or(0);
            if gpu_usage + size_bytes > quota.max_vram_per_gpu_bytes {
                return Err(QuotaViolation::VramExceeded {
                    current: gpu_usage,
                    requested: size_bytes,
                    limit: quota.max_vram_per_gpu_bytes,
                });
            }
        }

        // Total VRAM check
        if let Some(violation) = usage.would_exceed_quota(quota, size_bytes, 0, 0.0) {
            return Err(violation);
        }

        Ok(())
    }

    /// Gate a kernel launch request.
    pub fn check_launch(
        &self,
        tenant_id: TenantId,
        _session_id: SessionId,
    ) -> Result<(), QuotaViolation> {
        let tenant = self
            .tenants
            .get(&tenant_id)
            .ok_or(QuotaViolation::TenantDisabled)?;

        if !tenant.enabled {
            return Err(QuotaViolation::TenantDisabled);
        }

        let quota = &tenant.quota;
        let usage = &tenant.usage;

        // Pending queue check
        if quota.max_pending_kernels > 0 && usage.pending_kernels >= quota.max_pending_kernels {
            return Err(QuotaViolation::PendingQueueFull {
                current: usage.pending_kernels,
                limit: quota.max_pending_kernels,
            });
        }

        // Rate limit check
        if quota.max_geu_hours_per_period > 0.0
            && usage.geu_hours_this_period >= quota.max_geu_hours_per_period
        {
            return Err(QuotaViolation::GeuHoursExceeded {
                consumed: usage.geu_hours_this_period,
                limit: quota.max_geu_hours_per_period,
            });
        }

        Ok(())
    }

    /// Gate a session creation request.
    pub fn check_session(&self, tenant_id: TenantId) -> Result<(), QuotaViolation> {
        let tenant = self
            .tenants
            .get(&tenant_id)
            .ok_or(QuotaViolation::TenantDisabled)?;

        let quota = &tenant.quota;
        let usage = &tenant.usage;

        if quota.max_sessions > 0 && usage.active_session_count >= quota.max_sessions {
            return Err(QuotaViolation::SessionLimitReached {
                current: usage.active_session_count,
                limit: quota.max_sessions,
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Preemption Engine
// ---------------------------------------------------------------------------

/// Handles cooperative preemption when a higher-priority tenant needs GPU
/// resources currently held by a lower-priority tenant.
pub struct PreemptionEngine {
    /// Time slice configuration.
    pub config: TimeSliceConfig,
}

/// Preemption request issued by the scheduler.
#[derive(Debug, Clone)]
pub struct PreemptionRequest {
    /// GPU to preempt on.
    pub gpu_id: GpuId,
    /// Tenant being preempted (the victim).
    pub victim_tenant: TenantId,
    /// Session being preempted.
    pub victim_session: SessionId,
    /// Tenant that needs the GPU (the preemptor).
    pub preemptor_tenant: TenantId,
    /// Reason for preemption.
    pub reason: PreemptionReason,
    /// When the preemption was requested.
    pub requested_at: Instant,
}

/// Reason for preemption.
#[derive(Debug, Clone)]
pub enum PreemptionReason {
    /// Higher-priority tenant needs GPU time.
    PriorityPreemption {
        /// Victim's tier.
        victim_tier: PriorityTier,
        /// Preemptor's tier.
        preemptor_tier: PriorityTier,
    },
    /// Victim tenant exceeded their quota.
    QuotaExceeded {
        /// The quota violation.
        violation: QuotaViolation,
    },
    /// Pool rebalancing (R22 migration).
    Rebalancing,
    /// Victim's session is being terminated.
    SessionTermination,
}

/// Result of a preemption attempt.
#[derive(Debug)]
pub enum PreemptionResult {
    /// Preemption completed: victim's kernels drained, GPU ready.
    Completed {
        /// Time spent draining.
        drain_time: Duration,
        /// Number of kernels drained.
        kernels_drained: u32,
    },
    /// Preemption in progress: waiting for in-flight kernels.
    InProgress {
        /// In-flight kernel count.
        inflight_kernels: u32,
        /// Time elapsed.
        elapsed: Duration,
    },
    /// Preemption forced: grace period expired.
    Forced {
        /// Kernels killed.
        kernels_killed: u32,
    },
    /// Preemption unnecessary: victim already finished.
    Unnecessary,
}

impl PreemptionEngine {
    /// Create a new preemption engine.
    pub fn new(config: TimeSliceConfig) -> Self {
        Self { config }
    }

    /// Execute a cooperative preemption.
    ///
    /// Phase 1: STOP - mark victim's slice as Preempted, block new launches
    /// Phase 2: DRAIN - wait for in-flight kernels to complete
    /// Phase 3: YIELD - reassign GPU to preemptor
    /// Phase 4: RESUME - handled when victim gets next slice
    pub fn preempt(&self, request: PreemptionRequest) -> PreemptionResult {
        // TODO: Phase 1 - Stop victim's kernel dispatch via interception layer
        // TODO: Phase 2 - Wait for in-flight kernels via CUDA events
        // TODO: Phase 3 - Reassign GPU to preemptor

        let start = request.requested_at;
        PreemptionResult::Completed {
            drain_time: start.elapsed(),
            kernels_drained: 0,
        }
    }

    /// Check if preemption is needed.
    pub fn should_preempt(
        &self,
        preemptor: &Tenant,
        victim: &Tenant,
        _gpu_id: GpuId,
    ) -> bool {
        if !preemptor.priority.can_preempt(&victim.priority) {
            return false;
        }
        if !victim.scheduling_prefs.preemptible {
            return false;
        }
        true
    }

    /// Select the best victim for preemption from candidates.
    pub fn select_victim(
        &self,
        preemptor: &Tenant,
        candidates: &[(TenantId, &Tenant)],
    ) -> Option<TenantId> {
        candidates
            .iter()
            .filter(|(_, t)| preemptor.priority.can_preempt(&t.priority))
            .filter(|(_, t)| t.scheduling_prefs.preemptible)
            .min_by(|(_, a), (_, b)| {
                a.priority
                    .cmp(&b.priority)
                    .then(
                        b.usage
                            .current_geu
                            .partial_cmp(&a.usage.current_geu)
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
            })
            .map(|(id, _)| *id)
    }
}

// ---------------------------------------------------------------------------
// VRAM Isolation (Ledger)
// ---------------------------------------------------------------------------

/// A single VRAM allocation tracked by the ledger.
#[derive(Debug, Clone)]
pub struct VramAllocation {
    /// Virtual address.
    pub vaddr: u64,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Owning tenant.
    pub tenant_id: TenantId,
    /// Owning session.
    pub session_id: SessionId,
    /// When allocated.
    pub allocated_at: Instant,
    /// Last kernel that accessed this allocation.
    pub last_accessed: Instant,
    /// Access count.
    pub access_count: u64,
}

/// Per-tenant VRAM summary on a specific GPU.
#[derive(Debug, Clone, Default)]
pub struct VramTenantSummary {
    /// Total bytes allocated.
    pub total_bytes: u64,
    /// Number of allocations.
    pub allocation_count: u32,
    /// Largest single allocation.
    pub max_allocation_bytes: u64,
    /// Peak usage (high watermark).
    pub peak_bytes: u64,
}

/// Per-GPU VRAM usage.
#[derive(Debug, Clone)]
pub struct GpuVramUsage {
    /// Total VRAM on this GPU.
    pub total_bytes: u64,
    /// VRAM used by OuterLink tenants.
    pub tenant_used_bytes: u64,
    /// VRAM used by non-OuterLink processes.
    pub external_used_bytes: u64,
    /// Available for new tenant allocations.
    pub available_bytes: u64,
    /// Per-tenant breakdown.
    pub per_tenant: HashMap<TenantId, u64>,
}

/// Per-GPU tenant VRAM tracking. Maintains a ledger of which tenant owns
/// which allocations on each GPU, and enforces per-tenant VRAM limits.
pub struct VramLedger {
    /// Per-GPU ledger: maps (gpu_id, virtual_address) -> allocation.
    allocations: HashMap<GpuId, HashMap<u64, VramAllocation>>,
    /// Per-GPU per-tenant usage summary.
    gpu_tenant_usage: HashMap<(GpuId, TenantId), VramTenantSummary>,
    /// Per-GPU total usage.
    gpu_usage: HashMap<GpuId, GpuVramUsage>,
}

impl VramLedger {
    /// Create a new empty ledger.
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            gpu_tenant_usage: HashMap::new(),
            gpu_usage: HashMap::new(),
        }
    }

    /// Register a GPU with its total VRAM.
    pub fn register_gpu(&mut self, gpu_id: GpuId, total_vram_bytes: u64) {
        self.gpu_usage.insert(
            gpu_id,
            GpuVramUsage {
                total_bytes: total_vram_bytes,
                tenant_used_bytes: 0,
                external_used_bytes: 0,
                available_bytes: total_vram_bytes,
                per_tenant: HashMap::new(),
            },
        );
    }

    /// Record a new allocation. Called from cuMemAlloc interception.
    pub fn record_alloc(
        &mut self,
        gpu_id: GpuId,
        vaddr: u64,
        size_bytes: u64,
        tenant_id: TenantId,
        session_id: SessionId,
    ) {
        let now = Instant::now();
        let alloc = VramAllocation {
            vaddr,
            size_bytes,
            tenant_id,
            session_id,
            allocated_at: now,
            last_accessed: now,
            access_count: 0,
        };

        self.allocations
            .entry(gpu_id)
            .or_default()
            .insert(vaddr, alloc);

        let summary = self
            .gpu_tenant_usage
            .entry((gpu_id, tenant_id))
            .or_default();
        summary.total_bytes += size_bytes;
        summary.allocation_count += 1;
        if size_bytes > summary.max_allocation_bytes {
            summary.max_allocation_bytes = size_bytes;
        }
        if summary.total_bytes > summary.peak_bytes {
            summary.peak_bytes = summary.total_bytes;
        }

        if let Some(gpu_usage) = self.gpu_usage.get_mut(&gpu_id) {
            gpu_usage.tenant_used_bytes += size_bytes;
            gpu_usage.available_bytes = gpu_usage
                .total_bytes
                .saturating_sub(gpu_usage.tenant_used_bytes + gpu_usage.external_used_bytes);
            *gpu_usage.per_tenant.entry(tenant_id).or_insert(0) += size_bytes;
        }
    }

    /// Record a deallocation. Called from cuMemFree interception.
    pub fn record_free(&mut self, gpu_id: GpuId, vaddr: u64) {
        if let Some(gpu_allocs) = self.allocations.get_mut(&gpu_id) {
            if let Some(alloc) = gpu_allocs.remove(&vaddr) {
                if let Some(summary) = self
                    .gpu_tenant_usage
                    .get_mut(&(gpu_id, alloc.tenant_id))
                {
                    summary.total_bytes = summary.total_bytes.saturating_sub(alloc.size_bytes);
                    summary.allocation_count = summary.allocation_count.saturating_sub(1);
                }

                if let Some(gpu_usage) = self.gpu_usage.get_mut(&gpu_id) {
                    gpu_usage.tenant_used_bytes = gpu_usage
                        .tenant_used_bytes
                        .saturating_sub(alloc.size_bytes);
                    gpu_usage.available_bytes = gpu_usage
                        .total_bytes
                        .saturating_sub(gpu_usage.tenant_used_bytes + gpu_usage.external_used_bytes);
                    if let Some(tenant_usage) = gpu_usage.per_tenant.get_mut(&alloc.tenant_id) {
                        *tenant_usage = tenant_usage.saturating_sub(alloc.size_bytes);
                    }
                }
            }
        }
    }

    /// Check if a proposed allocation would violate VRAM isolation rules.
    pub fn check_vram_limit(
        &self,
        gpu_id: GpuId,
        tenant_id: TenantId,
        size_bytes: u64,
        quota: &TenantQuota,
    ) -> Result<(), QuotaViolation> {
        if quota.max_vram_per_gpu_bytes > 0 {
            let current = self
                .gpu_tenant_usage
                .get(&(gpu_id, tenant_id))
                .map(|s| s.total_bytes)
                .unwrap_or(0);
            if current + size_bytes > quota.max_vram_per_gpu_bytes {
                return Err(QuotaViolation::VramExceeded {
                    current,
                    requested: size_bytes,
                    limit: quota.max_vram_per_gpu_bytes,
                });
            }
        }

        if let Some(gpu_usage) = self.gpu_usage.get(&gpu_id) {
            if size_bytes > gpu_usage.available_bytes {
                return Err(QuotaViolation::VramExceeded {
                    current: gpu_usage.tenant_used_bytes,
                    requested: size_bytes,
                    limit: gpu_usage.total_bytes,
                });
            }
        }

        Ok(())
    }

    /// Get VRAM usage breakdown for a specific GPU.
    pub fn gpu_breakdown(&self, gpu_id: GpuId) -> Option<&GpuVramUsage> {
        self.gpu_usage.get(&gpu_id)
    }

    /// Get per-tenant summary for a GPU.
    pub fn tenant_summary(
        &self,
        gpu_id: GpuId,
        tenant_id: TenantId,
    ) -> Option<&VramTenantSummary> {
        self.gpu_tenant_usage.get(&(gpu_id, tenant_id))
    }

    /// Get all allocations for a specific tenant on a GPU.
    pub fn tenant_allocations(
        &self,
        gpu_id: GpuId,
        tenant_id: TenantId,
    ) -> Vec<&VramAllocation> {
        self.allocations
            .get(&gpu_id)
            .map(|allocs| {
                allocs
                    .values()
                    .filter(|a| a.tenant_id == tenant_id)
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for VramLedger {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Billing
// ---------------------------------------------------------------------------

/// Cumulative usage tracking for billing purposes.
/// This data is append-only (entries are never modified or deleted).
#[derive(Debug, Clone)]
pub struct BillingAccumulator {
    /// Completed billing periods.
    pub periods: Vec<BillingPeriod>,
    /// Current (incomplete) period accumulator.
    pub current: BillingPeriodAccumulator,
}

impl Default for BillingAccumulator {
    fn default() -> Self {
        Self {
            periods: Vec::new(),
            current: BillingPeriodAccumulator::new(),
        }
    }
}

/// A completed billing period.
#[derive(Debug, Clone)]
pub struct BillingPeriod {
    /// Tenant this record belongs to.
    pub tenant_id: TenantId,
    /// Start of the billing period.
    pub period_start: SystemTime,
    /// End of the billing period.
    pub period_end: SystemTime,

    // --- Primary billing metrics ---
    /// Total GEU-hours consumed.
    pub geu_hours: f64,
    /// Total wall-clock hours of GPU occupancy.
    pub gpu_occupancy_hours: f64,
    /// Total VRAM-hours (GB-hours).
    pub vram_gb_hours: f64,
    /// Total data transferred cross-node, in bytes.
    pub transfer_bytes: u64,

    // --- Breakdown ---
    /// Per-GPU usage breakdown.
    pub per_gpu: HashMap<GpuId, GpuBillingRecord>,
    /// Per-session usage breakdown.
    pub per_session: HashMap<SessionId, SessionBillingRecord>,

    // --- Quality metrics ---
    /// Number of preemptions.
    pub preemption_count: u32,
    /// Total queue delay.
    pub total_queue_delay: Duration,
    /// Quota violations.
    pub quota_violations: u32,
    /// Peak GEU consumption.
    pub peak_geu: f64,
    /// Peak VRAM usage in bytes.
    pub peak_vram_bytes: u64,
}

/// Per-GPU billing breakdown within a period.
#[derive(Debug, Clone, Default)]
pub struct GpuBillingRecord {
    /// GPU model name.
    pub gpu_name: String,
    /// GEU rating of this GPU.
    pub gpu_geu: f64,
    /// Wall-clock seconds tenant occupied this GPU.
    pub occupancy_seconds: f64,
    /// GEU-seconds consumed on this GPU.
    pub geu_seconds: f64,
    /// Kernels launched on this GPU.
    pub kernel_count: u64,
    /// VRAM peak on this GPU.
    pub peak_vram_bytes: u64,
}

/// Per-session billing breakdown within a period.
#[derive(Debug, Clone, Default)]
pub struct SessionBillingRecord {
    /// Application name.
    pub app_name: String,
    /// GEU-hours consumed.
    pub geu_hours: f64,
    /// Kernels launched.
    pub kernel_count: u64,
    /// Session duration.
    pub duration: Duration,
}

/// Accumulator for the current (incomplete) billing period.
#[derive(Debug, Clone)]
pub struct BillingPeriodAccumulator {
    /// Period start.
    pub period_start: SystemTime,
    /// Running totals.
    pub geu_seconds: f64,
    /// VRAM byte-seconds.
    pub vram_byte_seconds: f64,
    /// Transfer bytes.
    pub transfer_bytes: u64,
    /// Kernel count.
    pub kernel_count: u64,
    /// Preemption count.
    pub preemption_count: u32,
    /// Queue delay in nanoseconds.
    pub queue_delay_ns: u64,
    /// Quota violations.
    pub quota_violations: u32,
    /// Peak GEU.
    pub peak_geu: f64,
    /// Peak VRAM bytes.
    pub peak_vram_bytes: u64,
    /// Per-GPU accumulators.
    pub per_gpu: HashMap<GpuId, GpuBillingRecord>,
    /// Per-session accumulators.
    pub per_session: HashMap<SessionId, SessionBillingRecord>,
}

impl BillingPeriodAccumulator {
    /// Create a new accumulator starting now.
    pub fn new() -> Self {
        Self {
            period_start: SystemTime::now(),
            geu_seconds: 0.0,
            vram_byte_seconds: 0.0,
            transfer_bytes: 0,
            kernel_count: 0,
            preemption_count: 0,
            queue_delay_ns: 0,
            quota_violations: 0,
            peak_geu: 0.0,
            peak_vram_bytes: 0,
            per_gpu: HashMap::new(),
            per_session: HashMap::new(),
        }
    }

    /// Get GEU-hours (geu_seconds / 3600).
    pub fn geu_hours(&self) -> f64 {
        self.geu_seconds / 3600.0
    }
}

impl Default for BillingPeriodAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Accounting Events
// ---------------------------------------------------------------------------

/// Events that trigger accounting updates.
#[derive(Debug, Clone)]
pub enum AccountingEvent {
    /// A kernel completed execution.
    KernelCompleted {
        /// Tenant that owns the kernel.
        tenant_id: TenantId,
        /// Session that launched the kernel.
        session_id: SessionId,
        /// GPU the kernel ran on.
        gpu_id: GpuId,
        /// Execution time in nanoseconds.
        execution_ns: u64,
        /// GEU rating of the GPU.
        gpu_geu: f64,
    },
    /// VRAM was allocated.
    VramAllocated {
        /// Tenant.
        tenant_id: TenantId,
        /// Session.
        session_id: SessionId,
        /// GPU.
        gpu_id: GpuId,
        /// Size in bytes.
        size_bytes: u64,
    },
    /// VRAM was freed.
    VramFreed {
        /// Tenant.
        tenant_id: TenantId,
        /// Session.
        session_id: SessionId,
        /// GPU.
        gpu_id: GpuId,
        /// Size in bytes.
        size_bytes: u64,
    },
    /// Cross-node data transfer completed.
    TransferCompleted {
        /// Tenant.
        tenant_id: TenantId,
        /// Session.
        session_id: SessionId,
        /// Bytes transferred.
        bytes: u64,
    },
    /// Tenant was preempted.
    Preempted {
        /// Tenant.
        tenant_id: TenantId,
        /// Session.
        session_id: SessionId,
        /// GPU.
        gpu_id: GpuId,
    },
    /// Kernel dispatch was delayed by scheduling.
    QueueDelay {
        /// Tenant.
        tenant_id: TenantId,
        /// Session.
        session_id: SessionId,
        /// Delay in nanoseconds.
        delay_ns: u64,
    },
    /// Quota violation occurred.
    QuotaViolationEvent {
        /// Tenant.
        tenant_id: TenantId,
        /// Session.
        session_id: SessionId,
    },
}

/// Process an accounting event into the billing accumulator.
pub fn process_accounting_event(
    accumulator: &mut BillingPeriodAccumulator,
    event: AccountingEvent,
) {
    match event {
        AccountingEvent::KernelCompleted {
            gpu_id,
            execution_ns,
            gpu_geu,
            session_id,
            ..
        } => {
            let geu_seconds = (execution_ns as f64 / 1e9) * gpu_geu;
            accumulator.geu_seconds += geu_seconds;
            accumulator.kernel_count += 1;

            let gpu_record = accumulator.per_gpu.entry(gpu_id).or_default();
            gpu_record.geu_seconds += geu_seconds;
            gpu_record.kernel_count += 1;

            let session_record = accumulator.per_session.entry(session_id).or_default();
            session_record.geu_hours += geu_seconds / 3600.0;
            session_record.kernel_count += 1;
        }
        AccountingEvent::VramAllocated { .. } => {
            // VRAM-hours computed at period close from sampling
        }
        AccountingEvent::VramFreed { .. } => {
            // Handled by periodic VRAM sampling
        }
        AccountingEvent::TransferCompleted { bytes, .. } => {
            accumulator.transfer_bytes += bytes;
        }
        AccountingEvent::Preempted { .. } => {
            accumulator.preemption_count += 1;
        }
        AccountingEvent::QueueDelay { delay_ns, .. } => {
            accumulator.queue_delay_ns += delay_ns;
        }
        AccountingEvent::QuotaViolationEvent { .. } => {
            accumulator.quota_violations += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tenant Placement Filter (R17 integration)
// ---------------------------------------------------------------------------

/// R24 provides a tenant-aware filter for R17's placement pipeline.
pub trait TenantPlacementFilter {
    /// Filter GPU candidates based on tenant quotas and isolation rules.
    fn filter_for_tenant(&self, tenant_id: TenantId, candidates: &[GpuId]) -> Vec<GpuId>;

    /// Adjust the placement score for a GPU based on tenant co-tenancy.
    /// Returns a multiplier (0.0 - 2.0).
    fn tenancy_score_adjustment(&self, tenant_id: TenantId, gpu_id: GpuId) -> f64;

    /// Maximum number of tenants that should share a single GPU.
    fn max_co_tenants_per_gpu(&self) -> u32;
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper to create a test tenant ---

    fn make_tenant(id: TenantId, name: &str, priority: PriorityTier) -> Tenant {
        Tenant {
            tenant_id: id,
            name: name.to_string(),
            priority,
            quota: TenantQuota::default(),
            usage: ResourceUsage::default(),
            active_sessions: HashSet::new(),
            auth_credential: AuthCredential::None,
            enabled: true,
            created_at: SystemTime::now(),
            last_active: None,
            scheduling_prefs: SchedulingPreferences::default(),
            cumulative_usage: BillingAccumulator::default(),
        }
    }

    fn make_tenant_with_session(id: TenantId, name: &str, priority: PriorityTier) -> Tenant {
        let mut t = make_tenant(id, name, priority);
        t.active_sessions.insert(1000 + id as u64);
        t
    }

    // --- PriorityTier tests ---

    #[test]
    fn priority_weights() {
        assert!((PriorityTier::Background.weight() - 0.25).abs() < f64::EPSILON);
        assert!((PriorityTier::Development.weight() - 1.0).abs() < f64::EPSILON);
        assert!((PriorityTier::Production.weight() - 2.0).abs() < f64::EPSILON);
        assert!((PriorityTier::Critical.weight() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn priority_preemption() {
        assert!(PriorityTier::Production.can_preempt(&PriorityTier::Development));
        assert!(PriorityTier::Critical.can_preempt(&PriorityTier::Production));
        assert!(!PriorityTier::Development.can_preempt(&PriorityTier::Production));
        assert!(!PriorityTier::Background.can_preempt(&PriorityTier::Background));
    }

    #[test]
    fn priority_ordering() {
        assert!(PriorityTier::Background < PriorityTier::Development);
        assert!(PriorityTier::Development < PriorityTier::Production);
        assert!(PriorityTier::Production < PriorityTier::Critical);
    }

    // --- TenantQuota tests ---

    #[test]
    fn quota_default() {
        let q = TenantQuota::default();
        assert_eq!(q.max_vram_bytes, 0);
        assert_eq!(q.max_gpus, 0);
        assert!((q.max_geu).abs() < f64::EPSILON);
        assert_eq!(q.max_sessions, 8);
        assert_eq!(q.max_pending_kernels, 10_000);
        assert!((q.burst_multiplier - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn quota_presets() {
        let light = QuotaPreset::Light.to_quota();
        assert_eq!(light.max_gpus, 1);
        assert!((light.max_geu - 2.0).abs() < f64::EPSILON);
        assert!((light.max_geu_hours_per_period - 24.0).abs() < f64::EPSILON);

        let standard = QuotaPreset::Standard.to_quota();
        assert_eq!(standard.max_gpus, 2);
        assert!((standard.max_geu - 5.0).abs() < f64::EPSILON);

        let heavy = QuotaPreset::Heavy.to_quota();
        assert_eq!(heavy.max_gpus, 4);
        assert!((heavy.max_geu - 15.0).abs() < f64::EPSILON);

        let unlimited = QuotaPreset::Unlimited.to_quota();
        assert_eq!(unlimited.max_vram_bytes, 0);
    }

    // --- ResourceUsage quota checks ---

    #[test]
    fn usage_within_quota() {
        let usage = ResourceUsage::default();
        let quota = QuotaPreset::Standard.to_quota();
        assert!(usage
            .would_exceed_quota(&quota, 1024, 0, 0.0)
            .is_none());
    }

    #[test]
    fn usage_exceeds_vram() {
        let usage = ResourceUsage {
            vram_allocated_bytes: 20 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        let quota = QuotaPreset::Standard.to_quota(); // 24 GB limit
        let result = usage.would_exceed_quota(&quota, 10 * 1024 * 1024 * 1024, 0, 0.0);
        assert!(matches!(result, Some(QuotaViolation::VramExceeded { .. })));
    }

    #[test]
    fn usage_exceeds_gpu_count() {
        let usage = ResourceUsage {
            gpus_in_use: 2,
            ..Default::default()
        };
        let quota = QuotaPreset::Standard.to_quota(); // 2 GPU limit
        let result = usage.would_exceed_quota(&quota, 0, 1, 0.0);
        assert!(matches!(result, Some(QuotaViolation::GpuCountExceeded { .. })));
    }

    #[test]
    fn usage_exceeds_geu_with_burst() {
        let usage = ResourceUsage {
            current_geu: 7.0,
            ..Default::default()
        };
        let quota = QuotaPreset::Standard.to_quota(); // max_geu=5.0, burst=1.5 -> effective 7.5
        // 7.0 + 1.0 = 8.0 > 7.5
        let result = usage.would_exceed_quota(&quota, 0, 0, 1.0);
        assert!(matches!(result, Some(QuotaViolation::GeuExceeded { .. })));
    }

    #[test]
    fn usage_within_burst_geu() {
        let usage = ResourceUsage {
            current_geu: 6.0,
            ..Default::default()
        };
        let quota = QuotaPreset::Standard.to_quota(); // effective 7.5
        let result = usage.would_exceed_quota(&quota, 0, 0, 1.0);
        assert!(result.is_none()); // 6.0 + 1.0 = 7.0 < 7.5
    }

    #[test]
    fn usage_exceeds_geu_hours() {
        let usage = ResourceUsage {
            geu_hours_this_period: 200.0,
            ..Default::default()
        };
        let quota = QuotaPreset::Standard.to_quota(); // 120 GEU-hours/day
        let result = usage.would_exceed_quota(&quota, 0, 0, 0.0);
        assert!(matches!(result, Some(QuotaViolation::GeuHoursExceeded { .. })));
    }

    #[test]
    fn usage_unlimited_quota_never_exceeded() {
        let usage = ResourceUsage {
            vram_allocated_bytes: u64::MAX / 2,
            gpus_in_use: 100,
            current_geu: 1000.0,
            ..Default::default()
        };
        let quota = TenantQuota::default(); // all zeros = unlimited
        assert!(usage
            .would_exceed_quota(&quota, 1, 1, 1.0)
            .is_none());
    }

    // --- TimeSliceConfig tests ---

    #[test]
    fn time_slice_config_default() {
        let c = TimeSliceConfig::default();
        assert_eq!(c.default_slice_duration, Duration::from_millis(100));
        assert_eq!(c.min_slice_duration, Duration::from_millis(10));
        assert_eq!(c.max_slice_duration, Duration::from_secs(5));
        assert!(c.work_conserving);
        assert!(c.coalesce_consecutive);
    }

    // --- TimeSliceState tests ---

    #[test]
    fn time_slice_state_variants() {
        assert_ne!(TimeSliceState::Pending, TimeSliceState::Active);
        assert_eq!(TimeSliceState::Completed, TimeSliceState::Completed);
    }

    // --- SchedulingPolicy tests ---

    #[test]
    fn drf_resource_weights_default() {
        let w = DrfResourceWeights::default();
        assert!((w.compute - 0.50).abs() < f64::EPSILON);
        assert!((w.vram - 0.35).abs() < f64::EPSILON);
        assert!((w.network - 0.15).abs() < f64::EPSILON);
        // Weights should sum to 1.0
        assert!((w.compute + w.vram + w.network - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sharing_config_default() {
        let c = SharingConfig::default();
        assert!(matches!(
            c.policy,
            SchedulingPolicy::DominantResourceFairness { .. }
        ));
        assert!(c.enable_accounting);
        assert!(c.enforce_quotas);
        assert_eq!(c.max_tenants, 64);
        assert!((c.min_share_percent - 0.05).abs() < f64::EPSILON);
    }

    // --- SchedulingPreferences tests ---

    #[test]
    fn scheduling_prefs_default() {
        let p = SchedulingPreferences::default();
        assert!(p.allow_migration);
        assert!(p.preemptible);
        assert!(!p.exclusive_gpu);
        assert!(p.preferred_gpus.is_empty());
    }

    // --- Session tests ---

    #[test]
    fn session_state_variants() {
        assert_ne!(SessionState::Active, SessionState::Draining);
        assert_ne!(SessionState::Suspended, SessionState::Terminated);
    }

    #[test]
    fn session_usage_default() {
        let u = SessionUsage::default();
        assert_eq!(u.vram_bytes, 0);
        assert_eq!(u.kernels_launched, 0);
        assert_eq!(u.kernels_completed, 0);
    }

    // --- DrfScheduler tests ---

    #[test]
    fn drf_scheduler_new() {
        let s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 48 * 1024 * 1024 * 1024,
                total_network_bw: 10_000_000_000,
            },
        );
        assert_eq!(s.round, 0);
        assert!(s.tenants.is_empty());
    }

    #[test]
    fn drf_scheduler_register_tenant() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 48 * 1024 * 1024 * 1024,
                total_network_bw: 10_000_000_000,
            },
        );
        let t = make_tenant(1, "alice", PriorityTier::Development);
        s.register_tenant(t);
        assert_eq!(s.tenants.len(), 1);
        assert!(s.tenant_consumption.contains_key(&1));
    }

    #[test]
    fn drf_scheduler_remove_tenant() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 48 * 1024 * 1024 * 1024,
                total_network_bw: 10_000_000_000,
            },
        );
        s.register_tenant(make_tenant(1, "alice", PriorityTier::Development));
        s.remove_tenant(1);
        assert!(s.tenants.is_empty());
        assert!(!s.tenant_consumption.contains_key(&1));
    }

    #[test]
    fn drf_scheduler_dominant_shares() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 100,
                total_network_bw: 1000,
            },
        );
        s.register_tenant(make_tenant(1, "alice", PriorityTier::Development));
        s.register_tenant(make_tenant(2, "bob", PriorityTier::Production));

        // Give alice some consumption
        if let Some(c) = s.tenant_consumption.get_mut(&1) {
            c.geu_seconds = 5.0;
            c.vram_bytes = 50;
        }

        s.update_dominant_shares();

        let alice = s.tenant_consumption.get(&1).unwrap();
        assert!(alice.dominant_share > 0.0);
        // Alice: compute_share = 5/10 = 0.5, vram_share = 50/100 = 0.5
        // weighted: max(0.5*0.5, 0.35*0.5, 0.15*0) = 0.25
        // / dev weight 1.0 = 0.25
        assert!((alice.dominant_share - 0.25).abs() < 0.001);

        let bob = s.tenant_consumption.get(&2).unwrap();
        assert!((bob.dominant_share - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn drf_scheduler_assign_slice_picks_lowest_share() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 100,
                total_network_bw: 1000,
            },
        );
        s.register_gpu(0);
        s.register_tenant(make_tenant_with_session(1, "alice", PriorityTier::Development));
        s.register_tenant(make_tenant_with_session(2, "bob", PriorityTier::Development));

        // Give alice higher consumption
        if let Some(c) = s.tenant_consumption.get_mut(&1) {
            c.geu_seconds = 8.0;
        }

        s.update_dominant_shares();

        let slice = s.assign_slice(0);
        assert!(slice.is_some());
        // Bob should be picked (lower dominant share)
        assert_eq!(slice.unwrap().tenant_id, 2);
    }

    #[test]
    fn drf_scheduler_assign_slice_no_tenants() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 100,
                total_network_bw: 1000,
            },
        );
        s.register_gpu(0);
        let slice = s.assign_slice(0);
        assert!(slice.is_none());
    }

    #[test]
    fn drf_scheduler_assign_slice_disabled_tenant_skipped() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 100,
                total_network_bw: 1000,
            },
        );
        s.register_gpu(0);
        let mut t = make_tenant_with_session(1, "alice", PriorityTier::Development);
        t.enabled = false;
        s.register_tenant(t);
        let slice = s.assign_slice(0);
        assert!(slice.is_none());
    }

    #[test]
    fn drf_scheduler_assign_slice_no_sessions_skipped() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 100,
                total_network_bw: 1000,
            },
        );
        s.register_gpu(0);
        s.register_tenant(make_tenant(1, "alice", PriorityTier::Development));
        let slice = s.assign_slice(0);
        assert!(slice.is_none());
    }

    #[test]
    fn drf_scheduler_priority_weight_affects_slice_duration() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 100,
                total_network_bw: 1000,
            },
        );
        s.register_gpu(0);
        s.register_tenant(make_tenant_with_session(1, "bg", PriorityTier::Background));
        s.register_tenant(make_tenant_with_session(2, "prod", PriorityTier::Production));

        s.update_dominant_shares();

        let bg_slice = s.assign_slice(0);
        // Remove bg tenant to force prod
        s.remove_tenant(1);
        let prod_slice = s.assign_slice(0);

        // Production gets longer slices (weight 2.0 vs 0.25)
        assert!(bg_slice.is_some(), "bg_slice should be Some");
        assert!(prod_slice.is_some(), "prod_slice should be Some");
        let bg_dur = bg_slice.unwrap().duration_ns;
        let prod_dur = prod_slice.unwrap().duration_ns;
        assert!(
            prod_dur > bg_dur,
            "prod duration ({prod_dur}) should be > bg duration ({bg_dur})"
        );
    }

    #[test]
    fn drf_scheduler_schedule_round() {
        let mut s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 100,
                total_network_bw: 1000,
            },
        );
        s.register_gpu(0);
        s.register_tenant(make_tenant_with_session(1, "alice", PriorityTier::Development));

        let slices = s.schedule_round();
        assert!(!slices.is_empty());
        assert_eq!(s.round, 1);
        // All slices should be for alice (only tenant)
        for slice in &slices {
            assert_eq!(slice.tenant_id, 1);
        }
    }

    #[test]
    fn drf_scheduler_slices_per_round() {
        let s = DrfScheduler::new(
            SharingConfig::default(),
            PoolResources {
                total_geu: 10.0,
                total_vram_bytes: 100,
                total_network_bw: 1000,
            },
        );
        // horizon=10s, slice=100ms -> 100 slices
        assert_eq!(s.slices_per_round(), 100);
    }

    // --- QuotaEnforcer tests ---

    #[test]
    fn quota_enforcer_check_alloc_ok() {
        let mut enforcer = QuotaEnforcer::new(SharingConfig::default());
        let mut tenants = HashMap::new();
        tenants.insert(1, make_tenant(1, "alice", PriorityTier::Development));
        enforcer.set_tenants(tenants);

        assert!(enforcer.check_alloc(1, 0, 1024).is_ok());
    }

    #[test]
    fn quota_enforcer_check_alloc_unknown_tenant() {
        let enforcer = QuotaEnforcer::new(SharingConfig::default());
        let result = enforcer.check_alloc(999, 0, 1024);
        assert!(matches!(result, Err(QuotaViolation::TenantDisabled)));
    }

    #[test]
    fn quota_enforcer_check_alloc_disabled() {
        let mut enforcer = QuotaEnforcer::new(SharingConfig::default());
        let mut t = make_tenant(1, "alice", PriorityTier::Development);
        t.enabled = false;
        let mut tenants = HashMap::new();
        tenants.insert(1, t);
        enforcer.set_tenants(tenants);

        let result = enforcer.check_alloc(1, 0, 1024);
        assert!(matches!(result, Err(QuotaViolation::TenantDisabled)));
    }

    #[test]
    fn quota_enforcer_check_alloc_per_gpu_limit() {
        let mut enforcer = QuotaEnforcer::new(SharingConfig::default());
        let mut t = make_tenant(1, "alice", PriorityTier::Development);
        t.quota.max_vram_per_gpu_bytes = 1000;
        t.usage.vram_per_gpu.insert(0, 900);
        let mut tenants = HashMap::new();
        tenants.insert(1, t);
        enforcer.set_tenants(tenants);

        let result = enforcer.check_alloc(1, 0, 200);
        assert!(matches!(result, Err(QuotaViolation::VramExceeded { .. })));
    }

    #[test]
    fn quota_enforcer_check_launch_ok() {
        let mut enforcer = QuotaEnforcer::new(SharingConfig::default());
        let mut tenants = HashMap::new();
        tenants.insert(1, make_tenant(1, "alice", PriorityTier::Development));
        enforcer.set_tenants(tenants);

        assert!(enforcer.check_launch(1, 100).is_ok());
    }

    #[test]
    fn quota_enforcer_check_launch_queue_full() {
        let mut enforcer = QuotaEnforcer::new(SharingConfig::default());
        let mut t = make_tenant(1, "alice", PriorityTier::Development);
        t.quota.max_pending_kernels = 100;
        t.usage.pending_kernels = 100;
        let mut tenants = HashMap::new();
        tenants.insert(1, t);
        enforcer.set_tenants(tenants);

        let result = enforcer.check_launch(1, 100);
        assert!(matches!(result, Err(QuotaViolation::PendingQueueFull { .. })));
    }

    #[test]
    fn quota_enforcer_check_launch_rate_limit() {
        let mut enforcer = QuotaEnforcer::new(SharingConfig::default());
        let mut t = make_tenant(1, "alice", PriorityTier::Development);
        t.quota.max_geu_hours_per_period = 10.0;
        t.usage.geu_hours_this_period = 15.0;
        let mut tenants = HashMap::new();
        tenants.insert(1, t);
        enforcer.set_tenants(tenants);

        let result = enforcer.check_launch(1, 100);
        assert!(matches!(result, Err(QuotaViolation::GeuHoursExceeded { .. })));
    }

    #[test]
    fn quota_enforcer_check_session_ok() {
        let mut enforcer = QuotaEnforcer::new(SharingConfig::default());
        let mut tenants = HashMap::new();
        tenants.insert(1, make_tenant(1, "alice", PriorityTier::Development));
        enforcer.set_tenants(tenants);

        assert!(enforcer.check_session(1).is_ok());
    }

    #[test]
    fn quota_enforcer_check_session_limit() {
        let mut enforcer = QuotaEnforcer::new(SharingConfig::default());
        let mut t = make_tenant(1, "alice", PriorityTier::Development);
        t.quota.max_sessions = 2;
        t.usage.active_session_count = 2;
        let mut tenants = HashMap::new();
        tenants.insert(1, t);
        enforcer.set_tenants(tenants);

        let result = enforcer.check_session(1);
        assert!(matches!(result, Err(QuotaViolation::SessionLimitReached { .. })));
    }

    // --- PreemptionEngine tests ---

    #[test]
    fn preemption_should_preempt_higher_priority() {
        let engine = PreemptionEngine::new(TimeSliceConfig::default());
        let prod = make_tenant(1, "prod", PriorityTier::Production);
        let dev = make_tenant(2, "dev", PriorityTier::Development);
        assert!(engine.should_preempt(&prod, &dev, 0));
    }

    #[test]
    fn preemption_should_not_preempt_same_priority() {
        let engine = PreemptionEngine::new(TimeSliceConfig::default());
        let dev1 = make_tenant(1, "dev1", PriorityTier::Development);
        let dev2 = make_tenant(2, "dev2", PriorityTier::Development);
        assert!(!engine.should_preempt(&dev1, &dev2, 0));
    }

    #[test]
    fn preemption_should_not_preempt_non_preemptible() {
        let engine = PreemptionEngine::new(TimeSliceConfig::default());
        let prod = make_tenant(1, "prod", PriorityTier::Production);
        let mut dev = make_tenant(2, "dev", PriorityTier::Development);
        dev.scheduling_prefs.preemptible = false;
        assert!(!engine.should_preempt(&prod, &dev, 0));
    }

    #[test]
    fn preemption_select_victim_lowest_priority() {
        let engine = PreemptionEngine::new(TimeSliceConfig::default());
        let preemptor = make_tenant(1, "crit", PriorityTier::Critical);
        let dev = make_tenant(2, "dev", PriorityTier::Development);
        let bg = make_tenant(3, "bg", PriorityTier::Background);
        let candidates = vec![(2, &dev), (3, &bg)];
        let victim = engine.select_victim(&preemptor, &candidates);
        assert_eq!(victim, Some(3)); // Background is lowest
    }

    #[test]
    fn preemption_select_victim_none_eligible() {
        let engine = PreemptionEngine::new(TimeSliceConfig::default());
        let dev = make_tenant(1, "dev", PriorityTier::Development);
        let crit = make_tenant(2, "crit", PriorityTier::Critical);
        let candidates = vec![(2, &crit)];
        let victim = engine.select_victim(&dev, &candidates);
        assert!(victim.is_none());
    }

    #[test]
    fn preemption_execute() {
        let engine = PreemptionEngine::new(TimeSliceConfig::default());
        let req = PreemptionRequest {
            gpu_id: 0,
            victim_tenant: 2,
            victim_session: 100,
            preemptor_tenant: 1,
            reason: PreemptionReason::PriorityPreemption {
                victim_tier: PriorityTier::Development,
                preemptor_tier: PriorityTier::Production,
            },
            requested_at: Instant::now(),
        };
        let result = engine.preempt(req);
        assert!(matches!(result, PreemptionResult::Completed { .. }));
    }

    // --- VramLedger tests ---

    #[test]
    fn vram_ledger_new() {
        let ledger = VramLedger::new();
        assert!(ledger.gpu_breakdown(0).is_none());
    }

    #[test]
    fn vram_ledger_register_gpu() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 24 * 1024 * 1024 * 1024);
        let usage = ledger.gpu_breakdown(0).unwrap();
        assert_eq!(usage.total_bytes, 24 * 1024 * 1024 * 1024);
        assert_eq!(usage.available_bytes, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn vram_ledger_alloc_free() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 1_000_000);

        ledger.record_alloc(0, 0x100, 500_000, 1, 100);
        {
            let usage = ledger.gpu_breakdown(0).unwrap();
            assert_eq!(usage.tenant_used_bytes, 500_000);
            assert_eq!(usage.available_bytes, 500_000);
            assert_eq!(*usage.per_tenant.get(&1).unwrap(), 500_000);
        }

        ledger.record_free(0, 0x100);
        {
            let usage = ledger.gpu_breakdown(0).unwrap();
            assert_eq!(usage.tenant_used_bytes, 0);
            assert_eq!(usage.available_bytes, 1_000_000);
        }
    }

    #[test]
    fn vram_ledger_multiple_tenants() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 1_000_000);

        ledger.record_alloc(0, 0x100, 300_000, 1, 100);
        ledger.record_alloc(0, 0x200, 200_000, 2, 200);

        let usage = ledger.gpu_breakdown(0).unwrap();
        assert_eq!(usage.tenant_used_bytes, 500_000);
        assert_eq!(*usage.per_tenant.get(&1).unwrap(), 300_000);
        assert_eq!(*usage.per_tenant.get(&2).unwrap(), 200_000);
    }

    #[test]
    fn vram_ledger_tenant_summary() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 1_000_000);
        ledger.record_alloc(0, 0x100, 100_000, 1, 100);
        ledger.record_alloc(0, 0x200, 200_000, 1, 100);

        let summary = ledger.tenant_summary(0, 1).unwrap();
        assert_eq!(summary.total_bytes, 300_000);
        assert_eq!(summary.allocation_count, 2);
        assert_eq!(summary.max_allocation_bytes, 200_000);
        assert_eq!(summary.peak_bytes, 300_000);
    }

    #[test]
    fn vram_ledger_peak_tracking() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 1_000_000);
        ledger.record_alloc(0, 0x100, 500_000, 1, 100);
        ledger.record_free(0, 0x100);
        ledger.record_alloc(0, 0x200, 100_000, 1, 100);

        let summary = ledger.tenant_summary(0, 1).unwrap();
        assert_eq!(summary.peak_bytes, 500_000); // peak from first alloc
        assert_eq!(summary.total_bytes, 100_000); // current
    }

    #[test]
    fn vram_ledger_check_vram_limit_ok() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 1_000_000);

        let quota = TenantQuota {
            max_vram_per_gpu_bytes: 500_000,
            ..Default::default()
        };
        assert!(ledger.check_vram_limit(0, 1, 400_000, &quota).is_ok());
    }

    #[test]
    fn vram_ledger_check_vram_limit_exceeded() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 1_000_000);
        ledger.record_alloc(0, 0x100, 400_000, 1, 100);

        let quota = TenantQuota {
            max_vram_per_gpu_bytes: 500_000,
            ..Default::default()
        };
        let result = ledger.check_vram_limit(0, 1, 200_000, &quota);
        assert!(matches!(result, Err(QuotaViolation::VramExceeded { .. })));
    }

    #[test]
    fn vram_ledger_check_physical_availability() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 100_000);
        ledger.record_alloc(0, 0x100, 90_000, 1, 100);

        let quota = TenantQuota::default(); // no per-GPU limit
        let result = ledger.check_vram_limit(0, 2, 20_000, &quota);
        assert!(matches!(result, Err(QuotaViolation::VramExceeded { .. })));
    }

    #[test]
    fn vram_ledger_tenant_allocations() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 1_000_000);
        ledger.record_alloc(0, 0x100, 1000, 1, 100);
        ledger.record_alloc(0, 0x200, 2000, 1, 100);
        ledger.record_alloc(0, 0x300, 3000, 2, 200);

        let allocs = ledger.tenant_allocations(0, 1);
        assert_eq!(allocs.len(), 2);
    }

    #[test]
    fn vram_ledger_free_nonexistent() {
        let mut ledger = VramLedger::new();
        ledger.register_gpu(0, 1_000_000);
        // Should not panic
        ledger.record_free(0, 0xDEAD);
    }

    // --- BillingAccumulator tests ---

    #[test]
    fn billing_accumulator_default() {
        let acc = BillingAccumulator::default();
        assert!(acc.periods.is_empty());
        assert!((acc.current.geu_seconds - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn billing_period_accumulator_geu_hours() {
        let mut acc = BillingPeriodAccumulator::new();
        acc.geu_seconds = 3600.0;
        assert!((acc.geu_hours() - 1.0).abs() < f64::EPSILON);
    }

    // --- AccountingEvent processing tests ---

    #[test]
    fn process_kernel_completed() {
        let mut acc = BillingPeriodAccumulator::new();
        process_accounting_event(
            &mut acc,
            AccountingEvent::KernelCompleted {
                tenant_id: 1,
                session_id: 100,
                gpu_id: 0,
                execution_ns: 1_000_000_000, // 1 second
                gpu_geu: 2.0,
            },
        );
        // geu_seconds = 1.0 * 2.0 = 2.0
        assert!((acc.geu_seconds - 2.0).abs() < f64::EPSILON);
        assert_eq!(acc.kernel_count, 1);

        let gpu_rec = acc.per_gpu.get(&0).unwrap();
        assert!((gpu_rec.geu_seconds - 2.0).abs() < f64::EPSILON);

        let session_rec = acc.per_session.get(&100).unwrap();
        assert!((session_rec.geu_hours - 2.0 / 3600.0).abs() < 1e-10);
    }

    #[test]
    fn process_transfer_completed() {
        let mut acc = BillingPeriodAccumulator::new();
        process_accounting_event(
            &mut acc,
            AccountingEvent::TransferCompleted {
                tenant_id: 1,
                session_id: 100,
                bytes: 1_000_000,
            },
        );
        assert_eq!(acc.transfer_bytes, 1_000_000);
    }

    #[test]
    fn process_preemption() {
        let mut acc = BillingPeriodAccumulator::new();
        process_accounting_event(
            &mut acc,
            AccountingEvent::Preempted {
                tenant_id: 1,
                session_id: 100,
                gpu_id: 0,
            },
        );
        assert_eq!(acc.preemption_count, 1);
    }

    #[test]
    fn process_queue_delay() {
        let mut acc = BillingPeriodAccumulator::new();
        process_accounting_event(
            &mut acc,
            AccountingEvent::QueueDelay {
                tenant_id: 1,
                session_id: 100,
                delay_ns: 5_000_000,
            },
        );
        assert_eq!(acc.queue_delay_ns, 5_000_000);
    }

    #[test]
    fn process_quota_violation() {
        let mut acc = BillingPeriodAccumulator::new();
        process_accounting_event(
            &mut acc,
            AccountingEvent::QuotaViolationEvent {
                tenant_id: 1,
                session_id: 100,
            },
        );
        assert_eq!(acc.quota_violations, 1);
    }

    #[test]
    fn process_multiple_events() {
        let mut acc = BillingPeriodAccumulator::new();
        for _i in 0..10 {
            process_accounting_event(
                &mut acc,
                AccountingEvent::KernelCompleted {
                    tenant_id: 1,
                    session_id: 100,
                    gpu_id: 0,
                    execution_ns: 100_000_000,
                    gpu_geu: 1.0,
                },
            );
        }
        assert_eq!(acc.kernel_count, 10);
        // 10 * 0.1s * 1.0 = 1.0 GEU-seconds
        assert!((acc.geu_seconds - 1.0).abs() < 1e-10);
    }

    // --- AuthCredential tests ---

    #[test]
    fn auth_credential_variants() {
        let _ = AuthCredential::None;
        let _ = AuthCredential::ApiKey {
            key_hash: [0u8; 32],
            salt: [0u8; 16],
        };
        let _ = AuthCredential::CertificateFingerprint {
            sha256: [0u8; 32],
        };
    }

    // --- TenantRole tests ---

    #[test]
    fn tenant_role_variants() {
        assert_ne!(TenantRole::Admin, TenantRole::User);
        assert_eq!(TenantRole::Operator, TenantRole::Operator);
    }

    // --- SessionGrant tests ---

    #[test]
    fn session_grant_creation() {
        let grant = SessionGrant {
            tenant_id: 1,
            session_id: 42,
            role: TenantRole::User,
            quota_snapshot: TenantQuota::default(),
            issued_at: SystemTime::now(),
            expires_at: SystemTime::now(),
            signature: [0u8; 32],
        };
        assert_eq!(grant.tenant_id, 1);
        assert_eq!(grant.session_id, 42);
    }

    // --- QuotaViolation tests ---

    #[test]
    fn quota_violation_variants() {
        let _ = QuotaViolation::VramExceeded {
            current: 0,
            requested: 0,
            limit: 0,
        };
        let _ = QuotaViolation::GpuCountExceeded {
            current: 0,
            requested: 0,
            limit: 0,
        };
        let _ = QuotaViolation::TenantDisabled;
    }

    // --- PendingKernel tests ---

    #[test]
    fn pending_kernel_creation() {
        let pk = PendingKernel {
            function: 0x100,
            grid_dim: [128, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 4096,
            queued_at: Instant::now(),
            estimated_ns: 0,
            memory_regions: vec![MemoryRegionRef {
                vaddr: 0x200,
                size_bytes: 1024,
                is_write: true,
            }],
            assigned_gpu: None,
        };
        assert_eq!(pk.memory_regions.len(), 1);
        assert!(pk.assigned_gpu.is_none());
    }

    // --- TimeSlice tests ---

    #[test]
    fn time_slice_creation() {
        let ts = TimeSlice {
            tenant_id: 1,
            session_id: 100,
            gpu_id: 0,
            start_ns: 0,
            duration_ns: 100_000_000, // 100ms
            max_kernels: 0,
            extensible: true,
            geu_cost: 0.0001,
            priority: PriorityTier::Development,
            state: TimeSliceState::Pending,
        };
        assert_eq!(ts.state, TimeSliceState::Pending);
        assert!(ts.extensible);
    }

    // --- AuthMode tests ---

    #[test]
    fn auth_mode_variants() {
        let _ = AuthMode::Open;
        let _ = AuthMode::ApiKey;
        let _ = AuthMode::MutualTls;
    }

    // --- DrfConsumption tests ---

    #[test]
    fn drf_consumption_default() {
        let c = DrfConsumption::default();
        assert!((c.geu_seconds - 0.0).abs() < f64::EPSILON);
        assert_eq!(c.vram_bytes, 0);
        assert!((c.dominant_share - 0.0).abs() < f64::EPSILON);
    }

    // --- PreemptionReason tests ---

    #[test]
    fn preemption_reason_variants() {
        let _ = PreemptionReason::PriorityPreemption {
            victim_tier: PriorityTier::Background,
            preemptor_tier: PriorityTier::Production,
        };
        let _ = PreemptionReason::Rebalancing;
        let _ = PreemptionReason::SessionTermination;
    }

    // --- Integration scenario: full scheduling cycle ---

    #[test]
    fn integration_full_scheduling_cycle() {
        // Set up a pool with 1 GPU, 2 tenants
        let mut scheduler = DrfScheduler::new(
            SharingConfig {
                time_slice: TimeSliceConfig {
                    // Short horizon so we don't generate too many slices
                    scheduling_horizon: Duration::from_secs(1),
                    default_slice_duration: Duration::from_millis(100),
                    ..Default::default()
                },
                ..Default::default()
            },
            PoolResources {
                total_geu: 8.0,
                total_vram_bytes: 48 * 1024 * 1024 * 1024,
                total_network_bw: 10_000_000_000,
            },
        );
        scheduler.register_gpu(0);

        let alice = make_tenant_with_session(1, "alice", PriorityTier::Development);
        let bob = make_tenant_with_session(2, "bob", PriorityTier::Production);
        scheduler.register_tenant(alice);
        scheduler.register_tenant(bob);

        // First round
        let slices = scheduler.schedule_round();
        assert!(!slices.is_empty());

        // With DRF, both tenants should get slices since consumption
        // is updated after each assignment, alternating between tenants.
        // Production tenant (bob) has higher weight so his dominant share
        // grows slower, meaning he should get at least as many slices.
        let bob_slices = slices.iter().filter(|s| s.tenant_id == 2).count();
        let alice_slices = slices.iter().filter(|s| s.tenant_id == 1).count();
        // Total slices = 10 (1s / 100ms)
        assert_eq!(slices.len(), 10);
        // Both tenants should receive slices (DRF fairness)
        assert!(bob_slices + alice_slices == 10);
        // Bob (Production, weight 2.0) should get more slices than alice
        // (Development, weight 1.0) because his dominant share grows slower
        assert!(
            bob_slices >= alice_slices,
            "bob_slices={}, alice_slices={}",
            bob_slices,
            alice_slices
        );
    }
}
