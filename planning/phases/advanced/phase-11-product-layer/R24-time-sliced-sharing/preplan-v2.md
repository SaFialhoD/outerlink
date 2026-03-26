# R24: Time-Sliced GPU Sharing -- Pre-Plan v2

**Created:** 2026-03-26
**Last Updated:** 2026-03-26
**Status:** Draft
**Priority:** MEDIUM
**Supersedes:** (none -- first detailed pre-plan for R24)

## Purpose

Defines the complete architecture for multi-tenant GPU sharing in OuterLink. This document specifies Rust data structures for tenants, quotas, time slices, and scheduling policies; details the DRF (Dominant Resource Fairness) scheduling algorithm adapted for heterogeneous GPU pools; establishes VRAM isolation and memory limit enforcement; describes the authentication/authorization model; defines usage accounting and billing-ready metrics; and maps all integration points with R17, R13, R22, and R23.

The goal: turn a LAN of gaming PCs into a GPU cloud where multiple users submit workloads with fair sharing, priority preemption, and resource isolation -- all without requiring datacenter GPUs (no MIG, no vGPU).

---

## 1. Dependencies and Context

### 1.1 Hard Dependencies

| Dependency | What R24 Needs From It |
|---|---|
| **P10 (Multi-Node)** | Basic multi-node pool with coordinator. R24 builds tenant isolation on top of an already-working pool. |
| **R17 (Topology-Aware Scheduling)** | `PlacementDecision` pipeline. R24 injects a tenant-aware filter and quota check into the placement pipeline. |
| **R23 (GPU Mixing)** | `GpuProfile`, `GeuWeights`, `geu` field. R24 uses GEU as the fairness currency for DRF. Without GEU normalization, a user holding an RTX 4090 and a user holding an RTX 2060 cannot be compared fairly. |

### 1.2 Soft Dependencies

| Dependency | What R24 Gains |
|---|---|
| **R13 (CUDA Graphs)** | Graph-level scheduling allows R24 to preempt at graph boundaries rather than mid-kernel, reducing context switch cost. |
| **R22 (Live Migration)** | Enables R24 to rebalance tenant workloads when quotas change or new tenants join. Without R22, rebalancing requires workload restart. |
| **R15 (Fault Tolerance)** | Tenant workload recovery after GPU failure. Without R15, a failed GPU means the tenant's work is lost. |

### 1.3 What R24 Does NOT Do

- **Hardware partitioning (MIG):** OuterLink targets GeForce GPUs which do not support MIG. Isolation is software-enforced.
- **Container/VM isolation:** OuterLink runs as `LD_PRELOAD` in user processes. Isolation is at the CUDA API interception layer, not at the OS level.
- **Network bandwidth isolation:** Network QoS between tenants is deferred to R26 (if needed). R24 focuses on GPU compute and VRAM.

---

## 2. Research Summary

### 2.1 GPU Sharing Mechanisms (Industry)

**NVIDIA MIG:** Hardware partitioning into up to 7 isolated instances with dedicated HBM, cache, and compute. Only on A30/A100/H100+. Provides strong isolation but requires static pre-configuration and GPU reset to change layouts. Not applicable to GeForce GPUs.

**NVIDIA MPS (Multi-Process Service):** Software sharing via a daemon that merges multiple CUDA contexts into one, enabling concurrent kernel execution. Supports `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` for SM limiting and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` for memory capping. Works on all CUDA GPUs (CC 3.5+). No memory protection between clients -- designed for trusted, cooperative processes.

**NVIDIA Time-Slicing:** The GPU driver interleaves contexts via round-robin scheduling. No memory isolation, no fault isolation. Works on all GPUs. Context switch overhead is 200us-2ms depending on register file size and cache state.

**OuterLink's approach:** We operate above the CUDA driver. We intercept `cuLaunchKernel`, `cuMemAlloc`, and all other driver API calls. This gives us a unique position: we can implement time-slicing at the API interception layer, controlling when each tenant's kernels are submitted to the real GPU. Combined with VRAM tracking (we already intercept all allocations), we can enforce memory limits without hardware support.

### 2.2 Fair Scheduling: Dominant Resource Fairness (DRF)

DRF (Ghodsi et al., NSDI 2011) generalizes max-min fairness to multiple resource types. Key properties:

1. **Sharing incentive:** Each user gets at least 1/n of each resource (where n = number of users).
2. **Strategy-proofness:** Users cannot game the system by lying about resource needs.
3. **Pareto efficiency:** No allocation increase is possible without decreasing another user's allocation.
4. **Envy-freeness:** No user prefers another user's allocation.

For OuterLink, the resources are:
- **GPU compute time** (measured in GEU-seconds: time * GEU rating of the GPU)
- **VRAM capacity** (bytes, normalized by GEU to account for heterogeneous GPU memory sizes)
- **Network bandwidth** (bytes/sec for cross-node transfers)

DRF limitation for ML workloads (THEMIS, NSDI 2020): DRF does not account for placement preferences or long-running task durations. OuterLink addresses this by integrating DRF with R17's topology-aware placement -- the scheduler first filters by placement constraints, then applies DRF among eligible GPUs.

### 2.3 Preemption and Context Switching

GPU preemption is expensive: context switch costs 200us-2ms due to large register files (tens of MB per context). Research (GPREEMPT, USENIX ATC 2025) shows switch-based preemption achieves ~200us average latency.

OuterLink's advantage: since we intercept at the API level, we can implement **cooperative preemption** at kernel boundaries rather than mid-kernel. When a preemption is needed:
1. Stop submitting new kernels for the preempted tenant
2. Wait for in-flight kernels to complete (typically <10ms for inference, <100ms for training steps)
3. Flush the tenant's pending queue
4. Begin submitting the preempting tenant's kernels

This avoids GPU-level context switching entirely. The cost is the drain time for in-flight kernels.

### 2.4 Usage Accounting

Industry has moved away from "GPU hours" as a billing metric because it ignores GPU heterogeneity. The academic community (ACM HotOS 2022) proposes "GPU core hours" as a fairer metric. OuterLink uses **GEU-hours** (GPU Equivalent Unit hours), which already normalizes for GPU capability via R23's scoring system. One GEU-hour on an RTX 4090 (GEU ~4.04) costs the same as 4.04 GEU-hours on an RTX 3060 (GEU 1.0).

---

## 3. Core Data Structures

### 3.1 Tenant Identity

```rust
use std::collections::{HashMap, HashSet};
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

// ---------------------------------------------------------------------------
// Tenant
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

/// Priority tiers determine preemption rights and scheduling weight.
/// Higher tiers can preempt lower tiers during resource contention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PriorityTier {
    /// Background: lowest priority. Preempted by all other tiers.
    /// Suitable for: experiments, nightly batch jobs, speculative work.
    /// Scheduling weight: 0.25x
    Background = 0,

    /// Development: normal priority for interactive development.
    /// Preempted by Production and Critical only.
    /// Scheduling weight: 1.0x
    Development = 1,

    /// Production: high priority for serving workloads.
    /// Preempted only by Critical.
    /// Scheduling weight: 2.0x
    Production = 2,

    /// Critical: highest priority. Never preempted.
    /// Reserved for latency-sensitive inference serving.
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

/// Authentication credential stored for a tenant.
#[derive(Debug, Clone)]
pub enum AuthCredential {
    /// Pre-shared API key (stored as argon2 hash).
    ApiKey {
        key_hash: [u8; 32],
        salt: [u8; 16],
    },
    /// TLS client certificate (stored as SHA-256 fingerprint).
    CertificateFingerprint {
        sha256: [u8; 32],
    },
    /// No authentication (pool is in "open" mode -- LAN-only trust model).
    /// Default for single-user or trusted LAN setups.
    None,
}

/// Per-tenant scheduling preferences.
#[derive(Debug, Clone)]
pub struct SchedulingPreferences {
    /// Preferred GPU models (empty = no preference).
    /// The scheduler favors these GPUs but does not guarantee them.
    pub preferred_gpus: Vec<GpuId>,

    /// Preferred nodes (empty = no preference).
    pub preferred_nodes: Vec<NodeId>,

    /// Maximum acceptable latency for kernel dispatch (0 = no constraint).
    /// If a GPU cannot serve the kernel within this time, skip it.
    pub max_dispatch_latency_us: u64,

    /// Whether this tenant's workloads can be migrated for rebalancing (R22).
    pub allow_migration: bool,

    /// Whether this tenant's workloads can be preempted by higher-priority tenants.
    /// Critical-tier tenants always set this to false.
    pub preemptible: bool,

    /// Exclusive GPU access: if true, the scheduler never co-locates another
    /// tenant's work on GPUs assigned to this tenant. Costs more quota.
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
```

### 3.2 Quota and Resource Limits

```rust
// ---------------------------------------------------------------------------
// Quota
// ---------------------------------------------------------------------------

/// Resource quota for a tenant. Defines the maximum resources the tenant
/// can consume simultaneously (capacity limits) and over time (rate limits).
#[derive(Debug, Clone)]
pub struct TenantQuota {
    // --- Capacity limits (simultaneous usage caps) ---

    /// Maximum VRAM the tenant can allocate across all GPUs, in bytes.
    /// 0 = unlimited (pool default).
    pub max_vram_bytes: u64,

    /// Maximum number of GPUs the tenant can use simultaneously.
    /// 0 = unlimited.
    pub max_gpus: u32,

    /// Maximum GEU the tenant can consume simultaneously.
    /// This is the primary fairness metric. A tenant with max_geu = 5.0
    /// can hold one RTX 4090 (GEU 4.04) + one RTX 3060 (GEU 1.0), approximately.
    /// 0.0 = unlimited.
    pub max_geu: f64,

    /// Maximum VRAM per single GPU, in bytes.
    /// Prevents one tenant from consuming all VRAM on a single GPU.
    /// 0 = limited only by GPU total VRAM.
    pub max_vram_per_gpu_bytes: u64,

    /// Maximum number of concurrent sessions (application instances).
    /// 0 = unlimited.
    pub max_sessions: u32,

    /// Maximum number of kernels in the pending queue per session.
    /// Prevents queue flooding. 0 = unlimited.
    pub max_pending_kernels: u32,

    // --- Rate limits (over time) ---

    /// Maximum GEU-hours per billing period (typically 24h or 1 month).
    /// 0.0 = unlimited.
    pub max_geu_hours_per_period: f64,

    /// Billing period duration.
    pub billing_period: Duration,

    /// Maximum data transfer volume per billing period, in bytes.
    /// Covers cross-node transfers (host-staged and OpenDMA).
    /// 0 = unlimited.
    pub max_transfer_bytes_per_period: u64,

    // --- Burst allowance ---

    /// Burst multiplier: the tenant can temporarily exceed max_geu by this
    /// factor if the pool has spare capacity and no other tenant is waiting.
    /// 1.0 = no burst allowed. 2.0 = can use up to 2x max_geu when idle.
    pub burst_multiplier: f64,

    /// Maximum burst duration before enforcement kicks in.
    pub max_burst_duration: Duration,
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_vram_bytes: 0,                          // unlimited
            max_gpus: 0,                                 // unlimited
            max_geu: 0.0,                                // unlimited
            max_vram_per_gpu_bytes: 0,                  // unlimited
            max_sessions: 8,                             // reasonable default
            max_pending_kernels: 10_000,                // prevent queue flooding
            max_geu_hours_per_period: 0.0,              // unlimited
            billing_period: Duration::from_secs(86400), // 24 hours
            max_transfer_bytes_per_period: 0,            // unlimited
            burst_multiplier: 1.5,                       // 50% burst
            max_burst_duration: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Quota presets for common use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaPreset {
    /// Unlimited: no caps (single-user or admin).
    Unlimited,
    /// Light: suitable for development and experimentation.
    /// 1 GPU, 8 GB VRAM, 2 GEU max, 24 GEU-hours/day.
    Light,
    /// Standard: suitable for regular training workloads.
    /// 2 GPUs, 24 GB VRAM, 5 GEU max, 120 GEU-hours/day.
    Standard,
    /// Heavy: suitable for large-scale training.
    /// 4 GPUs, 96 GB VRAM, 15 GEU max, 360 GEU-hours/day.
    Heavy,
}

impl QuotaPreset {
    pub fn to_quota(&self) -> TenantQuota {
        match self {
            QuotaPreset::Unlimited => TenantQuota::default(),
            QuotaPreset::Light => TenantQuota {
                max_vram_bytes: 8 * 1024 * 1024 * 1024,        // 8 GB
                max_gpus: 1,
                max_geu: 2.0,
                max_vram_per_gpu_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
                max_sessions: 2,
                max_pending_kernels: 1_000,
                max_geu_hours_per_period: 24.0,
                billing_period: Duration::from_secs(86400),
                max_transfer_bytes_per_period: 100 * 1024 * 1024 * 1024, // 100 GB
                burst_multiplier: 1.5,
                max_burst_duration: Duration::from_secs(300),
            },
            QuotaPreset::Standard => TenantQuota {
                max_vram_bytes: 24 * 1024 * 1024 * 1024,        // 24 GB
                max_gpus: 2,
                max_geu: 5.0,
                max_vram_per_gpu_bytes: 24 * 1024 * 1024 * 1024, // 24 GB
                max_sessions: 4,
                max_pending_kernels: 5_000,
                max_geu_hours_per_period: 120.0,
                billing_period: Duration::from_secs(86400),
                max_transfer_bytes_per_period: 500 * 1024 * 1024 * 1024, // 500 GB
                burst_multiplier: 1.5,
                max_burst_duration: Duration::from_secs(300),
            },
            QuotaPreset::Heavy => TenantQuota {
                max_vram_bytes: 96 * 1024 * 1024 * 1024,        // 96 GB
                max_gpus: 4,
                max_geu: 15.0,
                max_vram_per_gpu_bytes: 24 * 1024 * 1024 * 1024, // 24 GB
                max_sessions: 8,
                max_pending_kernels: 10_000,
                max_geu_hours_per_period: 360.0,
                billing_period: Duration::from_secs(86400),
                max_transfer_bytes_per_period: 2 * 1024 * 1024 * 1024 * 1024, // 2 TB
                burst_multiplier: 1.25,
                max_burst_duration: Duration::from_secs(300),
            },
        }
    }
}
```

### 3.3 Resource Usage Tracking

```rust
// ---------------------------------------------------------------------------
// Resource Usage (real-time)
// ---------------------------------------------------------------------------

/// Current resource usage for a tenant, updated on every allocation/deallocation
/// and every kernel launch/completion.
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// Total VRAM currently allocated by this tenant across all GPUs, in bytes.
    pub vram_allocated_bytes: u64,

    /// Per-GPU VRAM usage breakdown.
    pub vram_per_gpu: HashMap<GpuId, u64>,

    /// Number of GPUs currently in use by this tenant.
    pub gpus_in_use: u32,

    /// Set of GPU IDs currently assigned to this tenant.
    pub assigned_gpus: HashSet<GpuId>,

    /// Current GEU consumption (sum of GEU values for assigned GPUs,
    /// weighted by the fraction of each GPU this tenant is using).
    pub current_geu: f64,

    /// Number of kernels currently in-flight on GPUs.
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

    /// Whether this tenant is currently in burst mode (exceeding max_geu).
    pub in_burst: bool,

    /// When burst mode started (None if not in burst).
    pub burst_start: Option<Instant>,
}

impl ResourceUsage {
    /// Check if a proposed allocation would exceed the tenant's quota.
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
    VramExceeded {
        current: u64,
        requested: u64,
        limit: u64,
    },
    GpuCountExceeded {
        current: u32,
        requested: u32,
        limit: u32,
    },
    GeuExceeded {
        current: f64,
        requested: f64,
        limit: f64,
        burst_limit: f64,
    },
    GeuHoursExceeded {
        consumed: f64,
        limit: f64,
    },
    TransferBytesExceeded {
        consumed: u64,
        limit: u64,
    },
    SessionLimitReached {
        current: u32,
        limit: u32,
    },
    PendingQueueFull {
        current: u32,
        limit: u32,
    },
    TenantDisabled,
}
```

### 3.4 Time Slice

```rust
// ---------------------------------------------------------------------------
// Time Slice
// ---------------------------------------------------------------------------

/// A time slice represents a scheduled execution window on a specific GPU
/// for a specific tenant. The scheduler produces a sequence of time slices
/// that determines who gets to run on each GPU and when.
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

    /// Maximum number of kernels the tenant can launch during this slice.
    /// 0 = unlimited (bounded only by time).
    pub max_kernels: u32,

    /// Whether this slice can be extended if no other tenant is waiting.
    pub extensible: bool,

    /// The GEU cost of this slice (duration_ns * gpu_geu / 1e9 / 3600 = GEU-hours).
    pub geu_cost: f64,

    /// Priority of the work in this slice (inherited from tenant).
    pub priority: PriorityTier,

    /// State of this time slice.
    pub state: TimeSliceState,
}

/// Life cycle of a time slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSliceState {
    /// Scheduled but not yet started.
    Pending,
    /// Currently active: the tenant's kernels are being dispatched.
    Active,
    /// Completed normally (duration expired or all work done).
    Completed,
    /// Preempted by a higher-priority tenant before duration expired.
    Preempted,
    /// Cancelled (tenant disconnected or session ended).
    Cancelled,
}

/// Configuration for the time-slicing scheduler.
#[derive(Debug, Clone)]
pub struct TimeSliceConfig {
    /// Default time slice duration per tenant per GPU.
    /// This is the base quantum. Actual slices may be longer or shorter
    /// depending on priority weight and GPU contention.
    pub default_slice_duration: Duration,

    /// Minimum time slice duration. Slices shorter than this are not worth
    /// the scheduling overhead.
    pub min_slice_duration: Duration,

    /// Maximum time slice duration. Prevents starvation of waiting tenants.
    pub max_slice_duration: Duration,

    /// Grace period after a time slice expires before forceful preemption.
    /// During this period, in-flight kernels are allowed to complete.
    pub preemption_grace_period: Duration,

    /// How far ahead the scheduler plans time slices.
    pub scheduling_horizon: Duration,

    /// How often the scheduler re-evaluates and adjusts the schedule.
    pub reschedule_interval: Duration,

    /// Whether to enable work-conserving scheduling: if a tenant has no
    /// pending work during its slice, the slice is given to the next tenant.
    pub work_conserving: bool,

    /// Whether to enable slice coalescing: if the same tenant has consecutive
    /// slices on the same GPU, merge them into one longer slice.
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
```

### 3.5 Scheduling Policy

```rust
// ---------------------------------------------------------------------------
// Scheduling Policy
// ---------------------------------------------------------------------------

/// The scheduling policy determines how time slices are allocated to tenants.
#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    /// Dominant Resource Fairness: each tenant's dominant resource share is
    /// equalized (weighted by priority tier). This is the default and
    /// recommended policy for multi-tenant pools.
    DominantResourceFairness {
        /// Resource weights for computing the dominant share.
        resource_weights: DrfResourceWeights,
    },

    /// Weighted Fair Queuing: time slices are distributed proportional to
    /// each tenant's priority weight. Simpler than DRF but does not account
    /// for heterogeneous resource demands.
    WeightedFairQueue,

    /// Strict Priority: higher-priority tenants always run first.
    /// Lower-priority tenants only get slices when higher-priority tenants
    /// have no pending work. Risk of starvation.
    StrictPriority,

    /// FIFO: first-come first-served. No fairness guarantees.
    /// Suitable for single-tenant or testing scenarios.
    Fifo,

    /// Reserved: specific GPUs are permanently assigned to specific tenants.
    /// No time-slicing. Maximum isolation, minimum utilization.
    Reserved {
        assignments: HashMap<TenantId, Vec<GpuId>>,
    },
}

/// Resource weights for DRF dominant share computation.
/// These determine how much each resource type contributes to the
/// "dominant share" calculation.
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

    /// Whether to enable quota enforcement (false = quotas are advisory only).
    pub enforce_quotas: bool,

    /// Default quota for new tenants.
    pub default_quota_preset: QuotaPreset,

    /// Maximum number of tenants in the pool.
    pub max_tenants: u32,

    /// Starvation prevention: minimum percentage of each scheduling round
    /// allocated to the lowest-priority tier with pending work.
    /// 0.0 = no starvation prevention (strict priority possible).
    /// 0.05 = at least 5% of GPU time goes to lowest tier.
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

/// Authentication mode for the pool.
#[derive(Debug, Clone)]
pub enum AuthMode {
    /// No authentication. Any client that connects is assigned a tenant
    /// based on OS username or a default tenant. Suitable for trusted LANs.
    Open,
    /// API key authentication. Clients must present a valid key in the
    /// connection handshake. Keys are pre-shared by the pool admin.
    ApiKey,
    /// TLS mutual authentication. Clients must present a valid client
    /// certificate. The certificate's CN or SAN determines the tenant.
    MutualTls,
}
```

### 3.6 Session (Application Instance)

```rust
// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A session represents one application instance connected to the OuterLink pool.
/// Each session belongs to exactly one tenant. Sessions are created when a
/// CUDA application connects (via LD_PRELOAD handshake) and destroyed when
/// the application exits or the connection drops.
#[derive(Debug, Clone)]
pub struct Session {
    /// Unique session identifier.
    pub session_id: SessionId,

    /// Owning tenant.
    pub tenant_id: TenantId,

    /// Process ID on the originating node (for debugging/audit).
    pub pid: u32,

    /// Node where the application is running.
    pub origin_node: NodeId,

    /// Application name (from /proc/pid/comm or equivalent).
    pub app_name: String,

    /// When this session started.
    pub started_at: SystemTime,

    /// Current state of the session.
    pub state: SessionState,

    /// Per-session resource usage (subset of tenant's total).
    pub usage: SessionUsage,

    /// CUDA contexts created by this session.
    pub cuda_contexts: Vec<CudaContextHandle>,

    /// VRAM allocations owned by this session.
    /// Maps virtual address to (gpu_id, size_bytes).
    pub allocations: HashMap<u64, (GpuId, u64)>,

    /// Pending kernel queue for this session.
    pub pending_kernels: VecDeque<PendingKernel>,
}

use std::collections::VecDeque;

/// Opaque handle to a CUDA context (mirrors CUcontext from the driver API).
pub type CudaContextHandle = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is active and can submit work.
    Active,
    /// Session is being drained (no new work accepted, waiting for in-flight to complete).
    Draining,
    /// Session is suspended (tenant over quota or preempted).
    Suspended,
    /// Session has ended.
    Terminated,
}

/// Per-session resource usage.
#[derive(Debug, Clone, Default)]
pub struct SessionUsage {
    /// Total VRAM allocated by this session.
    pub vram_bytes: u64,
    /// Kernels launched by this session.
    pub kernels_launched: u64,
    /// Kernels completed by this session.
    pub kernels_completed: u64,
    /// Total GPU time consumed by this session's kernels, in nanoseconds.
    pub gpu_time_ns: u64,
    /// GEU-seconds consumed (gpu_time_ns * gpu_geu for each kernel's GPU).
    pub geu_seconds: f64,
    /// Data transferred cross-node, in bytes.
    pub transfer_bytes: u64,
}

/// A kernel waiting to be dispatched to a GPU.
#[derive(Debug, Clone)]
pub struct PendingKernel {
    /// The CUDA function to launch.
    pub function: u64, // CUfunction as u64
    /// Grid dimensions.
    pub grid_dim: [u32; 3],
    /// Block dimensions.
    pub block_dim: [u32; 3],
    /// Shared memory size.
    pub shared_mem_bytes: u32,
    /// When the kernel was queued.
    pub queued_at: Instant,
    /// Estimated execution time (from R23 profiling, 0 if unknown).
    pub estimated_ns: u64,
    /// Memory regions this kernel will access (from R13 shadow graph analysis).
    pub memory_regions: Vec<MemoryRegionRef>,
    /// The target GPU (assigned by the scheduler, None if not yet assigned).
    pub assigned_gpu: Option<GpuId>,
}

/// Reference to a memory region accessed by a kernel (for VRAM accounting).
#[derive(Debug, Clone)]
pub struct MemoryRegionRef {
    /// Virtual address of the allocation.
    pub vaddr: u64,
    /// Size of the access (may be less than allocation size).
    pub size_bytes: u64,
    /// Whether the kernel writes to this region.
    pub is_write: bool,
}
```

---

## 4. Key Algorithms

### 4.1 DRF Scheduling Algorithm

The DRF scheduler runs at the coordinator and produces a sequence of `TimeSlice` assignments for each GPU. It runs every `reschedule_interval` (default 100ms).

```rust
// ---------------------------------------------------------------------------
// DRF Scheduler
// ---------------------------------------------------------------------------

/// The DRF scheduler maintains per-tenant dominant shares and picks the
/// tenant with the lowest weighted dominant share for each scheduling slot.
struct DrfScheduler {
    /// All registered tenants.
    tenants: HashMap<TenantId, Tenant>,

    /// Pool-level resource totals (denominator for share computation).
    pool_resources: PoolResources,

    /// Per-tenant resource consumption (numerator for share computation).
    tenant_consumption: HashMap<TenantId, DrfConsumption>,

    /// Configuration.
    config: SharingConfig,

    /// Per-GPU current tenant assignment.
    gpu_assignments: HashMap<GpuId, Option<TenantId>>,

    /// Time slice schedule (produced by each scheduling round).
    schedule: Vec<TimeSlice>,

    /// Round counter (for starvation detection).
    round: u64,
}

/// Total pool resources (the "capacity" in DRF).
struct PoolResources {
    /// Total GEU across all GPUs in the pool.
    total_geu: f64,
    /// Total VRAM across all GPUs in the pool, in bytes.
    total_vram_bytes: u64,
    /// Total network bandwidth across all links, in bytes/sec.
    total_network_bw: u64,
}

/// Per-tenant consumption tracking for DRF.
#[derive(Debug, Clone, Default)]
struct DrfConsumption {
    /// GEU-seconds consumed in the current scheduling window.
    geu_seconds: f64,
    /// VRAM bytes currently held.
    vram_bytes: u64,
    /// Network bytes/sec currently used.
    network_bps: u64,
    /// Computed shares (fraction of pool total).
    compute_share: f64,
    vram_share: f64,
    network_share: f64,
    /// Dominant share (the maximum of the weighted shares).
    dominant_share: f64,
}

impl DrfScheduler {
    /// Core scheduling loop: produce time slices for all GPUs for the next
    /// scheduling horizon.
    ///
    /// Algorithm (weighted DRF with starvation prevention):
    ///
    /// 1. Compute each tenant's resource shares:
    ///    compute_share = geu_seconds / pool_total_geu_seconds
    ///    vram_share = vram_bytes / pool_total_vram
    ///    network_share = network_bps / pool_total_network_bw
    ///
    /// 2. Compute weighted dominant share:
    ///    dominant_share = max(
    ///        resource_weights.compute * compute_share,
    ///        resource_weights.vram * vram_share,
    ///        resource_weights.network * network_share,
    ///    ) / tenant.priority.weight()
    ///
    /// 3. For each GPU with available scheduling slots:
    ///    a. Filter tenants: must have pending work, must pass quota check,
    ///       must be compatible with this GPU (R23 binary compat).
    ///    b. Among eligible tenants, pick the one with the LOWEST dominant_share.
    ///    c. Assign a time slice of duration = default_duration * priority_weight.
    ///    d. Update the selected tenant's consumption.
    ///    e. If work_conserving and no tenants have pending work, extend the
    ///       current tenant's slice.
    ///
    /// 4. Starvation prevention: if any tier has received less than
    ///    min_share_percent of total GPU time over the last N rounds, force
    ///    at least one slice for that tier's highest-waiting tenant.
    fn schedule_round(&mut self) -> Vec<TimeSlice> {
        let mut slices = Vec::new();

        // Step 1-2: compute dominant shares for all tenants
        self.update_dominant_shares();

        // Step 3: for each GPU, assign time slices
        let gpu_ids: Vec<GpuId> = self.gpu_assignments.keys().copied().collect();
        for gpu_id in gpu_ids {
            let slice_count = self.slices_per_round();
            for _ in 0..slice_count {
                if let Some(slice) = self.assign_slice(gpu_id) {
                    // Update consumption after assignment
                    let consumption = self.tenant_consumption
                        .entry(slice.tenant_id)
                        .or_default();
                    consumption.geu_seconds += slice.geu_cost * 3600.0; // convert GEU-hours to GEU-seconds
                    slices.push(slice);
                }
            }
        }

        // Step 4: starvation prevention
        self.enforce_min_share(&mut slices);

        self.round += 1;
        slices
    }

    /// Compute dominant share for each tenant.
    fn update_dominant_shares(&mut self) {
        let weights = match &self.config.policy {
            SchedulingPolicy::DominantResourceFairness { resource_weights } => {
                resource_weights.clone()
            }
            _ => DrfResourceWeights::default(),
        };

        let pool = &self.pool_resources;
        let total_geu_seconds = pool.total_geu; // per-second capacity
        let total_vram = pool.total_vram_bytes as f64;
        let total_network = pool.total_network_bw as f64;

        for (tenant_id, consumption) in &mut self.tenant_consumption {
            let tenant = match self.tenants.get(tenant_id) {
                Some(t) => t,
                None => continue,
            };

            // Compute shares (fraction of pool total)
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

            // Dominant share = max weighted share / priority weight
            consumption.dominant_share = f64::max(
                f64::max(
                    weights.compute * consumption.compute_share,
                    weights.vram * consumption.vram_share,
                ),
                weights.network * consumption.network_share,
            ) / tenant.priority.weight();
        }
    }

    /// Pick the best tenant for a time slice on the given GPU.
    fn assign_slice(&self, gpu_id: GpuId) -> Option<TimeSlice> {
        // Collect eligible tenants (have pending work, pass quota, GPU compatible)
        let mut candidates: Vec<(TenantId, f64)> = Vec::new();

        for (tenant_id, tenant) in &self.tenants {
            if !tenant.enabled { continue; }
            if tenant.active_sessions.is_empty() { continue; }

            // Check if tenant has pending work on any session
            let has_pending = true; // simplified: checked via session pending queues

            if !has_pending { continue; }

            // Quota check
            if self.config.enforce_quotas {
                if tenant.usage.would_exceed_quota(&tenant.quota, 0, 0, 0.0).is_some() {
                    continue;
                }
            }

            let dominant_share = self.tenant_consumption
                .get(tenant_id)
                .map(|c| c.dominant_share)
                .unwrap_or(0.0);

            candidates.push((*tenant_id, dominant_share));
        }

        // Pick tenant with lowest dominant share
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let (tenant_id, _) = candidates.first()?;
        let tenant = self.tenants.get(tenant_id)?;

        // Compute slice duration based on priority weight
        let base_duration = self.config.time_slice.default_slice_duration;
        let weighted_duration = Duration::from_nanos(
            (base_duration.as_nanos() as f64 * tenant.priority.weight()) as u64
        );
        let clamped_duration = weighted_duration
            .max(self.config.time_slice.min_slice_duration)
            .min(self.config.time_slice.max_slice_duration);

        // Pick a session from this tenant (round-robin among active sessions)
        let session_id = *tenant.active_sessions.iter().next()?;

        // Compute GEU cost
        // geu_cost = duration_seconds * gpu_geu / 3600 (to get GEU-hours)
        let gpu_geu = 1.0; // obtained from R23 GpuProfile
        let geu_cost = clamped_duration.as_secs_f64() * gpu_geu / 3600.0;

        Some(TimeSlice {
            tenant_id: *tenant_id,
            session_id,
            gpu_id,
            start_ns: 0, // filled in by the schedule assembler
            duration_ns: clamped_duration.as_nanos() as u64,
            max_kernels: 0, // unlimited
            extensible: self.config.time_slice.work_conserving,
            geu_cost,
            priority: tenant.priority,
            state: TimeSliceState::Pending,
        })
    }

    /// Enforce minimum share for starved tiers.
    fn enforce_min_share(&self, slices: &mut Vec<TimeSlice>) {
        if self.config.min_share_percent <= 0.0 { return; }

        // Count slices per priority tier
        let mut tier_counts: HashMap<PriorityTier, u32> = HashMap::new();
        for slice in slices.iter() {
            *tier_counts.entry(slice.priority).or_insert(0) += 1;
        }

        let total_slices = slices.len() as f64;
        if total_slices == 0.0 { return; }

        // Check if any tier with pending work got less than min_share_percent
        let min_count = (total_slices * self.config.min_share_percent).ceil() as u32;

        for tier in [PriorityTier::Background, PriorityTier::Development] {
            let count = tier_counts.get(&tier).copied().unwrap_or(0);
            if count < min_count {
                // Find a tenant at this tier with pending work and inject slices
                // (implementation detail: replace lowest-priority excess slices)
            }
        }
    }

    fn slices_per_round(&self) -> usize {
        let horizon = self.config.time_slice.scheduling_horizon;
        let slice_dur = self.config.time_slice.default_slice_duration;
        (horizon.as_nanos() / slice_dur.as_nanos()) as usize
    }
}
```

### 4.2 Quota Enforcement

Quota enforcement happens at three points in the request lifecycle:

```rust
// ---------------------------------------------------------------------------
// Quota Enforcer
// ---------------------------------------------------------------------------

/// The QuotaEnforcer is called at interception points to gate resource access.
struct QuotaEnforcer {
    tenants: HashMap<TenantId, Tenant>,
    config: SharingConfig,
}

/// Enforcement points and their actions:
///
/// 1. cuMemAlloc / cuMemAllocManaged / cuMemAllocPitch:
///    - Check: would this allocation exceed the tenant's max_vram_bytes
///      or max_vram_per_gpu_bytes?
///    - Action on violation: return CUDA_ERROR_OUT_OF_MEMORY to the application.
///    - The application sees a "normal" OOM, which most frameworks handle gracefully.
///
/// 2. cuLaunchKernel / cuLaunchCooperativeKernel:
///    - Check: is the tenant's pending queue full (max_pending_kernels)?
///    - Check: is the tenant over their GEU-hours rate limit?
///    - Action on violation: block the thread (backpressure) until quota frees up,
///      or return CUDA_ERROR_LAUNCH_TIMEOUT after a configurable wait.
///
/// 3. Session creation (cuCtxCreate / cuDevicePrimaryCtxRetain):
///    - Check: has the tenant reached max_sessions?
///    - Action on violation: return CUDA_ERROR_UNKNOWN with a log message.
///
/// 4. Periodic (every 1 second):
///    - Check: is the tenant in burst mode past max_burst_duration?
///    - Action: begin draining excess workloads (stop accepting new kernels
///      on the lowest-priority sessions until usage drops below max_geu).
///
/// 5. Billing period rollover:
///    - Reset geu_hours_this_period and transfer_bytes_this_period.
///    - Log the period's usage to the billing accumulator.

impl QuotaEnforcer {
    /// Gate a memory allocation request.
    fn check_alloc(
        &self,
        tenant_id: TenantId,
        gpu_id: GpuId,
        size_bytes: u64,
    ) -> Result<(), QuotaViolation> {
        let tenant = self.tenants.get(&tenant_id)
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
    fn check_launch(
        &self,
        tenant_id: TenantId,
        session_id: SessionId,
    ) -> Result<(), QuotaViolation> {
        let tenant = self.tenants.get(&tenant_id)
            .ok_or(QuotaViolation::TenantDisabled)?;

        if !tenant.enabled {
            return Err(QuotaViolation::TenantDisabled);
        }

        let quota = &tenant.quota;
        let usage = &tenant.usage;

        // Pending queue check
        if quota.max_pending_kernels > 0
            && usage.pending_kernels >= quota.max_pending_kernels
        {
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
    fn check_session(
        &self,
        tenant_id: TenantId,
    ) -> Result<(), QuotaViolation> {
        let tenant = self.tenants.get(&tenant_id)
            .ok_or(QuotaViolation::TenantDisabled)?;

        let quota = &tenant.quota;
        let usage = &tenant.usage;

        if quota.max_sessions > 0
            && usage.active_session_count >= quota.max_sessions
        {
            return Err(QuotaViolation::SessionLimitReached {
                current: usage.active_session_count,
                limit: quota.max_sessions,
            });
        }

        Ok(())
    }
}
```

### 4.3 Priority Preemption

```rust
// ---------------------------------------------------------------------------
// Preemption Engine
// ---------------------------------------------------------------------------

/// Handles cooperative preemption when a higher-priority tenant needs GPU
/// resources currently held by a lower-priority tenant.
struct PreemptionEngine {
    config: TimeSliceConfig,
}

/// Preemption request issued by the scheduler.
#[derive(Debug, Clone)]
struct PreemptionRequest {
    /// GPU to preempt on.
    gpu_id: GpuId,
    /// Tenant being preempted (the victim).
    victim_tenant: TenantId,
    /// Session being preempted.
    victim_session: SessionId,
    /// Tenant that needs the GPU (the preemptor).
    preemptor_tenant: TenantId,
    /// Reason for preemption.
    reason: PreemptionReason,
    /// When the preemption was requested.
    requested_at: Instant,
}

#[derive(Debug, Clone)]
enum PreemptionReason {
    /// Higher-priority tenant needs GPU time.
    PriorityPreemption { victim_tier: PriorityTier, preemptor_tier: PriorityTier },
    /// Victim tenant exceeded their quota.
    QuotaExceeded { violation: QuotaViolation },
    /// Pool rebalancing (R22 migration).
    Rebalancing,
    /// Victim tenant's session is being terminated.
    SessionTermination,
}

/// Result of a preemption attempt.
#[derive(Debug)]
enum PreemptionResult {
    /// Preemption completed: victim's kernels drained, GPU ready.
    Completed {
        drain_time: Duration,
        kernels_drained: u32,
    },
    /// Preemption in progress: waiting for in-flight kernels.
    InProgress {
        inflight_kernels: u32,
        elapsed: Duration,
    },
    /// Preemption forced: grace period expired, remaining kernels killed.
    Forced {
        kernels_killed: u32,
    },
    /// Preemption unnecessary: victim already finished.
    Unnecessary,
}

impl PreemptionEngine {
    /// Execute a preemption. This is a multi-step cooperative process:
    ///
    /// Phase 1: STOP (immediate, <1us)
    ///   - Mark the victim's time slice as Preempted.
    ///   - Stop dequeuing kernels from the victim's pending queue.
    ///   - The interception layer will block any new cuLaunchKernel calls
    ///     from the victim's session.
    ///
    /// Phase 2: DRAIN (cooperative, typically <100ms)
    ///   - Wait for the victim's in-flight kernels to complete on the GPU.
    ///   - Monitor via CUDA events placed after each kernel launch.
    ///   - Each completed kernel triggers a callback that decrements the
    ///     victim's inflight count.
    ///
    /// Phase 3: YIELD (after drain or grace period expiry)
    ///   - Once all in-flight kernels complete (or grace period expires):
    ///     a. The victim's session enters Suspended state.
    ///     b. The GPU is reassigned to the preemptor.
    ///     c. The preemptor's pending kernels begin dispatching.
    ///   - If grace period expires with kernels still in-flight:
    ///     a. Record the forced preemption in the victim's session log.
    ///     b. The remaining kernels will complete but their results may not
    ///        be consumed (the victim should handle this gracefully).
    ///
    /// Phase 4: RESUME (when the victim gets their next time slice)
    ///   - The victim's session moves from Suspended back to Active.
    ///   - Pending kernels resume dispatching from where they left off.
    ///   - VRAM allocations are NOT freed during preemption (only compute
    ///     time is preempted, not memory). This avoids the massive cost
    ///     of serializing/restoring GPU memory state.
    ///
    /// Key insight: we do NOT swap VRAM during preemption. The victim's
    /// memory stays resident on the GPU. Only compute scheduling changes.
    /// This limits the number of tenants that can co-exist on one GPU to
    /// those whose combined VRAM fits, but avoids the >100ms cost of
    /// VRAM migration per preemption.
    fn preempt(
        &self,
        request: PreemptionRequest,
    ) -> PreemptionResult {
        // Phase 1: STOP
        // Set victim's time slice state to Preempted
        // Block victim's kernel dispatch

        // Phase 2: DRAIN
        let grace = self.config.preemption_grace_period;
        let start = Instant::now();

        // Poll for in-flight kernel completion
        // (In practice: check CUDA events, or rely on the kernel completion
        //  callback that we already have for timing/profiling)

        // Phase 3: YIELD
        // Reassign GPU to preemptor

        // Return result
        PreemptionResult::Completed {
            drain_time: start.elapsed(),
            kernels_drained: 0, // actual count from drain loop
        }
    }

    /// Check if preemption is needed for a GPU.
    /// Called when a higher-priority tenant has pending work and all GPUs
    /// are occupied by lower-priority tenants.
    fn should_preempt(
        &self,
        preemptor: &Tenant,
        victim: &Tenant,
        gpu_id: GpuId,
    ) -> bool {
        // Only preempt if:
        // 1. Preemptor has higher priority
        if !preemptor.priority.can_preempt(&victim.priority) {
            return false;
        }

        // 2. Victim is preemptible (opt-out check)
        if !victim.scheduling_prefs.preemptible {
            return false;
        }

        // 3. Preemptor actually has pending work for this GPU
        // (checked by caller)

        true
    }

    /// Select the best victim for preemption when multiple lower-priority
    /// tenants occupy the target GPU.
    fn select_victim(
        &self,
        preemptor: &Tenant,
        candidates: &[(TenantId, &Tenant)],
    ) -> Option<TenantId> {
        // Selection criteria (in order):
        // 1. Lowest priority tier first
        // 2. Among same tier: highest dominant share (they've used the most)
        // 3. Among equal dominant share: least pending work (least impact)
        candidates.iter()
            .filter(|(_, t)| preemptor.priority.can_preempt(&t.priority))
            .filter(|(_, t)| t.scheduling_prefs.preemptible)
            .min_by(|(_, a), (_, b)| {
                a.priority.cmp(&b.priority)
                    .then(b.usage.current_geu.partial_cmp(&a.usage.current_geu)
                        .unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|(id, _)| *id)
    }
}
```

### 4.4 VRAM Isolation

```rust
// ---------------------------------------------------------------------------
// VRAM Isolation
// ---------------------------------------------------------------------------

/// Per-GPU tenant VRAM tracking. Maintains a ledger of which tenant owns
/// which allocations on each GPU, and enforces per-tenant VRAM limits.
///
/// Since OuterLink intercepts ALL cuMemAlloc/cuMemFree calls, we have
/// perfect visibility into every allocation. No hardware support needed.
struct VramLedger {
    /// Per-GPU ledger: maps (gpu_id, virtual_address) -> (tenant_id, size).
    allocations: HashMap<GpuId, HashMap<u64, VramAllocation>>,

    /// Per-GPU per-tenant usage summary.
    gpu_tenant_usage: HashMap<(GpuId, TenantId), VramTenantSummary>,

    /// Per-GPU total usage.
    gpu_usage: HashMap<GpuId, GpuVramUsage>,
}

/// A single VRAM allocation tracked by the ledger.
#[derive(Debug, Clone)]
struct VramAllocation {
    /// Virtual address (the pointer returned by cuMemAlloc).
    vaddr: u64,
    /// Size in bytes.
    size_bytes: u64,
    /// Owning tenant.
    tenant_id: TenantId,
    /// Owning session.
    session_id: SessionId,
    /// When allocated.
    allocated_at: Instant,
    /// Last kernel that accessed this allocation (for LRU eviction).
    last_accessed: Instant,
    /// Access count (for frequency-based eviction).
    access_count: u64,
}

/// Per-tenant VRAM summary on a specific GPU.
#[derive(Debug, Clone, Default)]
struct VramTenantSummary {
    /// Total bytes allocated by this tenant on this GPU.
    total_bytes: u64,
    /// Number of allocations.
    allocation_count: u32,
    /// Largest single allocation.
    max_allocation_bytes: u64,
    /// Peak usage (high watermark).
    peak_bytes: u64,
}

/// Per-GPU VRAM usage.
#[derive(Debug, Clone)]
struct GpuVramUsage {
    /// Total VRAM on this GPU.
    total_bytes: u64,
    /// VRAM used by OuterLink tenants.
    tenant_used_bytes: u64,
    /// VRAM used by non-OuterLink processes (detected via NVML).
    external_used_bytes: u64,
    /// Available for new tenant allocations.
    available_bytes: u64,
    /// Per-tenant breakdown.
    per_tenant: HashMap<TenantId, u64>,
}

impl VramLedger {
    /// Record a new allocation. Called from cuMemAlloc interception.
    fn record_alloc(
        &mut self,
        gpu_id: GpuId,
        vaddr: u64,
        size_bytes: u64,
        tenant_id: TenantId,
        session_id: SessionId,
    ) {
        let alloc = VramAllocation {
            vaddr,
            size_bytes,
            tenant_id,
            session_id,
            allocated_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
        };

        self.allocations
            .entry(gpu_id)
            .or_default()
            .insert(vaddr, alloc);

        let summary = self.gpu_tenant_usage
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

        // Update GPU-level usage
        if let Some(gpu_usage) = self.gpu_usage.get_mut(&gpu_id) {
            gpu_usage.tenant_used_bytes += size_bytes;
            gpu_usage.available_bytes = gpu_usage.total_bytes
                .saturating_sub(gpu_usage.tenant_used_bytes + gpu_usage.external_used_bytes);
            *gpu_usage.per_tenant.entry(tenant_id).or_insert(0) += size_bytes;
        }
    }

    /// Record a deallocation. Called from cuMemFree interception.
    fn record_free(&mut self, gpu_id: GpuId, vaddr: u64) {
        if let Some(gpu_allocs) = self.allocations.get_mut(&gpu_id) {
            if let Some(alloc) = gpu_allocs.remove(&vaddr) {
                // Update per-tenant summary
                if let Some(summary) = self.gpu_tenant_usage
                    .get_mut(&(gpu_id, alloc.tenant_id))
                {
                    summary.total_bytes = summary.total_bytes.saturating_sub(alloc.size_bytes);
                    summary.allocation_count = summary.allocation_count.saturating_sub(1);
                }

                // Update GPU-level usage
                if let Some(gpu_usage) = self.gpu_usage.get_mut(&gpu_id) {
                    gpu_usage.tenant_used_bytes = gpu_usage.tenant_used_bytes
                        .saturating_sub(alloc.size_bytes);
                    gpu_usage.available_bytes = gpu_usage.total_bytes
                        .saturating_sub(gpu_usage.tenant_used_bytes + gpu_usage.external_used_bytes);
                    if let Some(tenant_usage) = gpu_usage.per_tenant.get_mut(&alloc.tenant_id) {
                        *tenant_usage = tenant_usage.saturating_sub(alloc.size_bytes);
                    }
                }
            }
        }
    }

    /// Check if a proposed allocation would violate VRAM isolation rules.
    fn check_vram_limit(
        &self,
        gpu_id: GpuId,
        tenant_id: TenantId,
        size_bytes: u64,
        quota: &TenantQuota,
    ) -> Result<(), QuotaViolation> {
        // Per-GPU per-tenant limit
        if quota.max_vram_per_gpu_bytes > 0 {
            let current = self.gpu_tenant_usage
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

        // Check physical availability (don't over-commit VRAM)
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

    /// Get VRAM usage breakdown for a specific GPU (for admin/monitoring).
    fn gpu_breakdown(&self, gpu_id: GpuId) -> Option<&GpuVramUsage> {
        self.gpu_usage.get(&gpu_id)
    }
}
```

---

## 5. Integration Points

### 5.1 R17 (Topology-Aware Scheduling) Integration

R24 injects into R17's `PlacementDecision` pipeline as a filter and weight modifier.

```rust
/// R24 provides a TenantAwareFilter that R17 calls before scoring GPUs.
///
/// R17's current scoring (from R17 v2 preplan):
///   total_score = 0.35 * locality_score
///               + 0.25 * network_score
///               + 0.25 * load_score
///               + 0.15 * capability_score  (from R23)
///
/// With R24, the pipeline becomes:
///   1. [R24 Filter] Remove GPUs where tenant would exceed VRAM quota
///   2. [R24 Filter] Remove GPUs reserved by other tenants (exclusive mode)
///   3. [R23 Filter] Remove GPUs with incompatible CC/binary
///   4. [R17 Score]  Score remaining GPUs with the existing formula
///   5. [R24 Adjust] Multiply score by tenant affinity bonus for GPUs the
///      tenant already has allocations on (data locality within tenant)
///   6. [R24 Adjust] Penalize GPUs where adding this tenant would exceed
///      a soft co-tenancy limit (e.g., max 3 tenants per GPU)
///
/// The filter+adjust approach means R17's core scoring is unchanged.
/// R24 only narrows the candidate set and applies minor adjustments.
trait TenantPlacementFilter {
    /// Filter GPU candidates based on tenant quotas and isolation rules.
    /// Returns the subset of candidate GPUs that this tenant can use.
    fn filter_for_tenant(
        &self,
        tenant_id: TenantId,
        candidates: &[GpuId],
    ) -> Vec<GpuId>;

    /// Adjust the placement score for a GPU based on tenant co-tenancy.
    /// Returns a multiplier (0.0 - 2.0) applied to R17's computed score.
    fn tenancy_score_adjustment(
        &self,
        tenant_id: TenantId,
        gpu_id: GpuId,
    ) -> f64;

    /// Get the maximum number of tenants that should share a single GPU.
    /// Above this, the scheduler prefers other GPUs.
    fn max_co_tenants_per_gpu(&self) -> u32;
}

/// Implementation sketch:
///
/// filter_for_tenant:
///   1. Check tenant VRAM quota against GPU's available VRAM.
///   2. Check if GPU is exclusively reserved by another tenant.
///   3. Check if tenant is at max_gpus.
///   Return only GPUs that pass all checks.
///
/// tenancy_score_adjustment:
///   - If tenant already has allocations on this GPU: 1.5x (data locality bonus)
///   - If GPU has >= max_co_tenants other tenants: 0.5x (co-tenancy penalty)
///   - If GPU has 0 other tenants: 1.2x (isolation bonus)
///   - Otherwise: 1.0x (neutral)
```

### 5.2 R13 (CUDA Graphs) Integration

R24 adds tenant identity to the CUDA graph execution pipeline.

```rust
/// When R13 intercepts cuGraphLaunch, R24 wraps the graph execution:
///
/// 1. Identify the tenant from the session that created the graph.
/// 2. Check tenant's time slice:
///    - If the tenant has an active time slice on the target GPU(s),
///      proceed with graph launch.
///    - If the tenant's time slice has expired, queue the graph launch
///      as a PendingKernel (the graph is treated as a single "mega-kernel"
///      for scheduling purposes).
/// 3. For partitioned graphs (HEFT split across GPUs):
///    - Each partition is tracked as a separate resource claim.
///    - GEU cost = sum of (partition_duration_estimate * gpu_geu) for each GPU.
///    - The scheduler may assign different partitions to different time slices
///      if the graph spans multiple GPUs.
/// 4. Graph completion callback updates the tenant's usage accounting.
///
/// R13 already tracks per-kernel execution time. R24 annotates these
/// measurements with tenant_id so they contribute to the correct tenant's
/// GEU-hours accounting.
///
/// Key integration point: R13's ShadowGraph gets a tenant_id field.
/// R13's GraphPartition gets a tenant_id field.
///
/// No changes needed to R13's core HEFT algorithm -- it operates on GPUs
/// without caring about tenants. R24 controls WHEN a tenant's graph is
/// submitted, not HOW it's partitioned.
```

### 5.3 R22 (Live Migration) Integration

R24 uses R22's migration capability for tenant rebalancing.

```rust
/// Rebalancing scenarios where R24 triggers R22 migration:
///
/// Scenario 1: New tenant joins with high priority
///   - Current: GPU 0 has 20GB of tenant A's data.
///   - Event: Tenant B (Production tier) joins and needs GPU 0.
///   - Action: R24 asks R22 to migrate tenant A's allocations to GPU 1
///     (if tenant A allows migration and GPU 1 has space).
///   - Result: GPU 0 freed for tenant B. Tenant A continues on GPU 1.
///
/// Scenario 2: Quota reduction
///   - Current: Tenant A uses 4 GPUs.
///   - Event: Admin reduces tenant A's max_gpus to 2.
///   - Action: R24 identifies the 2 least-utilized GPUs for tenant A,
///     migrates their data to the 2 remaining GPUs (via R22), then
///     releases the freed GPUs.
///
/// Scenario 3: Load balancing
///   - Current: Tenant A has 16GB on GPU 0, Tenant B has 16GB on GPU 0.
///     GPU 1 is idle.
///   - Action: R24 asks R22 to migrate tenant B to GPU 1.
///   - Result: Each tenant gets an exclusive GPU (better isolation).
///
/// R24 triggers migration via:
trait TenantMigrationRequester {
    /// Request migration of a tenant's allocations from one GPU to another.
    /// Returns the migration handle for tracking progress.
    fn request_migration(
        &self,
        tenant_id: TenantId,
        session_id: SessionId,
        source_gpu: GpuId,
        target_gpu: GpuId,
    ) -> MigrationHandle;

    /// Check if migration is still in progress.
    fn migration_status(&self, handle: MigrationHandle) -> MigrationStatus;
}

type MigrationHandle = u64;

#[derive(Debug, Clone, Copy)]
enum MigrationStatus {
    InProgress { bytes_migrated: u64, bytes_total: u64 },
    Completed { duration: Duration },
    Failed { reason: String },
}

/// Migration constraints from R24:
/// - Only migrate tenants who have allow_migration = true.
/// - Suspend the tenant's kernel dispatch during migration.
/// - After migration, update all virtual address mappings in the tenant's
///   session so the application sees no change.
/// - Migration priority: Background > Development > Production (least-disruptive first).
```

### 5.4 R23 (GPU Mixing) Integration

R24 is a heavy consumer of R23's GEU system.

```rust
/// R24 uses R23's GpuProfile and GEU for ALL fairness computations.
///
/// Key consumption points:
///
/// 1. GEU as fairness currency:
///    - DRF computes shares in GEU-seconds, not raw seconds.
///    - A tenant using 1 second on an RTX 4090 (GEU 4.04) consumes
///      4.04 GEU-seconds, equivalent to 4.04 seconds on an RTX 3060.
///    - This prevents gaming: a tenant cannot get more by requesting
///      the weakest GPU (it costs less GEU but also delivers less work).
///
/// 2. VRAM normalization:
///    - 24 GB on an RTX 3090 is "different" from 24 GB on an RTX 4090
///      (different bandwidth, different effective throughput for paging).
///    - R24 uses raw bytes for VRAM quota enforcement (a byte is a byte)
///      but uses GEU-weighted VRAM for DRF share computation.
///
/// 3. Quota presets reference GEU:
///    - "Light" quota = 2 GEU means roughly one RTX 3090 (GEU 2.50) or
///      two RTX 3060s (GEU 1.0 each).
///
/// 4. Billing in GEU-hours:
///    - billing_record.amount = sum over all kernels of
///      (kernel_execution_ns * gpu_geu / 1e9 / 3600).
///    - This makes billing GPU-model-agnostic.
///
/// R24 accesses R23 via the existing GpuCapabilityProvider trait:
///   - geu(gpu_id) -> f64: get GEU for a GPU
///   - all_profiles() -> &[GpuProfile]: iterate pool GPUs
///   - capability_score(gpu_id, workload_class) -> f64: for cost estimation
```

---

## 6. Authentication and Authorization Model

### 6.1 Authentication Flow

```
Client startup (LD_PRELOAD init):
  |
  v
[1. Read auth config from environment]
  |  OUTERLINK_AUTH_TOKEN or OUTERLINK_CERT_PATH
  |  If neither set and pool is Open mode: use OS username as tenant name
  |
  v
[2. TCP/TLS handshake with coordinator]
  |  If AuthMode::MutualTls: present client certificate
  |  If AuthMode::ApiKey: send token in handshake message
  |  If AuthMode::Open: send OS username
  |
  v
[3. Coordinator validates credentials]
  |  ApiKey: argon2_verify(token, stored_hash)
  |  MutualTls: check certificate fingerprint against registered tenants
  |  Open: lookup or create tenant by username
  |
  v
[4. Coordinator returns session grant]
  |  Contains: tenant_id, session_id, assigned quota, pool capabilities
  |  Session grant is signed with coordinator's key (HMAC-SHA256)
  |
  v
[5. Client stores session grant]
  |  Included in every subsequent request as proof of authentication
  |  Validated server-side on first use, then cached
  |
  v
[6. Session active]
     All CUDA API calls tagged with (tenant_id, session_id)
     Coordinator tracks per-session resource usage
```

### 6.2 Authorization Matrix

```rust
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

/// Permission matrix:
///
/// | Action                        | Admin | Operator | User |
/// |-------------------------------|-------|----------|------|
/// | Submit GPU work               | Y     | Y        | Y    |
/// | View own usage                | Y     | Y        | Y    |
/// | View all tenant usage         | Y     | Y        | N    |
/// | Create tenant                 | Y     | Y        | N    |
/// | Modify tenant quota           | Y     | Y        | N    |
/// | Delete tenant                 | Y     | N         | N    |
/// | Change pool scheduling policy | Y     | N         | N    |
/// | Modify pool config            | Y     | N         | N    |
/// | Force preempt any tenant      | Y     | N         | N    |
/// | View billing records          | Y     | Y        | own  |

/// Session grant issued by the coordinator after authentication.
#[derive(Debug, Clone)]
pub struct SessionGrant {
    pub tenant_id: TenantId,
    pub session_id: SessionId,
    pub role: TenantRole,
    pub quota_snapshot: TenantQuota,
    pub issued_at: SystemTime,
    pub expires_at: SystemTime,
    /// HMAC-SHA256 of the above fields, keyed with coordinator's secret.
    pub signature: [u8; 32],
}
```

### 6.3 Environment Variables

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `OUTERLINK_AUTH_TOKEN` | string | None | API key for authentication |
| `OUTERLINK_CERT_PATH` | path | None | Path to client certificate for mTLS |
| `OUTERLINK_TENANT` | string | OS username | Override tenant name (Admin/Operator only) |

---

## 7. Usage Accounting and Billing Metrics

### 7.1 Billing Accumulator

```rust
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

    /// Total GEU-hours consumed (the primary billing unit).
    pub geu_hours: f64,
    /// Total wall-clock hours of GPU occupancy (regardless of utilization).
    pub gpu_occupancy_hours: f64,
    /// Total VRAM-hours (vram_bytes * hours / 1e9, in GB-hours).
    pub vram_gb_hours: f64,
    /// Total data transferred cross-node, in bytes.
    pub transfer_bytes: u64,

    // --- Breakdown metrics ---

    /// Per-GPU usage breakdown.
    pub per_gpu: HashMap<GpuId, GpuBillingRecord>,
    /// Per-session usage breakdown.
    pub per_session: HashMap<SessionId, SessionBillingRecord>,

    // --- Quality metrics ---

    /// Number of times the tenant was preempted.
    pub preemption_count: u32,
    /// Total time spent waiting for scheduling (queue delay).
    pub total_queue_delay: Duration,
    /// Number of quota violations (blocked requests).
    pub quota_violations: u32,
    /// Peak simultaneous GEU consumption.
    pub peak_geu: f64,
    /// Peak VRAM usage in bytes.
    pub peak_vram_bytes: u64,
}

/// Per-GPU billing breakdown within a period.
#[derive(Debug, Clone, Default)]
pub struct GpuBillingRecord {
    /// GPU model name (for the billing report).
    pub gpu_name: String,
    /// GEU rating of this GPU.
    pub gpu_geu: f64,
    /// Wall-clock seconds this tenant occupied this GPU.
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
    /// GEU-hours consumed by this session.
    pub geu_hours: f64,
    /// Kernels launched.
    pub kernel_count: u64,
    /// Session duration (wall clock).
    pub duration: Duration,
}

/// Accumulator for the current (incomplete) billing period.
#[derive(Debug, Clone)]
pub struct BillingPeriodAccumulator {
    /// Period start.
    pub period_start: SystemTime,
    /// Running totals.
    pub geu_seconds: f64,
    pub vram_byte_seconds: f64,
    pub transfer_bytes: u64,
    pub kernel_count: u64,
    pub preemption_count: u32,
    pub queue_delay_ns: u64,
    pub quota_violations: u32,
    pub peak_geu: f64,
    pub peak_vram_bytes: u64,
    /// Per-GPU accumulators.
    pub per_gpu: HashMap<GpuId, GpuBillingRecord>,
    /// Per-session accumulators.
    pub per_session: HashMap<SessionId, SessionBillingRecord>,
}
```

### 7.2 Accounting Events

```rust
/// Events that trigger accounting updates.
/// Each event is processed atomically by the billing system.
#[derive(Debug, Clone)]
pub enum AccountingEvent {
    /// A kernel completed execution.
    KernelCompleted {
        tenant_id: TenantId,
        session_id: SessionId,
        gpu_id: GpuId,
        execution_ns: u64,
        gpu_geu: f64,
    },
    /// VRAM was allocated.
    VramAllocated {
        tenant_id: TenantId,
        session_id: SessionId,
        gpu_id: GpuId,
        size_bytes: u64,
    },
    /// VRAM was freed.
    VramFreed {
        tenant_id: TenantId,
        session_id: SessionId,
        gpu_id: GpuId,
        size_bytes: u64,
    },
    /// Cross-node data transfer completed.
    TransferCompleted {
        tenant_id: TenantId,
        session_id: SessionId,
        bytes: u64,
    },
    /// Tenant was preempted.
    Preempted {
        tenant_id: TenantId,
        session_id: SessionId,
        gpu_id: GpuId,
        reason: PreemptionReason,
    },
    /// Kernel dispatch was delayed by scheduling.
    QueueDelay {
        tenant_id: TenantId,
        session_id: SessionId,
        delay_ns: u64,
    },
    /// Quota violation occurred (request was blocked).
    QuotaViolation {
        tenant_id: TenantId,
        session_id: SessionId,
        violation: QuotaViolation,
    },
}

/// Process an accounting event into the billing accumulator.
fn process_accounting_event(
    accumulator: &mut BillingPeriodAccumulator,
    event: AccountingEvent,
) {
    match event {
        AccountingEvent::KernelCompleted {
            gpu_id, execution_ns, gpu_geu, session_id, ..
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
        AccountingEvent::VramAllocated { size_bytes, .. } => {
            // VRAM-hours are computed at period close from sampling
            // (vram_byte_seconds accumulated every 1s by the monitor)
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
        AccountingEvent::QuotaViolation { .. } => {
            accumulator.quota_violations += 1;
        }
        _ => {}
    }
}
```

### 7.3 Telemetry Export

```rust
/// Telemetry channels exposed by R24 for monitoring and billing systems.
///
/// Channel 1: Real-time tenant metrics (sampled every 1 second)
///   - Per-tenant: current_geu, vram_bytes, inflight_kernels, pending_kernels
///   - Per-GPU: tenant_breakdown (which tenants, how much VRAM each)
///   - Pool-level: total_tenants, total_geu_used, scheduling_policy
///   - Format: Prometheus-compatible gauge metrics.
///
/// Channel 2: Accounting events (event-driven, per kernel/alloc/transfer)
///   - AccountingEvent stream (see above)
///   - Written to an append-only log file for audit trail.
///   - Format: JSON lines (one event per line).
///
/// Channel 3: Billing summaries (per billing period)
///   - BillingPeriod records (see above)
///   - Written to a structured output (JSON or CSV).
///   - Can be consumed by external billing systems.

/// Prometheus-style metric names:
///
/// outerlink_tenant_geu_current{tenant="alice"}           -- current GEU
/// outerlink_tenant_vram_bytes{tenant="alice",gpu="0"}    -- VRAM per GPU
/// outerlink_tenant_kernels_total{tenant="alice"}         -- cumulative kernels
/// outerlink_tenant_geu_hours_total{tenant="alice"}       -- cumulative GEU-hours
/// outerlink_tenant_preemptions_total{tenant="alice"}     -- cumulative preemptions
/// outerlink_tenant_queue_delay_seconds{tenant="alice"}   -- cumulative queue delay
/// outerlink_gpu_tenants{gpu="0"}                         -- number of tenants on GPU
/// outerlink_pool_total_geu                               -- pool total GEU
/// outerlink_pool_used_geu                                -- pool used GEU
/// outerlink_pool_tenant_count                            -- active tenant count
```

---

## 8. Configuration

### 8.1 Pool Configuration File

```toml
# outerlink-pool.toml -- sharing configuration

[sharing]
enabled = true
policy = "drf"                    # drf | weighted_fair | strict_priority | fifo | reserved
auth_mode = "open"                # open | api_key | mutual_tls
enforce_quotas = true
enable_accounting = true
max_tenants = 64
min_share_percent = 0.05          # 5% minimum for lowest tier

[sharing.drf]
compute_weight = 0.50
vram_weight = 0.35
network_weight = 0.15

[sharing.time_slice]
default_duration_ms = 100
min_duration_ms = 10
max_duration_ms = 5000
grace_period_ms = 50
scheduling_horizon_ms = 10000
reschedule_interval_ms = 100
work_conserving = true
coalesce_consecutive = true

[sharing.defaults]
quota_preset = "standard"         # unlimited | light | standard | heavy

# Tenant definitions (can also be managed via CLI)
[[tenant]]
name = "alice"
priority = "development"
quota_preset = "standard"

[[tenant]]
name = "bob"
priority = "production"
quota_preset = "heavy"

[[tenant]]
name = "ml-team"
priority = "production"
quota_preset = "heavy"
max_gpus = 4                     # override preset
max_vram_gb = 96                 # override preset
exclusive_gpu = true

# Reserved GPU assignments (only for policy = "reserved")
# [sharing.reserved]
# alice = [0, 1]
# bob = [2, 3]
```

### 8.2 Client Environment Variables

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `OUTERLINK_AUTH_TOKEN` | string | None | API key for tenant authentication |
| `OUTERLINK_CERT_PATH` | path | None | Client TLS certificate path |
| `OUTERLINK_TENANT` | string | OS user | Override tenant identity |
| `OUTERLINK_PRIORITY` | string | None | Override priority tier (requires Admin) |
| `OUTERLINK_MAX_PENDING` | u32 | quota | Override max pending kernels |
| `OUTERLINK_EXCLUSIVE_GPU` | bool | false | Request exclusive GPU access |

### 8.3 CLI Management Commands

```
outerlink tenant list                    -- list all tenants
outerlink tenant add <name> [--priority <tier>] [--quota <preset>]
outerlink tenant remove <name>
outerlink tenant quota <name> [--max-geu <val>] [--max-vram <val>] ...
outerlink tenant usage <name>            -- show current usage
outerlink tenant billing <name> [--period <date>]  -- show billing record

outerlink pool sharing status            -- show scheduling policy, active tenants
outerlink pool sharing policy <policy>   -- change scheduling policy
outerlink pool sharing config            -- show sharing config

outerlink gpu tenants                    -- show per-GPU tenant breakdown
outerlink gpu preempt <gpu> <tenant>     -- force preempt a tenant (Admin only)
```

---

## 9. Tenant Lifecycle

```
Admin creates tenant (CLI or config file):
  |
  v
[1. Tenant record created in coordinator]
  |  Assigned: tenant_id, quota, priority, auth credentials
  |
  v
[2. Client application starts with LD_PRELOAD]
  |  Reads OUTERLINK_AUTH_TOKEN or OUTERLINK_CERT_PATH
  |
  v
[3. Client authenticates with coordinator]
  |  Coordinator validates credentials -> maps to tenant_id
  |  Creates Session record
  |  Returns SessionGrant with quota snapshot
  |
  v
[4. Client creates CUDA context]
  |  Intercepted by OuterLink
  |  Session state -> Active
  |  Quota check: max_sessions
  |
  v
[5. Client allocates VRAM]
  |  Each cuMemAlloc checked against tenant's VRAM quota
  |  VramLedger updated
  |
  v
[6. Client launches kernels]
  |  Each cuLaunchKernel goes through:
  |    a. Quota check (pending queue, rate limit)
  |    b. Added to session's PendingKernel queue
  |    c. DRF scheduler assigns to GPU time slice
  |    d. Dispatched to real GPU during tenant's time slice
  |    e. Completion event -> accounting update
  |
  v
[7. Client exits or disconnects]
  |  Session state -> Terminated
  |  All VRAM allocations freed (recorded in ledger)
  |  Final accounting event flushed
  |  Session removed from tenant's active_sessions
  |
  v
[8. Admin removes tenant (optional)]
     All sessions terminated
     Final billing period closed
     Tenant record retained for audit (marked disabled)
```

---

## 10. Scheduling Execution Flow (Per-GPU)

This is the runtime hot path -- how kernel dispatching actually works with time-slicing.

```
Every reschedule_interval (100ms):
  |
  v
[1. DRF scheduler computes dominant shares for all tenants]
  |
  v
[2. For each GPU, produce time slice assignments]
  |  Result: ordered list of (tenant, session, duration) per GPU
  |
  v
[3. Time slice executor runs on each GPU's dispatch thread]
  |
  v
[4. Executor loop for current GPU]:
  |
  |  a. Dequeue next TimeSlice from schedule
  |  b. Mark TimeSlice as Active
  |  c. Set current_tenant for this GPU
  |  d. WHILE time remaining in slice AND tenant has pending work:
  |     i.   Dequeue PendingKernel from tenant's session
  |     ii.  Call R17 PlacementDecision (already resolved to this GPU)
  |     iii. Submit cuLaunchKernel to real GPU
  |     iv.  Record CUDA event after kernel for timing
  |     v.   Decrement time remaining (estimated from kernel profiling)
  |  e. IF tenant has no more pending work AND work_conserving:
  |     i.   Check if next tenant in schedule has pending work
  |     ii.  If yes: yield early, advance to next time slice
  |     iii. If no: extend current slice (current tenant keeps running)
  |  f. Mark TimeSlice as Completed
  |  g. Process kernel completion events:
  |     - Update tenant's ResourceUsage
  |     - Emit AccountingEvent::KernelCompleted
  |     - Update BillingPeriodAccumulator
  |
  v
[5. IF preemption requested (higher-priority tenant arrived)]:
     a. Stop dispatching from current slice
     b. Wait for in-flight kernels (grace period)
     c. Mark current slice as Preempted
     d. Jump to preemptor's time slice
```

---

## 11. Test Plan

### 11.1 Unit Tests

| Test | Validates |
|---|---|
| `test_drf_two_equal_tenants` | Two tenants with same priority get equal GEU-hours over 100 scheduling rounds |
| `test_drf_weighted_priority` | Production tenant (2x weight) gets 2x GEU-hours vs Development tenant |
| `test_drf_three_resources` | Tenant A (compute-heavy) and Tenant B (VRAM-heavy) get correct dominant shares |
| `test_drf_strategy_proofness` | Tenant cannot get more resources by declaring a different workload class |
| `test_quota_vram_enforcement` | cuMemAlloc returns CUDA_ERROR_OUT_OF_MEMORY when tenant exceeds max_vram |
| `test_quota_per_gpu_vram` | Per-GPU VRAM limit enforced independently of total VRAM limit |
| `test_quota_geu_limit` | Tenant at max_geu cannot acquire additional GPUs |
| `test_quota_rate_limit` | Tenant exceeding geu_hours_per_period has kernel launches blocked |
| `test_quota_burst_allowance` | Tenant can temporarily exceed max_geu by burst_multiplier when pool is idle |
| `test_quota_burst_timeout` | Burst mode expires after max_burst_duration and enforcement resumes |
| `test_quota_session_limit` | cuCtxCreate fails when tenant hits max_sessions |
| `test_quota_pending_queue` | Kernel launch blocked when pending queue reaches max_pending_kernels |
| `test_priority_preemption` | Production tenant preempts Background tenant within grace period |
| `test_no_preemption_same_tier` | Tenants at same priority tier do not preempt each other |
| `test_non_preemptible_tenant` | Tenant with preemptible=false is never preempted |
| `test_preemption_grace_period` | In-flight kernels complete before preemptor starts |
| `test_preemption_forced_timeout` | Forced preemption after grace period expires |
| `test_vram_ledger_alloc_free` | Allocation tracking correctly adjusts per-tenant totals |
| `test_vram_ledger_session_cleanup` | Session termination frees all of its allocations |
| `test_time_slice_duration_weighting` | Priority weight correctly scales slice duration |
| `test_work_conserving` | Idle tenant's slice is given to waiting tenant |
| `test_starvation_prevention` | Background tenant gets min_share_percent even under heavy Production load |
| `test_billing_accumulator` | GEU-hours computed correctly across heterogeneous GPUs |
| `test_billing_period_rollover` | Period rollover preserves cumulative data and resets counters |
| `test_auth_api_key` | Valid API key authenticates, invalid key rejected |
| `test_auth_open_mode` | OS username maps to correct tenant in open mode |
| `test_quota_preset_values` | Light/Standard/Heavy presets produce expected quota values |

### 11.2 Integration Tests

| Test | Validates |
|---|---|
| `test_two_tenants_fair_sharing` | Two clients with same priority get roughly equal GPU time (within 10%) over a 60-second workload |
| `test_priority_preemption_e2e` | Production client starts, gets GPU immediately, Background client waits. When Production stops, Background resumes. |
| `test_vram_isolation_e2e` | Tenant A allocates near its VRAM limit, Tenant B's allocation on same GPU succeeds within B's quota but fails if it would exceed B's quota |
| `test_quota_exhaustion_e2e` | Tenant exceeds GEU-hours limit, subsequent kernel launches are delayed until next billing period or admin intervention |
| `test_r17_placement_with_tenants` | R17 PlacementDecision respects tenant VRAM filter and co-tenancy preferences |
| `test_r13_graph_with_tenants` | CUDA graph launch is deferred until tenant's time slice on the target GPU |
| `test_r22_migration_rebalance` | Tenant migrated to different GPU when admin reduces their max_gpus |
| `test_session_disconnect_cleanup` | Client crash: session cleaned up, VRAM freed, accounting finalized within 10 seconds |
| `test_hot_add_tenant` | New tenant added while pool is under load; receives fair share within 2 scheduling rounds |
| `test_billing_report_accuracy` | Billing report GEU-hours matches sum of individual kernel accounting events (within 1%) |
| `test_concurrent_tenants_stress` | 8 tenants with mixed priorities, 100 sessions total, sustained for 300 seconds. No crashes, no resource leaks, billing balanced. |

### 11.3 Performance Benchmarks

| Benchmark | Target |
|---|---|
| DRF scheduling round (64 tenants, 8 GPUs) | < 0.5 ms |
| Quota check (single allocation) | < 0.005 ms (5 us) |
| Time slice transition (tenant A -> tenant B) | < 1 ms (excluding kernel drain) |
| Preemption latency (request to GPU available) | < 150 ms (50ms grace + drain) |
| Accounting event processing | < 0.01 ms per event |
| VRAM ledger update (alloc/free) | < 0.005 ms |
| Session creation (auth + grant) | < 5 ms |
| Billing period close (1000 events) | < 10 ms |
| Memory overhead per tenant | < 16 KB |
| Memory overhead per session | < 64 KB |
| Memory overhead per tracked allocation | < 128 bytes |

---

## 12. Open Questions

### Q1: VRAM overcommit policy

When total tenant VRAM quotas exceed physical VRAM, should R24 allow overcommit (relying on R10 memory tiering to page to host RAM) or should it strictly enforce physical VRAM limits?

**Arguments for overcommit:** Higher utilization. Most tenants do not use 100% of their VRAM allocation simultaneously. R10 already supports transparent paging.

**Arguments against overcommit:** Performance unpredictability. A tenant whose allocations get paged to host RAM will see 10-100x slower access. This violates the implicit performance SLA.

**Proposed resolution:** Configurable per-pool. Default: no overcommit (physical VRAM limits enforced). Optional: overcommit with a configurable ratio (e.g., 1.5x physical VRAM) and a warning to tenants when their data is paged.

### Q2: Context switching between tenants on the same GPU

OuterLink's cooperative preemption avoids GPU-level context switching, but multiple tenants on the same GPU still have separate CUDA contexts. Should R24 use MPS (where available) to merge contexts, or keep separate contexts?

**MPS advantages:** Concurrent kernel execution from different tenants on the same GPU. Higher SM utilization. Lower context switch overhead.

**MPS disadvantages:** No memory isolation. Error in one tenant's kernel can crash the MPS server, affecting all tenants. Adds a daemon dependency. MPS memory limits (CUDA_MPS_PINNED_DEVICE_MEM_LIMIT) are coarse.

**Proposed resolution:** Phase 1: separate contexts with time-slicing (simpler, safer). Phase 2: optional MPS mode for trusted multi-tenant environments (configurable via pool config). MPS mode would use CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for SM partitioning and time-slicing would only control the outer scheduling loop.

### Q3: Granularity of time slicing

Should time slicing operate at:
- **Per-kernel:** each kernel dispatch checks the schedule. Finest granularity but highest overhead.
- **Per-batch:** kernels are dispatched in batches (e.g., 10-100 at a time). The schedule is checked between batches.
- **Per-timeslice:** the standard approach -- all of a tenant's kernels during their time window.

**Proposed resolution:** Per-timeslice (100ms default) with per-batch fallback for long-running kernels. If a single kernel is estimated to exceed the time slice duration, it is dispatched anyway (cannot split a kernel) but the slice is extended only for that kernel.

### Q4: Tenant-to-tenant data sharing

Can two tenants share the same CUDA allocation (e.g., a shared model for inference)? This would be efficient for serving scenarios where multiple users run inference on the same model.

**Proposed resolution:** Deferred to v2. Phase 1 assumes all allocations are tenant-private. Phase 2 could introduce "shared allocations" owned by a special system tenant and read-accessible to specified tenants. This requires careful handling in the VRAM ledger (refcounted shared allocations).

### Q5: Billing granularity and export format

What level of detail should billing records provide, and in what format? Options:
- Per-kernel billing (finest, most data)
- Per-session billing (moderate)
- Per-period per-tenant billing (coarsest)

**Proposed resolution:** All three levels available. Per-kernel events are logged to the accounting event stream (for audit). Per-session summaries are computed at session end. Per-period summaries are computed at period rollover. Export formats: JSON (primary), CSV (for spreadsheets), Prometheus metrics (for real-time monitoring).

### Q6: Interaction with NVIDIA MPS on systems that have it enabled

If a system already has MPS running (e.g., user enabled it for another purpose), how should R24 behave? Should it detect MPS and change behavior?

**Proposed resolution:** R24 should detect MPS at node registration (check if nvidia-cuda-mps-control is running). If MPS is active, R24 logs a warning and operates in "MPS-aware mode" where time-slicing still controls the scheduling loop but context switching is handled by MPS. If MPS is not active, R24 uses its own cooperative time-slicing.

### Q7: Fair sharing when tenants have very different workload patterns

A batch training job (long-running, GPU-bound) and an inference service (short bursts, latency-sensitive) have fundamentally different needs. DRF treats them identically by dominant share. Should R24 distinguish between workload types for better QoS?

**Proposed resolution:** R24 supports this via priority tiers (inference = Production/Critical, training = Development/Background). Additionally, the `max_dispatch_latency_us` field in SchedulingPreferences lets inference tenants declare latency sensitivity. The scheduler can prioritize these tenants for immediate dispatch rather than strict DRF ordering, at the cost of slightly unfair compute allocation.

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Time-slicing overhead kills throughput for single-tenant use | Medium | High | Work-conserving mode: if only one tenant, they get 100% with zero overhead. Slice coalescing eliminates transitions. |
| VRAM fragmentation across tenants | Medium | Medium | Per-GPU per-tenant VRAM limits prevent any single tenant from fragmenting a GPU. R10's defragmentation can consolidate. |
| DRF scheduling round too slow for many tenants | Low | Medium | O(T * G) per round where T=tenants, G=GPUs. With 64 tenants and 8 GPUs: 512 iterations per round. Target <0.5ms. |
| Cooperative preemption too slow for latency-sensitive workloads | Medium | Medium | Grace period is configurable (default 50ms). For inference serving, set to 10ms. In-flight kernels are typically <5ms for inference. |
| Tenant gaming via many small sessions | Low | Low | max_sessions quota prevents session flooding. Pending queue limits prevent kernel flooding. |
| Clock drift between nodes affects billing accuracy | Low | Low | Billing uses local timestamps. Coordinator NTP sync recommended but not required. GEU-hours are computed per-GPU. |
| Memory leak in VRAM ledger if client crashes without freeing | Medium | Medium | Session cleanup (on disconnect) walks the ledger and frees all allocations owned by the terminated session. |

---

## 14. Estimated Effort

| Component | Complexity | Estimate |
|---|---|---|
| Tenant/Session data structures | Low | 2-3 days |
| Quota system (structs + enforcement) | Medium | 3-4 days |
| DRF scheduler | High | 5-7 days |
| Time slice executor (dispatch thread) | High | 5-7 days |
| Preemption engine | Medium | 3-4 days |
| VRAM ledger and isolation | Medium | 3-4 days |
| Authentication (API key + open mode) | Medium | 2-3 days |
| Authorization matrix | Low | 1-2 days |
| Billing accumulator + accounting events | Medium | 3-4 days |
| Telemetry export (Prometheus metrics) | Medium | 2-3 days |
| R17 integration (TenantPlacementFilter) | Medium | 2-3 days |
| R13 integration (tenant-tagged graphs) | Low | 1-2 days |
| R22 integration (migration for rebalancing) | Medium | 2-3 days |
| R23 integration (GEU consumption) | Low | 1-2 days |
| CLI management commands | Low | 2-3 days |
| Configuration file parsing | Low | 1-2 days |
| Unit tests | High | 4-5 days |
| Integration tests | High | 5-7 days |
| Performance benchmarks | Medium | 2-3 days |
| **Total** | | **48-71 days** |

---

## 15. Acceptance Criteria

1. Two tenants with equal priority and unlimited quotas receive GPU time within 10% of each other over a 60-second measurement window.
2. A Production-tier tenant preempts a Background-tier tenant and begins executing within 200ms of the preemption request.
3. A tenant at their VRAM quota sees `CUDA_ERROR_OUT_OF_MEMORY` for additional allocations, while other tenants on the same GPU can still allocate within their quotas.
4. GEU-hours in billing records match the sum of individual kernel accounting events within 1% accuracy.
5. A single-tenant pool has zero measurable overhead from the sharing system (work-conserving mode eliminates transitions).
6. The DRF scheduler completes a scheduling round for 64 tenants across 8 GPUs in under 0.5ms.
7. Session cleanup after client disconnect frees all VRAM and finalizes accounting within 10 seconds.
8. Quota enforcement latency (per allocation/launch check) is under 5 microseconds.
9. All quota presets (Light/Standard/Heavy/Unlimited) produce correct enforcement behavior.
10. Billing period rollover is seamless with no lost accounting events.

---

## Related Documents

- [README.md](./README.md) -- R24 overview and key questions
- [progress.md](./progress.md) -- Lifecycle tracker
- [R17 Topology-Aware Scheduling](../../phase-08-network-optimization/R17-topology-scheduling/) -- PlacementDecision integration
- [R13 CUDA Graph Interception](../../phase-10-compute-distribution/R13-cuda-graph-interception/) -- Graph-level scheduling
- [R22 Live Migration](../../phase-10-compute-distribution/R22-live-migration/) -- Tenant rebalancing
- [R23 Heterogeneous GPU Mixing](../../phase-10-compute-distribution/R23-heterogeneous-gpu-mixing/) -- GEU, GpuProfile, CapabilityScorer
- [R15 Fault Tolerance](../../phase-08-network-optimization/R15-fault-tolerance/) -- Tenant workload recovery
