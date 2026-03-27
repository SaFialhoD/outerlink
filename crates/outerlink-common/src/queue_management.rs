//! Multi-tenant queue management types.
//!
//! Priority queues and fair-share scheduling for multi-tenant GPU allocation.
//! Inspired by Run:AI and Volcano scheduling patterns, but pure types and
//! scheduling logic with no Kubernetes dependency.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Tenant
// ---------------------------------------------------------------------------

/// A tenant (team / org / user) with a GPU quota.
#[derive(Debug, Clone)]
pub struct Tenant {
    pub id: String,
    pub name: String,
    /// Priority weight: higher = more important. Default 128.
    pub priority: u8,
    /// Maximum number of GPUs this tenant may hold.
    pub gpu_quota: u32,
    /// Number of GPUs currently in use.
    pub gpu_used: u32,
    /// Whether this tenant may temporarily exceed its quota.
    pub burst_allowed: bool,
}

impl Tenant {
    /// Create a new tenant with sensible defaults.
    pub fn new(id: impl Into<String>, name: impl Into<String>, gpu_quota: u32) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            priority: 128,
            gpu_quota,
            gpu_used: 0,
            burst_allowed: true,
        }
    }

    /// GPUs remaining before hitting the quota.
    pub fn remaining_quota(&self) -> u32 {
        self.gpu_quota.saturating_sub(self.gpu_used)
    }

    /// Whether the tenant is at or over quota.
    pub fn is_over_quota(&self) -> bool {
        self.gpu_used >= self.gpu_quota
    }

    /// Current utilization as a fraction in `[0.0, ...]`.
    /// Can exceed 1.0 if burst is active.
    pub fn utilization(&self) -> f64 {
        if self.gpu_quota == 0 {
            return 0.0;
        }
        self.gpu_used as f64 / self.gpu_quota as f64
    }
}

// ---------------------------------------------------------------------------
// QueuedJob
// ---------------------------------------------------------------------------

/// A job waiting in (or running from) the scheduling queue.
#[derive(Debug, Clone)]
pub struct QueuedJob {
    pub job_id: String,
    pub tenant_id: String,
    /// Number of GPUs requested.
    pub gpu_count: u32,
    /// Scheduling priority (higher = more urgent).
    pub priority: u8,
    /// When the job was submitted.
    pub submitted_at: Instant,
    /// When the job started running (`None` if still queued).
    pub started_at: Option<Instant>,
    /// Whether this job may be preempted.
    pub preemptible: bool,
}

impl QueuedJob {
    /// Create a new queued job.
    pub fn new(
        job_id: impl Into<String>,
        tenant_id: impl Into<String>,
        gpu_count: u32,
        priority: u8,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            tenant_id: tenant_id.into(),
            gpu_count,
            priority,
            submitted_at: Instant::now(),
            started_at: None,
            preemptible: true,
        }
    }

    /// How long the job has been waiting (from submission to `now`).
    pub fn wait_duration(&self, now: Instant) -> std::time::Duration {
        now.duration_since(self.submitted_at)
    }

    /// Whether the job is currently running.
    pub fn is_running(&self) -> bool {
        self.started_at.is_some()
    }

    /// Whether this is a gang-scheduled job (needs more than 1 GPU).
    pub fn is_gang(&self) -> bool {
        self.gpu_count > 1
    }
}

// ---------------------------------------------------------------------------
// SchedulingPolicy
// ---------------------------------------------------------------------------

/// How the scheduler picks the next job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-in, first-out.
    Fifo,
    /// Highest-priority job first.
    Priority,
    /// Serve the most underserved tenant first.
    FairShare,
    /// Dominant Resource Fairness: equalize each tenant's maximum resource share.
    DominantResourceFairness,
}

impl SchedulingPolicy {
    /// Human-readable description of this policy.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Fifo => "First-in, first-out scheduling",
            Self::Priority => "Highest-priority job scheduled first",
            Self::FairShare => "Most underserved tenant scheduled first",
            Self::DominantResourceFairness => {
                "Equalize each tenant's dominant resource share"
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QueueConfig
// ---------------------------------------------------------------------------

/// Configuration for the scheduling queue.
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Maximum number of jobs allowed in the queue.
    pub max_queue_depth: u32,
    /// Whether running jobs may be preempted.
    pub preemption_enabled: bool,
    /// Whether gang scheduling (multi-GPU atomic allocation) is enabled.
    pub gang_scheduling: bool,
    /// Scheduling policy.
    pub policy: SchedulingPolicy,
    /// Seconds a job may wait before starvation prevention kicks in.
    pub starvation_timeout_secs: u64,
    /// How much a tenant may exceed its quota when bursting.
    pub burst_multiplier: f64,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 1000,
            preemption_enabled: true,
            gang_scheduling: true,
            policy: SchedulingPolicy::FairShare,
            starvation_timeout_secs: 3600,
            burst_multiplier: 1.5,
        }
    }
}

// ---------------------------------------------------------------------------
// PreemptionReason / PreemptionDecision
// ---------------------------------------------------------------------------

/// Why a running job was preempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionReason {
    /// A higher-priority job needs the resources.
    HigherPriority,
    /// The tenant's quota owner is reclaiming GPUs.
    QuotaReclaim,
    /// GPUs must be freed for a gang-scheduled job.
    GangScheduling,
    /// A job has been starving and needs resources.
    StarvationPrevention,
}

impl PreemptionReason {
    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::HigherPriority => "A higher-priority job needs the resources",
            Self::QuotaReclaim => "Quota owner is reclaiming GPUs",
            Self::GangScheduling => "GPUs freed for gang-scheduled job",
            Self::StarvationPrevention => "Starvation prevention triggered",
        }
    }
}

/// A decision to preempt a running job.
#[derive(Debug, Clone)]
pub struct PreemptionDecision {
    /// The job being evicted.
    pub victim_job_id: String,
    /// Why it was evicted.
    pub reason: PreemptionReason,
    /// The job that caused the eviction.
    pub preemptor_job_id: String,
}

// ---------------------------------------------------------------------------
// QueueStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the scheduling queue.
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub queued_jobs: u32,
    pub running_jobs: u32,
    pub preempted_jobs: u64,
    pub avg_wait_secs: f64,
    pub max_wait_secs: f64,
    pub total_gpu_hours: f64,
}

impl QueueStats {
    /// Total jobs in the system (queued + running).
    pub fn queue_depth(&self) -> u32 {
        self.queued_jobs + self.running_jobs
    }

    /// Whether the queue has reached its maximum depth.
    pub fn is_saturated(&self, config: &QueueConfig) -> bool {
        self.queue_depth() >= config.max_queue_depth
    }
}

// ---------------------------------------------------------------------------
// FairShareCalculator
// ---------------------------------------------------------------------------

/// Calculates fair GPU shares across tenants.
pub struct FairShareCalculator {
    pub tenants: Vec<Tenant>,
    pub total_gpus: u32,
}

impl FairShareCalculator {
    /// Compute each tenant's fair share as a fraction of total GPUs.
    ///
    /// Shares are proportional to `gpu_quota` and sum to 1.0.
    /// Returns `(tenant_id, share)` pairs.
    pub fn compute_shares(&self) -> Vec<(String, f64)> {
        let total_quota: u64 = self.tenants.iter().map(|t| t.gpu_quota as u64).sum();
        if total_quota == 0 {
            return self
                .tenants
                .iter()
                .map(|t| (t.id.clone(), 0.0))
                .collect();
        }
        self.tenants
            .iter()
            .map(|t| {
                let share = t.gpu_quota as f64 / total_quota as f64;
                (t.id.clone(), share)
            })
            .collect()
    }

    /// Dominant Resource Fairness: returns the tenant's dominant resource share.
    ///
    /// Currently only considers GPU share (gpu_used / total_gpus).
    /// When VRAM tracking is added, this will be `max(gpu_share, vram_share)`.
    pub fn dominant_resource_share(&self, tenant_id: &str) -> f64 {
        if self.total_gpus == 0 {
            return 0.0;
        }
        self.tenants
            .iter()
            .find(|t| t.id == tenant_id)
            .map(|t| t.gpu_used as f64 / self.total_gpus as f64)
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// select_next_job
// ---------------------------------------------------------------------------

/// Pick the next job to schedule from the queue based on the given policy.
///
/// Only considers jobs that are **not yet running**.
/// Returns the index into `queue` of the selected job, or `None` if no
/// eligible job exists.
pub fn select_next_job(
    queue: &[QueuedJob],
    policy: &SchedulingPolicy,
    tenants: &[Tenant],
    total_gpus: u32,
) -> Option<usize> {
    let pending: Vec<(usize, &QueuedJob)> = queue
        .iter()
        .enumerate()
        .filter(|(_, j)| !j.is_running())
        .collect();

    if pending.is_empty() {
        return None;
    }

    match policy {
        SchedulingPolicy::Fifo => {
            // Oldest submitted job first.
            pending
                .iter()
                .min_by_key(|(_, j)| j.submitted_at)
                .map(|(i, _)| *i)
        }
        SchedulingPolicy::Priority => {
            // Highest priority first; break ties by submission time (oldest first).
            pending
                .iter()
                .max_by(|(_, a), (_, b)| {
                    a.priority
                        .cmp(&b.priority)
                        .then_with(|| b.submitted_at.cmp(&a.submitted_at))
                })
                .map(|(i, _)| *i)
        }
        SchedulingPolicy::FairShare => {
            // Pick a job from the tenant with the lowest utilization.
            pending
                .iter()
                .min_by(|(_, a), (_, b)| {
                    let util_a = tenant_utilization(tenants, &a.tenant_id);
                    let util_b = tenant_utilization(tenants, &b.tenant_id);
                    util_a
                        .partial_cmp(&util_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.submitted_at.cmp(&b.submitted_at))
                })
                .map(|(i, _)| *i)
        }
        SchedulingPolicy::DominantResourceFairness => {
            // True DRF: pick tenant with lowest gpu_used / total_cluster_gpus.
            // This differs from FairShare which uses gpu_used / tenant_quota.
            let total = if total_gpus == 0 { 1.0 } else { total_gpus as f64 };
            pending
                .iter()
                .min_by(|(_, a), (_, b)| {
                    let drf_a = tenants.iter()
                        .find(|t| t.id == a.tenant_id)
                        .map(|t| t.gpu_used as f64 / total)
                        .unwrap_or(f64::MAX);
                    let drf_b = tenants.iter()
                        .find(|t| t.id == b.tenant_id)
                        .map(|t| t.gpu_used as f64 / total)
                        .unwrap_or(f64::MAX);
                    drf_a
                        .partial_cmp(&drf_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.submitted_at.cmp(&b.submitted_at))
                })
                .map(|(i, _)| *i)
        }
    }
}

/// Helper: find a tenant's utilization by id.
fn tenant_utilization(tenants: &[Tenant], tenant_id: &str) -> f64 {
    tenants
        .iter()
        .find(|t| t.id == tenant_id)
        .map(|t| t.utilization())
        .unwrap_or(f64::MAX)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // -- Tenant tests -------------------------------------------------------

    #[test]
    fn test_tenant_new_defaults() {
        let t = Tenant::new("t1", "Team Alpha", 8);
        assert_eq!(t.priority, 128);
        assert_eq!(t.gpu_quota, 8);
        assert_eq!(t.gpu_used, 0);
        assert!(t.burst_allowed);
    }

    #[test]
    fn test_tenant_remaining_quota() {
        let mut t = Tenant::new("t1", "A", 8);
        assert_eq!(t.remaining_quota(), 8);
        t.gpu_used = 5;
        assert_eq!(t.remaining_quota(), 3);
        t.gpu_used = 10; // over quota via burst
        assert_eq!(t.remaining_quota(), 0);
    }

    #[test]
    fn test_tenant_is_over_quota() {
        let mut t = Tenant::new("t1", "A", 4);
        assert!(!t.is_over_quota());
        t.gpu_used = 4;
        assert!(t.is_over_quota());
        t.gpu_used = 6;
        assert!(t.is_over_quota());
    }

    #[test]
    fn test_tenant_utilization() {
        let mut t = Tenant::new("t1", "A", 10);
        assert!((t.utilization() - 0.0).abs() < f64::EPSILON);
        t.gpu_used = 5;
        assert!((t.utilization() - 0.5).abs() < f64::EPSILON);
        t.gpu_used = 10;
        assert!((t.utilization() - 1.0).abs() < f64::EPSILON);
        t.gpu_used = 15; // burst
        assert!((t.utilization() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tenant_utilization_zero_quota() {
        let t = Tenant::new("t1", "A", 0);
        assert!((t.utilization() - 0.0).abs() < f64::EPSILON);
    }

    // -- QueuedJob tests ----------------------------------------------------

    #[test]
    fn test_queued_job_new_defaults() {
        let j = QueuedJob::new("j1", "t1", 2, 100);
        assert_eq!(j.gpu_count, 2);
        assert_eq!(j.priority, 100);
        assert!(j.preemptible);
        assert!(!j.is_running());
    }

    #[test]
    fn test_queued_job_wait_duration() {
        let base = Instant::now();
        let mut j = QueuedJob::new("j1", "t1", 1, 100);
        j.submitted_at = base;
        let later = base + Duration::from_secs(10);
        assert_eq!(j.wait_duration(later), Duration::from_secs(10));
    }

    #[test]
    fn test_queued_job_is_running() {
        let mut j = QueuedJob::new("j1", "t1", 1, 100);
        assert!(!j.is_running());
        j.started_at = Some(Instant::now());
        assert!(j.is_running());
    }

    #[test]
    fn test_queued_job_is_gang() {
        let j1 = QueuedJob::new("j1", "t1", 1, 100);
        assert!(!j1.is_gang());
        let j2 = QueuedJob::new("j2", "t1", 4, 100);
        assert!(j2.is_gang());
    }

    // -- SchedulingPolicy tests ---------------------------------------------

    #[test]
    fn test_scheduling_policy_descriptions() {
        assert!(!SchedulingPolicy::Fifo.description().is_empty());
        assert!(!SchedulingPolicy::Priority.description().is_empty());
        assert!(!SchedulingPolicy::FairShare.description().is_empty());
        assert!(!SchedulingPolicy::DominantResourceFairness.description().is_empty());
    }

    // -- QueueConfig tests --------------------------------------------------

    #[test]
    fn test_queue_config_defaults() {
        let c = QueueConfig::default();
        assert_eq!(c.max_queue_depth, 1000);
        assert!(c.preemption_enabled);
        assert!(c.gang_scheduling);
        assert_eq!(c.policy, SchedulingPolicy::FairShare);
        assert_eq!(c.starvation_timeout_secs, 3600);
        assert!((c.burst_multiplier - 1.5).abs() < f64::EPSILON);
    }

    // -- PreemptionReason tests ---------------------------------------------

    #[test]
    fn test_preemption_reason_descriptions() {
        assert!(!PreemptionReason::HigherPriority.description().is_empty());
        assert!(!PreemptionReason::QuotaReclaim.description().is_empty());
        assert!(!PreemptionReason::GangScheduling.description().is_empty());
        assert!(!PreemptionReason::StarvationPrevention.description().is_empty());
    }

    #[test]
    fn test_preemption_reason_equality() {
        assert_eq!(PreemptionReason::HigherPriority, PreemptionReason::HigherPriority);
        assert_ne!(PreemptionReason::HigherPriority, PreemptionReason::QuotaReclaim);
    }

    #[test]
    fn test_preemption_decision_fields() {
        let d = PreemptionDecision {
            victim_job_id: "j1".into(),
            reason: PreemptionReason::QuotaReclaim,
            preemptor_job_id: "j2".into(),
        };
        assert_eq!(d.victim_job_id, "j1");
        assert_eq!(d.reason, PreemptionReason::QuotaReclaim);
        assert_eq!(d.preemptor_job_id, "j2");
    }

    // -- QueueStats tests ---------------------------------------------------

    #[test]
    fn test_queue_stats_depth() {
        let s = QueueStats {
            queued_jobs: 10,
            running_jobs: 5,
            preempted_jobs: 0,
            avg_wait_secs: 0.0,
            max_wait_secs: 0.0,
            total_gpu_hours: 0.0,
        };
        assert_eq!(s.queue_depth(), 15);
    }

    #[test]
    fn test_queue_stats_is_saturated() {
        let cfg = QueueConfig {
            max_queue_depth: 20,
            ..QueueConfig::default()
        };
        let s = QueueStats {
            queued_jobs: 15,
            running_jobs: 5,
            preempted_jobs: 0,
            avg_wait_secs: 0.0,
            max_wait_secs: 0.0,
            total_gpu_hours: 0.0,
        };
        assert!(s.is_saturated(&cfg));

        let s2 = QueueStats {
            queued_jobs: 10,
            running_jobs: 5,
            ..s.clone()
        };
        assert!(!s2.is_saturated(&cfg));
    }

    #[test]
    fn test_queue_stats_not_saturated_at_zero() {
        let cfg = QueueConfig::default();
        let s = QueueStats {
            queued_jobs: 0,
            running_jobs: 0,
            preempted_jobs: 0,
            avg_wait_secs: 0.0,
            max_wait_secs: 0.0,
            total_gpu_hours: 0.0,
        };
        assert!(!s.is_saturated(&cfg));
    }

    // -- FairShareCalculator tests ------------------------------------------

    #[test]
    fn test_fair_share_two_equal_tenants() {
        let calc = FairShareCalculator {
            tenants: vec![
                Tenant::new("t1", "A", 8),
                Tenant::new("t2", "B", 8),
            ],
            total_gpus: 16,
        };
        let shares = calc.compute_shares();
        assert_eq!(shares.len(), 2);
        assert!((shares[0].1 - 0.5).abs() < f64::EPSILON);
        assert!((shares[1].1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fair_share_unequal_quotas() {
        let calc = FairShareCalculator {
            tenants: vec![
                Tenant::new("t1", "A", 6),
                Tenant::new("t2", "B", 2),
            ],
            total_gpus: 8,
        };
        let shares = calc.compute_shares();
        assert!((shares[0].1 - 0.75).abs() < f64::EPSILON);
        assert!((shares[1].1 - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fair_share_sums_to_one() {
        let calc = FairShareCalculator {
            tenants: vec![
                Tenant::new("t1", "A", 3),
                Tenant::new("t2", "B", 5),
                Tenant::new("t3", "C", 2),
            ],
            total_gpus: 10,
        };
        let shares = calc.compute_shares();
        let sum: f64 = shares.iter().map(|(_, s)| s).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fair_share_empty_tenants() {
        let calc = FairShareCalculator {
            tenants: vec![],
            total_gpus: 8,
        };
        let shares = calc.compute_shares();
        assert!(shares.is_empty());
    }

    #[test]
    fn test_fair_share_zero_quotas() {
        let calc = FairShareCalculator {
            tenants: vec![
                Tenant::new("t1", "A", 0),
                Tenant::new("t2", "B", 0),
            ],
            total_gpus: 8,
        };
        let shares = calc.compute_shares();
        assert!((shares[0].1).abs() < f64::EPSILON);
        assert!((shares[1].1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dominant_resource_share() {
        let mut t1 = Tenant::new("t1", "A", 8);
        t1.gpu_used = 4;
        let calc = FairShareCalculator {
            tenants: vec![t1],
            total_gpus: 16,
        };
        assert!((calc.dominant_resource_share("t1") - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dominant_resource_share_unknown_tenant() {
        let calc = FairShareCalculator {
            tenants: vec![Tenant::new("t1", "A", 8)],
            total_gpus: 16,
        };
        assert!((calc.dominant_resource_share("unknown") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dominant_resource_share_zero_gpus() {
        let calc = FairShareCalculator {
            tenants: vec![Tenant::new("t1", "A", 8)],
            total_gpus: 0,
        };
        assert!((calc.dominant_resource_share("t1") - 0.0).abs() < f64::EPSILON);
    }

    // -- select_next_job tests ----------------------------------------------

    fn make_jobs_with_times() -> (Vec<QueuedJob>, Instant) {
        let base = Instant::now();
        let mut j1 = QueuedJob::new("j1", "t1", 1, 100);
        j1.submitted_at = base;
        let mut j2 = QueuedJob::new("j2", "t2", 1, 200);
        j2.submitted_at = base + Duration::from_secs(1);
        let mut j3 = QueuedJob::new("j3", "t1", 1, 150);
        j3.submitted_at = base + Duration::from_secs(2);
        (vec![j1, j2, j3], base)
    }

    #[test]
    fn test_select_fifo() {
        let (jobs, _) = make_jobs_with_times();
        let tenants = vec![Tenant::new("t1", "A", 8), Tenant::new("t2", "B", 8)];
        let idx = select_next_job(&jobs, &SchedulingPolicy::Fifo, &tenants, 16);
        assert_eq!(idx, Some(0)); // j1 submitted first
    }

    #[test]
    fn test_select_priority() {
        let (jobs, _) = make_jobs_with_times();
        let tenants = vec![Tenant::new("t1", "A", 8), Tenant::new("t2", "B", 8)];
        let idx = select_next_job(&jobs, &SchedulingPolicy::Priority, &tenants, 16);
        assert_eq!(idx, Some(1)); // j2 has priority 200
    }

    #[test]
    fn test_select_fair_share() {
        let (jobs, _) = make_jobs_with_times();
        let mut t1 = Tenant::new("t1", "A", 8);
        t1.gpu_used = 6; // high utilization
        let t2 = Tenant::new("t2", "B", 8); // gpu_used=0, underserved
        let tenants = vec![t1, t2];
        let idx = select_next_job(&jobs, &SchedulingPolicy::FairShare, &tenants, 16);
        assert_eq!(idx, Some(1)); // j2 belongs to t2 (underserved)
    }

    #[test]
    fn test_select_drf() {
        let (jobs, _) = make_jobs_with_times();
        let mut t1 = Tenant::new("t1", "A", 8);
        t1.gpu_used = 4;
        let t2 = Tenant::new("t2", "B", 8); // 0 used
        let tenants = vec![t1, t2];
        let idx = select_next_job(&jobs, &SchedulingPolicy::DominantResourceFairness, &tenants, 16);
        assert_eq!(idx, Some(1)); // t2 has lowest DRF
    }

    #[test]
    fn test_select_empty_queue() {
        let tenants = vec![Tenant::new("t1", "A", 8)];
        let idx = select_next_job(&[], &SchedulingPolicy::Fifo, &tenants, 16);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_select_all_running() {
        let (mut jobs, _) = make_jobs_with_times();
        for j in &mut jobs {
            j.started_at = Some(Instant::now());
        }
        let tenants = vec![Tenant::new("t1", "A", 8), Tenant::new("t2", "B", 8)];
        let idx = select_next_job(&jobs, &SchedulingPolicy::Fifo, &tenants, 16);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_select_skips_running_jobs() {
        let (mut jobs, _) = make_jobs_with_times();
        jobs[0].started_at = Some(Instant::now()); // j1 already running
        let tenants = vec![Tenant::new("t1", "A", 8), Tenant::new("t2", "B", 8)];
        let idx = select_next_job(&jobs, &SchedulingPolicy::Fifo, &tenants, 16);
        assert_eq!(idx, Some(1)); // j2 is next pending (FIFO by submit time)
    }
}
