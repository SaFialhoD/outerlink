//! Web dashboard data model types.
//!
//! Pure data types for the OutterLink web dashboard. No HTTP, no templates,
//! no framework dependencies -- just the structs and enums that feed the UI.
//! Stack decision: axum + htmx + rust-embed (single binary), but this module
//! is the DATA LAYER only.

use serde::{Deserialize, Serialize};

// ── Dashboard Configuration ──────────────────────────────────────────────

/// Configuration for the web dashboard HTTP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Address to bind the dashboard HTTP server to.
    pub listen_addr: String,
    /// How often the dashboard auto-refreshes, in milliseconds.
    pub refresh_interval_ms: u64,
    /// Whether to enable Server-Sent Events for live updates.
    pub enable_sse: bool,
    /// Title displayed in the dashboard header.
    pub title: String,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:8080".to_string(),
            refresh_interval_ms: 2000,
            enable_sse: true,
            title: "OutterLink GPU Pool".to_string(),
        }
    }
}

// ── Cluster Overview ─────────────────────────────────────────────────────

/// Aggregate view of the entire GPU cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterOverview {
    /// Human-readable cluster name.
    pub cluster_name: String,
    /// Total number of nodes in the cluster.
    pub total_nodes: u32,
    /// Number of nodes currently reporting healthy.
    pub healthy_nodes: u32,
    /// Total GPUs across all nodes.
    pub total_gpus: u32,
    /// GPUs not currently assigned to any context.
    pub available_gpus: u32,
    /// Total VRAM across all GPUs, in bytes.
    pub total_vram_bytes: u64,
    /// VRAM currently in use, in bytes.
    pub used_vram_bytes: u64,
    /// Number of active CUDA contexts cluster-wide.
    pub active_contexts: u32,
    /// Cluster uptime in seconds.
    pub uptime_secs: u64,
}

impl ClusterOverview {
    /// Percentage of nodes that are healthy (0.0 -- 100.0).
    pub fn health_percent(&self) -> f64 {
        if self.total_nodes == 0 {
            return 0.0;
        }
        (self.healthy_nodes as f64 / self.total_nodes as f64) * 100.0
    }

    /// VRAM utilization as a percentage (0.0 -- 100.0).
    pub fn vram_utilization(&self) -> f64 {
        if self.total_vram_bytes == 0 {
            return 0.0;
        }
        (self.used_vram_bytes as f64 / self.total_vram_bytes as f64) * 100.0
    }

    /// Percentage of GPUs currently in use (0.0 -- 100.0).
    ///
    /// "In use" = total_gpus - available_gpus.
    pub fn gpu_utilization_percent(&self) -> f64 {
        if self.total_gpus == 0 {
            return 0.0;
        }
        let in_use = self.total_gpus.saturating_sub(self.available_gpus);
        (in_use as f64 / self.total_gpus as f64) * 100.0
    }
}

// ── Node View ────────────────────────────────────────────────────────────

/// Status of a single node in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Online,
    Degraded,
    Offline,
}

impl NodeStatus {
    /// Returns the CSS class name for frontend styling.
    pub fn css_class(&self) -> &'static str {
        match self {
            NodeStatus::Online => "status-online",
            NodeStatus::Degraded => "status-degraded",
            NodeStatus::Offline => "status-offline",
        }
    }
}

/// Per-node view for the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeView {
    /// Unique node identifier.
    pub node_id: String,
    /// Human-readable hostname.
    pub hostname: String,
    /// Network address (ip:port).
    pub address: String,
    /// Current node status.
    pub status: NodeStatus,
    /// Number of GPUs on this node.
    pub gpu_count: u32,
    /// Per-GPU details.
    pub gpus: Vec<GpuView>,
    /// Milliseconds since the last heartbeat was received.
    pub last_heartbeat_ago_ms: u64,
}

// ── GPU View ─────────────────────────────────────────────────────────────

/// Per-GPU view for the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuView {
    /// GPU index on the node (0-based).
    pub gpu_index: u32,
    /// GPU product name (e.g. "RTX 3090").
    pub name: String,
    /// Total VRAM in bytes.
    pub vram_total_bytes: u64,
    /// VRAM currently in use, in bytes.
    pub vram_used_bytes: u64,
    /// GPU temperature in degrees Celsius.
    pub temperature_c: f64,
    /// GPU core utilization (0--100).
    pub utilization_percent: u32,
    /// Current power draw in watts.
    pub power_watts: f64,
    /// Health state label (e.g. "healthy", "degraded", "error").
    pub health_state: String,
    /// Number of active CUDA contexts on this GPU.
    pub active_contexts: u32,
}

impl GpuView {
    /// VRAM utilization as a percentage (0.0 -- 100.0).
    pub fn vram_utilization(&self) -> f64 {
        if self.vram_total_bytes == 0 {
            return 0.0;
        }
        (self.vram_used_bytes as f64 / self.vram_total_bytes as f64) * 100.0
    }

    /// Free VRAM in bytes.
    pub fn vram_free_bytes(&self) -> u64 {
        self.vram_total_bytes.saturating_sub(self.vram_used_bytes)
    }

    /// Whether this GPU is in a healthy state.
    pub fn is_healthy(&self) -> bool {
        self.health_state == "healthy"
    }
}

// ── Transfer View ────────────────────────────────────────────────────────

/// Active or recently completed data transfer between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferView {
    /// Node that initiated the transfer.
    pub source_node: String,
    /// Destination node.
    pub dest_node: String,
    /// Total bytes transferred (or to transfer).
    pub bytes: u64,
    /// Current bandwidth in bits per second.
    pub bandwidth_bps: u64,
    /// Transfer mode label (e.g. "host-staged", "opendma").
    pub transfer_mode: String,
    /// Whether the transfer is still in progress.
    pub in_progress: bool,
}

// ── SSE Events ───────────────────────────────────────────────────────────

/// Server-Sent Event types pushed to the dashboard frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SseEvent {
    /// Full cluster overview update.
    ClusterUpdate(ClusterOverview),
    /// Single node update.
    NodeUpdate(NodeView),
    /// Single GPU update on a specific node.
    GpuUpdate { node_id: String, gpu: GpuView },
    /// Transfer activity update.
    TransferUpdate(TransferView),
    /// Alert notification.
    Alert { severity: String, message: String },
}

impl SseEvent {
    /// Returns the SSE event name string used in the `event:` field.
    pub fn event_type(&self) -> &'static str {
        match self {
            SseEvent::ClusterUpdate(_) => "cluster_update",
            SseEvent::NodeUpdate(_) => "node_update",
            SseEvent::GpuUpdate { .. } => "gpu_update",
            SseEvent::TransferUpdate(_) => "transfer_update",
            SseEvent::Alert { .. } => "alert",
        }
    }
}

// ── Dashboard Page Routing ───────────────────────────────────────────────

/// Which page the dashboard should render.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardPage {
    /// Cluster-wide overview.
    Overview,
    /// Node listing.
    Nodes,
    /// Detail view for a specific GPU (by index).
    GpuDetail(u32),
    /// Active/recent transfers.
    Transfers,
    /// Dashboard settings.
    Settings,
}

#[cfg(test)]
mod tests {
    // Tests will reference types that don't exist yet -- they should fail to compile.
    use super::*;

    // ── DashboardConfig defaults ──────────────────────────────────────

    #[test]
    fn dashboard_config_default_listen_addr() {
        let cfg = DashboardConfig::default();
        assert_eq!(cfg.listen_addr, "0.0.0.0:8080");
    }

    #[test]
    fn dashboard_config_default_refresh_interval() {
        let cfg = DashboardConfig::default();
        assert_eq!(cfg.refresh_interval_ms, 2000);
    }

    #[test]
    fn dashboard_config_default_enable_sse() {
        let cfg = DashboardConfig::default();
        assert!(cfg.enable_sse);
    }

    #[test]
    fn dashboard_config_default_title() {
        let cfg = DashboardConfig::default();
        assert_eq!(cfg.title, "OutterLink GPU Pool");
    }

    #[test]
    fn dashboard_config_custom_values() {
        let cfg = DashboardConfig {
            listen_addr: "127.0.0.1:9090".to_string(),
            refresh_interval_ms: 5000,
            enable_sse: false,
            title: "My Cluster".to_string(),
        };
        assert_eq!(cfg.listen_addr, "127.0.0.1:9090");
        assert_eq!(cfg.refresh_interval_ms, 5000);
        assert!(!cfg.enable_sse);
        assert_eq!(cfg.title, "My Cluster");
    }

    // ── ClusterOverview health / utilization math ─────────────────────

    #[test]
    fn cluster_health_percent_all_healthy() {
        let c = ClusterOverview {
            cluster_name: "test".into(),
            total_nodes: 4,
            healthy_nodes: 4,
            total_gpus: 8,
            available_gpus: 8,
            total_vram_bytes: 1024,
            used_vram_bytes: 0,
            active_contexts: 0,
            uptime_secs: 100,
        };
        assert!((c.health_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cluster_health_percent_half_healthy() {
        let c = ClusterOverview {
            cluster_name: "test".into(),
            total_nodes: 4,
            healthy_nodes: 2,
            total_gpus: 8,
            available_gpus: 4,
            total_vram_bytes: 1024,
            used_vram_bytes: 512,
            active_contexts: 0,
            uptime_secs: 100,
        };
        assert!((c.health_percent() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cluster_health_percent_zero_nodes() {
        let c = ClusterOverview {
            cluster_name: "empty".into(),
            total_nodes: 0,
            healthy_nodes: 0,
            total_gpus: 0,
            available_gpus: 0,
            total_vram_bytes: 0,
            used_vram_bytes: 0,
            active_contexts: 0,
            uptime_secs: 0,
        };
        assert!((c.health_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cluster_vram_utilization_half() {
        let c = ClusterOverview {
            cluster_name: "test".into(),
            total_nodes: 1,
            healthy_nodes: 1,
            total_gpus: 1,
            available_gpus: 1,
            total_vram_bytes: 1000,
            used_vram_bytes: 500,
            active_contexts: 0,
            uptime_secs: 100,
        };
        assert!((c.vram_utilization() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cluster_vram_utilization_zero_total() {
        let c = ClusterOverview {
            cluster_name: "empty".into(),
            total_nodes: 0,
            healthy_nodes: 0,
            total_gpus: 0,
            available_gpus: 0,
            total_vram_bytes: 0,
            used_vram_bytes: 0,
            active_contexts: 0,
            uptime_secs: 0,
        };
        assert!((c.vram_utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cluster_gpu_utilization_percent() {
        let c = ClusterOverview {
            cluster_name: "test".into(),
            total_nodes: 2,
            healthy_nodes: 2,
            total_gpus: 8,
            available_gpus: 2,
            total_vram_bytes: 1024,
            used_vram_bytes: 0,
            active_contexts: 0,
            uptime_secs: 100,
        };
        // 6 in use out of 8 = 75%
        assert!((c.gpu_utilization_percent() - 75.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cluster_gpu_utilization_percent_zero_gpus() {
        let c = ClusterOverview {
            cluster_name: "empty".into(),
            total_nodes: 0,
            healthy_nodes: 0,
            total_gpus: 0,
            available_gpus: 0,
            total_vram_bytes: 0,
            used_vram_bytes: 0,
            active_contexts: 0,
            uptime_secs: 0,
        };
        assert!((c.gpu_utilization_percent() - 0.0).abs() < f64::EPSILON);
    }

    // ── NodeStatus css_class ──────────────────────────────────────────

    #[test]
    fn node_status_online_css() {
        assert_eq!(NodeStatus::Online.css_class(), "status-online");
    }

    #[test]
    fn node_status_degraded_css() {
        assert_eq!(NodeStatus::Degraded.css_class(), "status-degraded");
    }

    #[test]
    fn node_status_offline_css() {
        assert_eq!(NodeStatus::Offline.css_class(), "status-offline");
    }

    // ── GpuView vram math ─────────────────────────────────────────────

    #[test]
    fn gpu_view_vram_utilization() {
        let g = GpuView {
            gpu_index: 0,
            name: "RTX 3090".into(),
            vram_total_bytes: 24_000_000_000,
            vram_used_bytes: 12_000_000_000,
            temperature_c: 65.0,
            utilization_percent: 50,
            power_watts: 300.0,
            health_state: "healthy".into(),
            active_contexts: 1,
        };
        assert!((g.vram_utilization() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gpu_view_vram_utilization_zero_total() {
        let g = GpuView {
            gpu_index: 0,
            name: "bad".into(),
            vram_total_bytes: 0,
            vram_used_bytes: 0,
            temperature_c: 0.0,
            utilization_percent: 0,
            power_watts: 0.0,
            health_state: "unknown".into(),
            active_contexts: 0,
        };
        assert!((g.vram_utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gpu_view_vram_free_bytes() {
        let g = GpuView {
            gpu_index: 0,
            name: "RTX 3090".into(),
            vram_total_bytes: 24_000_000_000,
            vram_used_bytes: 10_000_000_000,
            temperature_c: 65.0,
            utilization_percent: 40,
            power_watts: 280.0,
            health_state: "healthy".into(),
            active_contexts: 2,
        };
        assert_eq!(g.vram_free_bytes(), 14_000_000_000);
    }

    #[test]
    fn gpu_view_is_healthy_true() {
        let g = GpuView {
            gpu_index: 0,
            name: "RTX 3090".into(),
            vram_total_bytes: 24_000_000_000,
            vram_used_bytes: 0,
            temperature_c: 55.0,
            utilization_percent: 0,
            power_watts: 30.0,
            health_state: "healthy".into(),
            active_contexts: 0,
        };
        assert!(g.is_healthy());
    }

    #[test]
    fn gpu_view_is_healthy_false() {
        let g = GpuView {
            gpu_index: 0,
            name: "RTX 3090".into(),
            vram_total_bytes: 24_000_000_000,
            vram_used_bytes: 0,
            temperature_c: 55.0,
            utilization_percent: 0,
            power_watts: 30.0,
            health_state: "degraded".into(),
            active_contexts: 0,
        };
        assert!(!g.is_healthy());
    }

    // ── SseEvent event_type ───────────────────────────────────────────

    #[test]
    fn sse_event_type_cluster_update() {
        let overview = ClusterOverview {
            cluster_name: "test".into(),
            total_nodes: 1,
            healthy_nodes: 1,
            total_gpus: 1,
            available_gpus: 1,
            total_vram_bytes: 1024,
            used_vram_bytes: 0,
            active_contexts: 0,
            uptime_secs: 1,
        };
        let ev = SseEvent::ClusterUpdate(overview);
        assert_eq!(ev.event_type(), "cluster_update");
    }

    #[test]
    fn sse_event_type_node_update() {
        let node = NodeView {
            node_id: "n1".into(),
            hostname: "host1".into(),
            address: "1.2.3.4:5000".into(),
            status: NodeStatus::Online,
            gpu_count: 1,
            gpus: vec![],
            last_heartbeat_ago_ms: 100,
        };
        let ev = SseEvent::NodeUpdate(node);
        assert_eq!(ev.event_type(), "node_update");
    }

    #[test]
    fn sse_event_type_gpu_update() {
        let gpu = GpuView {
            gpu_index: 0,
            name: "RTX 3090".into(),
            vram_total_bytes: 24_000_000_000,
            vram_used_bytes: 0,
            temperature_c: 40.0,
            utilization_percent: 0,
            power_watts: 30.0,
            health_state: "healthy".into(),
            active_contexts: 0,
        };
        let ev = SseEvent::GpuUpdate {
            node_id: "n1".into(),
            gpu,
        };
        assert_eq!(ev.event_type(), "gpu_update");
    }

    #[test]
    fn sse_event_type_transfer_update() {
        let t = TransferView {
            source_node: "n1".into(),
            dest_node: "n2".into(),
            bytes: 1024,
            bandwidth_bps: 10_000_000,
            transfer_mode: "host-staged".into(),
            in_progress: true,
        };
        let ev = SseEvent::TransferUpdate(t);
        assert_eq!(ev.event_type(), "transfer_update");
    }

    #[test]
    fn sse_event_type_alert() {
        let ev = SseEvent::Alert {
            severity: "warning".into(),
            message: "GPU overheating".into(),
        };
        assert_eq!(ev.event_type(), "alert");
    }

    // ── DashboardPage variants ────────────────────────────────────────

    #[test]
    fn dashboard_page_overview_variant() {
        let page = DashboardPage::Overview;
        assert!(matches!(page, DashboardPage::Overview));
    }

    #[test]
    fn dashboard_page_gpu_detail_variant() {
        let page = DashboardPage::GpuDetail(2);
        assert!(matches!(page, DashboardPage::GpuDetail(2)));
    }

    #[test]
    fn dashboard_page_all_variants_exist() {
        // Ensure all five variants compile
        let _pages = vec![
            DashboardPage::Overview,
            DashboardPage::Nodes,
            DashboardPage::GpuDetail(0),
            DashboardPage::Transfers,
            DashboardPage::Settings,
        ];
    }

    // ── NodeView construction ─────────────────────────────────────────

    #[test]
    fn node_view_with_gpus() {
        let gpu = GpuView {
            gpu_index: 0,
            name: "RTX 3090".into(),
            vram_total_bytes: 24_000_000_000,
            vram_used_bytes: 6_000_000_000,
            temperature_c: 60.0,
            utilization_percent: 25,
            power_watts: 200.0,
            health_state: "healthy".into(),
            active_contexts: 1,
        };
        let node = NodeView {
            node_id: "node-1".into(),
            hostname: "gpu-box-1".into(),
            address: "192.168.1.100:5000".into(),
            status: NodeStatus::Online,
            gpu_count: 1,
            gpus: vec![gpu],
            last_heartbeat_ago_ms: 50,
        };
        assert_eq!(node.gpus.len(), 1);
        assert_eq!(node.gpu_count, 1);
    }

    // ── TransferView construction ─────────────────────────────────────

    #[test]
    fn transfer_view_construction() {
        let t = TransferView {
            source_node: "node-1".into(),
            dest_node: "node-2".into(),
            bytes: 1_000_000,
            bandwidth_bps: 10_000_000_000,
            transfer_mode: "opendma".into(),
            in_progress: false,
        };
        assert!(!t.in_progress);
        assert_eq!(t.transfer_mode, "opendma");
    }
}
