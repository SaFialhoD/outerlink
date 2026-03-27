//! Kubernetes Device Plugin types for OuterLink.
//!
//! Pure type definitions and response building for the K8s device plugin interface.
//! No gRPC or kubelet API calls -- just types that represent device registration,
//! health reporting, allocation requests/responses, and CRD definitions.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The Kubernetes extended resource name for OuterLink virtual GPUs.
/// Per R42 research: outerlink.io/vgpu (not .dev, vgpu not gpu).
pub const RESOURCE_NAME: &str = "outerlink.io/vgpu";

/// Default socket path where the device plugin registers with kubelet.
pub const PLUGIN_SOCKET_PATH: &str = "/var/lib/kubelet/device-plugins/outerlink.sock";

// ---------------------------------------------------------------------------
// Device Health
// ---------------------------------------------------------------------------

/// Health status of a GPU device as reported to the kubelet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceHealth {
    /// Device is operational and available for scheduling.
    Healthy,
    /// Device has a problem described by the inner message.
    Unhealthy(String),
}

impl DeviceHealth {
    /// Returns `true` when the device is healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self, DeviceHealth::Healthy)
    }
}

// ---------------------------------------------------------------------------
// GpuDevice
// ---------------------------------------------------------------------------

/// One allocatable GPU device exposed to Kubernetes.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Unique device identifier (e.g. "node1-gpu0").
    pub id: String,
    /// Physical GPU index on the host.
    pub gpu_index: u32,
    /// Current health status.
    pub health: DeviceHealth,
    /// Total VRAM in bytes.
    pub vram_total_bytes: u64,
    /// Name of the Kubernetes node hosting this GPU.
    pub node_name: String,
    /// Optional topology zone for scheduling affinity.
    pub topology_zone: Option<String>,
}

// ---------------------------------------------------------------------------
// DeviceList
// ---------------------------------------------------------------------------

/// A collection of GPU devices with convenience query methods.
#[derive(Debug, Clone)]
pub struct DeviceList {
    pub devices: Vec<GpuDevice>,
}

impl DeviceList {
    /// Number of healthy devices.
    pub fn healthy_count(&self) -> usize {
        self.devices.iter().filter(|d| d.health.is_healthy()).count()
    }

    /// Number of unhealthy devices.
    pub fn unhealthy_count(&self) -> usize {
        self.devices.iter().filter(|d| !d.health.is_healthy()).count()
    }

    /// Find a device by its unique `id`.
    pub fn find_by_id(&self, id: &str) -> Option<&GpuDevice> {
        self.devices.iter().find(|d| d.id == id)
    }

    /// Find a device by GPU index.
    pub fn find_by_index(&self, idx: u32) -> Option<&GpuDevice> {
        self.devices.iter().find(|d| d.gpu_index == idx)
    }
}

// ---------------------------------------------------------------------------
// Allocation types
// ---------------------------------------------------------------------------

/// What the kubelet sends when a pod requests GPU resources.
#[derive(Debug, Clone)]
pub struct AllocateRequest {
    /// Device IDs the kubelet selected for this pod.
    pub device_ids: Vec<String>,
}

/// A bind-mount to inject into the container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMount {
    /// Path on the host.
    pub host_path: String,
    /// Path inside the container.
    pub container_path: String,
    /// Whether the mount is read-only.
    pub read_only: bool,
}

/// A device node (e.g. `/dev/nvidia0`) to expose inside the container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceNode {
    /// Device path (e.g. `/dev/nvidia0`).
    pub path: String,
    /// Permission string, defaults to `"rw"`.
    pub permissions: String,
}

impl Default for DeviceNode {
    fn default() -> Self {
        Self {
            path: String::new(),
            permissions: "rw".to_string(),
        }
    }
}

/// The response returned to kubelet after an allocate call.
#[derive(Debug, Clone)]
pub struct AllocateResponse {
    /// Environment variables to inject into the container.
    pub envs: HashMap<String, String>,
    /// Bind-mounts to add.
    pub mounts: Vec<DeviceMount>,
    /// Device nodes to expose.
    pub devices: Vec<DeviceNode>,
}

// ---------------------------------------------------------------------------
// build_allocate_response
// ---------------------------------------------------------------------------

/// Build an [`AllocateResponse`] for the given allocation request.
///
/// Sets up:
/// - `LD_PRELOAD` pointing to the OuterLink interception library
/// - `OUTERLINK_SERVER` with the server address
/// - `OUTERLINK_GPU_INDICES` with comma-separated GPU indices derived from device IDs
/// - Library mount (read-only)
///
/// If `local_gpu` is true (server-side plugin with co-located GPUs), also injects
/// `/dev/nvidia*` device nodes. For remote GPU allocation (OutterLink's primary use
/// case), set `local_gpu = false` — the client talks to the GPU via the network,
/// not via local device nodes.
pub fn build_allocate_response(
    request: &AllocateRequest,
    server_addr: &str,
    library_path: &str,
    local_gpu: bool,
) -> AllocateResponse {
    // Extract GPU indices from device IDs.  Convention: id ends with the index
    // after the last '-', e.g. "node1-gpu0" -> "0".  If parsing fails, use the
    // positional index in the request list.
    let gpu_indices: Vec<String> = request
        .device_ids
        .iter()
        .enumerate()
        .map(|(pos, id)| {
            id.rsplit('-')
                .next()
                .and_then(|s| s.strip_prefix("gpu"))
                .and_then(|n| n.parse::<u32>().ok())
                .map(|n| n.to_string())
                .unwrap_or_else(|| pos.to_string())
        })
        .collect();

    let mut envs = HashMap::new();
    envs.insert("LD_PRELOAD".to_string(), library_path.to_string());
    envs.insert("OUTERLINK_SERVER".to_string(), server_addr.to_string());
    envs.insert(
        "OUTERLINK_GPU_INDICES".to_string(),
        gpu_indices.join(","),
    );

    // Mount the interception library into the container (read-only).
    let mounts = vec![DeviceMount {
        host_path: library_path.to_string(),
        container_path: library_path.to_string(),
        read_only: true,
    }];

    // Only expose device nodes for local GPU passthrough (server-side plugin).
    // Remote GPU allocations don't need /dev/nvidia* — the client talks via network.
    let mut devices = Vec::new();
    if local_gpu {
        devices.push(DeviceNode {
            path: "/dev/nvidiactl".to_string(),
            permissions: "rw".to_string(),
        });
        devices.push(DeviceNode {
            path: "/dev/nvidia-uvm".to_string(),
            permissions: "rw".to_string(),
        });
        for idx in &gpu_indices {
            devices.push(DeviceNode {
                path: format!("/dev/nvidia{}", idx),
                permissions: "rw".to_string(),
            });
        }
    }

    AllocateResponse {
        envs,
        mounts,
        devices,
    }
}

// ---------------------------------------------------------------------------
// CRD types
// ---------------------------------------------------------------------------

/// A node participating in a GPU pool.
#[derive(Debug, Clone)]
pub struct PoolNode {
    /// Kubernetes node name.
    pub node_name: String,
    /// Number of GPUs on this node.
    pub gpu_count: u32,
    /// Whether the node is healthy.
    pub healthy: bool,
    /// Network address of the OuterLink server on this node.
    pub address: String,
}

/// Custom Resource Definition representing a pool of GPUs across nodes.
#[derive(Debug, Clone)]
pub struct GpuPoolCrd {
    /// CRD resource name.
    pub name: String,
    /// Kubernetes namespace.
    pub namespace: String,
    /// Nodes in this pool.
    pub nodes: Vec<PoolNode>,
    /// Total GPU count across all nodes.
    pub total_gpus: u32,
    /// GPUs currently available for scheduling.
    pub available_gpus: u32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DeviceHealth -------------------------------------------------------

    #[test]
    fn healthy_returns_true() {
        assert!(DeviceHealth::Healthy.is_healthy());
    }

    #[test]
    fn unhealthy_returns_false() {
        let h = DeviceHealth::Unhealthy("thermal throttle".into());
        assert!(!h.is_healthy());
    }

    #[test]
    fn unhealthy_preserves_reason() {
        let reason = "ECC memory error";
        let h = DeviceHealth::Unhealthy(reason.into());
        match h {
            DeviceHealth::Unhealthy(msg) => assert_eq!(msg, reason),
            _ => panic!("expected Unhealthy"),
        }
    }

    #[test]
    fn health_equality() {
        assert_eq!(DeviceHealth::Healthy, DeviceHealth::Healthy);
        assert_ne!(
            DeviceHealth::Healthy,
            DeviceHealth::Unhealthy("x".into()),
        );
    }

    // -- GpuDevice ----------------------------------------------------------

    fn make_device(id: &str, index: u32, healthy: bool) -> GpuDevice {
        GpuDevice {
            id: id.to_string(),
            gpu_index: index,
            health: if healthy {
                DeviceHealth::Healthy
            } else {
                DeviceHealth::Unhealthy("bad".into())
            },
            vram_total_bytes: 24 * 1024 * 1024 * 1024,
            node_name: "node1".into(),
            topology_zone: None,
        }
    }

    #[test]
    fn gpu_device_fields() {
        let d = make_device("node1-gpu0", 0, true);
        assert_eq!(d.id, "node1-gpu0");
        assert_eq!(d.gpu_index, 0);
        assert!(d.health.is_healthy());
        assert_eq!(d.vram_total_bytes, 24 * 1024 * 1024 * 1024);
        assert_eq!(d.node_name, "node1");
        assert!(d.topology_zone.is_none());
    }

    #[test]
    fn gpu_device_with_topology_zone() {
        let d = GpuDevice {
            topology_zone: Some("zone-a".into()),
            ..make_device("node1-gpu1", 1, true)
        };
        assert_eq!(d.topology_zone.as_deref(), Some("zone-a"));
    }

    // -- DeviceList ---------------------------------------------------------

    fn make_list() -> DeviceList {
        DeviceList {
            devices: vec![
                make_device("node1-gpu0", 0, true),
                make_device("node1-gpu1", 1, true),
                make_device("node1-gpu2", 2, false),
            ],
        }
    }

    #[test]
    fn healthy_count() {
        assert_eq!(make_list().healthy_count(), 2);
    }

    #[test]
    fn unhealthy_count() {
        assert_eq!(make_list().unhealthy_count(), 1);
    }

    #[test]
    fn find_by_id_found() {
        let list = make_list();
        let d = list.find_by_id("node1-gpu1").unwrap();
        assert_eq!(d.gpu_index, 1);
    }

    #[test]
    fn find_by_id_not_found() {
        assert!(make_list().find_by_id("nonexistent").is_none());
    }

    #[test]
    fn find_by_index_found() {
        let list = make_list();
        let d = list.find_by_index(2).unwrap();
        assert_eq!(d.id, "node1-gpu2");
    }

    #[test]
    fn find_by_index_not_found() {
        assert!(make_list().find_by_index(99).is_none());
    }

    #[test]
    fn empty_device_list() {
        let list = DeviceList { devices: vec![] };
        assert_eq!(list.healthy_count(), 0);
        assert_eq!(list.unhealthy_count(), 0);
        assert!(list.find_by_id("x").is_none());
        assert!(list.find_by_index(0).is_none());
    }

    // -- AllocateRequest / AllocateResponse ---------------------------------

    #[test]
    fn allocate_request_holds_ids() {
        let req = AllocateRequest {
            device_ids: vec!["a".into(), "b".into()],
        };
        assert_eq!(req.device_ids.len(), 2);
    }

    #[test]
    fn allocate_response_construction() {
        let resp = AllocateResponse {
            envs: HashMap::new(),
            mounts: vec![],
            devices: vec![],
        };
        assert!(resp.envs.is_empty());
        assert!(resp.mounts.is_empty());
        assert!(resp.devices.is_empty());
    }

    // -- DeviceMount / DeviceNode -------------------------------------------

    #[test]
    fn device_mount_fields() {
        let m = DeviceMount {
            host_path: "/host/lib".into(),
            container_path: "/container/lib".into(),
            read_only: true,
        };
        assert_eq!(m.host_path, "/host/lib");
        assert!(m.read_only);
    }

    #[test]
    fn device_node_default_permissions() {
        let n = DeviceNode::default();
        assert_eq!(n.permissions, "rw");
        assert!(n.path.is_empty());
    }

    #[test]
    fn device_node_custom() {
        let n = DeviceNode {
            path: "/dev/nvidia0".into(),
            permissions: "rwm".into(),
        };
        assert_eq!(n.path, "/dev/nvidia0");
        assert_eq!(n.permissions, "rwm");
    }

    // -- build_allocate_response --------------------------------------------

    #[test]
    fn build_response_sets_ld_preload() {
        let req = AllocateRequest {
            device_ids: vec!["node1-gpu0".into()],
        };
        let resp = build_allocate_response(&req, "10.0.0.1:9000", "/usr/lib/outerlink.so", true);
        assert_eq!(resp.envs["LD_PRELOAD"], "/usr/lib/outerlink.so");
    }

    #[test]
    fn build_response_sets_server_addr() {
        let req = AllocateRequest {
            device_ids: vec!["node1-gpu0".into()],
        };
        let resp = build_allocate_response(&req, "10.0.0.1:9000", "/usr/lib/outerlink.so", true);
        assert_eq!(resp.envs["OUTERLINK_SERVER"], "10.0.0.1:9000");
    }

    #[test]
    fn build_response_gpu_indices() {
        let req = AllocateRequest {
            device_ids: vec!["node1-gpu0".into(), "node1-gpu2".into()],
        };
        let resp = build_allocate_response(&req, "addr", "/lib.so", true);
        assert_eq!(resp.envs["OUTERLINK_GPU_INDICES"], "0,2");
    }

    #[test]
    fn build_response_library_mount() {
        let req = AllocateRequest {
            device_ids: vec!["x-gpu0".into()],
        };
        let resp = build_allocate_response(&req, "addr", "/usr/lib/outerlink.so", true);
        assert_eq!(resp.mounts.len(), 1);
        assert_eq!(resp.mounts[0].host_path, "/usr/lib/outerlink.so");
        assert!(resp.mounts[0].read_only);
    }

    #[test]
    fn build_response_device_nodes() {
        let req = AllocateRequest {
            device_ids: vec!["n-gpu0".into(), "n-gpu1".into()],
        };
        let resp = build_allocate_response(&req, "addr", "/lib.so", true);
        // 2 control devices + 2 per-GPU devices = 4
        assert_eq!(resp.devices.len(), 4);
        assert_eq!(resp.devices[0].path, "/dev/nvidiactl");
        assert_eq!(resp.devices[1].path, "/dev/nvidia-uvm");
        assert_eq!(resp.devices[2].path, "/dev/nvidia0");
        assert_eq!(resp.devices[3].path, "/dev/nvidia1");
    }

    #[test]
    fn build_response_empty_request() {
        let req = AllocateRequest {
            device_ids: vec![],
        };
        let resp = build_allocate_response(&req, "addr", "/lib.so", true);
        assert_eq!(resp.envs["OUTERLINK_GPU_INDICES"], "");
        // Still has control devices
        assert_eq!(resp.devices.len(), 2);
    }

    #[test]
    fn build_response_fallback_index() {
        // IDs without the gpu<N> convention fall back to positional index
        let req = AllocateRequest {
            device_ids: vec!["arbitrary-id".into()],
        };
        let resp = build_allocate_response(&req, "addr", "/lib.so", true);
        assert_eq!(resp.envs["OUTERLINK_GPU_INDICES"], "0");
    }

    // -- CRD types ----------------------------------------------------------

    #[test]
    fn pool_node_construction() {
        let pn = PoolNode {
            node_name: "worker-1".into(),
            gpu_count: 4,
            healthy: true,
            address: "192.168.1.10:9000".into(),
        };
        assert_eq!(pn.node_name, "worker-1");
        assert_eq!(pn.gpu_count, 4);
        assert!(pn.healthy);
    }

    #[test]
    fn gpu_pool_crd_construction() {
        let crd = GpuPoolCrd {
            name: "default-pool".into(),
            namespace: "outerlink-system".into(),
            nodes: vec![
                PoolNode {
                    node_name: "w1".into(),
                    gpu_count: 2,
                    healthy: true,
                    address: "10.0.0.1:9000".into(),
                },
                PoolNode {
                    node_name: "w2".into(),
                    gpu_count: 2,
                    healthy: false,
                    address: "10.0.0.2:9000".into(),
                },
            ],
            total_gpus: 4,
            available_gpus: 2,
        };
        assert_eq!(crd.name, "default-pool");
        assert_eq!(crd.namespace, "outerlink-system");
        assert_eq!(crd.nodes.len(), 2);
        assert_eq!(crd.total_gpus, 4);
        assert_eq!(crd.available_gpus, 2);
    }

    // -- Constants ----------------------------------------------------------

    #[test]
    fn resource_name_format() {
        assert!(RESOURCE_NAME.contains('/'));
        assert!(RESOURCE_NAME.starts_with("outerlink.io/"));
        assert_eq!(RESOURCE_NAME, "outerlink.io/vgpu");
    }

    #[test]
    fn plugin_socket_path() {
        assert!(PLUGIN_SOCKET_PATH.starts_with("/var/lib/kubelet/"));
        assert!(PLUGIN_SOCKET_PATH.ends_with(".sock"));
    }
}
