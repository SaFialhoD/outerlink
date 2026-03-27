//! NUMA topology awareness types for OuterLink.
//!
//! Provides pure data types for representing NUMA topology, device placement,
//! and thread/memory affinity planning. No sysfs or hwloc calls -- these types
//! are populated by platform-specific discovery code elsewhere.

// ---------------------------------------------------------------------------
// Device types
// ---------------------------------------------------------------------------

/// Type of PCI device relevant to OuterLink placement decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    /// NVIDIA GPU (or any GPGPU accelerator).
    Gpu,
    /// Network interface card (Mellanox ConnectX, Intel E810, etc.).
    Nic,
    /// NVMe solid-state drive.
    NvmeSsd,
    /// Any other PCI device, with a freeform description.
    Other(String),
}

impl DeviceType {
    /// Returns `true` if this device is a GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self, DeviceType::Gpu)
    }

    /// Returns `true` if this device is a network interface card.
    pub fn is_nic(&self) -> bool {
        matches!(self, DeviceType::Nic)
    }
}

/// A PCI device with its NUMA placement information.
#[derive(Debug, Clone)]
pub struct PciDevice {
    /// PCI bus address, e.g. `"0000:41:00.0"`.
    pub pci_address: String,
    /// What kind of device this is.
    pub device_type: DeviceType,
    /// NUMA node this device is attached to.
    pub numa_node: u32,
    /// PCI vendor ID (e.g. `0x10de` for NVIDIA).
    pub vendor_id: u16,
    /// PCI device ID.
    pub device_id: u16,
}

// ---------------------------------------------------------------------------
// NUMA node
// ---------------------------------------------------------------------------

/// A single NUMA node with its resources.
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// NUMA node identifier (typically 0-based).
    pub node_id: u32,
    /// CPU core indices belonging to this node.
    pub cpu_cores: Vec<u32>,
    /// Total memory in bytes attached to this node.
    pub memory_bytes: u64,
    /// PCI devices attached to this node.
    pub devices: Vec<PciDevice>,
}

impl NumaNode {
    /// Returns `true` if at least one GPU is attached to this node.
    pub fn has_gpu(&self) -> bool {
        self.devices.iter().any(|d| d.device_type.is_gpu())
    }

    /// Returns `true` if at least one NIC is attached to this node.
    pub fn has_nic(&self) -> bool {
        self.devices.iter().any(|d| d.device_type.is_nic())
    }

    /// Number of CPU cores on this node.
    pub fn core_count(&self) -> usize {
        self.cpu_cores.len()
    }
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

/// Complete NUMA topology of a machine.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// All NUMA nodes in the system.
    pub nodes: Vec<NumaNode>,
    /// Whether this is a single-socket system (all devices share one NUMA domain).
    pub is_single_socket: bool,
}

impl NumaTopology {
    /// Number of NUMA nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Find which NUMA node a PCI device belongs to, by its PCI address.
    /// Returns `None` if the device is not found in any node.
    pub fn find_node_for_device(&self, pci_addr: &str) -> Option<u32> {
        for node in &self.nodes {
            if node.devices.iter().any(|d| d.pci_address == pci_addr) {
                return Some(node.node_id);
            }
        }
        None
    }

    /// Returns `true` if the GPU and NIC are on the same NUMA node.
    /// Returns `None` if either device is not found in the topology.
    pub fn gpu_nic_same_node(&self, gpu_pci: &str, nic_pci: &str) -> Option<bool> {
        let gpu_node = self.find_node_for_device(gpu_pci)?;
        let nic_node = self.find_node_for_device(nic_pci)?;
        Some(gpu_node == nic_node)
    }
}

// ---------------------------------------------------------------------------
// NUMA distance
// ---------------------------------------------------------------------------

/// Distance metric between two NUMA nodes.
///
/// Follows the Linux SLIT convention: 10 = local, 20+ = remote.
#[derive(Debug, Clone)]
pub struct NumaDistance {
    /// Source node.
    pub from_node: u32,
    /// Destination node.
    pub to_node: u32,
    /// Distance value (10 = local, 20 = one hop, 30 = two hops, etc.).
    pub distance: u32,
}

impl NumaDistance {
    /// Returns `true` if the distance represents a local (same-node) access.
    pub fn is_local(&self) -> bool {
        self.distance == 10
    }

    /// Returns the penalty factor relative to local access.
    ///
    /// Local access (distance 10) returns 1.0. Remote access returns
    /// `distance / 10.0`, so distance 20 yields 2.0x penalty.
    pub fn penalty_factor(&self) -> f64 {
        self.distance as f64 / 10.0
    }
}

// ---------------------------------------------------------------------------
// Affinity configuration
// ---------------------------------------------------------------------------

/// Configuration controlling how OuterLink pins threads and memory.
#[derive(Debug, Clone)]
pub struct AffinityConfig {
    /// Whether to pin worker/network threads to specific cores.
    pub enable_thread_pinning: bool,
    /// Whether to pin memory allocations to a specific NUMA node.
    pub enable_memory_pinning: bool,
    /// When GPU and NIC are on different NUMA nodes, prefer pinning memory
    /// to the GPU's node (better for compute-heavy workloads).
    pub prefer_gpu_node: bool,
    /// When cross-NUMA, interleave memory across nodes instead of pinning
    /// to one. Only takes effect if `prefer_gpu_node` would otherwise pin
    /// to a single node on a cross-NUMA setup.
    pub fallback_interleave: bool,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            enable_thread_pinning: true,
            enable_memory_pinning: true,
            prefer_gpu_node: true,
            fallback_interleave: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Thread pinning plan
// ---------------------------------------------------------------------------

/// Describes where to pin a single thread.
#[derive(Debug, Clone)]
pub struct ThreadPinning {
    /// Human-readable name for the thread (e.g. `"worker-0"`).
    pub thread_name: String,
    /// CPU cores this thread should be pinned to. Empty means no pinning.
    pub target_cores: Vec<u32>,
    /// NUMA node this thread is associated with.
    pub numa_node: u32,
}

/// Complete affinity plan for an OuterLink process.
///
/// Describes how worker threads (GPU interaction), network threads (NIC I/O),
/// and memory should be placed across NUMA nodes.
#[derive(Debug, Clone)]
pub struct AffinityPlan {
    /// Threads that perform GPU-side work (compute, memcpy staging).
    pub worker_threads: Vec<ThreadPinning>,
    /// Threads that perform network I/O.
    pub network_threads: Vec<ThreadPinning>,
    /// NUMA node for memory allocations (pinned host buffers, etc.).
    pub memory_node: u32,
}

impl AffinityPlan {
    /// Total number of threads in the plan.
    pub fn total_threads(&self) -> usize {
        self.worker_threads.len() + self.network_threads.len()
    }

    /// Returns `true` if worker and network threads span different NUMA nodes.
    pub fn is_cross_numa(&self) -> bool {
        let worker_nodes: std::collections::HashSet<u32> =
            self.worker_threads.iter().map(|t| t.numa_node).collect();
        let network_nodes: std::collections::HashSet<u32> =
            self.network_threads.iter().map(|t| t.numa_node).collect();
        // Cross-NUMA if the union of worker and network nodes has more than one distinct node
        let all_nodes: std::collections::HashSet<u32> =
            worker_nodes.union(&network_nodes).copied().collect();
        all_nodes.len() > 1
    }
}

// ---------------------------------------------------------------------------
// Affinity plan builder
// ---------------------------------------------------------------------------

/// Build an affinity plan given the system topology, GPU/NIC PCI addresses,
/// and configuration.
///
/// This is the main entry point for NUMA-aware placement decisions. It:
/// 1. Locates the GPU and NIC NUMA nodes from the topology.
/// 2. Assigns worker threads to the GPU's NUMA node cores.
/// 3. Assigns network threads to the NIC's NUMA node cores.
/// 4. Chooses a memory node based on `config.prefer_gpu_node`.
///
/// If a device is not found in the topology, its threads default to node 0.
pub fn build_affinity_plan(
    topology: &NumaTopology,
    gpu_pci: &str,
    nic_pci: &str,
    config: &AffinityConfig,
) -> AffinityPlan {
    let gpu_node_id = topology.find_node_for_device(gpu_pci).unwrap_or(0);
    let nic_node_id = topology.find_node_for_device(nic_pci).unwrap_or(0);

    let gpu_node = topology
        .nodes
        .iter()
        .find(|n| n.node_id == gpu_node_id);
    let nic_node = topology
        .nodes
        .iter()
        .find(|n| n.node_id == nic_node_id);

    // Gather cores for each role
    let gpu_cores: Vec<u32> = gpu_node
        .map(|n| n.cpu_cores.clone())
        .unwrap_or_default();
    let nic_cores: Vec<u32> = nic_node
        .map(|n| n.cpu_cores.clone())
        .unwrap_or_default();

    // Decide how many threads of each type. Heuristic: half the cores for
    // workers, half for network, minimum 1 each.
    let worker_count = (gpu_cores.len() / 2).max(1);
    let network_count = (nic_cores.len() / 2).max(1);

    let pin_threads = config.enable_thread_pinning;

    let worker_threads: Vec<ThreadPinning> = (0..worker_count)
        .map(|i| ThreadPinning {
            thread_name: format!("worker-{}", i),
            target_cores: if pin_threads {
                // Assign one core per thread, cycling if needed
                if gpu_cores.is_empty() { vec![] } else { vec![gpu_cores[i % gpu_cores.len()]] }
            } else {
                vec![]
            },
            numa_node: gpu_node_id,
        })
        .collect();

    let network_threads: Vec<ThreadPinning> = (0..network_count)
        .map(|i| ThreadPinning {
            thread_name: format!("net-{}", i),
            target_cores: if pin_threads {
                // Use the second half of cores for network
                let offset = nic_cores.len() / 2;
                if nic_cores.is_empty() { vec![] } else { vec![nic_cores[(offset + i) % nic_cores.len()]] }
            } else {
                vec![]
            },
            numa_node: nic_node_id,
        })
        .collect();

    // Memory node: prefer GPU node unless interleave is requested on cross-NUMA
    let is_cross = gpu_node_id != nic_node_id;
    let memory_node = if config.prefer_gpu_node || !is_cross {
        gpu_node_id
    } else if config.fallback_interleave {
        // In a real implementation, interleave would be signaled differently.
        // For the type layer we just pick the GPU node as the "primary".
        gpu_node_id
    } else {
        nic_node_id
    };

    AffinityPlan {
        worker_threads,
        network_threads,
        memory_node,
    }
}
