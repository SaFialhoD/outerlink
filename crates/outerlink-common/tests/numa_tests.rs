//! Tests for NUMA topology awareness types (R50).

use outerlink_common::numa::*;

// ---------------------------------------------------------------------------
// Helper builders
// ---------------------------------------------------------------------------

fn gpu_device(pci: &str, node: u32) -> PciDevice {
    PciDevice {
        pci_address: pci.to_string(),
        device_type: DeviceType::Gpu,
        numa_node: node,
        vendor_id: 0x10de, // NVIDIA
        device_id: 0x2204, // RTX 3090
    }
}

fn nic_device(pci: &str, node: u32) -> PciDevice {
    PciDevice {
        pci_address: pci.to_string(),
        device_type: DeviceType::Nic,
        numa_node: node,
        vendor_id: 0x15b3, // Mellanox
        device_id: 0x101b, // ConnectX-5
    }
}

fn single_socket_topology() -> NumaTopology {
    NumaTopology {
        nodes: vec![NumaNode {
            node_id: 0,
            cpu_cores: vec![0, 1, 2, 3, 4, 5, 6, 7],
            memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GiB
            devices: vec![
                gpu_device("0000:01:00.0", 0),
                nic_device("0000:02:00.0", 0),
            ],
        }],
        is_single_socket: true,
    }
}

fn dual_socket_topology() -> NumaTopology {
    NumaTopology {
        nodes: vec![
            NumaNode {
                node_id: 0,
                cpu_cores: vec![0, 1, 2, 3],
                memory_bytes: 16 * 1024 * 1024 * 1024,
                devices: vec![gpu_device("0000:41:00.0", 0)],
            },
            NumaNode {
                node_id: 1,
                cpu_cores: vec![4, 5, 6, 7],
                memory_bytes: 16 * 1024 * 1024 * 1024,
                devices: vec![nic_device("0000:81:00.0", 1)],
            },
        ],
        is_single_socket: false,
    }
}

// ---------------------------------------------------------------------------
// DeviceType tests
// ---------------------------------------------------------------------------

#[test]
fn device_type_is_gpu() {
    assert!(DeviceType::Gpu.is_gpu());
    assert!(!DeviceType::Gpu.is_nic());
}

#[test]
fn device_type_is_nic() {
    assert!(DeviceType::Nic.is_nic());
    assert!(!DeviceType::Nic.is_gpu());
}

#[test]
fn device_type_other_is_neither() {
    let other = DeviceType::Other("USB controller".to_string());
    assert!(!other.is_gpu());
    assert!(!other.is_nic());
}

#[test]
fn device_type_nvme_ssd() {
    assert!(!DeviceType::NvmeSsd.is_gpu());
    assert!(!DeviceType::NvmeSsd.is_nic());
}

// ---------------------------------------------------------------------------
// PciDevice tests
// ---------------------------------------------------------------------------

#[test]
fn pci_device_fields() {
    let dev = gpu_device("0000:01:00.0", 0);
    assert_eq!(dev.pci_address, "0000:01:00.0");
    assert_eq!(dev.numa_node, 0);
    assert_eq!(dev.vendor_id, 0x10de);
    assert!(dev.device_type.is_gpu());
}

// ---------------------------------------------------------------------------
// NumaNode tests
// ---------------------------------------------------------------------------

#[test]
fn numa_node_has_gpu() {
    let topo = single_socket_topology();
    assert!(topo.nodes[0].has_gpu());
}

#[test]
fn numa_node_has_nic() {
    let topo = single_socket_topology();
    assert!(topo.nodes[0].has_nic());
}

#[test]
fn numa_node_no_gpu() {
    let topo = dual_socket_topology();
    // Node 1 has only NIC
    assert!(!topo.nodes[1].has_gpu());
    assert!(topo.nodes[1].has_nic());
}

#[test]
fn numa_node_core_count() {
    let topo = single_socket_topology();
    assert_eq!(topo.nodes[0].core_count(), 8);
}

#[test]
fn numa_node_core_count_dual() {
    let topo = dual_socket_topology();
    assert_eq!(topo.nodes[0].core_count(), 4);
    assert_eq!(topo.nodes[1].core_count(), 4);
}

// ---------------------------------------------------------------------------
// NumaTopology tests
// ---------------------------------------------------------------------------

#[test]
fn topology_node_count() {
    assert_eq!(single_socket_topology().node_count(), 1);
    assert_eq!(dual_socket_topology().node_count(), 2);
}

#[test]
fn topology_find_device_single_socket() {
    let topo = single_socket_topology();
    let node = topo.find_node_for_device("0000:01:00.0");
    assert_eq!(node, Some(0));
}

#[test]
fn topology_find_device_dual_socket() {
    let topo = dual_socket_topology();
    assert_eq!(topo.find_node_for_device("0000:41:00.0"), Some(0));
    assert_eq!(topo.find_node_for_device("0000:81:00.0"), Some(1));
}

#[test]
fn topology_find_device_missing() {
    let topo = single_socket_topology();
    assert_eq!(topo.find_node_for_device("0000:ff:00.0"), None);
}

#[test]
fn topology_gpu_nic_same_node_true() {
    // Single socket -- everything is on node 0
    let topo = single_socket_topology();
    assert!(topo.gpu_nic_same_node(0, 0));
}

#[test]
fn topology_gpu_nic_same_node_false() {
    assert!(!dual_socket_topology().gpu_nic_same_node(0, 1));
}

#[test]
fn topology_single_socket_flag() {
    assert!(single_socket_topology().is_single_socket);
    assert!(!dual_socket_topology().is_single_socket);
}

// ---------------------------------------------------------------------------
// NumaDistance tests
// ---------------------------------------------------------------------------

#[test]
fn numa_distance_local() {
    let d = NumaDistance {
        from_node: 0,
        to_node: 0,
        distance: 10,
    };
    assert!(d.is_local());
    assert!((d.penalty_factor() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn numa_distance_remote() {
    let d = NumaDistance {
        from_node: 0,
        to_node: 1,
        distance: 20,
    };
    assert!(!d.is_local());
    assert!((d.penalty_factor() - 2.0).abs() < f64::EPSILON);
}

#[test]
fn numa_distance_far_remote() {
    let d = NumaDistance {
        from_node: 0,
        to_node: 3,
        distance: 30,
    };
    assert!(!d.is_local());
    assert!((d.penalty_factor() - 3.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// AffinityConfig tests
// ---------------------------------------------------------------------------

#[test]
fn affinity_config_defaults() {
    let cfg = AffinityConfig::default();
    assert!(cfg.enable_thread_pinning);
    assert!(cfg.enable_memory_pinning);
    assert!(cfg.prefer_gpu_node);
    assert!(!cfg.fallback_interleave);
}

// ---------------------------------------------------------------------------
// ThreadPinning tests
// ---------------------------------------------------------------------------

#[test]
fn thread_pinning_fields() {
    let tp = ThreadPinning {
        thread_name: "worker-0".to_string(),
        target_cores: vec![0, 1],
        numa_node: 0,
    };
    assert_eq!(tp.thread_name, "worker-0");
    assert_eq!(tp.target_cores.len(), 2);
}

// ---------------------------------------------------------------------------
// AffinityPlan tests
// ---------------------------------------------------------------------------

#[test]
fn affinity_plan_total_threads() {
    let plan = AffinityPlan {
        worker_threads: vec![
            ThreadPinning { thread_name: "w0".into(), target_cores: vec![0], numa_node: 0 },
            ThreadPinning { thread_name: "w1".into(), target_cores: vec![1], numa_node: 0 },
        ],
        network_threads: vec![
            ThreadPinning { thread_name: "n0".into(), target_cores: vec![2], numa_node: 0 },
        ],
        memory_node: 0,
    };
    assert_eq!(plan.total_threads(), 3);
}

#[test]
fn affinity_plan_is_cross_numa_false() {
    let plan = AffinityPlan {
        worker_threads: vec![
            ThreadPinning { thread_name: "w0".into(), target_cores: vec![0], numa_node: 0 },
        ],
        network_threads: vec![
            ThreadPinning { thread_name: "n0".into(), target_cores: vec![1], numa_node: 0 },
        ],
        memory_node: 0,
    };
    assert!(!plan.is_cross_numa());
}

#[test]
fn affinity_plan_is_cross_numa_true() {
    let plan = AffinityPlan {
        worker_threads: vec![
            ThreadPinning { thread_name: "w0".into(), target_cores: vec![0], numa_node: 0 },
        ],
        network_threads: vec![
            ThreadPinning { thread_name: "n0".into(), target_cores: vec![4], numa_node: 1 },
        ],
        memory_node: 0,
    };
    assert!(plan.is_cross_numa());
}

// ---------------------------------------------------------------------------
// build_affinity_plan tests
// ---------------------------------------------------------------------------

#[test]
fn build_plan_single_socket() {
    let topo = single_socket_topology();
    let cfg = AffinityConfig::default();
    let plan = build_affinity_plan(&topo, "0000:01:00.0", "0000:02:00.0", &cfg);

    // Same node -- not cross-NUMA
    assert!(!plan.is_cross_numa());
    assert_eq!(plan.memory_node, 0);
    assert!(!plan.worker_threads.is_empty());
    assert!(!plan.network_threads.is_empty());
}

#[test]
fn build_plan_dual_socket_cross_numa() {
    let topo = dual_socket_topology();
    let cfg = AffinityConfig::default();
    let plan = build_affinity_plan(&topo, "0000:41:00.0", "0000:81:00.0", &cfg);

    // GPU on node 0, NIC on node 1 -- cross-NUMA
    assert!(plan.is_cross_numa());
    // prefer_gpu_node=true -> memory pinned to GPU node
    assert_eq!(plan.memory_node, 0);
    // Worker threads on GPU node (node 0 cores)
    for w in &plan.worker_threads {
        assert_eq!(w.numa_node, 0);
    }
    // Network threads on NIC node (node 1 cores)
    for n in &plan.network_threads {
        assert_eq!(n.numa_node, 1);
    }
}

#[test]
fn build_plan_pinning_disabled() {
    let topo = single_socket_topology();
    let cfg = AffinityConfig {
        enable_thread_pinning: false,
        enable_memory_pinning: false,
        ..AffinityConfig::default()
    };
    let plan = build_affinity_plan(&topo, "0000:01:00.0", "0000:02:00.0", &cfg);

    // With pinning disabled, threads have empty core lists
    for w in &plan.worker_threads {
        assert!(w.target_cores.is_empty());
    }
    for n in &plan.network_threads {
        assert!(n.target_cores.is_empty());
    }
}

#[test]
fn build_plan_fallback_interleave_ignored_when_same_node() {
    let topo = single_socket_topology();
    let cfg = AffinityConfig {
        fallback_interleave: true,
        ..AffinityConfig::default()
    };
    let plan = build_affinity_plan(&topo, "0000:01:00.0", "0000:02:00.0", &cfg);
    // Same node, so memory_node is just 0 (no interleave needed)
    assert_eq!(plan.memory_node, 0);
}

// ---------------------------------------------------------------------------
// Clone / Debug derive tests
// ---------------------------------------------------------------------------

#[test]
fn types_are_clone_and_debug() {
    let topo = single_socket_topology();
    let _cloned = topo.clone();
    let _debug = format!("{:?}", topo);

    let cfg = AffinityConfig::default();
    let _cloned_cfg = cfg.clone();
    let _debug_cfg = format!("{:?}", cfg);

    let d = NumaDistance { from_node: 0, to_node: 1, distance: 20 };
    let _cloned_d = d.clone();
    let _debug_d = format!("{:?}", d);
}
