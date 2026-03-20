# P10: Multi-Node Scaling

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** MEDIUM - Phase 6 Implementation

## Goal

Extend OutterLink from a 2-PC system to support 3+ nodes with a unified GPU pool, NVLink-aware scheduling, system RAM pooling, and dynamic cluster membership. After this phase, a CUDA application on any node sees every GPU and RAM resource across the entire cluster as locally available.

## Milestone

- 3+ PCs can join a cluster and expose all GPUs to any node
- GPU pool API provides unified view: list, allocate, release
- NVLink-aware scheduler prefers local GPU pairs for bandwidth-heavy operations
- System RAM on remote nodes is accessible via RDMA reads/writes
- Nodes can be added and removed dynamically without restarting the cluster
- A node going offline is handled gracefully (affected contexts return CUDA errors, rest continues)

## Prerequisites

- [ ] P6: Core Transport complete (memory transfers + kernel launch)
- [ ] P7: CUDA Completeness complete (streams, events, multi-GPU)
- [ ] P8: Performance Optimization complete (UCX with RDMA, batching)
- [ ] 3+ PCs with ConnectX-5 and RDMA verified
- [ ] NVLink bridges installed and verified on at least one node

---

## 1. Node Discovery

### Option A: Static Configuration File

```toml
# /etc/outterlink/cluster.toml

[cluster]
name = "lab-cluster"

[[nodes]]
id = "node-01"
address = "192.168.100.1"
port = 9700
role = "server"  # serves its GPUs to the pool

[[nodes]]
id = "node-02"
address = "192.168.100.2"
port = 9700
role = "server"

[[nodes]]
id = "node-03"
address = "192.168.100.3"
port = 9700
role = "server"
```

**Pros:** Simplest to implement, no external dependencies, deterministic.
**Cons:** Manual editing, does not scale past ~10 nodes, requires config distribution.

### Option B: mDNS / DNS-SD Auto-Discovery

Nodes announce themselves on the local network using mDNS service records:

```
Service: _outterlink._tcp.local
TXT record: version=0.1.0, gpus=4, vram_gb=96, rdma=true
```

```rust
// crates/outterlink-common/src/discovery/mdns.rs

use mdns_sd::{ServiceDaemon, ServiceInfo, ServiceEvent};

const SERVICE_TYPE: &str = "_outterlink._tcp.local.";

pub struct MdnsDiscovery {
    daemon: ServiceDaemon,
    known_nodes: Arc<RwLock<HashMap<String, NodeInfo>>>,
}

impl MdnsDiscovery {
    pub fn new(local_node: &NodeInfo) -> Result<Self, DiscoveryError> {
        let daemon = ServiceDaemon::new()?;

        // Register ourselves
        let service = ServiceInfo::new(
            SERVICE_TYPE,
            &local_node.id,
            &local_node.hostname,
            local_node.address,
            local_node.port,
            &[
                ("version", env!("CARGO_PKG_VERSION")),
                ("gpus", &local_node.gpu_count.to_string()),
                ("vram_gb", &local_node.total_vram_gb.to_string()),
                ("rdma", &local_node.rdma_capable.to_string()),
            ],
        )?;
        daemon.register(service)?;

        // Browse for other nodes
        daemon.browse(SERVICE_TYPE)?;

        Ok(Self {
            daemon,
            known_nodes: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn run(&self) {
        let receiver = self.daemon.browse(SERVICE_TYPE).unwrap();
        loop {
            match receiver.recv_async().await {
                Ok(ServiceEvent::ServiceResolved(info)) => {
                    let node = NodeInfo::from_mdns(&info);
                    tracing::info!("Discovered node: {} at {}", node.id, node.address);
                    self.known_nodes.write().await.insert(node.id.clone(), node);
                }
                Ok(ServiceEvent::ServiceRemoved(_, name)) => {
                    tracing::warn!("Node left: {}", name);
                    self.known_nodes.write().await.remove(&name);
                }
                _ => {}
            }
        }
    }
}
```

**Pros:** Zero-configuration, nodes auto-discover each other, handles joins/leaves.
**Cons:** Limited to single L2 broadcast domain (same subnet), mDNS can be unreliable on some networks.

### Option C: Central Registry Service

A dedicated registry process (can run on any node) that all nodes register with:

```
Registry (any node, port 9701)
  |
  |-- POST /register { id, address, gpus, ... }
  |-- GET /nodes -> [ { id, address, gpus, ... }, ... ]
  |-- WebSocket /events -> stream of join/leave events
```

**Pros:** Works across subnets, centralized state, easy to monitor.
**Cons:** Single point of failure (unless replicated), extra service to run.

### Recommendation: Static Config + mDNS Hybrid

**Phase 1 (this phase):** Static configuration file. Reason: simplest, no external dependencies, our initial cluster is 3 nodes on the same desk. Get multi-node working first.

**Phase 2 (future):** Add mDNS as optional auto-discovery layer. Nodes found via mDNS are added to the pool alongside static config. This gives zero-config convenience for small deployments.

**Do NOT build a central registry** unless OutterLink reaches a scale (50+ nodes) where mDNS is insufficient. This is premature for the current scope.

```rust
// crates/outterlink-common/src/discovery/mod.rs

pub enum DiscoveryBackend {
    Static(StaticConfig),
    Mdns(MdnsDiscovery),
    Hybrid {
        static_config: StaticConfig,
        mdns: MdnsDiscovery,
    },
}

impl DiscoveryBackend {
    pub async fn get_nodes(&self) -> Vec<NodeInfo> {
        match self {
            Self::Static(s) => s.nodes.clone(),
            Self::Mdns(m) => m.known_nodes.read().await.values().cloned().collect(),
            Self::Hybrid { static_config, mdns } => {
                let mut nodes = static_config.nodes.clone();
                let discovered = mdns.known_nodes.read().await;
                for (id, node) in discovered.iter() {
                    if !nodes.iter().any(|n| n.id == *id) {
                        nodes.push(node.clone());
                    }
                }
                nodes
            }
        }
    }
}
```

---

## 2. GPU Pool Management

### GPU Registration

When a node starts, it queries local GPUs and registers them with the cluster:

```rust
// crates/outterlink-common/src/pool/mod.rs

/// Metadata for a single GPU in the pool
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    /// Globally unique ID: "{node_id}:gpu{local_index}"
    pub global_id: String,
    /// Node this GPU physically resides on
    pub node_id: String,
    /// Local device index on that node (0, 1, 2, ...)
    pub local_index: u32,
    /// GPU model name (e.g., "NVIDIA GeForce RTX 3090 Ti")
    pub name: String,
    /// Total VRAM in bytes
    pub vram_total: u64,
    /// Currently available VRAM in bytes
    pub vram_available: u64,
    /// CUDA compute capability (major, minor)
    pub compute_capability: (u32, u32),
    /// PCIe bus ID for topology awareness
    pub pcie_bus_id: String,
    /// NVLink connections to other GPUs on the same node
    pub nvlink_peers: Vec<NvLinkPeer>,
    /// Current allocation status
    pub status: GpuStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NvLinkPeer {
    /// Global ID of the peer GPU
    pub peer_global_id: String,
    /// NVLink bandwidth in GB/s
    pub bandwidth_gbps: f64,
    /// Number of NVLink connections
    pub link_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GpuStatus {
    /// Available for allocation
    Available,
    /// Allocated to a specific session
    Allocated { session_id: String },
    /// Offline or in error state
    Offline { reason: String },
}

/// The global GPU pool, maintained by each node with eventual consistency
pub struct GpuPool {
    /// All known GPUs across all nodes
    gpus: Arc<RwLock<HashMap<String, GpuInfo>>>,
    /// This node's ID
    local_node_id: String,
}

impl GpuPool {
    /// Register local GPUs by querying CUDA and nvidia-smi
    pub fn register_local_gpus(&self) -> Result<Vec<GpuInfo>, PoolError> {
        let mut device_count: i32 = 0;
        unsafe { cuDeviceGetCount(&mut device_count); }

        let mut local_gpus = Vec::new();
        for i in 0..device_count {
            let gpu = GpuInfo {
                global_id: format!("{}:gpu{}", self.local_node_id, i),
                node_id: self.local_node_id.clone(),
                local_index: i as u32,
                name: get_device_name(i)?,
                vram_total: get_device_total_mem(i)?,
                vram_available: get_device_free_mem(i)?,
                compute_capability: get_compute_capability(i)?,
                pcie_bus_id: get_pcie_bus_id(i)?,
                nvlink_peers: detect_nvlink_peers(i, &self.local_node_id)?,
                status: GpuStatus::Available,
            };
            local_gpus.push(gpu);
        }

        // Add to pool
        let mut pool = self.gpus.write().unwrap();
        for gpu in &local_gpus {
            pool.insert(gpu.global_id.clone(), gpu.clone());
        }

        Ok(local_gpus)
    }

    /// Merge GPU info received from a remote node
    pub fn merge_remote_gpus(&self, remote_gpus: Vec<GpuInfo>) {
        let mut pool = self.gpus.write().unwrap();
        for gpu in remote_gpus {
            pool.insert(gpu.global_id.clone(), gpu);
        }
    }

    /// List all GPUs in the pool, optionally filtered
    pub fn list_gpus(&self, filter: Option<GpuFilter>) -> Vec<GpuInfo> {
        let pool = self.gpus.read().unwrap();
        pool.values()
            .filter(|g| match &filter {
                None => true,
                Some(f) => f.matches(g),
            })
            .cloned()
            .collect()
    }

    /// Allocate a GPU for a session
    pub fn allocate(&self, global_id: &str, session_id: &str) -> Result<(), PoolError> {
        let mut pool = self.gpus.write().unwrap();
        let gpu = pool.get_mut(global_id).ok_or(PoolError::GpuNotFound)?;
        match &gpu.status {
            GpuStatus::Available => {
                gpu.status = GpuStatus::Allocated {
                    session_id: session_id.to_string(),
                };
                Ok(())
            }
            GpuStatus::Allocated { session_id: owner } => {
                Err(PoolError::AlreadyAllocated(owner.clone()))
            }
            GpuStatus::Offline { reason } => {
                Err(PoolError::GpuOffline(reason.clone()))
            }
        }
    }

    /// Release a GPU back to the pool
    pub fn release(&self, global_id: &str) -> Result<(), PoolError> {
        let mut pool = self.gpus.write().unwrap();
        let gpu = pool.get_mut(global_id).ok_or(PoolError::GpuNotFound)?;
        gpu.status = GpuStatus::Available;
        Ok(())
    }
}

pub struct GpuFilter {
    pub min_vram: Option<u64>,
    pub compute_capability_min: Option<(u32, u32)>,
    pub node_id: Option<String>,
    pub status: Option<GpuStatus>,
    pub prefer_nvlink_pair: bool,
}
```

### Pool Synchronization Between Nodes

Each node maintains a local copy of the full pool. Synchronization happens via gossip:

1. On join, a node sends its GPU list to all known nodes.
2. Periodically (every 5s), nodes exchange heartbeats with GPU status updates.
3. On allocation/release, the change is broadcast to all nodes.

This is eventually consistent. Conflicts (two nodes allocate the same GPU) are resolved by the node that owns the GPU -- it is the authority.

---

## 3. NVLink-Aware Scheduling

### Detecting NVLink Topology

```rust
// crates/outterlink-common/src/pool/nvlink.rs

use std::process::Command;

/// Detect NVLink connections for a given GPU
pub fn detect_nvlink_peers(
    device_index: i32,
    node_id: &str,
) -> Result<Vec<NvLinkPeer>, PoolError> {
    // Parse nvidia-smi topo -m output
    // Example output:
    //         GPU0  GPU1  GPU2  GPU3
    // GPU0     X    NV12  PHB   SYS
    // GPU1    NV12   X    SYS   PHB
    // GPU2    PHB   SYS    X    NV12
    // GPU3    SYS   PHB   NV12   X

    let output = Command::new("nvidia-smi")
        .args(["topo", "-m"])
        .output()?;
    let topo = String::from_utf8_lossy(&output.stdout);

    let mut peers = Vec::new();
    for (peer_idx, connection) in parse_topo_row(&topo, device_index)?.iter().enumerate() {
        if let Some(nvlink_info) = parse_nvlink_connection(connection) {
            peers.push(NvLinkPeer {
                peer_global_id: format!("{}:gpu{}", node_id, peer_idx),
                bandwidth_gbps: nvlink_info.bandwidth,
                link_count: nvlink_info.link_count,
            });
        }
    }

    Ok(peers)
}

struct NvLinkConnectionInfo {
    /// Number of NVLink connections (e.g., NV12 = 12 links)
    link_count: u32,
    /// Estimated bandwidth in GB/s
    bandwidth: f64,
}

fn parse_nvlink_connection(connection: &str) -> Option<NvLinkConnectionInfo> {
    // "NV12" -> 12 links, "NV4" -> 4 links, etc.
    if connection.starts_with("NV") {
        let links: u32 = connection[2..].parse().ok()?;
        // NVLink 3 (3090 Ti): ~28 GB/s per link
        let bandwidth = links as f64 * 28.125;
        Some(NvLinkConnectionInfo {
            link_count: links,
            bandwidth,
        })
    } else {
        None
    }
}
```

### Scheduling Decision Logic

When a CUDA application requests GPU operations, the scheduler decides which GPU to use:

```rust
// crates/outterlink-common/src/pool/scheduler.rs

/// Decide which GPU to use for a new allocation or operation
pub struct Scheduler {
    pool: Arc<GpuPool>,
    local_node_id: String,
}

impl Scheduler {
    /// Select the best GPU for an allocation request
    pub fn select_gpu(&self, request: &AllocationRequest) -> Result<String, SchedulerError> {
        let candidates = self.pool.list_gpus(Some(GpuFilter {
            min_vram: Some(request.min_vram),
            compute_capability_min: request.min_compute_capability,
            status: Some(GpuStatus::Available),
            ..Default::default()
        }));

        if candidates.is_empty() {
            return Err(SchedulerError::NoGpuAvailable);
        }

        // Score each candidate
        let mut scored: Vec<(String, f64)> = candidates
            .iter()
            .map(|gpu| {
                let score = self.score_gpu(gpu, request);
                (gpu.global_id.clone(), score)
            })
            .collect();

        // Highest score wins
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scored[0].0.clone())
    }

    fn score_gpu(&self, gpu: &GpuInfo, request: &AllocationRequest) -> f64 {
        let mut score = 0.0;

        // Prefer local GPUs (no network overhead)
        if gpu.node_id == self.local_node_id {
            score += 100.0;
        }

        // Prefer GPUs with NVLink to already-allocated GPUs in this session
        if let Some(ref partner_gpu) = request.partner_gpu_id {
            for peer in &gpu.nvlink_peers {
                if peer.peer_global_id == *partner_gpu {
                    // NVLink peer: massive bandwidth bonus
                    score += 200.0 + peer.bandwidth_gbps;
                }
            }
        }

        // Prefer GPUs on the same node as partner (PCIe peer-to-peer)
        if let Some(ref partner_gpu) = request.partner_gpu_id {
            if let Some(partner) = self.pool.list_gpus(None)
                .iter()
                .find(|g| g.global_id == *partner_gpu)
            {
                if partner.node_id == gpu.node_id {
                    score += 50.0; // same node, PCIe P2P possible
                }
            }
        }

        // Prefer GPUs with more free VRAM
        score += (gpu.vram_available as f64 / gpu.vram_total as f64) * 30.0;

        // Prefer higher compute capability
        score += (gpu.compute_capability.0 * 10 + gpu.compute_capability.1) as f64;

        // Penalize GPUs on nodes with high latency (if measured)
        // score -= measured_latency_ms * 10.0;

        score
    }
}

pub struct AllocationRequest {
    pub min_vram: u64,
    pub min_compute_capability: Option<(u32, u32)>,
    /// If this allocation is a partner to an existing GPU (e.g., multi-GPU training),
    /// prefer NVLink-connected or same-node GPUs
    pub partner_gpu_id: Option<String>,
    /// Workload type hint for scheduling
    pub workload_hint: WorkloadHint,
}

pub enum WorkloadHint {
    /// Inference: latency-sensitive, moderate bandwidth
    Inference,
    /// Training: bandwidth-heavy, benefits from NVLink
    Training,
    /// General: no preference
    General,
}
```

### When to Use Local vs Remote GPU

| Scenario | Decision | Reason |
|----------|----------|--------|
| Single GPU needed, local available | Local | Zero network overhead |
| Multi-GPU, NVLink pair available locally | Both local | 112.5 GB/s NVLink >> 12.5 GB/s network |
| Multi-GPU, no NVLink available locally | One local + one remote | Maximize total VRAM |
| All local GPUs busy | Remote | Something is better than nothing |
| Training workload, partner GPU exists | Same node as partner | Minimize inter-GPU bandwidth bottleneck |
| Inference workload | Any available | Latency per call matters more than bandwidth |

---

## 4. 3+ Node Support

### Network Topology: Full Mesh

For 3-10 nodes (our target scale), every node connects to every other node:

```
       Node A
      /      \
     /        \
Node B ------  Node C
```

Each edge is one or more 100GbE links. With 4 ConnectX-5 ports per node, we can do:

| Nodes | Links per pair | Total links | Bandwidth per pair |
|-------|---------------|-------------|-------------------|
| 2 | 4 | 4 | 50 GB/s |
| 3 | 2 | 6 | 25 GB/s |
| 4 | 1 | 6 (some via switch) | 12.5 GB/s |
| 5+ | Need switch | N*(N-1)/2 | 12.5 GB/s per pair |

For 3 nodes with 4 ports each: 2 direct links per pair, 25 GB/s per pair. No switch needed.

### Cross-Node Memory Transfer Routing

When GPU_A on Node 1 needs data from GPU_D on Node 3, the transfer is point-to-point:

```
GPU_A (Node 1) -> pinned host (Node 1) -> RDMA -> pinned host (Node 3) -> GPU_D (Node 3)
```

There is NO multi-hop routing. Every node has a direct link to every other node. If a direct link does not exist (e.g., 5+ nodes without full mesh), transfers go through a relay node, but this is a future concern.

```rust
// crates/outterlink-common/src/cluster/routing.rs

pub struct ClusterRouter {
    /// Direct connections to other nodes
    connections: HashMap<String, NodeConnection>,
    local_node_id: String,
}

impl ClusterRouter {
    /// Get the transport connection to reach a target node
    pub fn route_to(&self, target_node_id: &str) -> Result<&NodeConnection, RoutingError> {
        // Direct connection?
        if let Some(conn) = self.connections.get(target_node_id) {
            return Ok(conn);
        }

        // No direct route. For now: error.
        // Future: find relay node with connections to both us and target
        Err(RoutingError::NoRoute(target_node_id.to_string()))
    }
}
```

### Consistency Model: Owner-Authoritative

Each GPU has exactly one owner: the node it physically resides on. That node is the authority for:
- GPU status (available, allocated, offline)
- Memory allocations on that GPU
- Kernel execution on that GPU

When a node goes down:
1. Other nodes detect the failure via heartbeat timeout (default: 10 seconds).
2. All GPUs on the failed node are marked `Offline` in the pool.
3. Active CUDA contexts using those GPUs receive `CUDA_ERROR_DEVICE_UNAVAILABLE` on the next call.
4. Other GPUs on other nodes continue operating normally -- no cluster-wide failure.

```rust
// crates/outterlink-common/src/cluster/health.rs

pub struct HealthMonitor {
    pool: Arc<GpuPool>,
    connections: Arc<RwLock<HashMap<String, NodeConnection>>>,
    heartbeat_interval: Duration,
    heartbeat_timeout: Duration,
}

impl HealthMonitor {
    pub async fn run(&self) {
        let mut interval = tokio::time::interval(self.heartbeat_interval);
        loop {
            interval.tick().await;

            let connections = self.connections.read().await;
            for (node_id, conn) in connections.iter() {
                match tokio::time::timeout(
                    self.heartbeat_timeout,
                    conn.send_heartbeat(),
                ).await {
                    Ok(Ok(response)) => {
                        // Update GPU status from heartbeat response
                        self.pool.merge_remote_gpus(response.gpu_updates);
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("Heartbeat to {} failed: {}", node_id, e);
                        self.mark_node_offline(node_id).await;
                    }
                    Err(_timeout) => {
                        tracing::warn!("Heartbeat to {} timed out", node_id);
                        self.mark_node_offline(node_id).await;
                    }
                }
            }
        }
    }

    async fn mark_node_offline(&self, node_id: &str) {
        let gpus = self.pool.list_gpus(Some(GpuFilter {
            node_id: Some(node_id.to_string()),
            ..Default::default()
        }));

        for gpu in gpus {
            self.pool.set_status(
                &gpu.global_id,
                GpuStatus::Offline {
                    reason: format!("Node {} unreachable", node_id),
                },
            );
        }
    }
}
```

---

## 5. System RAM Pooling

### Use Case

A 70B-parameter LLM with FP16 weights needs ~140 GB of VRAM. No single GPU has that much, but across 3 nodes we have 512 GB of system RAM. Strategy: keep model weights in remote system RAM, stream them to GPU for compute layer-by-layer.

### Design

System RAM on remote nodes is exposed as RDMA-accessible memory regions. The local node can RDMA READ weight tensors on-demand.

```rust
// crates/outterlink-common/src/pool/ram_pool.rs

/// A region of system RAM exposed for remote access via RDMA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RamRegion {
    /// Globally unique ID
    pub id: String,
    /// Node this RAM resides on
    pub node_id: String,
    /// Size in bytes
    pub size: u64,
    /// RDMA remote key for direct access
    pub rkey: u32,
    /// Remote virtual address
    pub remote_addr: u64,
    /// What is stored here (for the pool manager)
    pub content_tag: Option<String>,
}

pub struct RamPool {
    /// All registered RAM regions across nodes
    regions: Arc<RwLock<Vec<RamRegion>>>,
    local_node_id: String,
}

impl RamPool {
    /// Allocate a region of local system RAM and register it for RDMA access
    pub fn allocate_local(
        &self,
        size: u64,
        content_tag: Option<String>,
    ) -> Result<RamRegion, PoolError> {
        // 1. Allocate pinned memory (needed for RDMA registration)
        let ptr = allocate_pinned_memory(size as usize)?;

        // 2. Register with RDMA subsystem (libibverbs mr_reg or UCX mem_map)
        let (rkey, remote_addr) = register_rdma_region(ptr, size as usize)?;

        let region = RamRegion {
            id: uuid::Uuid::new_v4().to_string(),
            node_id: self.local_node_id.clone(),
            size,
            rkey,
            remote_addr,
            content_tag,
        };

        self.regions.write().unwrap().push(region.clone());
        Ok(region)
    }

    /// Read data from a remote RAM region into local memory
    pub async fn read_remote(
        &self,
        region: &RamRegion,
        offset: u64,
        local_buf: &mut [u8],
    ) -> Result<(), PoolError> {
        // RDMA READ from remote pinned memory
        let conn = get_rdma_connection(&region.node_id)?;
        conn.rdma_read(
            local_buf.as_mut_ptr(),
            local_buf.len(),
            region.remote_addr + offset,
            region.rkey,
        ).await?;
        Ok(())
    }

    /// Write data to a remote RAM region from local memory
    pub async fn write_remote(
        &self,
        region: &RamRegion,
        offset: u64,
        local_buf: &[u8],
    ) -> Result<(), PoolError> {
        let conn = get_rdma_connection(&region.node_id)?;
        conn.rdma_write(
            local_buf.as_ptr(),
            local_buf.len(),
            region.remote_addr + offset,
            region.rkey,
        ).await?;
        Ok(())
    }
}
```

### RDMA for System Memory

ConnectX-5 supports standard RDMA for system memory without any GPUDirect requirement. This is the standard RDMA use case and works on all hardware:

```
Node A (has GPU):
  1. Compute needs weights for layer N
  2. RDMA READ from Node B's RAM pool region
  3. Data arrives in local pinned host memory
  4. cudaMemcpyAsync from pinned host to GPU VRAM
  5. Compute layer N
  6. Release pinned host buffer

Node B (has RAM):
  - Weights pre-loaded into RDMA-registered pinned memory
  - No CPU involvement during reads (NIC DMA handles it)
```

### Bandwidth Expectations

| Operation | Bandwidth | Latency |
|-----------|-----------|---------|
| Local RAM read | ~45 GB/s (DDR5 single channel) | <100ns |
| RDMA READ from remote RAM (100GbE) | ~12 GB/s | ~2us |
| RDMA READ from remote RAM (bonded 2x100GbE) | ~24 GB/s | ~2us |
| cudaMemcpy pinned -> GPU VRAM | ~25 GB/s (PCIe 4.0 x16) | ~1us |

End-to-end remote RAM -> local GPU: limited by the slowest stage (network at 12 GB/s for single link).

---

## 6. Dynamic Scaling

### Adding a Node to a Running Cluster

```
1. New node starts outterlink-server with cluster.toml pointing to existing nodes
   (or: new node broadcasts via mDNS)

2. New node sends JOIN request to any existing node

3. Existing node responds with current cluster state:
   - List of all nodes
   - List of all GPUs
   - List of all RAM regions

4. New node registers its own GPUs and RAM regions

5. New node info is broadcast to all existing nodes via gossip

6. All nodes update their pool views

7. New GPUs are immediately available for allocation
```

```rust
// crates/outterlink-server/src/cluster/membership.rs

pub struct ClusterMembership {
    pool: Arc<GpuPool>,
    ram_pool: Arc<RamPool>,
    router: Arc<ClusterRouter>,
}

impl ClusterMembership {
    /// Handle a JOIN request from a new node
    pub async fn handle_join(
        &self,
        new_node: NodeInfo,
    ) -> Result<JoinResponse, ClusterError> {
        // 1. Add the new node's GPUs to our pool
        self.pool.merge_remote_gpus(new_node.gpus.clone());

        // 2. Add connection to the new node
        let conn = self.router.connect_to(&new_node).await?;
        self.router.add_connection(new_node.id.clone(), conn);

        // 3. Broadcast new node to all existing nodes
        self.broadcast_node_joined(&new_node).await;

        // 4. Return current cluster state to the new node
        Ok(JoinResponse {
            nodes: self.router.list_nodes(),
            gpus: self.pool.list_gpus(None),
            ram_regions: self.ram_pool.list_regions(),
        })
    }

    /// Handle a node leaving gracefully (LEAVE request)
    pub async fn handle_leave(
        &self,
        leaving_node_id: &str,
    ) -> Result<(), ClusterError> {
        tracing::info!("Node {} leaving cluster", leaving_node_id);

        // 1. Mark all GPUs on this node as Offline
        // (active sessions will get errors on next call)
        let gpus = self.pool.list_gpus(Some(GpuFilter {
            node_id: Some(leaving_node_id.to_string()),
            ..Default::default()
        }));

        for gpu in &gpus {
            if let GpuStatus::Allocated { session_id } = &gpu.status {
                tracing::warn!(
                    "GPU {} was allocated to session {} -- session will fail",
                    gpu.global_id, session_id
                );
            }
            self.pool.remove_gpu(&gpu.global_id);
        }

        // 2. Remove RAM regions from this node
        self.ram_pool.remove_node_regions(leaving_node_id);

        // 3. Disconnect
        self.router.remove_connection(leaving_node_id);

        // 4. Broadcast removal to remaining nodes
        self.broadcast_node_left(leaving_node_id).await;

        Ok(())
    }
}
```

### Removing a Node Gracefully

1. Node sends LEAVE to all peers.
2. If node has allocated GPUs, those sessions are notified (they will fail on next CUDA call).
3. Node's GPUs and RAM regions are removed from all pool views.
4. Connections to the node are closed.

### GPU Hot-Add / Hot-Remove

CUDA contexts are bound to a specific GPU. If a GPU is removed (node leaves, GPU fails):

- All `CUcontext` handles pointing to that GPU become invalid.
- All subsequent CUDA calls on those contexts return `CUDA_ERROR_DEVICE_UNAVAILABLE`.
- The application must handle this -- OutterLink does not migrate CUDA contexts between GPUs.

CUDA context migration (moving a running kernel to a different GPU) is not supported by CUDA and is out of scope.

**What we CAN do:** If a GPU goes offline, new allocation requests are routed to remaining GPUs. The pool automatically adapts.

---

## 7. Multi-Node Device Numbering

When a CUDA application calls `cuDeviceGetCount` and `cuDeviceGet`, it sees all GPUs in the pool as local devices:

```
Physical layout:
  Node A: GPU0 (3090 Ti), GPU1 (3090 Ti)
  Node B: GPU0 (5090), GPU1 (5090)
  Node C: GPU0 (3090 Ti)

What the app sees on Node A:
  Device 0: GPU0@A (local, 3090 Ti)
  Device 1: GPU1@A (local, 3090 Ti, NVLink to Device 0)
  Device 2: GPU0@B (remote, 5090)
  Device 3: GPU1@B (remote, 5090)
  Device 4: GPU0@C (remote, 3090 Ti)
```

Ordering: local GPUs first (sorted by local index), then remote GPUs sorted by node ID then local index. This ensures the app's device 0 is always a local GPU if one exists.

---

## Implementation Phases

### Phase 10a: Static Discovery + GPU Pool

**Files to create:**
- `crates/outterlink-common/src/discovery/mod.rs`
- `crates/outterlink-common/src/discovery/static_config.rs`
- `crates/outterlink-common/src/pool/mod.rs`
- `crates/outterlink-common/src/pool/gpu_info.rs`
- Config file format: `cluster.toml`

**Acceptance criteria:**
- [ ] 3 nodes join a cluster via static config
- [ ] All GPUs appear in the unified pool
- [ ] `outterlink-cli pool list` shows all GPUs with node location

### Phase 10b: NVLink-Aware Scheduling

**Files to create:**
- `crates/outterlink-common/src/pool/nvlink.rs`
- `crates/outterlink-common/src/pool/scheduler.rs`

**Acceptance criteria:**
- [ ] NVLink topology detected from nvidia-smi
- [ ] Scheduler prefers NVLink-connected GPU pairs
- [ ] Multi-GPU allocation on same node when NVLink available

### Phase 10c: Health Monitoring + Fault Tolerance

**Files to create:**
- `crates/outterlink-common/src/cluster/health.rs`
- `crates/outterlink-common/src/cluster/membership.rs`

**Acceptance criteria:**
- [ ] Node failure detected within 10 seconds
- [ ] Failed node's GPUs marked offline
- [ ] Other nodes continue operating
- [ ] Graceful leave works without errors

### Phase 10d: System RAM Pooling

**Files to create:**
- `crates/outterlink-common/src/pool/ram_pool.rs`

**Acceptance criteria:**
- [ ] RAM regions registered and accessible via RDMA READ/WRITE
- [ ] Remote RAM read achieves >10 GB/s on 100GbE

### Phase 10e: Dynamic Scaling + mDNS

**Files to create/modify:**
- `crates/outterlink-common/src/discovery/mdns.rs`
- Modify membership to handle runtime joins/leaves

**Acceptance criteria:**
- [ ] New node joins a running cluster
- [ ] New node's GPUs immediately available in pool
- [ ] Node removal does not affect other nodes

---

## Test Plan

| Test | Expected |
|------|----------|
| 3 nodes start with static config | All GPUs visible on all nodes |
| Kill one node | Its GPUs marked offline within 10s, others continue |
| Restart killed node | Its GPUs come back online |
| Allocate GPU with NVLink partner | NVLink-connected GPU selected |
| RDMA READ 1GB from remote RAM | Completes in <100ms on 100GbE |
| Add 4th node to running cluster | Its GPUs appear in pool |
| Remove node with allocated GPU | Active session gets CUDA error |
| cuDeviceGetCount on 3-node cluster | Returns total GPU count across all nodes |

## Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Split-brain (network partition) | Two sub-clusters, inconsistent pool | Heartbeat timeout + reconnection logic; no leader election needed (owner-authoritative) |
| GPU allocation race (two nodes allocate same GPU) | Double allocation | Owner node is authority -- rejects second allocation |
| mDNS unreliable on some networks | Discovery fails | Fall back to static config (always available) |
| RAM pool security (RDMA exposes memory) | Data leak | Cluster is trusted (same lab/owner); future: RDMA key rotation |
| Node with allocated GPUs crashes | Sessions fail | Application must handle CUDA errors; no automatic migration |
| NVLink topology detection fails | Suboptimal scheduling | Fall back to VRAM-based scoring |

## Estimated Scope

| Component | New Files | Complexity |
|-----------|-----------|-----------|
| Static config discovery | 2 | Low |
| mDNS discovery | 1 | Medium |
| GPU pool management | 3 | Medium |
| NVLink detection + scheduler | 2 | Medium |
| Health monitoring | 1 | Medium |
| Cluster membership | 1 | High |
| RAM pooling | 1 | High |
| Cluster routing | 1 | Medium |
| Device numbering changes | 2 (modify intercept) | Medium |

## Related Documents

- [R4: ConnectX-5 + Transport Stack](../research/R4-connectx5-transport-stack.md)
- [R6: NVLink Cross-PC](../research/R6-nvlink-cross-pc.md)
- [Hardware Inventory](../pre-planning/01-hardware-inventory.md)
- [Research Consolidation](../research/CONSOLIDATION-all-research.md)
- [P8: Performance Optimization](P8-performance.md) (UCX, RDMA pipeline)

## Open Questions

- [ ] What is the maximum cluster size we want to support? (10 nodes? 100?)
- [ ] Should we support heterogeneous nodes (some without GPUs, RAM-only)?
- [ ] Do we need authentication between nodes? (currently: trusted LAN assumed)
- [ ] Should the scheduler account for network latency measurements?
- [ ] How do we handle CUDA unified memory (cuMemAllocManaged) across nodes?
