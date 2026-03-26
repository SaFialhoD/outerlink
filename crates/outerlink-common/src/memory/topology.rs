//! Topology-aware scheduling for OuterLink clusters.
//!
//! Models the cluster's network topology (nodes, links, routes) and provides
//! placement scoring for intelligent page migration and GPU selection.

use std::collections::HashMap;

use dashmap::DashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Information about a node in the cluster.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: u8,
    pub hostname: String,
    pub gpu_count: u8,
    pub vram_bytes: u64,
    pub dram_bytes: u64,
    pub nvme_bytes: u64,
    pub online: bool,
}

/// Information about a network link between two nodes.
#[derive(Debug, Clone)]
pub struct LinkInfo {
    pub from_node: u8,
    pub to_node: u8,
    pub link_type: LinkType,
    pub bandwidth_bps: u64,
    pub latency_ns: u64,
    pub utilization: f32,
}

/// The type of network link connecting two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkType {
    /// Same-node loopback (zero-copy).
    Loopback,
    /// ConnectX RDMA link.
    RdmaConnectX,
    /// USB4 / Thunderbolt.
    Usb4,
    /// OCuLink PCIe.
    OcuLink,
    /// Fallback TCP.
    Tcp,
}

/// A computed route between two nodes.
#[derive(Debug, Clone)]
pub struct Route {
    pub from_node: u8,
    pub to_node: u8,
    pub hops: Vec<u8>,
    pub total_latency_ns: u64,
    pub bottleneck_bandwidth_bps: u64,
    pub link_type: LinkType,
}

// ---------------------------------------------------------------------------
// TopologyGraph
// ---------------------------------------------------------------------------

/// Models the cluster network topology for routing and placement decisions.
///
/// All public methods are safe for concurrent access via `DashMap`.
pub struct TopologyGraph {
    nodes: DashMap<u8, NodeInfo>,
    links: DashMap<(u8, u8), LinkInfo>,
    routes: DashMap<(u8, u8), Route>,
}

impl TopologyGraph {
    /// Create an empty topology graph.
    pub fn new() -> Self {
        Self {
            nodes: DashMap::new(),
            links: DashMap::new(),
            routes: DashMap::new(),
        }
    }

    /// Add or update a node.
    pub fn add_node(&self, info: NodeInfo) {
        self.nodes.insert(info.node_id, info);
    }

    /// Remove a node and all links that reference it.
    pub fn remove_node(&self, node_id: u8) {
        self.nodes.remove(&node_id);
        // Remove all links touching this node.
        let keys_to_remove: Vec<(u8, u8)> = self
            .links
            .iter()
            .filter(|entry| {
                let (from, to) = *entry.key();
                from == node_id || to == node_id
            })
            .map(|entry| *entry.key())
            .collect();
        for key in keys_to_remove {
            self.links.remove(&key);
        }
        // Invalidate cached routes involving this node.
        let route_keys: Vec<(u8, u8)> = self
            .routes
            .iter()
            .filter(|entry| {
                let (from, to) = *entry.key();
                from == node_id
                    || to == node_id
                    || entry.value().hops.contains(&node_id)
            })
            .map(|entry| *entry.key())
            .collect();
        for key in route_keys {
            self.routes.remove(&key);
        }
    }

    /// Add or update a link between two nodes.
    pub fn add_link(&self, info: LinkInfo) {
        self.links.insert((info.from_node, info.to_node), info);
    }

    /// Remove a link.
    pub fn remove_link(&self, from: u8, to: u8) {
        self.links.remove(&(from, to));
    }

    /// Get a clone of a node's info.
    pub fn get_node(&self, id: u8) -> Option<NodeInfo> {
        self.nodes.get(&id).map(|r| r.value().clone())
    }

    /// Get a clone of a link's info.
    pub fn get_link(&self, from: u8, to: u8) -> Option<LinkInfo> {
        self.links.get(&(from, to)).map(|r| r.value().clone())
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// List all online node IDs.
    pub fn online_nodes(&self) -> Vec<u8> {
        self.nodes
            .iter()
            .filter(|entry| entry.value().online)
            .map(|entry| entry.value().node_id)
            .collect()
    }

    /// Update link utilization (0.0 to 1.0).
    pub fn update_utilization(&self, from: u8, to: u8, utilization: f32) {
        if let Some(mut link) = self.links.get_mut(&(from, to)) {
            link.utilization = utilization;
        }
    }

    /// Compute shortest-latency routes for all pairs of online nodes
    /// using Dijkstra's algorithm. Results are cached in the `routes` map.
    pub fn compute_routes(&self) {
        self.routes.clear();

        let node_ids: Vec<u8> = self
            .nodes
            .iter()
            .filter(|e| e.value().online)
            .map(|e| e.value().node_id)
            .collect();

        // Build adjacency list from links.
        let mut adj: HashMap<u8, Vec<(u8, u64, u64, LinkType)>> = HashMap::new();
        for entry in self.links.iter() {
            let link = entry.value();
            adj.entry(link.from_node)
                .or_default()
                .push((link.to_node, link.latency_ns, link.bandwidth_bps, link.link_type));
        }

        // Run Dijkstra from each node.
        for &source in &node_ids {
            let mut dist: HashMap<u8, u64> = HashMap::new();
            let mut prev: HashMap<u8, u8> = HashMap::new();
            let mut bw: HashMap<u8, u64> = HashMap::new();
            let mut lt: HashMap<u8, LinkType> = HashMap::new();
            let mut visited: HashMap<u8, bool> = HashMap::new();

            for &n in &node_ids {
                dist.insert(n, u64::MAX);
                visited.insert(n, false);
            }
            dist.insert(source, 0);
            bw.insert(source, u64::MAX);
            lt.insert(source, LinkType::Loopback);

            for _ in 0..node_ids.len() {
                // Find unvisited node with minimum distance.
                let current = node_ids
                    .iter()
                    .filter(|&&n| !visited.get(&n).copied().unwrap_or(true))
                    .min_by_key(|&&n| dist.get(&n).copied().unwrap_or(u64::MAX))
                    .copied();

                let current = match current {
                    Some(c) => c,
                    None => break,
                };

                if dist.get(&current).copied().unwrap_or(u64::MAX) == u64::MAX {
                    break;
                }

                visited.insert(current, true);

                if let Some(neighbors) = adj.get(&current) {
                    for &(neighbor, latency, bandwidth, link_type) in neighbors {
                        if visited.get(&neighbor).copied().unwrap_or(true) {
                            continue;
                        }
                        let new_dist = dist.get(&current).copied().unwrap_or(u64::MAX)
                            .saturating_add(latency);
                        if new_dist < dist.get(&neighbor).copied().unwrap_or(u64::MAX) {
                            dist.insert(neighbor, new_dist);
                            prev.insert(neighbor, current);
                            // Bottleneck bandwidth: min along the path.
                            let current_bw = bw.get(&current).copied().unwrap_or(u64::MAX);
                            bw.insert(neighbor, current_bw.min(bandwidth));
                            lt.insert(neighbor, link_type);
                        }
                    }
                }
            }

            // Reconstruct routes.
            for &dest in &node_ids {
                if dest == source {
                    continue;
                }
                if dist.get(&dest).copied().unwrap_or(u64::MAX) == u64::MAX {
                    continue; // Unreachable.
                }

                let mut hops = Vec::new();
                let mut current = dest;
                while let Some(&p) = prev.get(&current) {
                    if p == source {
                        break;
                    }
                    hops.push(p);
                    current = p;
                }
                hops.reverse();

                let route = Route {
                    from_node: source,
                    to_node: dest,
                    hops,
                    total_latency_ns: dist.get(&dest).copied().unwrap_or(0),
                    bottleneck_bandwidth_bps: bw.get(&dest).copied().unwrap_or(0),
                    link_type: lt.get(&dest).copied().unwrap_or(LinkType::Tcp),
                };
                self.routes.insert((source, dest), route);
            }
        }
    }

    /// Get the best (shortest-latency) cached route between two nodes.
    pub fn best_route(&self, from: u8, to: u8) -> Option<Route> {
        self.routes.get(&(from, to)).map(|r| r.value().clone())
    }

    /// Find the nearest online node (by latency) that has at least `min_bytes`
    /// of available VRAM. Returns `None` if no qualifying node exists.
    pub fn nearest_node_with_capacity(&self, from: u8, min_bytes: u64) -> Option<u8> {
        // Gather candidates: online nodes with enough VRAM, excluding `from`.
        let candidates: Vec<u8> = self
            .nodes
            .iter()
            .filter(|entry| {
                let n = entry.value();
                n.online && n.node_id != from && n.vram_bytes >= min_bytes
            })
            .map(|entry| entry.value().node_id)
            .collect();

        // Pick the one with the lowest latency route.
        candidates
            .into_iter()
            .filter_map(|id| {
                self.routes
                    .get(&(from, id))
                    .map(|r| (id, r.value().total_latency_ns))
            })
            .min_by_key(|&(_, latency)| latency)
            .map(|(id, _)| id)
    }
}

impl Default for TopologyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PlacementScorer
// ---------------------------------------------------------------------------

/// Configurable weights for placement scoring.
#[derive(Debug, Clone)]
pub struct PlacementWeights {
    /// How often this GPU accesses the page (0.35 default).
    pub affinity: f32,
    /// Link bandwidth to this node (0.25 default).
    pub bandwidth: f32,
    /// Available capacity on this node (0.15 default).
    pub capacity: f32,
    /// Cost to migrate the page (0.10 default).
    pub migration_cost: f32,
    /// GPU compute power (0.15 default).
    pub gpu_capability: f32,
}

impl PlacementWeights {
    /// Default weights as specified in the design doc.
    pub fn default_weights() -> Self {
        Self {
            affinity: 0.35,
            bandwidth: 0.25,
            capacity: 0.15,
            migration_cost: 0.10,
            gpu_capability: 0.15,
        }
    }

    /// Sum of all weights (should be 1.0 for properly normalised weights).
    pub fn sum(&self) -> f32 {
        self.affinity + self.bandwidth + self.capacity + self.migration_cost + self.gpu_capability
    }
}

/// Scores candidate nodes for page placement.
pub struct PlacementScorer {
    pub weights: PlacementWeights,
}

impl PlacementScorer {
    /// Create a scorer with the given weights.
    pub fn new(weights: PlacementWeights) -> Self {
        Self { weights }
    }

    /// Create a scorer with default weights.
    pub fn with_default_weights() -> Self {
        Self::new(PlacementWeights::default_weights())
    }

    /// Compute a weighted placement score.
    ///
    /// Each input value should be normalised to 0.0..=1.0.
    /// Higher score = better placement candidate.
    pub fn score(
        &self,
        affinity: f32,
        bandwidth: f32,
        capacity: f32,
        migration_cost: f32,
        gpu_capability: f32,
    ) -> f32 {
        self.weights.affinity * affinity
            + self.weights.bandwidth * bandwidth
            + self.weights.capacity * capacity
            + self.weights.migration_cost * migration_cost
            + self.weights.gpu_capability * gpu_capability
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: u8, vram: u64, online: bool) -> NodeInfo {
        NodeInfo {
            node_id: id,
            hostname: format!("node-{}", id),
            gpu_count: 1,
            vram_bytes: vram,
            dram_bytes: 64 * 1024 * 1024 * 1024,
            nvme_bytes: 1_000_000_000_000,
            online,
        }
    }

    fn make_link(from: u8, to: u8, link_type: LinkType, bw: u64, lat: u64) -> LinkInfo {
        LinkInfo {
            from_node: from,
            to_node: to,
            link_type,
            bandwidth_bps: bw,
            latency_ns: lat,
            utilization: 0.0,
        }
    }

    #[test]
    fn topology_add_remove_nodes() {
        let graph = TopologyGraph::new();
        assert_eq!(graph.node_count(), 0);

        graph.add_node(make_node(1, 24_000_000_000, true));
        graph.add_node(make_node(2, 24_000_000_000, true));
        graph.add_node(make_node(3, 12_000_000_000, false));
        assert_eq!(graph.node_count(), 3);

        let n1 = graph.get_node(1).expect("node 1 should exist");
        assert_eq!(n1.hostname, "node-1");
        assert!(n1.online);

        graph.remove_node(2);
        assert_eq!(graph.node_count(), 2);
        assert!(graph.get_node(2).is_none());
    }

    #[test]
    fn topology_add_remove_links() {
        let graph = TopologyGraph::new();
        graph.add_node(make_node(1, 24_000_000_000, true));
        graph.add_node(make_node(2, 24_000_000_000, true));

        graph.add_link(make_link(1, 2, LinkType::RdmaConnectX, 100_000_000_000, 1_000));
        graph.add_link(make_link(2, 1, LinkType::RdmaConnectX, 100_000_000_000, 1_000));

        assert!(graph.get_link(1, 2).is_some());
        assert!(graph.get_link(2, 1).is_some());

        graph.remove_link(1, 2);
        assert!(graph.get_link(1, 2).is_none());
        // Other direction still exists.
        assert!(graph.get_link(2, 1).is_some());
    }

    #[test]
    fn topology_compute_routes() {
        // 3-node topology: 1 --RDMA-- 2 --TCP-- 3
        // Also add direct 1->3 with higher latency.
        let graph = TopologyGraph::new();
        graph.add_node(make_node(1, 24_000_000_000, true));
        graph.add_node(make_node(2, 24_000_000_000, true));
        graph.add_node(make_node(3, 24_000_000_000, true));

        // 1 <-> 2: fast RDMA
        graph.add_link(make_link(1, 2, LinkType::RdmaConnectX, 100_000_000_000, 1_000));
        graph.add_link(make_link(2, 1, LinkType::RdmaConnectX, 100_000_000_000, 1_000));
        // 2 <-> 3: TCP
        graph.add_link(make_link(2, 3, LinkType::Tcp, 10_000_000_000, 5_000));
        graph.add_link(make_link(3, 2, LinkType::Tcp, 10_000_000_000, 5_000));
        // 1 -> 3: slow direct link (higher latency than going through 2)
        graph.add_link(make_link(1, 3, LinkType::Tcp, 1_000_000_000, 50_000));

        graph.compute_routes();

        // Route 1->3 via 2 should be preferred (1000+5000=6000 < 50000).
        let route = graph.best_route(1, 3).expect("route 1->3 should exist");
        assert_eq!(route.total_latency_ns, 6_000);
        assert_eq!(route.hops, vec![2]); // goes through node 2

        // Route 1->2 should be direct.
        let route12 = graph.best_route(1, 2).expect("route 1->2 should exist");
        assert_eq!(route12.total_latency_ns, 1_000);
        assert!(route12.hops.is_empty());
    }

    #[test]
    fn topology_best_route() {
        let graph = TopologyGraph::new();
        graph.add_node(make_node(1, 24_000_000_000, true));
        graph.add_node(make_node(2, 24_000_000_000, true));

        // Two different links from 1->2: RDMA (low latency) and TCP (high latency).
        // Only one link per pair is stored (last write wins), so we test with
        // the low-latency link.
        graph.add_link(make_link(1, 2, LinkType::RdmaConnectX, 100_000_000_000, 500));
        graph.add_link(make_link(2, 1, LinkType::RdmaConnectX, 100_000_000_000, 500));

        graph.compute_routes();

        let route = graph.best_route(1, 2).expect("should have route");
        assert_eq!(route.total_latency_ns, 500);
        assert_eq!(route.link_type, LinkType::RdmaConnectX);
        assert_eq!(route.bottleneck_bandwidth_bps, 100_000_000_000);
    }

    #[test]
    fn topology_nearest_with_capacity() {
        let graph = TopologyGraph::new();
        graph.add_node(make_node(1, 4_000_000_000, true)); // 4 GB
        graph.add_node(make_node(2, 24_000_000_000, true)); // 24 GB, far
        graph.add_node(make_node(3, 24_000_000_000, true)); // 24 GB, close

        // 1->2: high latency
        graph.add_link(make_link(1, 2, LinkType::Tcp, 10_000_000_000, 10_000));
        // 1->3: low latency
        graph.add_link(make_link(1, 3, LinkType::RdmaConnectX, 100_000_000_000, 1_000));

        graph.compute_routes();

        // Need 8 GB: node 1 doesn't qualify (only 4 GB), node 3 is closer.
        let nearest = graph
            .nearest_node_with_capacity(1, 8_000_000_000)
            .expect("should find a node");
        assert_eq!(nearest, 3);
    }

    #[test]
    fn topology_update_utilization() {
        let graph = TopologyGraph::new();
        graph.add_node(make_node(1, 24_000_000_000, true));
        graph.add_node(make_node(2, 24_000_000_000, true));
        graph.add_link(make_link(1, 2, LinkType::Tcp, 10_000_000_000, 5_000));

        let link = graph.get_link(1, 2).unwrap();
        assert_eq!(link.utilization, 0.0);

        graph.update_utilization(1, 2, 0.75);

        let link = graph.get_link(1, 2).unwrap();
        assert!((link.utilization - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn topology_link_types() {
        // Verify the link types are distinct and usable.
        assert_ne!(LinkType::Loopback, LinkType::RdmaConnectX);
        assert_ne!(LinkType::Usb4, LinkType::OcuLink);
        assert_ne!(LinkType::Tcp, LinkType::Loopback);

        // Verify different link types can coexist in the graph.
        let graph = TopologyGraph::new();
        graph.add_node(make_node(1, 24_000_000_000, true));
        graph.add_node(make_node(2, 24_000_000_000, true));
        graph.add_node(make_node(3, 24_000_000_000, true));

        graph.add_link(make_link(1, 2, LinkType::RdmaConnectX, 100_000_000_000, 1_000));
        graph.add_link(make_link(1, 3, LinkType::Usb4, 40_000_000_000, 2_000));

        let l12 = graph.get_link(1, 2).unwrap();
        let l13 = graph.get_link(1, 3).unwrap();
        assert_eq!(l12.link_type, LinkType::RdmaConnectX);
        assert_eq!(l13.link_type, LinkType::Usb4);
        assert!(l12.bandwidth_bps > l13.bandwidth_bps);
    }

    #[test]
    fn placement_scorer_weighted() {
        let scorer = PlacementScorer::with_default_weights();

        // All ones => weighted sum = sum of weights = 1.0.
        let score_all_ones = scorer.score(1.0, 1.0, 1.0, 1.0, 1.0);
        assert!((score_all_ones - 1.0).abs() < 0.001);

        // All zeros => 0.0.
        let score_all_zeros = scorer.score(0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(score_all_zeros.abs() < f32::EPSILON);

        // Affinity-heavy scenario: high affinity, low everything else.
        let score_affinity = scorer.score(1.0, 0.0, 0.0, 0.0, 0.0);
        assert!((score_affinity - 0.35).abs() < 0.001);

        // Compare two candidates.
        let candidate_a = scorer.score(0.9, 0.8, 0.5, 0.3, 0.7);
        let candidate_b = scorer.score(0.2, 0.3, 0.9, 0.8, 0.4);
        // A should score higher (strong affinity + bandwidth).
        assert!(candidate_a > candidate_b);
    }

    #[test]
    fn placement_default_weights() {
        let weights = PlacementWeights::default_weights();
        let sum = weights.sum();
        assert!(
            (sum - 1.0).abs() < 0.001,
            "default weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn topology_disconnected_nodes() {
        let graph = TopologyGraph::new();
        graph.add_node(make_node(1, 24_000_000_000, true));
        graph.add_node(make_node(2, 24_000_000_000, true));
        graph.add_node(make_node(3, 24_000_000_000, true));

        // Only link 1 <-> 2. Node 3 is isolated.
        graph.add_link(make_link(1, 2, LinkType::RdmaConnectX, 100_000_000_000, 1_000));
        graph.add_link(make_link(2, 1, LinkType::RdmaConnectX, 100_000_000_000, 1_000));

        graph.compute_routes();

        assert!(graph.best_route(1, 2).is_some());
        assert!(graph.best_route(1, 3).is_none(), "no route to disconnected node 3");
        assert!(graph.best_route(3, 1).is_none(), "no route from disconnected node 3");

        // nearest_node_with_capacity should not find node 3 either.
        // Node 2 qualifies.
        let nearest = graph.nearest_node_with_capacity(1, 1_000);
        assert_eq!(nearest, Some(2));
    }
}
