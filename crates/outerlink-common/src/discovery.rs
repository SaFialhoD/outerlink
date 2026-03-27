//! Node discovery via mDNS-SD.
//!
//! Each OutterLink node announces itself on the local network using
//! `_outterlink._tcp.local.` and discovers peers automatically.
//!
//! This module provides the data types and coordinator election logic.
//! Actual mDNS network I/O will be built on top of these types in the
//! server crate.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// mDNS service type for OutterLink nodes.
pub const SERVICE_TYPE: &str = "_outterlink._tcp.local.";

/// Default control plane port.
pub const DEFAULT_PORT: u16 = 14833;

/// Current protocol version.
pub const PROTOCOL_VERSION: u32 = 1;

// TXT record keys used in mDNS announcements.
pub(crate) const TXT_VERSION: &str = "version";
pub(crate) const TXT_CLUSTER: &str = "cluster";
pub(crate) const TXT_GPUS: &str = "gpus";
pub(crate) const TXT_VRAM: &str = "vram";
pub(crate) const TXT_RDMA: &str = "rdma";
pub(crate) const TXT_JOINED: &str = "joined";
pub(crate) const TXT_NODE_ID: &str = "node_id";

/// Parse a boolean from a TXT record value, accepting common spellings.
fn parse_bool_lenient(s: &str) -> Option<bool> {
    match s.to_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

/// Get a non-empty string from a TXT record.
fn non_empty(txt: &HashMap<String, String>, key: &str) -> Option<String> {
    let v = txt.get(key)?;
    if v.is_empty() { None } else { Some(v.clone()) }
}

/// Information about a discovered OutterLink node.
#[derive(Debug, Clone)]
pub struct DiscoveredNode {
    /// Unique node identifier (UUID v4).
    pub node_id: String,
    /// Node's control plane address.
    pub addr: SocketAddr,
    /// Hostname from mDNS.
    pub hostname: String,
    /// Protocol version.
    pub protocol_version: u32,
    /// Cluster identifier.
    pub cluster_id: String,
    /// Number of GPUs on this node.
    pub gpu_count: u32,
    /// Total VRAM in bytes.
    pub total_vram: u64,
    /// Whether RDMA is available.
    pub rdma_capable: bool,
    /// Unix timestamp (ms) when the node joined.
    pub joined_at: u64,
}

impl DiscoveredNode {
    /// Create from mDNS TXT records and resolved address.
    ///
    /// Returns `None` if any required TXT field is missing or unparseable.
    pub fn from_txt_records(
        hostname: String,
        addr: SocketAddr,
        txt: &HashMap<String, String>,
    ) -> Option<Self> {
        Some(Self {
            node_id: non_empty(txt, TXT_NODE_ID)?,
            addr,
            hostname,
            protocol_version: txt.get(TXT_VERSION)?.parse().ok()?,
            cluster_id: non_empty(txt, TXT_CLUSTER)?,
            gpu_count: txt.get(TXT_GPUS)?.parse().ok()?,
            total_vram: txt.get(TXT_VRAM)?.parse().ok()?,
            rdma_capable: parse_bool_lenient(txt.get(TXT_RDMA)?)?,
            joined_at: txt.get(TXT_JOINED)?.parse().ok()?,
        })
    }

    /// Build TXT record map for mDNS announcement.
    #[must_use]
    pub fn to_txt_records(&self) -> HashMap<String, String> {
        let mut txt = HashMap::new();
        txt.insert(TXT_NODE_ID.into(), self.node_id.clone());
        txt.insert(TXT_VERSION.into(), self.protocol_version.to_string());
        txt.insert(TXT_CLUSTER.into(), self.cluster_id.clone());
        txt.insert(TXT_GPUS.into(), self.gpu_count.to_string());
        txt.insert(TXT_VRAM.into(), self.total_vram.to_string());
        txt.insert(TXT_RDMA.into(), self.rdma_capable.to_string());
        txt.insert(TXT_JOINED.into(), self.joined_at.to_string());
        txt
    }
}

/// Static node configuration (fallback when mDNS is unavailable).
#[derive(Debug, Clone)]
pub struct StaticNodeConfig {
    /// Address to connect to.
    pub addr: SocketAddr,
    /// Optional cluster ID filter.
    pub cluster_id: Option<String>,
}

/// Discovery configuration.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Cluster ID to join/filter by.
    pub cluster_id: String,
    /// Control plane port to announce.
    pub port: u16,
    /// Whether to enable mDNS discovery.
    pub mdns_enabled: bool,
    /// Static peers (always connected regardless of mDNS).
    pub static_peers: Vec<StaticNodeConfig>,
    /// How long to wait for initial discovery before proceeding.
    pub discovery_timeout: Duration,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            cluster_id: "default".into(),
            port: DEFAULT_PORT,
            mdns_enabled: true,
            static_peers: Vec::new(),
            discovery_timeout: Duration::from_secs(5),
        }
    }
}

/// The local node's identity (what we announce to others).
#[derive(Debug, Clone)]
pub struct LocalNodeIdentity {
    /// Unique node identifier (UUID v4).
    pub node_id: String,
    /// Protocol version.
    pub protocol_version: u32,
    /// Cluster identifier.
    pub cluster_id: String,
    /// Number of GPUs on this node.
    pub gpu_count: u32,
    /// Total VRAM in bytes.
    pub total_vram: u64,
    /// Whether RDMA is available.
    pub rdma_capable: bool,
    /// Unix timestamp (ms) when the node joined.
    pub joined_at: u64,
}

impl LocalNodeIdentity {
    /// Create a new identity with current timestamp.
    pub fn new(cluster_id: String, gpu_count: u32, total_vram: u64, rdma_capable: bool) -> Self {
        let joined_at: u64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock predates UNIX epoch")
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);

        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            protocol_version: PROTOCOL_VERSION,
            cluster_id,
            gpu_count,
            total_vram,
            rdma_capable,
            joined_at,
        }
    }

    /// Convert to a [`DiscoveredNode`] (for self-representation in the peer list).
    pub fn to_discovered_node(&self, addr: SocketAddr, hostname: String) -> DiscoveredNode {
        DiscoveredNode {
            node_id: self.node_id.clone(),
            addr,
            hostname,
            protocol_version: self.protocol_version,
            cluster_id: self.cluster_id.clone(),
            gpu_count: self.gpu_count,
            total_vram: self.total_vram,
            rdma_capable: self.rdma_capable,
            joined_at: self.joined_at,
        }
    }
}

/// Determine the coordinator from a set of discovered nodes.
///
/// The coordinator is the node with the oldest (smallest) `joined_at` timestamp.
/// Ties are broken by `node_id` (lexicographic order).
///
/// Returns `None` if the slice is empty.
#[must_use]
pub fn elect_coordinator(nodes: &[DiscoveredNode]) -> Option<&DiscoveredNode> {
    nodes.iter().min_by(|a, b| {
        a.joined_at
            .cmp(&b.joined_at)
            .then_with(|| a.node_id.cmp(&b.node_id))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::net::{IpAddr, Ipv4Addr};

    /// Helper: build a valid TXT record map.
    fn valid_txt() -> HashMap<String, String> {
        let mut txt = HashMap::new();
        txt.insert(TXT_NODE_ID.into(), "aaaa-bbbb-cccc".into());
        txt.insert(TXT_VERSION.into(), "1".into());
        txt.insert(TXT_CLUSTER.into(), "test-cluster".into());
        txt.insert(TXT_GPUS.into(), "2".into());
        txt.insert(TXT_VRAM.into(), "25769803776".into()); // 24 GiB
        txt.insert(TXT_RDMA.into(), "true".into());
        txt.insert(TXT_JOINED.into(), "1700000000000".into());
        txt
    }

    fn test_addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10)), 14833)
    }

    // ---- DiscoveredNode::from_txt_records ----

    #[test]
    fn from_txt_records_valid() {
        let txt = valid_txt();
        let node =
            DiscoveredNode::from_txt_records("host1.local".into(), test_addr(), &txt).unwrap();

        assert_eq!(node.node_id, "aaaa-bbbb-cccc");
        assert_eq!(node.addr, test_addr());
        assert_eq!(node.hostname, "host1.local");
        assert_eq!(node.protocol_version, 1);
        assert_eq!(node.cluster_id, "test-cluster");
        assert_eq!(node.gpu_count, 2);
        assert_eq!(node.total_vram, 25_769_803_776);
        assert!(node.rdma_capable);
        assert_eq!(node.joined_at, 1_700_000_000_000);
    }

    #[test]
    fn from_txt_records_missing_node_id() {
        let mut txt = valid_txt();
        txt.remove(TXT_NODE_ID);
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_missing_version() {
        let mut txt = valid_txt();
        txt.remove(TXT_VERSION);
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_missing_cluster() {
        let mut txt = valid_txt();
        txt.remove(TXT_CLUSTER);
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_missing_gpus() {
        let mut txt = valid_txt();
        txt.remove(TXT_GPUS);
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_missing_vram() {
        let mut txt = valid_txt();
        txt.remove(TXT_VRAM);
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_missing_rdma() {
        let mut txt = valid_txt();
        txt.remove(TXT_RDMA);
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_missing_joined() {
        let mut txt = valid_txt();
        txt.remove(TXT_JOINED);
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_invalid_gpus_parse() {
        let mut txt = valid_txt();
        txt.insert(TXT_GPUS.into(), "not_a_number".into());
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_invalid_vram_parse() {
        let mut txt = valid_txt();
        txt.insert(TXT_VRAM.into(), "xyz".into());
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    #[test]
    fn from_txt_records_invalid_rdma_parse() {
        let mut txt = valid_txt();
        txt.insert(TXT_RDMA.into(), "maybe".into());
        assert!(DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).is_none());
    }

    // ---- to_txt_records roundtrip ----

    #[test]
    fn txt_records_roundtrip() {
        let original_txt = valid_txt();
        let node =
            DiscoveredNode::from_txt_records("host1.local".into(), test_addr(), &original_txt)
                .unwrap();
        let emitted = node.to_txt_records();

        // Re-parse from emitted records.
        let roundtripped =
            DiscoveredNode::from_txt_records("host1.local".into(), test_addr(), &emitted).unwrap();

        assert_eq!(roundtripped.node_id, node.node_id);
        assert_eq!(roundtripped.protocol_version, node.protocol_version);
        assert_eq!(roundtripped.cluster_id, node.cluster_id);
        assert_eq!(roundtripped.gpu_count, node.gpu_count);
        assert_eq!(roundtripped.total_vram, node.total_vram);
        assert_eq!(roundtripped.rdma_capable, node.rdma_capable);
        assert_eq!(roundtripped.joined_at, node.joined_at);
    }

    #[test]
    fn to_txt_records_contains_all_keys() {
        let txt = valid_txt();
        let node =
            DiscoveredNode::from_txt_records("h".into(), test_addr(), &txt).unwrap();
        let emitted = node.to_txt_records();

        let expected_keys: HashSet<&str> = [
            TXT_NODE_ID, TXT_VERSION, TXT_CLUSTER, TXT_GPUS, TXT_VRAM, TXT_RDMA, TXT_JOINED,
        ]
        .into_iter()
        .collect();

        let actual_keys: HashSet<&str> = emitted.keys().map(|k| k.as_str()).collect();
        assert_eq!(expected_keys, actual_keys);
    }

    // ---- elect_coordinator ----

    fn make_node(node_id: &str, joined_at: u64) -> DiscoveredNode {
        DiscoveredNode {
            node_id: node_id.into(),
            addr: test_addr(),
            hostname: "test".into(),
            protocol_version: 1,
            cluster_id: "c".into(),
            gpu_count: 1,
            total_vram: 1024,
            rdma_capable: false,
            joined_at,
        }
    }

    #[test]
    fn elect_coordinator_picks_oldest() {
        let nodes = vec![
            make_node("node-b", 2000),
            make_node("node-a", 1000),
            make_node("node-c", 3000),
        ];
        let coord = elect_coordinator(&nodes).unwrap();
        assert_eq!(coord.node_id, "node-a");
        assert_eq!(coord.joined_at, 1000);
    }

    #[test]
    fn elect_coordinator_tiebreak_by_node_id() {
        let nodes = vec![
            make_node("node-z", 1000),
            make_node("node-a", 1000),
            make_node("node-m", 1000),
        ];
        let coord = elect_coordinator(&nodes).unwrap();
        // All have same joined_at; lexicographically smallest node_id wins.
        assert_eq!(coord.node_id, "node-a");
    }

    #[test]
    fn elect_coordinator_empty_returns_none() {
        let nodes: Vec<DiscoveredNode> = vec![];
        assert!(elect_coordinator(&nodes).is_none());
    }

    #[test]
    fn elect_coordinator_single_node() {
        let nodes = vec![make_node("only-node", 5000)];
        let coord = elect_coordinator(&nodes).unwrap();
        assert_eq!(coord.node_id, "only-node");
    }

    // ---- LocalNodeIdentity ----

    #[test]
    fn local_node_identity_new_generates_unique_ids() {
        let id1 = LocalNodeIdentity::new("c".into(), 1, 1024, false);
        let id2 = LocalNodeIdentity::new("c".into(), 1, 1024, false);
        assert_ne!(id1.node_id, id2.node_id, "UUIDs must be unique");
    }

    #[test]
    fn local_node_identity_new_sets_fields() {
        let id = LocalNodeIdentity::new("my-cluster".into(), 4, 100_000_000, true);
        assert_eq!(id.cluster_id, "my-cluster");
        assert_eq!(id.gpu_count, 4);
        assert_eq!(id.total_vram, 100_000_000);
        assert!(id.rdma_capable);
        assert_eq!(id.protocol_version, 1);
        // joined_at should be a recent timestamp (within last 10 seconds).
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        assert!(id.joined_at <= now_ms);
        assert!(id.joined_at > now_ms - 10_000);
    }

    #[test]
    fn local_node_identity_to_discovered_node() {
        let id = LocalNodeIdentity::new("c".into(), 2, 2048, true);
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), 14833);
        let node = id.to_discovered_node(addr, "myhost".into());

        assert_eq!(node.node_id, id.node_id);
        assert_eq!(node.addr, addr);
        assert_eq!(node.hostname, "myhost");
        assert_eq!(node.protocol_version, id.protocol_version);
        assert_eq!(node.cluster_id, id.cluster_id);
        assert_eq!(node.gpu_count, id.gpu_count);
        assert_eq!(node.total_vram, id.total_vram);
        assert_eq!(node.rdma_capable, id.rdma_capable);
        assert_eq!(node.joined_at, id.joined_at);
    }

    // ---- DiscoveryConfig ----

    #[test]
    fn discovery_config_default_values() {
        let cfg = DiscoveryConfig::default();
        assert_eq!(cfg.cluster_id, "default");
        assert_eq!(cfg.port, DEFAULT_PORT);
        assert!(cfg.mdns_enabled);
        assert!(cfg.static_peers.is_empty());
        assert_eq!(cfg.discovery_timeout, Duration::from_secs(5));
    }

    // ---- StaticNodeConfig ----

    #[test]
    fn static_node_config_construction() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1)), 14833);
        let cfg = StaticNodeConfig {
            addr,
            cluster_id: Some("prod".into()),
        };
        assert_eq!(cfg.addr.port(), 14833);
        assert_eq!(cfg.cluster_id.as_deref(), Some("prod"));

        let cfg_none = StaticNodeConfig {
            addr,
            cluster_id: None,
        };
        assert!(cfg_none.cluster_id.is_none());
    }

    // ---- Constants ----

    #[test]
    fn service_type_format() {
        assert!(SERVICE_TYPE.starts_with('_'));
        assert!(SERVICE_TYPE.contains("._tcp."));
        assert!(SERVICE_TYPE.ends_with(".local."));
    }

    #[test]
    fn default_port_is_nonzero() {
        assert!(DEFAULT_PORT > 1024, "Should use an unprivileged port");
    }
}
