//! Memory management types and topology for OuterLink.

pub mod topology;

pub use topology::{
    LinkInfo, LinkType, NodeInfo, PlacementScorer, PlacementWeights, Route, TopologyGraph,
};
