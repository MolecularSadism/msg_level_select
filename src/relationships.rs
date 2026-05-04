//! Links between nodes, paths, and edges.
//!
//! The node↔path links (`PathFrom`/`PathTo` ↔ `OutgoingPaths`/`IncomingPaths`)
//! use Bevy's built-in relationship system — each path has exactly one
//! source and one target node.
//!
//! The path↔edge link is a true many-to-many: a single Voronoi adjacency
//! can appear in multiple routes. Bevy's relationship system is 1-to-many
//! only, so [`PathEdges`] and [`EdgePaths`] are plain `Vec<Entity>`
//! components, populated in lock-step by the spawn pipeline.

use bevy::prelude::*;

/// Source-side: a `MapPath`'s starting `MapNode`.
#[derive(Component, Reflect, Debug)]
#[relationship(relationship_target = OutgoingPaths)]
#[reflect(Component)]
pub struct PathFrom(pub Entity);

/// Target-side: list of `MapPath` entities leaving this node.
#[derive(Component, Reflect, Debug, Default)]
#[relationship_target(relationship = PathFrom)]
#[reflect(Component)]
pub struct OutgoingPaths(Vec<Entity>);

impl OutgoingPaths {
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.iter().copied()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Source-side: a `MapPath`'s ending `MapNode`.
#[derive(Component, Reflect, Debug)]
#[relationship(relationship_target = IncomingPaths)]
#[reflect(Component)]
pub struct PathTo(pub Entity);

/// Target-side: list of `MapPath` entities arriving at this node.
#[derive(Component, Reflect, Debug, Default)]
#[relationship_target(relationship = PathTo)]
#[reflect(Component)]
pub struct IncomingPaths(Vec<Entity>);

impl IncomingPaths {
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.iter().copied()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Edges that make up this `MapPath`, in traversal order.
///
/// Paired with [`EdgePaths`] to form a many-to-many link: one edge can
/// belong to several paths when Voronoi adjacencies are walked by more
/// than one route. The spawn pipeline populates both sides; consumers
/// should treat them as read-only.
#[derive(Component, Reflect, Debug, Default)]
#[reflect(Component)]
pub struct PathEdges(Vec<Entity>);

impl PathEdges {
    pub fn new(edges: Vec<Entity>) -> Self {
        Self(edges)
    }
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.iter().copied()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Paths that traverse this `MapEdge`.
///
/// Mirror of [`PathEdges`]. See its docs for why this isn't a Bevy
/// relationship.
#[derive(Component, Reflect, Debug, Default)]
#[reflect(Component)]
pub struct EdgePaths(Vec<Entity>);

impl EdgePaths {
    pub fn new(paths: Vec<Entity>) -> Self {
        Self(paths)
    }
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.0.iter().copied()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
