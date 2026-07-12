//! Marker and data components attached to map entities.

use bevy::prelude::*;

/// Marks an entity as one cell of the underlying Voronoi diagram.
///
/// Every cell in the generated diagram is spawned as a `MapNode` so
/// consumers can render the full mosaic, not just the chosen path. The
/// entity always carries:
/// - a `Transform` whose translation defaults to the cell's Voronoi site (post-rotation, no
///   jitter); the spawn pipeline overrides it with a jittered "point of interest" for chosen
///   [`Site`] cells.
/// - a [`VoronoiCell`] component holding the polygon vertices.
///
/// Subsets carry additional markers:
/// - [`Site`] — picked as a true campaign stop. Has [`crate::LocationState`].
/// - [`Waypoint`] — sits along a chosen path's traversal corridor. Has [`crate::LocationState`].
/// - bare `MapNode` — a "dead" cell, included so consumers can render the full Voronoi background.
///   No state, no Site, no Waypoint.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct MapNode;

/// Polygon describing the cell's Voronoi shape, in CCW world-space order
/// (post-rotation). Attached to every [`MapNode`] entity. Consumers can
/// render this with `Mesh2d`, `Gizmos`, or any 2D fill primitive.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component)]
pub struct VoronoiCell {
    pub vertices: Vec<Vec2>,
}

/// Tagged on [`MapNode`] entities that represent campaign sites.
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
pub struct Site {
    /// 0-indexed belt (depth position in the graph).
    pub belt: u32,
    /// 0-indexed site position within its belt.
    pub site: u32,
}

/// Tagged on [`MapNode`] entities introduced as intermediate cells by
/// [`crate::DesiredTraversals`] or `allow_extra_traversal`.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct Waypoint;

/// Marks an entity as a logical path between two sites.
///
/// A path is a composite that owns one or more [`MapEdge`] entities (one
/// for a direct neighbor, two or more when waypoints are threaded in).
/// Use [`crate::PathFrom`] / [`crate::PathTo`] to find its endpoints and
/// [`crate::PathEdges`] to enumerate its edges in spawn order.
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct MapPath;

/// The shared polygon boundary ("wall") between two adjacent Voronoi
/// cells.
///
/// One `MapEdge` exists per unordered cell pair in the diagram. Its
/// geometry is the wall segment itself — the two corner positions where
/// the two polygons meet — not a line between their sites. Edges that
/// lie along a chosen [`MapPath`] additionally carry
/// [`crate::EdgeOfPath`] + [`crate::LocationState`]; "dead" walls have
/// neither.
///
/// `from` and `to` are the two cell [`MapNode`] entities the wall
/// separates (not directional). `wall` holds the segment endpoints in
/// world coordinates (post-rotation).
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
pub struct MapEdge {
    pub from: Entity,
    pub to: Entity,
    pub wall: [Vec2; 2],
}

/// Root entity for a generated map. Children include every node, path,
/// and edge that was spawned for this map.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component)]
pub struct LevelMap {
    /// Width and height (in world units) of the visible region (excludes
    /// the buffered fluff outside the camera view).
    pub size: Vec2,
    /// Seed used for generation; useful for reproducing the layout. When
    /// an internal retry fires this is the offset seed that actually
    /// produced the layout, so it may differ from [`Self::requested_seed`].
    pub seed: u64,
    /// The un-offset input seed the caller asked for (before any internal
    /// retry offset). Stable for a given request, so consumers can gate
    /// "do I already have this map?" on it without spuriously matching or
    /// mismatching when a retry shifted the effective seed.
    pub requested_seed: u64,
    /// Rotation (radians) applied during alignment.
    pub rotation: f32,
    /// Y-offset subtracted from every world-space coordinate after
    /// rotation to y-center the traversable map on the visible window.
    pub y_offset: f32,
}
