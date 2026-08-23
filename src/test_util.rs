//! Shared scaffolding for the traversal test suites: hand-built minimal
//! maps (root, nodes, edges, paths) and an app that runs the real plugin
//! so state changes settle through the actual FSM.

use bevy::prelude::*;

use crate::LevelSelectPlugin;
use crate::components::{LevelMap, MapEdge, MapNode, MapPath, Site};
use crate::config::LevelMapPolicy;
use crate::relationships::{PathEdges, PathFrom, PathTo};
use crate::state::LocationState;

/// Root carrying a strict traversal policy: no teleport, no revisit,
/// path visits allowed.
pub(crate) fn spawn_root(world: &mut World) -> Entity {
    world
        .spawn((
            LevelMap {
                size: Vec2::splat(1000.0),
                seed: 0,
                requested_seed: 0,
                rotation: 0.0,
                y_offset: 0.0,
            },
            LevelMapPolicy {
                allow_revisit: false,
                allow_teleport: false,
                allow_path_visit: true,
            },
        ))
        .id()
}

pub(crate) fn node(
    world: &mut World,
    root: Entity,
    belt: u32,
    site: u32,
    state: LocationState,
) -> Entity {
    world
        .spawn((MapNode, Site { belt, site }, state, ChildOf(root)))
        .id()
}

pub(crate) fn edge(
    world: &mut World,
    root: Entity,
    from: Entity,
    to: Entity,
    state: LocationState,
) -> Entity {
    world
        .spawn((
            MapEdge {
                from,
                to,
                wall: [Vec2::ZERO, Vec2::ZERO],
            },
            state,
            ChildOf(root),
        ))
        .id()
}

pub(crate) fn path(
    world: &mut World,
    root: Entity,
    from: Entity,
    to: Entity,
    edges: Vec<Entity>,
    state: LocationState,
) -> Entity {
    world
        .spawn((
            MapPath,
            PathFrom(from),
            PathTo(to),
            PathEdges::new(edges),
            state,
            ChildOf(root),
        ))
        .id()
}

pub(crate) fn state_of(world: &mut World, entity: Entity) -> LocationState {
    *world
        .get::<LocationState>(entity)
        .expect("entity has LocationState")
}

/// Let the deferred `StateChangeRequest`s issued by the observers settle
/// through the FSM.
pub(crate) fn settle(app: &mut App) {
    for _ in 0..8 {
        app.update();
    }
}

pub(crate) fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(LevelSelectPlugin { seed: Some(0) });
    app
}
