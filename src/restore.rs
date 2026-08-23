//! Rebuild traversal state on an already-spawned map from a visited-site
//! history.
//!
//! A consumer that persists which sites a run has completed can hand that
//! history back as a [`RestoreTraversal`] event on each map open and get the
//! live traversal state back: intermediate completed sites `Visited`, the
//! current position `Active` with its outgoing corridor `Available`, and the
//! stale sibling corridors the spawn pipeline lit from the entry node
//! retracted to `Inactive` — mirroring the retraction the live
//! [`VisitLocation`](crate::VisitLocation) observer performs on each hop.
//!
//! Restoration writes through the same [`try_promote`]/[`try_retract`]
//! priority discipline as live traversal rather than replaying
//! [`VisitLocation`](crate::VisitLocation) hop by hop — the spawn pipeline
//! has already promoted the entry node to `Active`, so a replayed first hop
//! would either be a no-op or get rejected by the teleport gate.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::components::{MapEdge, MapNode, Site};
use crate::relationships::{OutgoingPaths, PathEdges};
use crate::state::LocationState;
use crate::visit::{collect_corridor, propagate_corridor, retract_corridor, try_promote};

/// Request to rebuild traversal state on the map under `root` from a set of
/// completed site keys.
///
/// `completed` holds the `(belt, site)` key of every site the run has
/// completed, in any order. The *current* position is the highest key (the
/// site most recently traveled to); with no completed sites the run is still
/// at the entry node `(0, 0)` and the spawn-time state is left untouched.
///
/// Restoration:
/// 1. demotes the entry node to `Visited`,
/// 2. marks every completed site short of the current position `Visited`,
/// 3. promotes the current position to `Active` and lights its outgoing
///    paths, their edges, and their waypoints `Available`,
/// 4. retracts the corridors leaving every passed site (the entry node and
///    each intermediate completed site) that the current position does not
///    re-light, so only the current node's successors stay `Available` —
///    sibling retraction, exactly as a live hop performs it.
#[derive(Event, Debug, Clone)]
pub struct RestoreTraversal {
    /// The [`LevelMap`](crate::LevelMap) root whose nodes to restore.
    pub root: Entity,
    /// `(belt, site)` keys of every completed site, in any order.
    pub completed: Vec<(u32, u32)>,
}

/// The `(belt, site)` key of the map node a run currently occupies: the
/// highest completed key (the site most recently traveled to), or the entry
/// node `(0, 0)` when nothing has been completed yet.
#[must_use]
pub fn current_site_key(completed: &[(u32, u32)]) -> (u32, u32) {
    completed.iter().max().copied().unwrap_or((0, 0))
}

pub(crate) fn on_restore_traversal(
    trigger: On<RestoreTraversal>,
    mut commands: Commands,
    q_sites: Query<(Entity, &Site, &ChildOf), With<MapNode>>,
    q_outgoing: Query<&OutgoingPaths>,
    q_path_edges: Query<&PathEdges>,
    q_map_edge: Query<&MapEdge>,
    q_state: Query<&LocationState>,
) {
    let event = trigger.event();

    let site_map: HashMap<(u32, u32), Entity> = q_sites
        .iter()
        .filter(|(_, _, child_of)| child_of.parent() == event.root)
        .map(|(e, s, _)| ((s.belt, s.site), e))
        .collect();

    let Some(&entry_entity) = site_map.get(&(0, 0)) else {
        warn!(
            "RestoreTraversal: map {:?} has no entry node (0, 0); ignoring",
            event.root
        );
        return;
    };

    // Sort and dedup so completed sites are processed in graph order and the
    // current position is the last entry.
    let mut completed = event.completed.clone();
    completed.sort_unstable();
    completed.dedup();

    let current_entity = site_map
        .get(&current_site_key(&completed))
        .copied()
        .unwrap_or(entry_entity);

    // Spawn already promoted entry -> Active and its outgoing corridor ->
    // Available. If the run is still at entry, that's the state we want.
    if current_entity == entry_entity {
        return;
    }

    // Demote the entry from Active to Visited.
    try_promote(
        &mut commands,
        entry_entity,
        &q_state,
        LocationState::Visited,
    );

    // Every completed site short of the current position is Visited.
    for &(belt, site) in completed.iter().take(completed.len().saturating_sub(1)) {
        if let Some(&entity) = site_map.get(&(belt, site)) {
            try_promote(&mut commands, entity, &q_state, LocationState::Visited);
        }
    }

    // Current position becomes Active.
    try_promote(
        &mut commands,
        current_entity,
        &q_state,
        LocationState::Active,
    );

    // Light up the corridor leaving the current position so the player can
    // pick a successor. Every entity lit here is recorded in `forward` so the
    // retraction below never dims a corridor the current node re-lights (a
    // shared edge or waypoint).
    let mut forward: HashSet<Entity> = HashSet::new();
    forward.insert(current_entity);
    if let Ok(outgoing) = q_outgoing.get(current_entity) {
        for path in outgoing.iter() {
            forward.insert(path);
            try_promote(&mut commands, path, &q_state, LocationState::Available);
            collect_corridor(path, &q_path_edges, &q_map_edge, &mut forward);
            propagate_corridor(
                &mut commands,
                path,
                &q_path_edges,
                &q_map_edge,
                &q_state,
                LocationState::Available,
            );
        }
    }

    // Retract the corridors leaving every passed site — the entry node and
    // each completed site short of the current position. The spawn pipeline
    // lit the entry's outgoing paths `Available`; those siblings must fall
    // back to `Inactive` now that the run has moved on, or they stay
    // reachable and highlighted. Only `Available` entities outside `forward`
    // are retracted, so `Visited` progress and the current node's own
    // successors are left intact.
    let passed = std::iter::once(entry_entity).chain(
        completed
            .iter()
            .take(completed.len().saturating_sub(1))
            .filter_map(|&(belt, site)| site_map.get(&(belt, site)).copied()),
    );
    for node in passed {
        let Ok(outgoing) = q_outgoing.get(node) else {
            continue;
        };
        for path in outgoing.iter() {
            retract_corridor(
                &mut commands,
                path,
                &q_path_edges,
                &q_map_edge,
                &q_state,
                &forward,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LevelSelectPlugin;
    use crate::components::{LevelMap, MapPath};
    use crate::config::LevelMapPolicy;
    use crate::relationships::{PathFrom, PathTo};

    #[test]
    fn defaults_to_entry_node_when_no_sites_completed() {
        assert_eq!(current_site_key(&[]), (0, 0));
    }

    #[test]
    fn picks_highest_belt_completed_site() {
        assert_eq!(current_site_key(&[(0, 0), (1, 2), (2, 0)]), (2, 0));
        // Within a belt, the higher site index wins.
        assert_eq!(current_site_key(&[(1, 0), (1, 2)]), (1, 2));
    }

    fn spawn_root(world: &mut World) -> Entity {
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

    fn node(world: &mut World, root: Entity, belt: u32, site: u32, state: LocationState) -> Entity {
        world
            .spawn((MapNode, Site { belt, site }, state, ChildOf(root)))
            .id()
    }

    fn edge(
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

    fn path(
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

    fn state_of(world: &mut World, entity: Entity) -> LocationState {
        *world
            .get::<LocationState>(entity)
            .expect("entity has LocationState")
    }

    fn settle(app: &mut App) {
        for _ in 0..8 {
            app.update();
        }
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(LevelSelectPlugin { seed: Some(0) });
        app
    }

    /// Reconstructing a map where the run has moved to a belt-1 site retracts
    /// the entry node's other outgoing corridors back to `Inactive`, while
    /// the current node's own successors light up `Available`.
    #[test]
    fn reconstruction_retracts_entry_siblings() {
        let mut app = test_app();
        let world = app.world_mut();

        let root = spawn_root(world);

        // Spawn-time state: entry Active with both belt-1 corridors lit.
        let entry = node(world, root, 0, 0, LocationState::Active);
        let b = node(world, root, 1, 0, LocationState::Available);
        let c = node(world, root, 1, 1, LocationState::Available);
        // B's own successor, dark until B is reached.
        let d = node(world, root, 2, 0, LocationState::Inactive);

        let edge_eb = edge(world, root, entry, b, LocationState::Available);
        let edge_ec = edge(world, root, entry, c, LocationState::Available);
        let edge_bd = edge(world, root, b, d, LocationState::Inactive);

        let path_eb = path(
            world,
            root,
            entry,
            b,
            vec![edge_eb],
            LocationState::Available,
        );
        let path_ec = path(
            world,
            root,
            entry,
            c,
            vec![edge_ec],
            LocationState::Available,
        );
        let path_bd = path(world, root, b, d, vec![edge_bd], LocationState::Inactive);

        // History says the run has traveled to belt-1 site 0 (node B).
        world.trigger(RestoreTraversal {
            root,
            completed: vec![(1, 0)],
        });
        settle(&mut app);
        let world = app.world_mut();

        // Entry retired, current node active.
        assert_eq!(state_of(world, entry), LocationState::Visited);
        assert_eq!(state_of(world, b), LocationState::Active);

        // The sibling corridor the run never took is fully retracted.
        assert_eq!(state_of(world, c), LocationState::Inactive);
        assert_eq!(state_of(world, path_ec), LocationState::Inactive);
        assert_eq!(state_of(world, edge_ec), LocationState::Inactive);

        // The current node's forward corridor is reachable.
        assert_eq!(state_of(world, d), LocationState::Available);
        assert_eq!(state_of(world, path_bd), LocationState::Available);
        assert_eq!(state_of(world, edge_bd), LocationState::Available);

        // The entry corridor the run did traverse is no longer left dangling
        // as a reachable option either.
        assert_eq!(state_of(world, path_eb), LocationState::Inactive);
    }

    /// Intermediate completed sites become `Visited`, and only the current
    /// position's successors stay lit.
    #[test]
    fn reconstruction_marks_intermediates_visited() {
        let mut app = test_app();
        let world = app.world_mut();

        let root = spawn_root(world);
        let entry = node(world, root, 0, 0, LocationState::Active);
        let b = node(world, root, 1, 0, LocationState::Available);
        let c = node(world, root, 2, 0, LocationState::Inactive);
        let d = node(world, root, 3, 0, LocationState::Inactive);

        let edge_eb = edge(world, root, entry, b, LocationState::Available);
        let edge_bc = edge(world, root, b, c, LocationState::Inactive);
        let edge_cd = edge(world, root, c, d, LocationState::Inactive);

        let _path_eb = path(
            world,
            root,
            entry,
            b,
            vec![edge_eb],
            LocationState::Available,
        );
        let _path_bc = path(world, root, b, c, vec![edge_bc], LocationState::Inactive);
        let path_cd = path(world, root, c, d, vec![edge_cd], LocationState::Inactive);

        world.trigger(RestoreTraversal {
            root,
            completed: vec![(1, 0), (2, 0)],
        });
        settle(&mut app);
        let world = app.world_mut();

        assert_eq!(state_of(world, entry), LocationState::Visited);
        assert_eq!(state_of(world, b), LocationState::Visited);
        assert_eq!(state_of(world, c), LocationState::Active);
        assert_eq!(state_of(world, d), LocationState::Available);
        assert_eq!(state_of(world, path_cd), LocationState::Available);
    }

    /// An empty history leaves the spawn-time state untouched: the entry is
    /// still `Active` and its corridor still `Available`.
    #[test]
    fn empty_history_leaves_spawn_state_alone() {
        let mut app = test_app();
        let world = app.world_mut();

        let root = spawn_root(world);
        let entry = node(world, root, 0, 0, LocationState::Active);
        let b = node(world, root, 1, 0, LocationState::Available);
        let edge_eb = edge(world, root, entry, b, LocationState::Available);
        let path_eb = path(
            world,
            root,
            entry,
            b,
            vec![edge_eb],
            LocationState::Available,
        );

        world.trigger(RestoreTraversal {
            root,
            completed: vec![],
        });
        settle(&mut app);
        let world = app.world_mut();

        assert_eq!(state_of(world, entry), LocationState::Active);
        assert_eq!(state_of(world, b), LocationState::Available);
        assert_eq!(state_of(world, path_eb), LocationState::Available);
    }
}
