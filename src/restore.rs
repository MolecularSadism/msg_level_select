//! Rebuild traversal state on an already-spawned map from a visited-site
//! history.
//!
//! A consumer that persists which sites a run has completed can hand that
//! history back as a [`RestoreTraversal`] event on each map open.
//! Restoration reproduces the run's *positional* state: completed site
//! nodes `Visited`, the current position `Active` with its outgoing
//! corridor lit `Available`, and the stale sibling corridors the spawn
//! pipeline lit from the entry node retracted to `Inactive` — mirroring
//! the retraction the live [`VisitLocation`](crate::VisitLocation)
//! observer performs on each hop.
//!
//! What restoration does *not* reproduce is the corridor trail: the paths,
//! edges, and interior waypoints the run traveled through are retracted
//! (or left at their spawn-time `Inactive`) along with the unchosen
//! siblings, not promoted to `Visited`. A consumer that renders `Visited`
//! corridors as a breadcrumb trail will not see that trail after a
//! restore — only the completed site nodes carry `Visited`.
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
/// completed, in any order. `current` names the site the run occupies; when
/// `None`, it is inferred as the highest completed key via
/// [`current_site_key`]. With no completed sites and no explicit `current`
/// the run is still at the entry node `(0, 0)` and the spawn-time state is
/// left untouched.
///
/// Restoration:
/// 1. demotes the entry node to `Visited` (unless it is the current
///    position),
/// 2. marks every completed site other than the current position `Visited`,
/// 3. promotes the current position to `Active` and lights its outgoing
///    paths, their edges, and their waypoints `Available`,
/// 4. retracts the corridors leaving every passed site (the entry node and
///    each other completed site) that the current position does not
///    re-light, so only the current node's successors stay `Available` —
///    sibling retraction, exactly as a live hop performs it.
///
/// A `current` key with no matching node on the map discards the whole
/// restore with a `warn!`; a completed key with no matching node is skipped
/// with a `warn!` and the rest of the restore proceeds.
#[derive(Event, Debug, Clone)]
pub struct RestoreTraversal {
    /// The [`LevelMap`](crate::LevelMap) root whose nodes to restore.
    pub root: Entity,
    /// `(belt, site)` keys of every completed site, in any order.
    pub completed: Vec<(u32, u32)>,
    /// `(belt, site)` key of the site the run currently occupies. `None`
    /// infers the highest completed key ([`current_site_key`]), which is
    /// only correct for monotone forward histories — a run that teleported
    /// or revisited backward (see
    /// [`LevelMapPolicy`](crate::LevelMapPolicy)) must name its position
    /// explicitly.
    pub current: Option<(u32, u32)>,
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

    // Sort and dedup so completed sites are processed in graph order.
    let mut completed = event.completed.clone();
    completed.sort_unstable();
    completed.dedup();

    let current_key = event
        .current
        .unwrap_or_else(|| current_site_key(&completed));
    let Some(&current_entity) = site_map.get(&current_key) else {
        warn!(
            "RestoreTraversal: map {:?} has no node at current position {:?}; ignoring",
            event.root, current_key
        );
        return;
    };

    // Every completed site other than the current position (and the entry
    // node, demoted explicitly below) becomes Visited.
    let mut passed_sites: Vec<Entity> = Vec::new();
    for &key in completed
        .iter()
        .filter(|&&key| key != current_key && key != (0, 0))
    {
        match site_map.get(&key) {
            Some(&entity) => passed_sites.push(entity),
            None => warn!(
                "RestoreTraversal: map {:?} has no node at completed site {:?}; skipping",
                event.root, key
            ),
        }
    }

    // Spawn already promoted entry -> Active and its outgoing corridor ->
    // Available. If the run is still at entry with no further history,
    // that's the state we want.
    if current_entity == entry_entity && passed_sites.is_empty() {
        return;
    }

    // Demote the entry from Active to Visited.
    if current_entity != entry_entity {
        try_promote(
            &mut commands,
            entry_entity,
            &q_state,
            LocationState::Visited,
        );
    }

    for &entity in &passed_sites {
        try_promote(&mut commands, entity, &q_state, LocationState::Visited);
    }

    // Current position becomes Active.
    try_promote(
        &mut commands,
        current_entity,
        &q_state,
        LocationState::Active,
    );

    // Light up the corridor leaving the current position so the player can
    // pick a successor. Every entity lit here — and every node this restore
    // just promoted — is recorded in `forward` so the retraction below never
    // dims a corridor the current node re-lights (a shared edge or waypoint)
    // and never targets a node whose promotion is still in flight.
    let mut forward: HashSet<Entity> = HashSet::new();
    forward.insert(entry_entity);
    forward.insert(current_entity);
    forward.extend(passed_sites.iter().copied());
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
    // each completed site other than the current position. The spawn
    // pipeline lit the entry's outgoing paths `Available`; those siblings
    // must fall back to `Inactive` now that the run has moved on, or they
    // stay reachable and highlighted. Only `Available` entities outside
    // `forward` are retracted, so `Visited` progress and the current node's
    // own successors are left intact.
    let passed = std::iter::once(entry_entity)
        .filter(|&entity| entity != current_entity)
        .chain(passed_sites.iter().copied());
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
    use crate::test_util::{edge, node, path, settle, spawn_root, state_of, test_app};

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
            current: None,
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
            current: None,
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
            current: None,
        });
        settle(&mut app);
        let world = app.world_mut();

        assert_eq!(state_of(world, entry), LocationState::Active);
        assert_eq!(state_of(world, b), LocationState::Available);
        assert_eq!(state_of(world, path_eb), LocationState::Available);
    }

    /// An explicit `current` below the highest completed key restores the
    /// run there: the higher completed site is `Visited`, not `Active`, and
    /// only the current node's corridor is lit.
    #[test]
    fn explicit_current_overrides_the_inferred_max_key() {
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

        let path_eb = path(
            world,
            root,
            entry,
            b,
            vec![edge_eb],
            LocationState::Available,
        );
        let path_bc = path(world, root, b, c, vec![edge_bc], LocationState::Inactive);
        let path_cd = path(world, root, c, d, vec![edge_cd], LocationState::Inactive);

        // The run completed (2, 0) but has since moved back to (1, 0).
        world.trigger(RestoreTraversal {
            root,
            completed: vec![(1, 0), (2, 0)],
            current: Some((1, 0)),
        });
        settle(&mut app);
        let world = app.world_mut();

        assert_eq!(state_of(world, entry), LocationState::Visited);
        assert_eq!(state_of(world, b), LocationState::Active);
        assert_eq!(state_of(world, c), LocationState::Visited);

        // Only the current node's corridor lights up; the visited node's own
        // successors stay dark, and the traveled entry corridor retracts.
        assert_eq!(state_of(world, path_bc), LocationState::Available);
        assert_eq!(state_of(world, edge_bc), LocationState::Available);
        assert_eq!(state_of(world, d), LocationState::Inactive);
        assert_eq!(state_of(world, path_cd), LocationState::Inactive);
        assert_eq!(state_of(world, path_eb), LocationState::Inactive);
    }

    /// An edge shared between a retracted entry sibling and the current
    /// node's forward reach stays `Available` — the restore analogue of the
    /// live observer's forward-set protection.
    #[test]
    fn reconstruction_preserves_edges_shared_with_the_forward_reach() {
        let mut app = test_app();
        let world = app.world_mut();

        let root = spawn_root(world);
        let entry = node(world, root, 0, 0, LocationState::Active);
        let b = node(world, root, 1, 0, LocationState::Available);
        let c = node(world, root, 1, 1, LocationState::Available);

        // `shared` is walked by both entry->C (a sibling of the traveled
        // entry->B) and B->C (B's forward route), so restoring onto B must
        // keep it lit.
        let shared = edge(world, root, b, c, LocationState::Available);
        let edge_eb = edge(world, root, entry, b, LocationState::Available);

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
            vec![shared],
            LocationState::Available,
        );
        let path_bc = path(world, root, b, c, vec![shared], LocationState::Inactive);

        world.trigger(RestoreTraversal {
            root,
            completed: vec![(1, 0)],
            current: None,
        });
        settle(&mut app);
        let world = app.world_mut();

        assert_eq!(state_of(world, b), LocationState::Active);
        // The sibling path entity itself retracts...
        assert_eq!(state_of(world, path_ec), LocationState::Inactive);
        // ...but the shared edge and its endpoint stay Available because B
        // re-lights them forward, and B's forward path is now Available.
        assert_eq!(state_of(world, shared), LocationState::Available);
        assert_eq!(state_of(world, c), LocationState::Available);
        assert_eq!(state_of(world, path_bc), LocationState::Available);
        // The traveled entry corridor retracts as usual.
        assert_eq!(state_of(world, path_eb), LocationState::Inactive);
    }

    /// A history that explicitly lists the entry key `(0, 0)` restores
    /// identically to one that omits it.
    #[test]
    fn entry_key_in_completed_history_is_a_no_op() {
        let mut app = test_app();
        let world = app.world_mut();

        let root = spawn_root(world);
        let entry = node(world, root, 0, 0, LocationState::Active);
        let b = node(world, root, 1, 0, LocationState::Available);
        let c = node(world, root, 1, 1, LocationState::Available);

        let edge_eb = edge(world, root, entry, b, LocationState::Available);
        let edge_ec = edge(world, root, entry, c, LocationState::Available);

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

        world.trigger(RestoreTraversal {
            root,
            completed: vec![(0, 0), (1, 0)],
            current: None,
        });
        settle(&mut app);
        let world = app.world_mut();

        assert_eq!(state_of(world, entry), LocationState::Visited);
        assert_eq!(state_of(world, b), LocationState::Active);
        assert_eq!(state_of(world, c), LocationState::Inactive);
        assert_eq!(state_of(world, path_eb), LocationState::Inactive);
        assert_eq!(state_of(world, path_ec), LocationState::Inactive);
    }

    /// A `current` key with no matching node discards the whole restore:
    /// the spawn-time state is left untouched (and a warning is logged).
    #[test]
    fn unknown_current_key_discards_the_restore() {
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
            completed: vec![(1, 0)],
            current: Some((9, 9)),
        });
        settle(&mut app);
        let world = app.world_mut();

        assert_eq!(state_of(world, entry), LocationState::Active);
        assert_eq!(state_of(world, b), LocationState::Available);
        assert_eq!(state_of(world, path_eb), LocationState::Available);
    }
}
