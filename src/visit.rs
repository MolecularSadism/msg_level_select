//! `VisitLocation` event and the observer that drives the traversal FSM.
//!
//! Traversal updates touch four layers in lock-step:
//! 1. The previously active node and the destination node.
//! 2. The connecting [`MapPath`], its [`MapEdge`] children, and the interior waypoint nodes those
//!    edges visit.
//! 3. The previous node's *non-chosen* outgoing corridors, retracted back to `Inactive` so the
//!    siblings the player passed over stop looking reachable.
//! 4. Every outgoing path from the new active node — and that path's edges and waypoint nodes — so
//!    the reachable corridor lights up as one.
//!
//! Promotions go through [`try_promote`], which respects the priority order
//! `Visited > Active > Available > Inactive`. A `Visited` entity is never
//! downgraded; the only transition out of `Visited` is back to `Active`
//! (revisit). This avoids the flicker where a path entity races between
//! `Available` and `Visited` based on iteration order, and it stops outgoing
//! `Available` writes from clobbering edges that are already `Visited` from a
//! prior traversal.
//!
//! Retraction goes through [`try_retract`], the one downward move: an
//! `Available` corridor leaving the previous node falls back to `Inactive`.
//! `Visited` and `Active` entities are left alone, and anything the new active
//! node re-lights forward (a shared edge or waypoint) is excluded from
//! retraction so it stays `Available`.

use std::collections::HashSet;

use bevy::prelude::*;
use bevy_fsm::StateChangeRequest;

use crate::components::{LevelMap, MapEdge, MapNode, MapPath};
use crate::config::LevelMapPolicy;
use crate::relationships::{IncomingPaths, OutgoingPaths, PathEdges, PathFrom, PathTo};
use crate::state::LocationState;

// The observer walks the path-edge chain via [`PathEdges`] (many-to-many),
// so shared Voronoi adjacencies propagate for every path they belong to —
// not just whichever route happened to be inserted last.

/// Request to move to a node OR enter a path.
///
/// Targets:
/// - A [`MapNode`]: the player jumps to that node. If a connecting [`MapPath`] exists, it is also
///   marked as `Visited`.
/// - A [`MapPath`]: the player enters the corridor. The path becomes `Visited` and its destination
///   becomes `Active`.
#[derive(Event, Debug, Clone, Copy)]
pub struct VisitLocation {
    pub target: Entity,
}

/// Internal state we collect before issuing FSM transitions, so we don't
/// borrow conflict on `Commands` + queries during the resolution step.
struct Decision {
    /// Node currently `Active` (will transition to `Visited` on success).
    previous_active: Option<Entity>,
    /// Node we are moving to (will transition to `Active`).
    new_active: Entity,
    /// Path between previous_active and new_active, if found.
    connecting_path: Option<Entity>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn on_visit_location(
    trigger: On<VisitLocation>,
    mut commands: Commands,
    q_node: Query<(), With<MapNode>>,
    q_path: Query<(), With<MapPath>>,
    q_path_endpoints: Query<(&PathFrom, &PathTo)>,
    q_path_edges: Query<&PathEdges>,
    q_outgoing: Query<&OutgoingPaths>,
    q_incoming: Query<&IncomingPaths>,
    q_state: Query<&LocationState>,
    q_map_edge: Query<&MapEdge>,
    q_child_of: Query<&ChildOf>,
    q_active_nodes: Query<(Entity, &LocationState, &ChildOf), With<MapNode>>,
    q_policy: Query<&LevelMapPolicy, With<LevelMap>>,
) {
    let target = trigger.target;
    let Some(target_kind) = classify_target(target, &q_node, &q_path) else {
        warn!(
            "VisitLocation: target {:?} is neither a MapNode nor a MapPath; ignoring",
            target
        );
        return;
    };

    // Scope the visit to the LevelMap that owns `target`. Resolving via
    // ChildOf (rather than via the Active node) means the very first
    // visit — fired at spawn when no node is Active yet — still lands
    // on the right policy, and multiple maps in one world stay isolated.
    let root = q_child_of.get(target).ok().map(|c| c.parent());

    let policy = root
        .and_then(|r| q_policy.get(r).ok())
        .copied()
        .unwrap_or_default();

    // Path-visit gate.
    if matches!(target_kind, TargetKind::Path) && !policy.allow_path_visit {
        warn!(
            "VisitLocation rejected: path visit disabled (target {:?})",
            target
        );
        return;
    }

    let previous_active = q_active_nodes
        .iter()
        .find(|(_, state, child_of)| {
            **state == LocationState::Active && Some(child_of.parent()) == root
        })
        .map(|(node, _, _)| node);

    let (path, dest_node) = match target_kind {
        TargetKind::Node => (
            resolve_path_between(previous_active, target, &q_incoming, &q_path_endpoints),
            target,
        ),
        TargetKind::Path => {
            let Ok((path_from, path_to)) = q_path_endpoints.get(target) else {
                warn!("VisitLocation: path {:?} missing PathFrom/PathTo", target);
                return;
            };
            // Only honor the path as the connector if it actually leaves the
            // currently active node. Otherwise treat as teleport-to-dest.
            let connector = match previous_active {
                Some(prev) if path_from.0 == prev => Some(target),
                _ => None,
            };
            (connector, path_to.0)
        }
    };

    // Revisit gate.
    if let Ok(state) = q_state.get(dest_node)
        && *state == LocationState::Visited
        && !policy.allow_revisit
    {
        warn!(
            "VisitLocation rejected: revisit disabled (target {:?})",
            dest_node
        );
        return;
    }

    // Connection gate.
    let connecting = match path {
        Some(p) => Some(p),
        None => {
            if previous_active.is_some() && !policy.allow_teleport {
                warn!(
                    "VisitLocation rejected: teleport disabled and no path to {:?}",
                    dest_node
                );
                return;
            }
            if previous_active.is_some() {
                warn!(
                    "VisitLocation: no connection from {:?} to {:?}; teleporting",
                    previous_active, dest_node
                );
            }
            None
        }
    };

    apply_decision(
        Decision {
            previous_active,
            new_active: dest_node,
            connecting_path: connecting,
        },
        &mut commands,
        &q_path_edges,
        &q_outgoing,
        &q_state,
        &q_map_edge,
    );
}

enum TargetKind {
    Node,
    Path,
}

fn classify_target(
    target: Entity,
    q_node: &Query<(), With<MapNode>>,
    q_path: &Query<(), With<MapPath>>,
) -> Option<TargetKind> {
    if q_node.contains(target) {
        Some(TargetKind::Node)
    } else if q_path.contains(target) {
        Some(TargetKind::Path)
    } else {
        None
    }
}

/// Find an incoming path to `dest` that originates from `previous_active`.
/// Returns `None` if either input is missing or no such path exists — the
/// caller then treats the move as a teleport candidate.
fn resolve_path_between(
    previous_active: Option<Entity>,
    dest: Entity,
    q_incoming: &Query<&IncomingPaths>,
    q_path_endpoints: &Query<(&PathFrom, &PathTo)>,
) -> Option<Entity> {
    let prev = previous_active?;
    let incoming = q_incoming.get(dest).ok()?;
    incoming.iter().find(|p| {
        q_path_endpoints
            .get(*p)
            .map(|(from, _)| from.0 == prev)
            .unwrap_or(false)
    })
}

fn apply_decision(
    d: Decision,
    commands: &mut Commands,
    q_path_edges: &Query<&PathEdges>,
    q_outgoing: &Query<&OutgoingPaths>,
    q_state: &Query<&LocationState>,
    q_map_edge: &Query<&MapEdge>,
) {
    // 1. Previous active -> Visited. Step 3 will then promote the new active node, so this can fire
    //    before we touch the corridor.
    if let Some(prev) = d.previous_active {
        try_promote(commands, prev, q_state, LocationState::Visited);
    }

    // 2. Connecting path + every edge + every interior waypoint -> Visited. try_promote ensures the
    //    new_active endpoint isn't stomped by the edge-walk (it will get bumped to Active in step 3
    //    via the Visited -> Active revisit transition).
    if let Some(path) = d.connecting_path {
        try_promote(commands, path, q_state, LocationState::Visited);
        propagate_corridor(
            commands,
            path,
            q_path_edges,
            q_map_edge,
            q_state,
            LocationState::Visited,
        );
    }

    // 3. Destination -> Active. Runs after step 2 so the revisit transition (Visited -> Active)
    //    handles the case where step 2's edge-walk promoted new_active to Visited as a corridor
    //    endpoint.
    try_promote(commands, d.new_active, q_state, LocationState::Active);

    // 4. Outgoing paths from new_active -> Available, plus their edges and interior waypoints.
    //    try_promote refuses to demote Visited or Active entities, so corridors that have already
    //    been traversed stay Visited and the new active node stays Active. Collect every entity we
    //    light forward so step 5's retraction can skip corridors shared with the new node's own
    //    forward reach (an edge or waypoint reused by more than one route).
    let mut forward: HashSet<Entity> = HashSet::new();
    if let Ok(outgoing) = q_outgoing.get(d.new_active) {
        for path in outgoing.iter() {
            forward.insert(path);
            try_promote(commands, path, q_state, LocationState::Available);
            collect_corridor(path, q_path_edges, q_map_edge, &mut forward);
            propagate_corridor(
                commands,
                path,
                q_path_edges,
                q_map_edge,
                q_state,
                LocationState::Available,
            );
        }
    }

    // 5. Retract the previous node's non-chosen outgoing corridors. The siblings the player passed
    //    over — the paths, edges, and destination nodes lit Available when they arrived at `prev` —
    //    fall back to Inactive now that a different route is committed. try_retract only touches
    //    Available entities, so the connecting corridor (now Visited) and the new active node are
    //    untouched; the `forward` set protects anything the new node re-lights.
    if let Some(prev) = d.previous_active
        && let Ok(outgoing) = q_outgoing.get(prev)
    {
        for path in outgoing.iter() {
            if Some(path) == d.connecting_path {
                continue;
            }
            retract_corridor(commands, path, q_path_edges, q_map_edge, q_state, &forward);
        }
    }
}

/// Promote every edge of `path` and every node those edges touch to
/// `target`, deferring to [`try_promote`] for priority discipline.
pub(crate) fn propagate_corridor(
    commands: &mut Commands,
    path: Entity,
    q_path_edges: &Query<&PathEdges>,
    q_map_edge: &Query<&MapEdge>,
    q_state: &Query<&LocationState>,
    target: LocationState,
) {
    let Ok(edges) = q_path_edges.get(path) else {
        return;
    };
    for edge in edges.iter() {
        try_promote(commands, edge, q_state, target);
        if let Ok(map_edge) = q_map_edge.get(edge) {
            try_promote(commands, map_edge.from, q_state, target);
            try_promote(commands, map_edge.to, q_state, target);
        }
    }
}

/// Retract `path` and every edge and node it touches back to `Inactive`,
/// deferring to [`try_retract`] for priority discipline. Entities in `keep`
/// (the new active node's forward reach) are skipped so a corridor shared
/// between the retracted route and the committed route stays `Available`.
pub(crate) fn retract_corridor(
    commands: &mut Commands,
    path: Entity,
    q_path_edges: &Query<&PathEdges>,
    q_map_edge: &Query<&MapEdge>,
    q_state: &Query<&LocationState>,
    keep: &HashSet<Entity>,
) {
    if !keep.contains(&path) {
        try_retract(commands, path, q_state);
    }
    let Ok(edges) = q_path_edges.get(path) else {
        return;
    };
    for edge in edges.iter() {
        if !keep.contains(&edge) {
            try_retract(commands, edge, q_state);
        }
        if let Ok(map_edge) = q_map_edge.get(edge) {
            for endpoint in [map_edge.from, map_edge.to] {
                if !keep.contains(&endpoint) {
                    try_retract(commands, endpoint, q_state);
                }
            }
        }
    }
}

/// Gather `path`, its edges, and the nodes those edges touch into `out`.
/// Mirrors the walk in [`propagate_corridor`] without issuing any state
/// change — used to record the corridors the new active node lights forward.
pub(crate) fn collect_corridor(
    path: Entity,
    q_path_edges: &Query<&PathEdges>,
    q_map_edge: &Query<&MapEdge>,
    out: &mut HashSet<Entity>,
) {
    out.insert(path);
    let Ok(edges) = q_path_edges.get(path) else {
        return;
    };
    for edge in edges.iter() {
        out.insert(edge);
        if let Ok(map_edge) = q_map_edge.get(edge) {
            out.insert(map_edge.from);
            out.insert(map_edge.to);
        }
    }
}

/// Issue a [`StateChangeRequest`] only when `target` is a *promotion* over
/// the entity's current state. Priority order is
/// `Visited > Active > Available > Inactive`, with the single special case
/// that `Visited -> Active` is allowed (revisit, gated upstream by
/// [`LevelMapPolicy::allow_revisit`]).
///
/// Entities without a [`LocationState`] component (dead Voronoi cells, dead
/// adjacency edges) are silently skipped — they are not part of any
/// traversable corridor.
pub(crate) fn try_promote(
    commands: &mut Commands,
    entity: Entity,
    q_state: &Query<&LocationState>,
    target: LocationState,
) {
    let Ok(&current) = q_state.get(entity) else {
        return;
    };
    use LocationState::*;
    let allowed = match (current, target) {
        // No-op: same state.
        (a, b) if a == b => false,
        // Visited is sticky — only revisit (-> Active) escapes.
        (Visited, Active) => true,
        (Visited, _) => false,
        // Active only steps forward to Visited.
        (Active, Visited) => true,
        (Active, _) => false,
        // Available climbs to Active or Visited; never demotes to Inactive.
        (Available, Active) => true,
        (Available, Visited) => true,
        (Available, _) => false,
        // Inactive can become anything.
        (Inactive, _) => true,
    };
    if allowed {
        commands.trigger(StateChangeRequest::<LocationState> {
            entity,
            next: target,
        });
    }
}

/// Retract an entity to `Inactive`, the one downward move in the priority
/// ladder. Only `Available` entities are demoted: `Visited` and `Active`
/// stay put (they carry real progress), and `Inactive` is already the floor.
/// Entities without a [`LocationState`] are silently skipped.
fn try_retract(commands: &mut Commands, entity: Entity, q_state: &Query<&LocationState>) {
    let Ok(&current) = q_state.get(entity) else {
        return;
    };
    if current == LocationState::Available {
        commands.trigger(StateChangeRequest::<LocationState> {
            entity,
            next: LocationState::Inactive,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{edge, node, path, settle, spawn_root, state_of, test_app};

    /// Committing to one successor retracts the previous node's other
    /// outgoing corridors to `Inactive`, while the chosen route becomes
    /// `Visited` and the new node's own forward corridor lights `Available`.
    #[test]
    fn visiting_a_sibling_retracts_the_other_outgoing_corridors() {
        let mut app = test_app();
        let world = app.world_mut();

        let root = spawn_root(world);
        // A is the active start; B and C are its two lit successors.
        let a = node(world, root, 0, 0, LocationState::Active);
        let b = node(world, root, 1, 0, LocationState::Available);
        let c = node(world, root, 1, 1, LocationState::Available);
        // D sits one belt beyond B and is dark until B is reached.
        let d = node(world, root, 2, 0, LocationState::Inactive);

        let edge_ab = edge(world, root, a, b, LocationState::Available);
        let edge_ac = edge(world, root, a, c, LocationState::Available);
        let edge_bd = edge(world, root, b, d, LocationState::Inactive);

        let path_ab = path(world, root, a, b, vec![edge_ab], LocationState::Available);
        let path_ac = path(world, root, a, c, vec![edge_ac], LocationState::Available);
        let path_bd = path(world, root, b, d, vec![edge_bd], LocationState::Inactive);

        world.trigger(VisitLocation { target: b });
        settle(&mut app);
        let world = app.world_mut();

        // Chosen route recorded as visited; new position active.
        assert_eq!(state_of(world, a), LocationState::Visited);
        assert_eq!(state_of(world, path_ab), LocationState::Visited);
        assert_eq!(state_of(world, edge_ab), LocationState::Visited);
        assert_eq!(state_of(world, b), LocationState::Active);

        // The unchosen sibling corridor is fully retracted.
        assert_eq!(state_of(world, c), LocationState::Inactive);
        assert_eq!(state_of(world, path_ac), LocationState::Inactive);
        assert_eq!(state_of(world, edge_ac), LocationState::Inactive);

        // The new node's forward corridor is now reachable.
        assert_eq!(state_of(world, d), LocationState::Available);
        assert_eq!(state_of(world, path_bd), LocationState::Available);
        assert_eq!(state_of(world, edge_bd), LocationState::Available);
    }

    /// An edge (or node) shared between a retracted sibling route and the new
    /// node's forward reach stays `Available` — retraction never dims a
    /// corridor the committed position re-lights.
    #[test]
    fn retraction_preserves_edges_shared_with_the_forward_reach() {
        let mut app = test_app();
        let world = app.world_mut();

        let root = spawn_root(world);
        let a = node(world, root, 0, 0, LocationState::Active);
        let b = node(world, root, 1, 0, LocationState::Available);
        let c = node(world, root, 1, 1, LocationState::Available);

        // `shared` is walked by both A->C (a sibling of the chosen A->B) and
        // B->C (B's forward route), so committing to B must keep it lit.
        let shared = edge(world, root, b, c, LocationState::Available);
        let edge_ab = edge(world, root, a, b, LocationState::Available);

        let path_ab = path(world, root, a, b, vec![edge_ab], LocationState::Available);
        let path_ac = path(world, root, a, c, vec![shared], LocationState::Available);
        let path_bc = path(world, root, b, c, vec![shared], LocationState::Inactive);

        world.trigger(VisitLocation { target: b });
        settle(&mut app);
        let world = app.world_mut();

        assert_eq!(state_of(world, b), LocationState::Active);
        // The chosen connecting corridor is visited...
        assert_eq!(state_of(world, path_ab), LocationState::Visited);
        // ...the sibling path entity itself retracts...
        assert_eq!(state_of(world, path_ac), LocationState::Inactive);
        // ...but the shared edge and its endpoint stay Available because B
        // re-lights them forward, and B's forward path is now Available.
        assert_eq!(state_of(world, shared), LocationState::Available);
        assert_eq!(state_of(world, c), LocationState::Available);
        assert_eq!(state_of(world, path_bc), LocationState::Available);
    }
}
