//! `VisitLocation` event and the observer that drives the traversal FSM.
//!
//! Traversal updates touch three layers in lock-step:
//! 1. The previously active node and the destination node.
//! 2. The connecting [`MapPath`], its [`MapEdge`] children, and the interior waypoint nodes those
//!    edges visit.
//! 3. Every outgoing path from the new active node — and that path's edges and waypoint nodes — so
//!    the reachable corridor lights up as one.
//!
//! All writes go through [`try_promote`], which respects the priority order
//! `Visited > Active > Available > Inactive`. A `Visited` entity is never
//! downgraded; the only transition out of `Visited` is back to `Active`
//! (revisit). This avoids the flicker where a path entity races between
//! `Available` and `Visited` based on iteration order, and it stops outgoing
//! `Available` writes from clobbering edges that are already `Visited` from a
//! prior traversal.

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
    //    been traversed stay Visited and the new active node stays Active.
    if let Ok(outgoing) = q_outgoing.get(d.new_active) {
        for path in outgoing.iter() {
            try_promote(commands, path, q_state, LocationState::Available);
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
}

/// Promote every edge of `path` and every node those edges touch to
/// `target`, deferring to [`try_promote`] for priority discipline.
fn propagate_corridor(
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

/// Issue a [`StateChangeRequest`] only when `target` is a *promotion* over
/// the entity's current state. Priority order is
/// `Visited > Active > Available > Inactive`, with the single special case
/// that `Visited -> Active` is allowed (revisit, gated upstream by
/// [`LevelMapPolicy::allow_revisit`]).
///
/// Entities without a [`LocationState`] component (dead Voronoi cells, dead
/// adjacency edges) are silently skipped — they are not part of any
/// traversable corridor.
fn try_promote(
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
