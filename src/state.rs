//! `LocationState` FSM driving traversal lifecycle for nodes, paths, and edges.

use bevy::prelude::*;
use bevy_fsm::{FSMState, FSMTransition};

/// Lifecycle of any visitable entity (node, path, edge).
///
/// Priority (high → low): `Visited` > `Active` > `Available` > `Inactive`.
/// Once `Visited`, an entity may only re-enter `Active` (a revisit, gated by
/// [`crate::LevelMapPolicy`]). It is never demoted to `Available` or
/// `Inactive`. The traversal observer uses a `try_promote` helper to enforce
/// this priority — the FSM transitions below are the underlying validation
/// layer that admits any move *up* the priority ladder (plus the revisit
/// escape from `Visited`).
///
/// Transitions allowed:
/// - `Inactive` -> `Available` | `Active` | `Visited`
/// - `Available` -> `Active` | `Visited`
/// - `Active`    -> `Visited`
/// - `Visited`   -> `Active` (revisit only)
#[derive(Component, FSMState, Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[reflect(Component)]
pub enum LocationState {
    #[default]
    Inactive,
    Available,
    Active,
    Visited,
}

impl FSMTransition for LocationState {
    fn can_transition(from: Self, to: Self) -> bool {
        use LocationState::*;
        from == to
            || matches!(
                (from, to),
                (Inactive, Available)
                    | (Inactive, Active)
                    | (Inactive, Visited)
                    | (Available, Active)
                    | (Available, Visited)
                    | (Active, Visited)
                    | (Visited, Active)
            )
    }
}
