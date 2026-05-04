//! Configuration for [`crate::LevelMapCommands::spawn_level_map`].

use bevy::prelude::*;

/// Generation + runtime configuration for one map.
#[derive(Clone, Debug)]
pub struct LevelMapConfig {
    /// `L(0)..L(S-1)`. The first value is the entry-level count, the
    /// last is the exit-level count. Must contain at least 2 entries.
    pub layout: Vec<u32>,
    /// Poisson-disc radius in world units. Smaller = denser, noisier
    /// cells. Larger = more regular cells.
    pub poisson_radius: f32,
    /// Extra stage columns of points on each side beyond the visible
    /// region so Voronoi clipping artifacts never reach the camera.
    pub stage_buffer: u32,
    /// `width / height` of the visible (non-buffered) rectangle.
    pub aspect_ratio: f32,
    /// Padding inside each chosen cell, expressed as a fraction of
    /// `poisson_radius`, that keeps the visitable point away from cell
    /// corners. `0.0` puts it at the cell center; `0.5` lets it wander
    /// across most of the inscribed disc.
    pub node_position_buffer: f32,
    /// Connectivity floor for [`crate::generation::selection::select`]:
    /// if no direct neighbor satisfies a stage jump, the selector may
    /// bridge with up to this many intermediate dead cells. Keep this as
    /// low as needed to ensure connectivity for tight layouts. When
    /// [`Self::desired_traversals`] is set, the selector silently raises
    /// its BFS budget above this value so the distance-bias has longer
    /// candidates to score against.
    pub allow_extra_traversal: u32,
    /// Optional: prefer farther stage-(N+1) target cells so corridors are
    /// naturally longer (rather than threading short-hop targets through
    /// detours).
    ///
    /// Drives two stages of the pipeline:
    /// 1. **Selection** ([`crate::generation::selection::select`]) — for each previous-stage cell,
    ///    samples a desired path length via [`DesiredTraversals::easing`] and biases target
    ///    picking toward candidates whose BFS-shortest path matches that length. Coverage of every
    ///    prev cell still dominates; distance is the next-strongest signal.
    /// 2. **Inflation** ([`crate::generation::waypoints::inflate`]) — runs after selection as a
    ///    per-corridor variance refiner, stretching individual paths that fell short of the
    ///    sampled target when the Voronoi graph permits it.
    pub desired_traversals: Option<DesiredTraversals>,
    /// Deterministic seed for this map. The same seed plus same config
    /// always produces the same map (modulo the internal retry chain).
    /// `None` pulls a fresh sub-seed from the plugin's
    /// [`crate::LevelMapRng`] resource at spawn time.
    pub seed: Option<u64>,
    /// Runtime policy. Stored on the [`crate::LevelMap`] root so
    /// consumers can mutate it after spawn without re-generating.
    pub policy: LevelMapPolicy,
    /// Maximum number of seeds the generator tries before giving up.
    /// Each attempt increments the seed by one, so a higher value trades
    /// CPU at spawn time for a better chance of satisfying tight layouts.
    pub max_attempts: u32,
}

impl Default for LevelMapConfig {
    fn default() -> Self {
        Self {
            layout: vec![1, 3, 1, 3, 3, 1],
            poisson_radius: 40.0,
            stage_buffer: 5,
            aspect_ratio: 16.0 / 9.0,
            node_position_buffer: 0.10,
            // 1 lets the BFS bridge the small gaps that appear when the
            // band width (~2.5 * poisson_radius) is just below twice a
            // cell radius — without it, a stage-i cell deep in its band
            // can have zero direct neighbors in the i+1 band.
            allow_extra_traversal: 1,
            desired_traversals: None,
            seed: None,
            policy: LevelMapPolicy::default(),
            max_attempts: 8,
        }
    }
}

/// Optional: inflate every path with extra waypoint cells.
#[derive(Clone, Debug)]
pub struct DesiredTraversals {
    /// Average number of intermediate waypoints per path.
    pub average: u8,
    /// Maps `t in [0,1]` to a multiplier in `[0,1]`. The actual count
    /// is `round(easing.sample(t) * 2 * average)`. The default uses an
    /// approximation of a Gaussian via a smooth-step S-curve.
    pub easing: EasingCurve<f32>,
}

impl Default for DesiredTraversals {
    fn default() -> Self {
        Self {
            average: 1,
            // SmoothStep approximates the integral of a Gaussian: sampling uniform
            // `t` and applying it concentrates outputs around 0.5.
            easing: EasingCurve::new(0.0, 1.0, EaseFunction::SmoothStep),
        }
    }
}

/// Runtime traversal policy. Lives on the [`crate::LevelMap`] root
/// entity; consumers may mutate it at any time and the next
/// [`crate::VisitLocation`] will see the new value.
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
pub struct LevelMapPolicy {
    /// When `false`, a node already in `Visited` may not transition
    /// back to `Active`; a clicked-revisit is rejected with a `warn!`.
    pub allow_revisit: bool,
    /// When `false`, a [`crate::VisitLocation`] targeting an
    /// unreachable node is rejected with a `warn!`. When `true`, the
    /// connection check is bypassed (the "cheat is OK" mode).
    pub allow_teleport: bool,
    /// When `false`, a [`crate::VisitLocation`] targeting a
    /// [`crate::MapPath`] is rejected with a `warn!`. Only
    /// [`crate::MapNode`] targets are accepted.
    pub allow_path_visit: bool,
}

impl Default for LevelMapPolicy {
    fn default() -> Self {
        Self {
            allow_revisit: false,
            allow_teleport: true,
            allow_path_visit: true,
        }
    }
}
