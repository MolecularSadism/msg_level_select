//! `msg_level_select` — procedural FTL-style level map generator.
//!
//! Hand it a layout (e.g. `[1, 3, 1, 3, 3, 1]`) and it produces a
//! Voronoi-based map where every level reaches at least one downstream
//! successor. The crate spawns purely logical ECS entities tagged with
//! [`MapNode`], [`MapPath`], [`MapEdge`], and the [`LocationState`]
//! FSM; consumers attach their own `Sprite`, `Mesh2d`, `UiNode`, or
//! `Gizmos`-based rendering.
//!
//! # Seeding
//!
//! The plugin owns a master [`LevelMapRng`] resource. Add it with a
//! fixed seed for run-to-run determinism (leaving the seed `None` to
//! draw OS entropy is a `dev`-feature-only convenience for demos). Each [`LevelMapConfig`] may
//! override the per-map seed; otherwise the spawn pulls a fresh sub-seed
//! from the resource.
//!
//! # Restoring a run
//!
//! A consumer that persists which `(belt, site)` keys a run has completed
//! can trigger [`RestoreTraversal`] after respawning the map to rebuild
//! the traversal state — completed sites `Visited`, the current position
//! `Active` with its outgoing corridor lit. See the [`restore`] module
//! docs for exactly what is (and is not) reproduced.
//!
//! # Feature flags
//!
//! - `serde` — derives `Serialize`/`Deserialize` on [`LevelMapConfig`],
//!   [`DesiredTraversals`], and [`LevelMapPolicy`] (struct-level
//!   `#[serde(default)]` backed by the `Default` impls, so partial config
//!   files work), letting a consumer embed the config in its own files.
//! - `dev` — enables the `bevy-inspector-egui` dependency the interactive
//!   example requires, plus the entropy-seeded conveniences
//!   (`LevelMapRng::from_entropy` and the plugin's `Default` impl).
//!
//! # Quick start
//!
//! ```
//! use bevy::prelude::*;
//! use msg_level_select::{
//!     LevelMapCommands, LevelMapConfig, LevelMapRng, LevelSelectPlugin,
//! };
//!
//! fn spawn_map(mut commands: Commands, mut rng: ResMut<LevelMapRng>) {
//!     // `seed: None` pulls a fresh sub-seed from the plugin's RNG.
//!     // Pass `seed: Some(42)` to pin this individual map.
//!     let _ = commands.spawn_level_map(&mut rng, LevelMapConfig::default());
//! }
//!
//! let mut app = App::new();
//! app.add_plugins(MinimalPlugins);
//! app.add_plugins(LevelSelectPlugin { seed: Some(42) });
//! app.add_systems(Startup, spawn_map);
//! ```

pub mod components;
pub mod config;
pub mod generation;
pub mod relationships;
pub mod restore;
pub mod spawn;
pub mod state;
pub mod visit;

#[cfg(test)]
pub(crate) mod test_util;

pub use components::{LevelMap, MapEdge, MapNode, MapPath, Site, VoronoiCell, Waypoint};
pub use config::{DesiredTraversals, LevelMapConfig, LevelMapPolicy};
pub use generation::{Generated, GenerationError};
pub use relationships::{EdgePaths, IncomingPaths, OutgoingPaths, PathEdges, PathFrom, PathTo};
pub use restore::{RestoreTraversal, current_site_key};
pub use spawn::{LevelMapCommands, LevelMapSpawner, SpawnProgress};
pub use state::LocationState;
pub use visit::VisitLocation;

use bevy::prelude::*;
use bevy_fsm::FSMPlugin;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;

/// Master RNG used to derive per-map seeds when [`LevelMapConfig::seed`]
/// is `None`. Installed by [`LevelSelectPlugin`]; consumers may also
/// reseed it at runtime by writing a fresh value into the resource.
#[derive(Resource)]
pub struct LevelMapRng(pub StdRng);

impl LevelMapRng {
    /// Construct a new RNG from a deterministic seed.
    pub fn from_seed(seed: u64) -> Self {
        Self(StdRng::seed_from_u64(seed))
    }

    /// Construct a new RNG from OS entropy.
    ///
    /// **Determinism trap** — a map seeded this way is different every run,
    /// which is never what a consumer whose maps must be seed-stable wants.
    /// Only available with the `dev` feature, for demos and examples; use
    /// [`Self::from_seed`] everywhere else.
    #[cfg(feature = "dev")]
    pub fn from_entropy() -> Self {
        Self(StdRng::seed_from_u64(rand::rng().random()))
    }

    /// Draw a fresh sub-seed for one map generation. Advances the
    /// underlying stream.
    pub fn next_seed(&mut self) -> u64 {
        self.0.random()
    }
}

/// Adds the [`LocationState`] FSM, registers types for reflection,
/// installs the [`VisitLocation`] and [`RestoreTraversal`] observers, and
/// provisions the [`LevelMapRng`] resource used to seed map generation.
///
/// # Panics
///
/// Building the plugin with `seed: None` draws OS entropy, which is only
/// supported with the `dev` feature (demos and examples); without it,
/// [`Plugin::build`] panics. The `Default` impl (which leaves `seed`
/// `None`) is therefore only derived with the `dev` feature — release
/// consumers must construct the plugin with an explicit seed.
#[cfg_attr(feature = "dev", derive(Default))]
pub struct LevelSelectPlugin {
    /// Master seed for the [`LevelMapRng`] resource. Pick a fixed value
    /// to make every run produce the same sequence of maps. `None` draws
    /// OS entropy at plugin build time and is only supported with the
    /// `dev` feature (demos); without it, `None` panics at build.
    pub seed: Option<u64>,
}

impl Plugin for LevelSelectPlugin {
    fn build(&self, app: &mut App) {
        let rng = match self.seed {
            Some(s) => LevelMapRng::from_seed(s),
            #[cfg(feature = "dev")]
            None => LevelMapRng::from_entropy(),
            #[cfg(not(feature = "dev"))]
            None => panic!(
                "LevelSelectPlugin {{ seed: None }} draws OS entropy, which makes every run's \
                 maps different — a determinism trap. Pass a fixed seed, or enable the `dev` \
                 feature for demos."
            ),
        };
        app.insert_resource(rng)
            .add_plugins(FSMPlugin::<LocationState>::default())
            .register_type::<MapNode>()
            .register_type::<VoronoiCell>()
            .register_type::<Site>()
            .register_type::<Waypoint>()
            .register_type::<MapPath>()
            .register_type::<MapEdge>()
            .register_type::<LevelMap>()
            .register_type::<LevelMapPolicy>()
            .register_type::<LocationState>()
            .register_type::<PathEdges>()
            .register_type::<EdgePaths>()
            .register_type::<PathFrom>()
            .register_type::<OutgoingPaths>()
            .register_type::<PathTo>()
            .register_type::<IncomingPaths>()
            .add_observer(visit::on_visit_location)
            .add_observer(restore::on_restore_traversal);
    }
}
