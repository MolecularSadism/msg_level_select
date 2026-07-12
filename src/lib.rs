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
//! fixed seed for run-to-run determinism, or leave the seed `None` to
//! draw entropy from the OS at app start. Each [`LevelMapConfig`] may
//! override the per-map seed; otherwise the spawn pulls a fresh sub-seed
//! from the resource.
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
pub mod spawn;
pub mod state;
pub mod visit;

pub use components::{LevelMap, MapEdge, MapNode, MapPath, Site, VoronoiCell, Waypoint};
pub use config::{DesiredTraversals, LevelMapConfig, LevelMapPolicy};
pub use generation::{Generated, GenerationError};
pub use relationships::{EdgePaths, IncomingPaths, OutgoingPaths, PathEdges, PathFrom, PathTo};
pub use spawn::LevelMapCommands;
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
/// installs the [`VisitLocation`] observer, and provisions the
/// [`LevelMapRng`] resource used to seed map generation.
#[derive(Default)]
pub struct LevelSelectPlugin {
    /// Master seed for the [`LevelMapRng`] resource. `None` draws OS
    /// entropy at plugin build time — pick a fixed value here to make
    /// every run produce the same sequence of maps.
    pub seed: Option<u64>,
}

impl Plugin for LevelSelectPlugin {
    fn build(&self, app: &mut App) {
        let rng = match self.seed {
            Some(s) => LevelMapRng::from_seed(s),
            None => LevelMapRng::from_entropy(),
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
            .add_observer(visit::on_visit_location);
    }
}
