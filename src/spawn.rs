//! Turn a [`crate::generation::Generated`] into ECS entities.
//!
//! Every spawned entity starts in [`LocationState::Inactive`]. A single
//! [`VisitLocation`] fired at the end of spawn targets the entry level;
//! the observer promotes it to `Active` and cascades `Available` down
//! its outgoing corridor. Keeping initial state out of the spawn
//! pipeline means there's exactly one code path for promotion, and
//! shared waypoints / edges can't be clobbered by last-write-wins
//! inserts.
//!
//! The spawn pipeline is a resumable state machine ([`LevelMapSpawner`])
//! so a large map can be materialized across several frames under a
//! per-call entity budget. The one-shot [`LevelMapCommands`] entry points
//! drive the same machine to completion in a single call.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::LevelMapRng;
use crate::components::{LevelMap, MapEdge, MapNode, MapPath, Site, VoronoiCell, Waypoint};
use crate::config::{LevelMapConfig, LevelMapPolicy};
use crate::generation::{self, Generated, GenerationError};
use crate::relationships::{EdgePaths, PathEdges, PathFrom, PathTo};
use crate::state::LocationState;
use crate::visit::VisitLocation;

/// Ergonomic command extension for spawning a map.
pub trait LevelMapCommands {
    /// Spawn a fully-wired level map. Returns the root entity (children
    /// are nodes, paths, and edges). On failure, no entities are
    /// created and `Err` is returned.
    ///
    /// `rng` is only consumed when [`LevelMapConfig::seed`] is `None`;
    /// pass it from a `ResMut<LevelMapRng>` system param. When `seed`
    /// is `Some`, the rng is left untouched.
    ///
    /// This is a thin wrapper over [`generation::generate`] followed by
    /// [`Self::spawn_generated_map`]. Split them when the (pure-CPU)
    /// generation should run off the main thread: generate on a task
    /// pool, hold the [`Generated`], then call `spawn_generated_map` on
    /// the main thread once the task completes.
    fn spawn_level_map(
        &mut self,
        rng: &mut LevelMapRng,
        cfg: LevelMapConfig,
    ) -> Result<Entity, GenerationError>;

    /// Spawn the ECS entities for an already-computed [`Generated`] in a
    /// single call. Returns the root entity (children are nodes, paths,
    /// and edges).
    ///
    /// Pairs with [`generation::generate`] to let the pure-CPU
    /// generation step run separately (e.g. on an
    /// `AsyncComputeTaskPool`) from the main-thread ECS spawn. To also
    /// spread the ECS spawn itself across frames, use
    /// [`LevelMapSpawner`] instead.
    fn spawn_generated_map(&mut self, cfg: &LevelMapConfig, generated: &Generated) -> Entity;
}

impl LevelMapCommands for Commands<'_, '_> {
    fn spawn_level_map(
        &mut self,
        rng: &mut LevelMapRng,
        cfg: LevelMapConfig,
    ) -> Result<Entity, GenerationError> {
        let seed = cfg.seed.unwrap_or_else(|| rng.next_seed());
        let generated = generation::generate(&cfg, seed)?;
        Ok(self.spawn_generated_map(&cfg, &generated))
    }

    fn spawn_generated_map(&mut self, cfg: &LevelMapConfig, generated: &Generated) -> Entity {
        let mut state = SpawnState::new();
        // usize::MAX budget: run the whole pipeline in one call.
        while !state.step(cfg.policy, generated, self, usize::MAX) {}
        state.root.expect("root is created on the first step")
    }
}

/// Progress reported by [`LevelMapSpawner::step`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpawnProgress {
    /// More entities remain; call [`LevelMapSpawner::step`] again next
    /// frame. Carries the root entity, which exists from the first step
    /// so callers can attach markers / visibility to it immediately.
    InProgress(Entity),
    /// Every entity is spawned and the entry visit has fired. Carries the
    /// root entity.
    Done(Entity),
}

impl SpawnProgress {
    /// The map root entity, valid in both variants.
    pub fn root(self) -> Entity {
        match self {
            SpawnProgress::InProgress(root) | SpawnProgress::Done(root) => root,
        }
    }

    /// `true` once the whole map has been spawned.
    pub fn is_done(self) -> bool {
        matches!(self, SpawnProgress::Done(_))
    }
}

/// A resumable spawner that materializes a [`Generated`] map across
/// multiple [`LevelMapSpawner::step`] calls under a per-call entity
/// budget.
///
/// Owns the [`Generated`] layout so it can be held in a `Component`
/// between frames. Drive it until [`LevelMapSpawner::step`] returns
/// [`SpawnProgress::Done`]:
///
/// ```
/// use bevy::prelude::*;
/// use msg_level_select::{
///     LevelMapConfig, LevelMapSpawner, SpawnProgress, generation,
/// };
///
/// #[derive(Component)]
/// struct PendingSpawn(LevelMapSpawner);
///
/// fn drive(mut commands: Commands, mut q: Query<(Entity, &mut PendingSpawn)>) {
///     for (task, mut pending) in &mut q {
///         if let SpawnProgress::Done(_root) = pending.0.step(&mut commands, 256) {
///             commands.entity(task).despawn();
///         }
///     }
/// }
///
/// let cfg = LevelMapConfig {
///     seed: Some(1),
///     ..LevelMapConfig::default()
/// };
/// // A finished layout would come from `generation::generate`; here we
/// // just show how the spawner is constructed and stored.
/// let _ = |generated: generation::Generated| {
///     PendingSpawn(LevelMapSpawner::new(&cfg, generated))
/// };
///
/// let mut app = App::new();
/// app.add_plugins(MinimalPlugins);
/// app.add_systems(Update, drive);
/// ```
pub struct LevelMapSpawner {
    policy: LevelMapPolicy,
    generated: Generated,
    state: SpawnState,
}

impl LevelMapSpawner {
    /// Create a spawner for an already-computed layout. No entities are
    /// created until the first [`LevelMapSpawner::step`].
    pub fn new(cfg: &LevelMapConfig, generated: Generated) -> Self {
        Self {
            policy: cfg.policy,
            generated,
            state: SpawnState::new(),
        }
    }

    /// Spawn up to `budget` entities this call (each spawned node, edge,
    /// path, and each many-to-many link insert counts as one). The root
    /// is created on the first call regardless of budget. Returns
    /// [`SpawnProgress::Done`] once the whole map — including the entry
    /// [`VisitLocation`] — has been spawned.
    pub fn step(&mut self, commands: &mut Commands, budget: usize) -> SpawnProgress {
        let done = self
            .state
            .step(self.policy, &self.generated, commands, budget);
        let root = self.state.root.expect("root is created on the first step");
        if done {
            SpawnProgress::Done(root)
        } else {
            SpawnProgress::InProgress(root)
        }
    }

    /// The map root, once the first step has created it.
    pub fn root(&self) -> Option<Entity> {
        self.state.root
    }
}

/// Cursor through the ordered spawn phases. Later phases reference the
/// entities created by earlier ones, so the order is fixed.
enum SpawnPhase {
    /// One [`MapNode`] per Voronoi cell. Field: next cell index.
    Cells(usize),
    /// One [`MapEdge`] per unique adjacency. Field: next edge index.
    Edges(usize),
    /// Promote chosen cells to [`Site`]s. Fields: stage, index-in-stage.
    Sites { stage: usize, idx: usize },
    /// Spawn paths, promote waypoints, record the many-to-many link.
    /// Fields: stage, prev-in-stage, route index.
    Paths {
        stage: usize,
        prev: usize,
        route: usize,
    },
    /// Attach [`PathEdges`] to each path. Field: next index into
    /// `link_paths`.
    LinkPaths(usize),
    /// Attach [`EdgePaths`] + [`LocationState`] to each path-bearing
    /// edge. Field: next index into `link_edges`.
    LinkEdges(usize),
    /// Fire the entry [`VisitLocation`].
    Visit,
    /// Everything is spawned.
    Done,
}

/// Mutable spawn progress: the entity handles produced so far plus the
/// phase cursor. Reused by the one-shot path and [`LevelMapSpawner`].
struct SpawnState {
    root: Option<Entity>,
    cell_entities: Vec<Entity>,
    edge_entities: HashMap<(usize, usize), Entity>,
    level_entities: Vec<Vec<Entity>>,
    path_edges: HashMap<Entity, Vec<Entity>>,
    edge_paths: HashMap<Entity, Vec<Entity>>,
    /// `path_edges` / `edge_paths` drained into ordered vecs when the
    /// `Paths` phase completes, so the link phases can resume by index.
    link_paths: Vec<(Entity, Vec<Entity>)>,
    link_edges: Vec<(Entity, Vec<Entity>)>,
    phase: SpawnPhase,
}

impl SpawnState {
    fn new() -> Self {
        Self {
            root: None,
            cell_entities: Vec::new(),
            edge_entities: HashMap::new(),
            level_entities: Vec::new(),
            path_edges: HashMap::new(),
            edge_paths: HashMap::new(),
            link_paths: Vec::new(),
            link_edges: Vec::new(),
            phase: SpawnPhase::Cells(0),
        }
    }

    /// Advance the pipeline, spending at most `budget` entity operations.
    /// Returns `true` when the map is fully spawned.
    fn step(
        &mut self,
        policy: LevelMapPolicy,
        g: &Generated,
        commands: &mut Commands,
        budget: usize,
    ) -> bool {
        let mut spent = 0usize;

        if self.root.is_none() {
            self.root = Some(
                commands
                    .spawn((
                        Name::new("LevelMap"),
                        LevelMap {
                            size: g.size,
                            seed: g.seed,
                            requested_seed: g.requested_seed,
                            rotation: g.rotation,
                            y_offset: g.y_offset,
                        },
                        policy,
                        Transform::default(),
                        Visibility::default(),
                    ))
                    .id(),
            );
            spent += 1;
        }
        let root = self.root.expect("root just created");

        loop {
            if spent >= budget {
                return false;
            }
            match self.phase {
                // 1. One MapNode per Voronoi cell. Default Transform = cell
                //    site; overridden for chosen Sites in the Sites phase.
                SpawnPhase::Cells(i) => {
                    if i >= g.cells.len() {
                        self.phase = SpawnPhase::Edges(0);
                        continue;
                    }
                    let cell = &g.cells[i];
                    let entity = commands
                        .spawn((
                            Name::new(format!("Cell {i}")),
                            MapNode,
                            VoronoiCell {
                                vertices: cell.vertices.clone(),
                            },
                            Transform::from_translation(cell.site.extend(0.0)),
                            Visibility::default(),
                            ChildOf(root),
                        ))
                        .id();
                    self.cell_entities.push(entity);
                    self.phase = SpawnPhase::Cells(i + 1);
                    spent += 1;
                }

                // 2. One MapEdge per unique Voronoi adjacency, keyed by the
                //    canonical (min, max) cell pair for later lookup.
                SpawnPhase::Edges(i) => {
                    if i >= g.edges.len() {
                        self.phase = SpawnPhase::Sites { stage: 0, idx: 0 };
                        continue;
                    }
                    let edge = &g.edges[i];
                    let (a, b) = edge.cells;
                    let entity = commands
                        .spawn((
                            Name::new("MapEdge"),
                            MapEdge {
                                from: self.cell_entities[a],
                                to: self.cell_entities[b],
                                wall: edge.wall,
                            },
                            ChildOf(root),
                        ))
                        .id();
                    self.edge_entities.insert((a, b), entity);
                    self.phase = SpawnPhase::Edges(i + 1);
                    spent += 1;
                }

                // 3. Promote chosen cells to Sites. Everyone starts
                //    Inactive — the entry visit propagates any non-Inactive
                //    state.
                SpawnPhase::Sites { stage, idx } => {
                    if stage >= g.levels.len() {
                        self.phase = SpawnPhase::Paths {
                            stage: 0,
                            prev: 0,
                            route: 0,
                        };
                        continue;
                    }
                    if idx == 0 {
                        self.level_entities
                            .push(Vec::with_capacity(g.levels[stage].len()));
                    }
                    if idx >= g.levels[stage].len() {
                        self.phase = SpawnPhase::Sites {
                            stage: stage + 1,
                            idx: 0,
                        };
                        continue;
                    }
                    let cell_idx = g.levels[stage][idx];
                    let entity = self.cell_entities[cell_idx];
                    let pos = g.level_jitters[stage][idx];
                    commands.entity(entity).insert((
                        Name::new(format!("Site {stage}.{idx}")),
                        Site {
                            belt: stage as u32,
                            site: idx as u32,
                        },
                        Transform::from_translation(pos.extend(0.0)),
                        LocationState::Inactive,
                    ));
                    self.level_entities[stage].push(entity);
                    self.phase = SpawnPhase::Sites {
                        stage,
                        idx: idx + 1,
                    };
                    spent += 1;
                }

                // 4. Spawn paths and walk each route's cell chain, recording
                //    both sides of the many-to-many link.
                SpawnPhase::Paths { stage, prev, route } => {
                    if stage >= g.connections.len() {
                        // Drain the accumulated link maps into ordered vecs
                        // so the link phases can resume by index.
                        self.link_paths =
                            std::mem::take(&mut self.path_edges).into_iter().collect();
                        self.link_edges =
                            std::mem::take(&mut self.edge_paths).into_iter().collect();
                        self.phase = SpawnPhase::LinkPaths(0);
                        continue;
                    }
                    if prev >= g.connections[stage].len() {
                        self.phase = SpawnPhase::Paths {
                            stage: stage + 1,
                            prev: 0,
                            route: 0,
                        };
                        continue;
                    }
                    if route >= g.connections[stage][prev].len() {
                        self.phase = SpawnPhase::Paths {
                            stage,
                            prev: prev + 1,
                            route: 0,
                        };
                        continue;
                    }

                    self.spawn_route(g, commands, root, stage, prev, route);
                    self.phase = SpawnPhase::Paths {
                        stage,
                        prev,
                        route: route + 1,
                    };
                    spent += 1;
                }

                // 5a. Populate PathEdges on every path.
                SpawnPhase::LinkPaths(i) => {
                    if i >= self.link_paths.len() {
                        self.phase = SpawnPhase::LinkEdges(0);
                        continue;
                    }
                    let (path, edges) = &self.link_paths[i];
                    commands.entity(*path).insert(PathEdges::new(edges.clone()));
                    self.phase = SpawnPhase::LinkPaths(i + 1);
                    spent += 1;
                }

                // 5b. Stamp EdgePaths + Inactive on every path-bearing edge.
                //     Edges not on any path stay "dead" (no EdgePaths, no
                //     LocationState).
                SpawnPhase::LinkEdges(i) => {
                    if i >= self.link_edges.len() {
                        self.phase = SpawnPhase::Visit;
                        continue;
                    }
                    let (edge, paths) = &self.link_edges[i];
                    commands
                        .entity(*edge)
                        .insert((EdgePaths::new(paths.clone()), LocationState::Inactive));
                    self.phase = SpawnPhase::LinkEdges(i + 1);
                    spent += 1;
                }

                // 6. Hand control to the observer via the entry visit.
                SpawnPhase::Visit => {
                    let entry_node = self.level_entities[0][0];
                    commands.trigger(VisitLocation { target: entry_node });
                    self.phase = SpawnPhase::Done;
                    return true;
                }

                SpawnPhase::Done => return true,
            }
        }
    }

    /// Spawn one route: its `MapPath`, promote its intermediate cells to
    /// `Waypoint`s, and record the edges it traverses on both sides of the
    /// many-to-many link.
    fn spawn_route(
        &mut self,
        g: &Generated,
        commands: &mut Commands,
        root: Entity,
        stage: usize,
        prev: usize,
        route: usize,
    ) {
        let r = &g.connections[stage][prev][route];
        let from_node = self.level_entities[stage][prev];
        let to_node = self.level_entities[stage + 1][r.target_in_next];

        let path_entity = commands
            .spawn((
                Name::new(format!(
                    "Path L{stage}.{prev}->L{next}.{target}",
                    next = stage + 1,
                    target = r.target_in_next,
                )),
                MapPath,
                PathFrom(from_node),
                PathTo(to_node),
                Transform::default(),
                Visibility::default(),
                LocationState::Inactive,
                ChildOf(root),
            ))
            .id();

        // Promote intermediate cells to Waypoints. Writing Inactive from
        // every route is idempotent, so a waypoint shared across routes
        // lands on a consistent state regardless of iteration order.
        for (wp_idx, &cell_idx) in r.intermediate_cells.iter().enumerate() {
            commands.entity(self.cell_entities[cell_idx]).insert((
                Name::new(format!(
                    "Waypoint {stage}-{next}{suffix}",
                    next = stage + 1,
                    suffix = char::from(b'a' + wp_idx as u8),
                )),
                Waypoint,
                LocationState::Inactive,
            ));
        }

        // Build the cell chain and record the edges this path traverses.
        let mut cell_chain: Vec<usize> = Vec::with_capacity(r.intermediate_cells.len() + 2);
        cell_chain.push(g.levels[stage][prev]);
        cell_chain.extend(r.intermediate_cells.iter().copied());
        cell_chain.push(g.levels[stage + 1][r.target_in_next]);

        let mut this_path_edges: Vec<Entity> =
            Vec::with_capacity(cell_chain.len().saturating_sub(1));
        for window in cell_chain.windows(2) {
            let (a, b) = (window[0], window[1]);
            let key = (a.min(b), a.max(b));
            let Some(&edge_entity) = self.edge_entities.get(&key) else {
                warn!("spawn_level_map: route walks edge ({a},{b}) not in Voronoi adjacency");
                continue;
            };
            this_path_edges.push(edge_entity);
            self.edge_paths
                .entry(edge_entity)
                .or_default()
                .push(path_entity);
        }
        self.path_edges.insert(path_entity, this_path_edges);
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use crate::components::{LevelMap, MapEdge, MapNode, MapPath, Site, Waypoint};
    use crate::config::{DesiredTraversals, LevelMapConfig};
    use crate::generation;
    use crate::spawn::{LevelMapCommands, LevelMapSpawner, SpawnProgress};
    use crate::{LevelMapRng, LevelSelectPlugin};

    /// A structural fingerprint of a spawned map: how many of each kind
    /// of entity exist plus the root's recorded generation parameters.
    /// Two spawns that agree on all of these produced the same entity
    /// graph.
    ///
    /// Deliberately excludes any [`LocationState`]-derived count. That
    /// state is driven by the entry [`VisitLocation`] observer, whose
    /// deferred FSM transitions resolve differently depending on whether
    /// the entry visit lands in the same command flush as the spawn
    /// (one-shot) or a later one (spread across frames). That timing
    /// artifact is orthogonal to the entity graph the spawner builds, so
    /// it has no place in a structural fingerprint.
    #[derive(Debug, PartialEq)]
    struct MapSignature {
        nodes: usize,
        sites: usize,
        waypoints: usize,
        edges: usize,
        paths: usize,
        seed: u64,
        requested_seed: u64,
        rotation: u32,
        y_offset: u32,
        size: (u32, u32),
    }

    fn signature(app: &mut App) -> MapSignature {
        let world = app.world_mut();
        let map = world
            .query::<&LevelMap>()
            .single(world)
            .expect("exactly one LevelMap root")
            .clone();
        MapSignature {
            nodes: world.query::<&MapNode>().iter(world).count(),
            sites: world.query::<&Site>().iter(world).count(),
            waypoints: world.query::<&Waypoint>().iter(world).count(),
            edges: world.query::<&MapEdge>().iter(world).count(),
            paths: world.query::<&MapPath>().iter(world).count(),
            seed: map.seed,
            requested_seed: map.requested_seed,
            rotation: map.rotation.to_bits(),
            y_offset: map.y_offset.to_bits(),
            size: (map.size.x.to_bits(), map.size.y.to_bits()),
        }
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(LevelSelectPlugin { seed: Some(0) });
        app
    }

    fn working_cfg(seed: u64) -> LevelMapConfig {
        LevelMapConfig {
            seed: Some(seed),
            allow_extra_traversal: 2,
            desired_traversals: Some(DesiredTraversals::default()),
            ..LevelMapConfig::default()
        }
    }

    /// Run enough update cycles for the entry `VisitLocation` observer's
    /// deferred FSM `StateChangeRequest`s to fully settle, so a signature
    /// taken afterward is independent of how many frames the spawn itself
    /// took.
    fn settle(app: &mut App) {
        for _ in 0..8 {
            app.update();
        }
    }

    #[test]
    fn spawn_generated_map_matches_spawn_level_map() {
        // Seed 6 with this config exercises the internal retry chain:
        // attempt 0 fails and attempt 1 succeeds, so the effective
        // `seed` lands at 7 while `requested_seed` stays 6. That makes
        // the un-offset request the interesting value to check.
        const SEED: u64 = 6;
        let cfg = working_cfg(SEED);

        // Path A: the all-in-one wrapper.
        let mut app_a = test_app();
        {
            let cfg = cfg.clone();
            app_a.add_systems(
                Startup,
                move |mut commands: Commands, mut rng: ResMut<LevelMapRng>| {
                    commands
                        .spawn_level_map(&mut rng, cfg.clone())
                        .expect("map generation succeeds");
                },
            );
        }
        settle(&mut app_a);
        let sig_a = signature(&mut app_a);

        // Path B: generate up front, then spawn the pre-computed layout.
        let generated = generation::generate(&cfg, SEED).expect("map generation succeeds");
        let mut app_b = test_app();
        {
            let cfg = cfg.clone();
            app_b.add_systems(Startup, move |mut commands: Commands| {
                commands.spawn_generated_map(&cfg, &generated);
            });
        }
        settle(&mut app_b);
        let sig_b = signature(&mut app_b);

        assert_eq!(sig_a, sig_b);
        // The wrapper reports the un-offset request as `requested_seed`
        // regardless of whether an internal retry shifted `seed`.
        assert_eq!(sig_a.requested_seed, SEED);
        // A retry fired for this seed, so the effective seed differs —
        // demonstrating why gating on `requested_seed` is the stable
        // choice.
        assert_ne!(sig_a.seed, sig_a.requested_seed);
    }

    #[test]
    fn incremental_spawn_matches_one_shot() {
        const SEED: u64 = 6;
        let cfg = working_cfg(SEED);

        // One-shot reference.
        let mut app_one = test_app();
        {
            let cfg = cfg.clone();
            app_one.add_systems(Startup, move |mut commands: Commands| {
                let generated = generation::generate(&cfg, SEED).expect("generation");
                commands.spawn_generated_map(&cfg, &generated);
            });
        }
        settle(&mut app_one);
        let sig_one = signature(&mut app_one);

        // Incremental: a tight budget of 1 entity per step forces the
        // spawner to resume across many `update()` cycles.
        let mut app_inc = test_app();
        {
            let cfg = cfg.clone();
            app_inc.add_systems(Startup, move |mut commands: Commands| {
                let generated = generation::generate(&cfg, SEED).expect("generation");
                commands.insert_resource(PendingSpawn(Some(LevelMapSpawner::new(&cfg, generated))));
            });
            app_inc.add_systems(Update, drive_pending);
        }
        // Startup + enough updates for a 1-per-frame budget to finish.
        app_inc.update();
        for _ in 0..(sig_one.nodes + sig_one.edges + sig_one.paths + 16) {
            let remaining = app_inc
                .world()
                .get_resource::<PendingSpawn>()
                .map(|p| p.0.is_some())
                .unwrap_or(false);
            if !remaining {
                break;
            }
            app_inc.update();
        }
        // Let the final flush + entry-visit cascade settle.
        settle(&mut app_inc);
        let sig_inc = signature(&mut app_inc);

        assert_eq!(sig_one, sig_inc);
    }

    #[derive(Resource)]
    struct PendingSpawn(Option<LevelMapSpawner>);

    fn drive_pending(mut commands: Commands, pending: Option<ResMut<PendingSpawn>>) {
        let Some(mut pending) = pending else {
            return;
        };
        let Some(spawner) = pending.0.as_mut() else {
            return;
        };
        if let SpawnProgress::Done(_) = spawner.step(&mut commands, 1) {
            pending.0 = None;
        }
    }
}
