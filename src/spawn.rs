//! Turn a [`crate::generation::Generated`] into ECS entities.
//!
//! Every spawned entity starts in [`LocationState::Inactive`]. A single
//! [`VisitLocation`] fired at the end of spawn targets the entry level;
//! the observer promotes it to `Active` and cascades `Available` down
//! its outgoing corridor. Keeping initial state out of the spawn
//! pipeline means there's exactly one code path for promotion, and
//! shared waypoints / edges can't be clobbered by last-write-wins
//! inserts.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::LevelMapRng;
use crate::components::{LevelMap, MapEdge, MapNode, MapPath, Site, VoronoiCell, Waypoint};
use crate::config::LevelMapConfig;
use crate::generation::{self, GenerationError};
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
    fn spawn_level_map(
        &mut self,
        rng: &mut LevelMapRng,
        cfg: LevelMapConfig,
    ) -> Result<Entity, GenerationError>;
}

impl LevelMapCommands for Commands<'_, '_> {
    fn spawn_level_map(
        &mut self,
        rng: &mut LevelMapRng,
        cfg: LevelMapConfig,
    ) -> Result<Entity, GenerationError> {
        let seed = cfg.seed.unwrap_or_else(|| rng.next_seed());
        let generated = generation::generate(&cfg, seed)?;
        Ok(spawn_generated(self, &cfg, &generated))
    }
}

fn spawn_generated(
    commands: &mut Commands,
    cfg: &LevelMapConfig,
    g: &generation::Generated,
) -> Entity {
    let root = commands
        .spawn((
            Name::new("LevelMap"),
            LevelMap {
                size: g.size,
                seed: g.seed,
                rotation: g.rotation,
                y_offset: g.y_offset,
            },
            cfg.policy,
            Transform::default(),
            Visibility::default(),
        ))
        .id();

    // 1. Spawn one MapNode per Voronoi cell. Default Transform = cell site; we override it for
    //    chosen Levels in step 3.
    let mut cell_entities: Vec<Entity> = Vec::with_capacity(g.cells.len());
    for (cell_idx, cell) in g.cells.iter().enumerate() {
        let entity = commands
            .spawn((
                Name::new(format!("Cell {cell_idx}")),
                MapNode,
                VoronoiCell {
                    vertices: cell.vertices.clone(),
                },
                Transform::from_translation(cell.site.extend(0.0)),
                Visibility::default(),
                ChildOf(root),
            ))
            .id();
        cell_entities.push(entity);
    }

    // 2. Spawn one MapEdge per unique Voronoi adjacency. Index by the canonical (min, max) cell
    //    pair so path construction can look them up regardless of traversal direction.
    let mut edge_entities: HashMap<(usize, usize), Entity> = HashMap::with_capacity(g.edges.len());
    for edge in &g.edges {
        let (a, b) = edge.cells;
        let entity = commands
            .spawn((
                Name::new("MapEdge"),
                MapEdge {
                    from: cell_entities[a],
                    to: cell_entities[b],
                    wall: edge.wall,
                },
                ChildOf(root),
            ))
            .id();
        edge_entities.insert((a, b), entity);
    }

    // 3. Promote chosen cells to Sites. Everyone starts Inactive — the entry visit fired in step 6
    //    propagates any non-Inactive state.
    let mut level_entities: Vec<Vec<Entity>> = Vec::with_capacity(g.levels.len());
    for (stage_idx, stage) in g.levels.iter().enumerate() {
        let mut row = Vec::with_capacity(stage.len());
        for (level_idx, &cell_idx) in stage.iter().enumerate() {
            let entity = cell_entities[cell_idx];
            let pos = g.level_jitters[stage_idx][level_idx];
            commands.entity(entity).insert((
                Name::new(format!("Site {stage_idx}.{level_idx}")),
                Site {
                    belt: stage_idx as u32,
                    site: level_idx as u32,
                },
                Transform::from_translation(pos.extend(0.0)),
                LocationState::Inactive,
            ));
            row.push(entity);
        }
        level_entities.push(row);
    }

    // 4. Spawn paths and walk each route's cell chain, collecting both sides of the many-to-many
    //    link so PathEdges / EdgePaths can be populated in step 5.
    let mut path_edges: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut edge_paths: HashMap<Entity, Vec<Entity>> = HashMap::new();

    for (stage_idx, stage_conns) in g.connections.iter().enumerate() {
        for (prev_idx, prev_conns) in stage_conns.iter().enumerate() {
            for route in prev_conns {
                let from_node = level_entities[stage_idx][prev_idx];
                let to_node = level_entities[stage_idx + 1][route.target_in_next];

                let path_entity = commands
                    .spawn((
                        Name::new(format!(
                            "Path L{stage}.{prev}->L{next}.{target}",
                            stage = stage_idx,
                            prev = prev_idx,
                            next = stage_idx + 1,
                            target = route.target_in_next,
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

                // Promote intermediate cells to Waypoints. Writing
                // Inactive from every route is idempotent, so a
                // waypoint shared across routes lands on a consistent
                // state regardless of iteration order.
                for (wp_idx, &cell_idx) in route.intermediate_cells.iter().enumerate() {
                    commands.entity(cell_entities[cell_idx]).insert((
                        Name::new(format!(
                            "Waypoint {stage}-{next}{suffix}",
                            stage = stage_idx,
                            next = stage_idx + 1,
                            suffix = char::from(b'a' + wp_idx as u8),
                        )),
                        Waypoint,
                        LocationState::Inactive,
                    ));
                }

                // Build the cell chain and record the edges this path
                // traverses on both sides of the many-to-many link.
                let mut cell_chain: Vec<usize> =
                    Vec::with_capacity(route.intermediate_cells.len() + 2);
                cell_chain.push(g.levels[stage_idx][prev_idx]);
                cell_chain.extend(route.intermediate_cells.iter().copied());
                cell_chain.push(g.levels[stage_idx + 1][route.target_in_next]);

                let mut this_path_edges: Vec<Entity> =
                    Vec::with_capacity(cell_chain.len().saturating_sub(1));
                for window in cell_chain.windows(2) {
                    let (a, b) = (window[0], window[1]);
                    let key = (a.min(b), a.max(b));
                    let Some(&edge_entity) = edge_entities.get(&key) else {
                        warn!(
                            "spawn_level_map: route walks edge ({a},{b}) not in Voronoi adjacency"
                        );
                        continue;
                    };
                    this_path_edges.push(edge_entity);
                    edge_paths.entry(edge_entity).or_default().push(path_entity);
                }
                path_edges.insert(path_entity, this_path_edges);
            }
        }
    }

    // 5. Populate the many-to-many link and stamp Inactive on every path-bearing edge so the
    //    observer has a LocationState to promote from. Edges that aren't part of any path stay
    //    "dead" (no EdgePaths, no LocationState).
    for (path, edges) in path_edges {
        commands.entity(path).insert(PathEdges::new(edges));
    }
    for (edge, paths) in edge_paths {
        commands
            .entity(edge)
            .insert((EdgePaths::new(paths), LocationState::Inactive));
    }

    // 6. Hand control to the observer. The entry level is levels[0][0]; the VisitLocation promotes
    //    it to Active and lights up its outgoing corridor through the same code path as every
    //    subsequent visit.
    let entry_node = level_entities[0][0];
    commands.trigger(VisitLocation { target: entry_node });

    root
}
