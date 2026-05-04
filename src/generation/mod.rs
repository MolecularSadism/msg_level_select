//! Seven-step generation pipeline orchestrator.
//!
//! 1. Determine bounds.
//! 2. Poisson-disc sample points.
//! 3. Build Voronoi diagram.
//! 4. Pick cells matching the layout.
//! 5. Compute axis alignment rotation.
//! 6. (Optional) inflate with extra waypoints.
//! 7. Center the map in y by subtracting the mean y of all chosen levels and path-edge wall
//!    endpoints, so the visible window catches as many traversable features as possible.
//!
//! Entity spawning lives in [`crate::spawn`] so this module stays free
//! of ECS dependencies.

pub mod align;
pub mod poisson;
pub mod selection;
pub mod voronoi;
pub mod waypoints;

use bevy::prelude::*;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;

use crate::config::LevelMapConfig;

/// Errors returned from the generation pipeline.
#[derive(Debug, thiserror::Error)]
pub enum GenerationError {
    #[error("layout must contain at least 2 stages")]
    LayoutTooShort,
    #[error("voronoi construction failed")]
    Voronoi,
    #[error("could not satisfy layout after {0} attempts")]
    Unsatisfiable(u32),
}

/// Pure (no-ECS) output of the generation pipeline. Spawned into the
/// world by [`crate::spawn`].
///
/// Every Voronoi cell and every Voronoi adjacency is exposed here so
/// the spawn pipeline can materialize the *entire* diagram as ECS
/// entities. The "chosen" cells (campaign levels and their waypoint
/// corridors) are addressed via cell indices into [`Generated::cells`].
pub struct Generated {
    /// Visible region size in world units (post-rotation, post-translate).
    pub size: Vec2,
    /// Every Voronoi cell, indexed by cell id. Sites and vertices are
    /// expressed in final world coordinates (post-rotation).
    pub cells: Vec<CellData>,
    /// Unique Voronoi adjacencies — each unordered pair appears once
    /// with `cells.0 < cells.1`. Indices into [`Generated::cells`].
    /// Adjacencies whose shared wall is clipped by the bounding box
    /// (fewer than two shared corner vertices) are skipped.
    pub edges: Vec<EdgeData>,
    /// `levels[stage][index_in_stage]` = cell id. The chosen subset.
    pub levels: Vec<Vec<usize>>,
    /// Same shape as [`Generated::levels`]. The jittered "point of
    /// interest" position for each chosen level (overrides the cell's
    /// site position when spawning the entity).
    pub level_jitters: Vec<Vec<Vec2>>,
    /// Routes between chosen levels. `connections[stage][prev_in_stage]`
    /// is a list of routes from that previous-stage cell to cells in
    /// stage+1.
    pub connections: Vec<Vec<Vec<PathRoute>>>,
    /// Rotation (radians) used during alignment.
    pub rotation: f32,
    /// Y-offset subtracted from every post-rotation world coordinate
    /// (cell sites, cell vertices, edge walls, level jitter points).
    /// Equal to the mean y of chosen levels and path-edge wall
    /// endpoints, i.e. the shift that minimizes the sum of squared
    /// y-deviations across the traversable map.
    pub y_offset: f32,
    /// Seed actually used (may differ from input if retries kicked in).
    pub seed: u64,
}

/// Geometry for one Voronoi cell, in final world coordinates.
pub struct CellData {
    /// Cell site (the Poisson sample). Used as the default node position.
    pub site: Vec2,
    /// Polygon vertices, CCW.
    pub vertices: Vec<Vec2>,
}

/// One Voronoi adjacency — the shared polygon wall between two cells,
/// in final world coordinates.
pub struct EdgeData {
    /// The two cell indices (into [`Generated::cells`]) sharing this
    /// wall, always with `cells.0 < cells.1`.
    pub cells: (usize, usize),
    /// Endpoints of the shared polygon edge.
    pub wall: [Vec2; 2],
}

/// One route from a chosen previous-stage level to a chosen next-stage
/// level, including the unchosen cells the corridor walks through.
pub struct PathRoute {
    /// Index into `levels[stage + 1]`.
    pub target_in_next: usize,
    /// Cell ids of the corridor's intermediate cells (excludes both
    /// endpoints). Empty for direct neighbors.
    pub intermediate_cells: Vec<usize>,
}

pub fn generate(cfg: &LevelMapConfig, seed: u64) -> Result<Generated, GenerationError> {
    if cfg.layout.len() < 2 {
        return Err(GenerationError::LayoutTooShort);
    }

    let mut last_err = GenerationError::Unsatisfiable(0);
    for attempt in 0..cfg.max_attempts {
        let try_seed = seed.wrapping_add(attempt as u64);
        match try_generate_once(cfg, try_seed) {
            Ok(g) => return Ok(g),
            Err(e) => {
                last_err = e;
            }
        }
    }
    match last_err {
        GenerationError::LayoutTooShort | GenerationError::Voronoi => Err(last_err),
        _ => Err(GenerationError::Unsatisfiable(cfg.max_attempts)),
    }
}

fn try_generate_once(cfg: &LevelMapConfig, seed: u64) -> Result<Generated, GenerationError> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Step 1: bounds.
    let total_stages_with_buffer = cfg.layout.len() as u32 + 2 * cfg.stage_buffer;
    let max_levels = *cfg.layout.iter().max().unwrap() as f32;
    let visible_width = cfg.layout.len() as f32 * cfg.poisson_radius * 2.5;
    let visible_height =
        (visible_width / cfg.aspect_ratio).max(max_levels * cfg.poisson_radius * 1.5);
    let buffered_width = total_stages_with_buffer as f32 * cfg.poisson_radius * 2.5;
    let buffered_height = visible_height + cfg.stage_buffer as f32 * cfg.poisson_radius * 2.0;

    let buffered_min = Vec2::new(-buffered_width * 0.5, -buffered_height * 0.5);
    let buffered_max = Vec2::new(buffered_width * 0.5, buffered_height * 0.5);
    let visible_min = Vec2::new(-visible_width * 0.5, -visible_height * 0.5);
    let visible_max = Vec2::new(visible_width * 0.5, visible_height * 0.5);

    // Step 2: Poisson-disc sampling over the buffered rect.
    let points = poisson::sample(buffered_min, buffered_max, cfg.poisson_radius, &mut rng);
    if points.len() < cfg.layout.iter().sum::<u32>() as usize * 2 {
        return Err(GenerationError::Unsatisfiable(0));
    }

    // Step 3: Voronoi.
    let diagram =
        voronoi::build(&points, buffered_min, buffered_max).ok_or(GenerationError::Voronoi)?;

    // Step 4: cell selection over the visible band. `desired_traversals`
    // is forwarded so the picker biases toward farther stage-(N+1) targets;
    // the post-selection `inflate` below then refines per-corridor variance.
    let mut selection = selection::select(
        &diagram,
        &cfg.layout,
        visible_min,
        visible_max,
        cfg.allow_extra_traversal,
        cfg.desired_traversals.as_ref(),
        &mut rng,
    )
    .map_err(|_| GenerationError::Unsatisfiable(0))?;

    // Step 5: rotation alignment.
    let entry_sites: Vec<Vec2> = selection.levels[0]
        .iter()
        .map(|c| diagram.sites[*c])
        .collect();
    let exit_sites: Vec<Vec2> = selection.levels[cfg.layout.len() - 1]
        .iter()
        .map(|c| diagram.sites[*c])
        .collect();
    let alignment = align::compute(&entry_sites, &exit_sites);

    // Optional: inflate paths.
    if let Some(traversals) = cfg.desired_traversals.as_ref() {
        waypoints::inflate(&diagram, &mut selection, traversals, &mut rng);
    }

    // Step 6 prep: rotate cell geometry into world space.
    let mut cells: Vec<CellData> = diagram
        .sites
        .iter()
        .zip(diagram.cell_vertices.iter())
        .map(|(site, verts)| CellData {
            site: align::rotate(*site, alignment.rotation),
            vertices: verts
                .iter()
                .map(|v| align::rotate(*v, alignment.rotation))
                .collect(),
        })
        .collect();

    // Deduplicate adjacencies as unordered pairs. Each edge carries the
    // shared polygon wall (two corner points), rotated into world space
    // to match `cells`. Adjacencies whose wall is clipped by the bounding
    // box — and therefore share fewer than two Voronoi vertices — are
    // dropped so every `MapEdge` has a well-defined segment.
    let mut edges: Vec<EdgeData> = Vec::new();
    let mut seen_edges: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for (i, ns) in diagram.neighbors.iter().enumerate() {
        for &j in ns {
            let key = (i.min(j), i.max(j));
            if !seen_edges.insert(key) {
                continue;
            }
            let Some(wall) = voronoi::shared_wall(&diagram, key.0, key.1) else {
                continue;
            };
            let wall = [
                align::rotate(wall[0], alignment.rotation),
                align::rotate(wall[1], alignment.rotation),
            ];
            edges.push(EdgeData { cells: key, wall });
        }
    }

    // Compute jittered points-of-interest for chosen levels. Waypoints
    // and dead cells stay at the cell site.
    let inscribed = (cfg.poisson_radius * 0.5) - (cfg.node_position_buffer * cfg.poisson_radius);
    let inscribed = inscribed.max(0.0);

    let jitter = |site: Vec2, rng: &mut StdRng| -> Vec2 {
        if inscribed <= 0.0 {
            return site;
        }
        let r = inscribed * (rng.random_range(0.0_f32..1.0)).sqrt();
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        site + Vec2::new(r * theta.cos(), r * theta.sin())
    };

    let mut level_jitters: Vec<Vec<Vec2>> = Vec::with_capacity(selection.levels.len());
    for stage in &selection.levels {
        let mut stage_pos = Vec::with_capacity(stage.len());
        for &c in stage {
            // jitter operates in pre-rotation space, then we rotate to
            // match the rest of the cell geometry.
            let p = jitter(diagram.sites[c], &mut rng);
            stage_pos.push(align::rotate(p, alignment.rotation));
        }
        level_jitters.push(stage_pos);
    }

    // Step 7: y-center the map on the traversable features. Collect the
    // set of edges walked by at least one route; those shared walls,
    // together with the chosen level positions, define "what the
    // visible window needs to fit." Subtracting the mean y from every
    // world-space coordinate minimizes the sum of squared y-deviations
    // and thus the worst-case offset from the window's horizontal
    // midline.
    let mut path_edge_keys: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for (stage_idx, stage_conns) in selection.connections.iter().enumerate() {
        for (prev_idx, prev_conns) in stage_conns.iter().enumerate() {
            for (target_in_next, path) in prev_conns {
                let prev_cell = selection.levels[stage_idx][prev_idx];
                let next_cell = selection.levels[stage_idx + 1][*target_in_next];
                let mut chain: Vec<usize> = Vec::with_capacity(path.len() + 2);
                chain.push(prev_cell);
                chain.extend(path.iter().copied());
                chain.push(next_cell);
                for window in chain.windows(2) {
                    let (a, b) = (window[0], window[1]);
                    path_edge_keys.insert((a.min(b), a.max(b)));
                }
            }
        }
    }

    let mut ys: Vec<f32> = Vec::new();
    for stage in &level_jitters {
        for p in stage {
            ys.push(p.y);
        }
    }
    for edge in &edges {
        if path_edge_keys.contains(&edge.cells) {
            ys.push(edge.wall[0].y);
            ys.push(edge.wall[1].y);
        }
    }
    let y_shift = align::y_centering_shift(&ys);

    for cell in &mut cells {
        cell.site.y -= y_shift;
        for v in &mut cell.vertices {
            v.y -= y_shift;
        }
    }
    for edge in &mut edges {
        edge.wall[0].y -= y_shift;
        edge.wall[1].y -= y_shift;
    }
    for stage in &mut level_jitters {
        for p in stage {
            p.y -= y_shift;
        }
    }

    let connections: Vec<Vec<Vec<PathRoute>>> = selection
        .connections
        .iter()
        .map(|stage_conns| {
            stage_conns
                .iter()
                .map(|prev_conns| {
                    prev_conns
                        .iter()
                        .map(|(target, path)| PathRoute {
                            target_in_next: *target,
                            intermediate_cells: path.clone(),
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    Ok(Generated {
        size: visible_max - visible_min,
        cells,
        edges,
        levels: selection.levels,
        level_jitters,
        connections,
        rotation: alignment.rotation,
        y_offset: y_shift,
        seed,
    })
}
