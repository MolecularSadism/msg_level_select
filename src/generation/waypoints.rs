//! Per-corridor variance refiner that runs after
//! [`super::selection::select`] and before entity spawning.
//!
//! Selection has already biased target picking toward farther stage-(N+1)
//! cells using `desired_traversals`, so most corridors arrive here with a
//! length close to the sampled target. This pass re-samples per corridor
//! and, when the new sample asks for a longer path than selection produced,
//! tries to extend it via BFS through still-unused Voronoi cells — the
//! selection-supplied intermediates are released back into the unused pool
//! so BFS can route through them again. Corridors selection already made
//! long enough are left alone; ones BFS can't extend keep their selection
//! geometry.

use std::collections::{HashSet, VecDeque};

use bevy::prelude::Curve;
use rand::prelude::*;

use crate::config::DesiredTraversals;

use super::selection::Selection;
use super::voronoi::Diagram;

pub fn inflate(
    diagram: &Diagram,
    selection: &mut Selection,
    cfg: &DesiredTraversals,
    rng: &mut impl Rng,
) {
    // Mark every cell that already belongs to selection as used.
    let mut used: HashSet<usize> = HashSet::new();
    for stage in &selection.levels {
        for &c in stage {
            used.insert(c);
        }
    }
    for stage_conns in &selection.connections {
        for prev_conns in stage_conns {
            for (_, path) in prev_conns {
                for &c in path {
                    used.insert(c);
                }
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    for stage in 0..selection.connections.len() {
        let from_cells: Vec<usize> = selection.levels[stage].clone();
        let to_cells: Vec<usize> = selection.levels[stage + 1].clone();
        // Intentionally walk an indexed loop so we can mutate the slice.
        for prev_idx in 0..selection.connections[stage].len() {
            for conn_idx in 0..selection.connections[stage][prev_idx].len() {
                let want = sample_count(cfg, rng) as usize;
                let existing_len = selection.connections[stage][prev_idx][conn_idx].1.len();

                // Selection already met or exceeded the desired length —
                // either through direct adjacency (existing_len == 0 and
                // want == 0) or through allow_extra_traversal bridging.
                // Leave it; we never *shorten* a path the selector built.
                if want <= existing_len {
                    continue;
                }

                let target_in_next = selection.connections[stage][prev_idx][conn_idx].0;
                let from = from_cells[prev_idx];
                let to = to_cells[target_in_next];

                // Release the selection-supplied intermediates so the BFS
                // can route through them again — otherwise a path that
                // selection bridged with `allow_extra_traversal` would have
                // those cells locked in `used` and BFS would be forced to
                // detour around them.
                let old_path =
                    std::mem::take(&mut selection.connections[stage][prev_idx][conn_idx].1);
                for &c in &old_path {
                    used.remove(&c);
                }

                match find_path_with_target_length(diagram, from, to, want, &used) {
                    Some(path) => {
                        for &c in &path {
                            used.insert(c);
                        }
                        selection.connections[stage][prev_idx][conn_idx].1 = path;
                    }
                    None => {
                        // No path of the requested length exists; keep what
                        // the selector gave us so connectivity is preserved.
                        for &c in &old_path {
                            used.insert(c);
                        }
                        selection.connections[stage][prev_idx][conn_idx].1 = old_path;
                    }
                }
            }
        }
    }
}

fn sample_count(cfg: &DesiredTraversals, rng: &mut impl Rng) -> u32 {
    let t: f32 = rng.random_range(0.0..1.0);
    let multiplier = Curve::sample(&cfg.easing, t).unwrap_or(0.5);
    (multiplier.clamp(0.0, 1.0) * 2.0 * cfg.average as f32).round() as u32
}

/// BFS exploring through unused cells looking for a path to `to` of
/// length close to `desired_intermediates + 1`. We accept the first
/// path within `[max(1, desired-1), desired+1]` intermediates.
fn find_path_with_target_length(
    diagram: &Diagram,
    from: usize,
    to: usize,
    desired_intermediates: usize,
    used: &HashSet<usize>,
) -> Option<Vec<usize>> {
    let min_len = desired_intermediates.saturating_sub(1).max(1);
    let max_len = desired_intermediates + 1;

    let mut q: VecDeque<(usize, Vec<usize>)> = VecDeque::new();
    q.push_back((from, Vec::new()));
    let mut best: Option<Vec<usize>> = None;

    let max_explore = max_len + 2;

    while let Some((node, path)) = q.pop_front() {
        if path.len() > max_explore {
            continue;
        }
        for &n in &diagram.neighbors[node] {
            if n == from {
                continue;
            }
            if n == to {
                if path.len() >= min_len && path.len() <= max_len {
                    return Some(path);
                } else if path.len() <= max_len && best.is_none() {
                    best = Some(path.clone());
                }
                continue;
            }
            if used.contains(&n) || path.contains(&n) {
                continue;
            }
            if path.len() + 1 > max_explore {
                continue;
            }
            let mut next = path.clone();
            next.push(n);
            q.push_back((n, next));
        }
    }
    best
}
