//! Pick concrete Voronoi cells matching the user's `layout` shape.
//!
//! Strategy: column banding by x-coordinate, then for each stage, derive
//! candidates from the Voronoi neighbors of previous-stage cells (with
//! optional `allow_extra_traversal` BFS through unused cells). Greedy
//! "set cover" picks ensure every previous-stage cell has at least one
//! chosen successor.
//!
//! When [`crate::config::DesiredTraversals`] is supplied, the picker also
//! biases each previous-stage cell toward a target whose BFS-shortest
//! path length matches that cell's sampled desired length. The BFS budget
//! is auto-raised so longer corridors are actually present in the
//! candidate set; otherwise the bias has nothing to choose from.

use std::collections::{HashMap, HashSet, VecDeque};

use bevy::prelude::*;
use rand::prelude::*;

use crate::config::DesiredTraversals;

use super::voronoi::Diagram;

/// Result of the selection step.
pub struct Selection {
    /// `levels[stage][index_in_stage]` = cell index in the diagram.
    pub levels: Vec<Vec<usize>>,
    /// `connections[stage][source_idx_in_stage]` = list of
    /// `(target_idx_in_next_stage, intermediate_cell_indices)`. The
    /// intermediates are the dead cells we routed through (empty for
    /// direct neighbors).
    pub connections: Vec<Vec<Vec<(usize, Vec<usize>)>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error("layout must have at least 2 stages")]
    LayoutTooShort,
    #[error("no candidates in stage {0}")]
    EmptyBand(usize),
    #[error("could not connect stage {from_stage} to {to_stage} for level {level}")]
    Unconnectable {
        from_stage: usize,
        to_stage: usize,
        level: usize,
    },
}

/// `bounds_min.x` and `bounds_max.x` define the visible x-range; the
/// buffer columns sit outside it. `layout.len()` stages map to bands
/// inside `[bounds_min.x, bounds_max.x]`.
#[allow(clippy::too_many_arguments)]
pub fn select(
    diagram: &Diagram,
    layout: &[u32],
    bounds_min: Vec2,
    bounds_max: Vec2,
    allow_extra_traversal: u32,
    desired_traversals: Option<&DesiredTraversals>,
    rng: &mut impl Rng,
) -> Result<Selection, SelectionError> {
    if layout.len() < 2 {
        return Err(SelectionError::LayoutTooShort);
    }

    let bands = build_bands(diagram, layout.len(), bounds_min.x, bounds_max.x);
    for (i, band) in bands.iter().enumerate() {
        if band.is_empty() {
            return Err(SelectionError::EmptyBand(i));
        }
    }

    let mut used: HashSet<usize> = HashSet::new();
    let mut levels: Vec<Vec<usize>> = Vec::with_capacity(layout.len());

    // Stage 0 seed: pick L(0) cells in band 0, distributed in y. Bias
    // toward cells that have at least one direct neighbor in band 1, so
    // we don't pick a cell deep in the band with no forward connectivity.
    let next_band: HashSet<usize> = bands[1].iter().copied().collect();
    let forward_capable: Vec<usize> = bands[0]
        .iter()
        .copied()
        .filter(|&c| diagram.neighbors[c].iter().any(|n| next_band.contains(n)))
        .collect();
    let stage0_pool: &[usize] = if forward_capable.len() >= layout[0] as usize {
        &forward_capable
    } else {
        &bands[0]
    };
    let stage0 = pick_evenly_in_y(diagram, stage0_pool, layout[0] as usize, rng);
    for &c in &stage0 {
        used.insert(c);
    }
    levels.push(stage0);

    let mut connections: Vec<Vec<Vec<(usize, Vec<usize>)>>> = Vec::with_capacity(layout.len() - 1);

    // BFS budget: `allow_extra_traversal` is the connectivity floor (the
    // selector's safety net for stages that have no direct neighbor in the
    // next band). When `desired_traversals` is set, raise the ceiling so
    // BFS can also surface paths long enough to satisfy a sampled-from-the-
    // top-of-the-curve desired length. Without this lift, the distance bias
    // below would have nothing longer than `allow_extra_traversal` to score.
    let desired_max = desired_traversals
        .map(|d| (d.average as usize) * 2 + 1)
        .unwrap_or(0);
    let max_intermediates = (allow_extra_traversal as usize).max(desired_max);

    for stage in 1..layout.len() {
        let want = layout[stage] as usize;
        let band_set: HashSet<usize> = bands[stage].iter().copied().collect();
        let prev = levels.last().unwrap().clone();

        // For each previous cell, BFS to find every band-i cell reachable
        // through up to `max_intermediates` unused non-band cells.
        let reachable_per_prev: Vec<Vec<(usize, Vec<usize>)>> = prev
            .iter()
            .map(|&p| reach_into_band(diagram, p, &band_set, &used, max_intermediates))
            .collect();

        // Candidate pool = union of reachable targets across all prevs.
        let mut pool: HashSet<usize> = HashSet::new();
        for paths in &reachable_per_prev {
            for (target, _) in paths {
                pool.insert(*target);
            }
        }
        if pool.len() < want {
            return Err(SelectionError::Unconnectable {
                from_stage: stage - 1,
                to_stage: stage,
                level: 0,
            });
        }

        // Forward-awareness: when not the final stage, score a cell as
        // "forward-capable" if it has any direct neighbor in band i+1.
        // The greedy and fill steps use this as a soft tiebreaker so we
        // pick cells that the next stage can still reach.
        let next_band: Option<HashSet<usize>> = if stage + 1 < layout.len() {
            Some(bands[stage + 1].iter().copied().collect())
        } else {
            None
        };
        let is_forward_capable = |c: usize| -> bool {
            match next_band.as_ref() {
                Some(nb) => diagram.neighbors[c].iter().any(|n| nb.contains(n)),
                None => true,
            }
        };

        // Per-prev desired path length, sampled fresh for this stage so
        // every corridor's length varies independently. `None` if the
        // user didn't supply `desired_traversals`, in which case the
        // distance-bias score below contributes 0.
        let desired_per_prev: Vec<Option<usize>> = match desired_traversals {
            Some(d) => (0..prev.len())
                .map(|_| Some(sample_desired_len(d, rng)))
                .collect(),
            None => vec![None; prev.len()],
        };

        // Coverage-first greedy: ensure every prev cell has at least one
        // chosen successor. Order by reachable-set size (most constrained
        // prev first) and at each step, pick the candidate that covers the
        // greatest number of still-uncovered prevs.
        let mut chosen_set: HashSet<usize> = HashSet::new();
        let mut covered: Vec<bool> = vec![false; prev.len()];

        let mut order: Vec<usize> = (0..prev.len()).collect();
        order.sort_by_key(|&i| reachable_per_prev[i].len());

        for pi in order {
            if covered[pi] {
                continue;
            }
            let want_len = desired_per_prev[pi];
            let mut best: Option<usize> = None;
            let mut best_score: i32 = i32::MIN;
            for (target, path) in &reachable_per_prev[pi] {
                if chosen_set.contains(target) {
                    continue;
                }
                let coverage = (0..prev.len())
                    .filter(|&i| {
                        !covered[i] && reachable_per_prev[i].iter().any(|(t, _)| t == target)
                    })
                    .count() as i32;
                let fwd = is_forward_capable(*target) as i32;
                // Distance bias: reward targets whose BFS path from `pi`
                // sits close to the sampled desired length. Saturated so
                // very-far paths aren't infinitely better than just-far.
                let dist_bonus = match want_len {
                    Some(want) => {
                        let delta = (path.len() as i32 - want as i32).abs();
                        (16 - delta).max(0)
                    }
                    None => 0,
                };
                // Coverage dominates everything (we must cover every prev);
                // distance bias dominates forward-capability so corridors
                // hit the desired shape even if it costs us a soft hint.
                let score = coverage * 1024 + dist_bonus * 8 + fwd;
                if score > best_score {
                    best = Some(*target);
                    best_score = score;
                }
            }
            match best {
                Some(c) => {
                    chosen_set.insert(c);
                    for i in 0..prev.len() {
                        if reachable_per_prev[i].iter().any(|(t, _)| *t == c) {
                            covered[i] = true;
                        }
                    }
                }
                None => {
                    return Err(SelectionError::Unconnectable {
                        from_stage: stage - 1,
                        to_stage: stage,
                        level: pi,
                    });
                }
            }
        }

        if chosen_set.len() > want {
            // Coverage requires more cells than the layout allows.
            return Err(SelectionError::Unconnectable {
                from_stage: stage - 1,
                to_stage: stage,
                level: 0,
            });
        }

        // Fill remaining slots from the pool, distributed in y. Prefer
        // forward-capable cells; only dip into non-forward-capable when
        // we run out.
        let need = want - chosen_set.len();
        if need > 0 {
            let unchosen: Vec<usize> = pool
                .iter()
                .filter(|c| !chosen_set.contains(c))
                .copied()
                .collect();
            if unchosen.len() < need {
                return Err(SelectionError::Unconnectable {
                    from_stage: stage - 1,
                    to_stage: stage,
                    level: 0,
                });
            }
            let mut forward: Vec<usize> = unchosen
                .iter()
                .copied()
                .filter(|c| is_forward_capable(*c))
                .collect();
            let mut backup: Vec<usize> = unchosen
                .iter()
                .copied()
                .filter(|c| !is_forward_capable(*c))
                .collect();
            forward.sort_by(|a, b| {
                diagram.sites[*a]
                    .y
                    .partial_cmp(&diagram.sites[*b].y)
                    .unwrap()
            });
            backup.sort_by(|a, b| {
                diagram.sites[*a]
                    .y
                    .partial_cmp(&diagram.sites[*b].y)
                    .unwrap()
            });

            let from_forward = forward.len().min(need);
            let from_backup = need - from_forward;

            pick_into(&forward, from_forward, &mut chosen_set);
            pick_into(&backup, from_backup, &mut chosen_set);
        }

        // Stable indexing: sort chosen by y.
        let mut chosen: Vec<usize> = chosen_set.into_iter().collect();
        chosen.sort_by(|a, b| {
            diagram.sites[*a]
                .y
                .partial_cmp(&diagram.sites[*b].y)
                .unwrap()
        });
        let chosen_lookup: HashMap<usize, usize> =
            chosen.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        // Build connections: for each prev, record every reachable chosen
        // target (with the path that BFS found).
        let mut stage_conns: Vec<Vec<(usize, Vec<usize>)>> = vec![Vec::new(); prev.len()];
        for (pi, paths) in reachable_per_prev.iter().enumerate() {
            for (target, path) in paths {
                if let Some(&ci) = chosen_lookup.get(target) {
                    stage_conns[pi].push((ci, path.clone()));
                }
            }
            if stage_conns[pi].is_empty() {
                return Err(SelectionError::Unconnectable {
                    from_stage: stage - 1,
                    to_stage: stage,
                    level: pi,
                });
            }
        }

        for &c in &chosen {
            used.insert(c);
        }
        for conns in &stage_conns {
            for (_, path) in conns {
                for &c in path {
                    used.insert(c);
                }
            }
        }

        levels.push(chosen);
        connections.push(stage_conns);
    }

    Ok(Selection {
        levels,
        connections,
    })
}

/// Sample a desired path length (intermediate-cell count) for a single
/// corridor. Mirrors [`crate::generation::waypoints::sample_count`] so
/// selection-time bias and post-selection inflation agree on what
/// `DesiredTraversals` means.
fn sample_desired_len(cfg: &DesiredTraversals, rng: &mut impl Rng) -> usize {
    let t: f32 = rng.random_range(0.0..1.0);
    let multiplier = bevy::prelude::Curve::sample(&cfg.easing, t).unwrap_or(0.5);
    (multiplier.clamp(0.0, 1.0) * 2.0 * cfg.average as f32).round() as usize
}

fn build_bands(diagram: &Diagram, stage_count: usize, x_min: f32, x_max: f32) -> Vec<Vec<usize>> {
    let mut bands: Vec<Vec<usize>> = vec![Vec::new(); stage_count];
    let band_width = (x_max - x_min) / stage_count as f32;
    for (i, p) in diagram.sites.iter().enumerate() {
        if p.x < x_min || p.x >= x_max {
            continue;
        }
        let bin = ((p.x - x_min) / band_width).floor() as usize;
        let bin = bin.min(stage_count - 1);
        bands[bin].push(i);
    }
    bands
}

fn pick_evenly_in_y(diagram: &Diagram, band: &[usize], n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut sorted: Vec<usize> = band.to_vec();
    sorted.sort_by(|a, b| {
        diagram.sites[*a]
            .y
            .partial_cmp(&diagram.sites[*b].y)
            .unwrap()
    });
    pick_evenly_in_y_indices(&sorted, n, rng)
}

/// Pick `n` cells from `sorted` (assumed sorted by y) into `out`, evenly
/// spaced along y. Skips cells already in `out`.
fn pick_into(sorted: &[usize], n: usize, out: &mut HashSet<usize>) {
    if n == 0 || sorted.is_empty() {
        return;
    }
    for k in 0..n {
        let frac = (k as f32 + 0.5) / n as f32;
        let mut idx = ((frac * sorted.len() as f32).floor() as usize).min(sorted.len() - 1);
        let start_idx = idx;
        while out.contains(&sorted[idx]) {
            idx = (idx + 1) % sorted.len();
            if idx == start_idx {
                return;
            }
        }
        out.insert(sorted[idx]);
    }
}

fn pick_evenly_in_y_indices(sorted_band: &[usize], n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let len = sorted_band.len();
    if n == 1 {
        // Pick something near the middle for the single-entry case.
        let jitter = if len > 2 {
            rng.random_range(0..len.min(2) + 1)
        } else {
            0
        };
        let mid = (len / 2 + jitter).min(len - 1);
        return vec![sorted_band[mid]];
    }
    if n >= len {
        return sorted_band.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let frac = (k as f32 + 0.5) / n as f32;
        let idx = (frac * len as f32).floor() as usize;
        let idx = idx.min(len - 1);
        out.push(sorted_band[idx]);
    }
    out
}

/// BFS from `start` looking for any cell in `band`, traversing only
/// through unused non-band cells. Returns one entry per reachable band
/// cell, with the intermediate cells on the BFS-shortest path
/// (excluding both `start` and the target).
///
/// `max_intermediates` caps the number of dead cells we may walk through;
/// 0 means direct neighbors only.
fn reach_into_band(
    diagram: &Diagram,
    start: usize,
    band: &HashSet<usize>,
    used: &HashSet<usize>,
    max_intermediates: usize,
) -> Vec<(usize, Vec<usize>)> {
    let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    visited.insert(start);

    let mut q: VecDeque<(usize, Vec<usize>)> = VecDeque::new();
    q.push_back((start, Vec::new()));

    while let Some((node, path)) = q.pop_front() {
        for &n in &diagram.neighbors[node] {
            if visited.contains(&n) {
                continue;
            }
            visited.insert(n);

            if band.contains(&n) {
                if !used.contains(&n) {
                    out.push((n, path.clone()));
                }
                // Don't traverse through band cells — they're our targets.
                continue;
            }
            if used.contains(&n) {
                continue;
            }
            if path.len() + 1 > max_intermediates {
                continue;
            }
            let mut new_path = path.clone();
            new_path.push(n);
            q.push_back((n, new_path));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::poisson;
    use crate::generation::voronoi;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Build a diagram with band geometry that matches what the
    /// generation orchestrator sets up: visible width = stages * r * 2.5,
    /// so each band is ~2.5 cell-radii wide.
    fn build_test_diagram(
        seed: u64,
        layout: &[u32],
        r: f32,
        stage_buffer: u32,
    ) -> (Diagram, Vec2, Vec2) {
        let total_stages = layout.len() as u32 + 2 * stage_buffer;
        let max_levels = *layout.iter().max().unwrap() as f32;
        let visible_width = layout.len() as f32 * r * 2.5;
        let visible_height = (visible_width / 1.7).max(max_levels * r * 1.5);
        let buffered_width = total_stages as f32 * r * 2.5;
        let buffered_height = visible_height + stage_buffer as f32 * r * 2.0;

        let bmin = Vec2::new(-buffered_width * 0.5, -buffered_height * 0.5);
        let bmax = Vec2::new(buffered_width * 0.5, buffered_height * 0.5);
        let vmin = Vec2::new(-visible_width * 0.5, -visible_height * 0.5);
        let vmax = Vec2::new(visible_width * 0.5, visible_height * 0.5);

        let mut rng = StdRng::seed_from_u64(seed);
        let pts = poisson::sample(bmin, bmax, r, &mut rng);
        let d = voronoi::build(&pts, bmin, bmax).unwrap();
        (d, vmin, vmax)
    }

    /// Try a few seeds — we don't expect every seed to satisfy a
    /// constraint-heavy layout, but at least one should.
    fn try_select(
        layout: &[u32],
        allow_extra_traversal: u32,
        max_attempts: u32,
    ) -> Result<Selection, SelectionError> {
        let mut last_err = SelectionError::LayoutTooShort;
        for attempt in 0..max_attempts {
            let (d, bmin, bmax) = build_test_diagram(attempt as u64 + 1, layout, 25.0, 5);
            let mut rng = StdRng::seed_from_u64(attempt as u64 + 1);
            match select(
                &d,
                layout,
                bmin,
                bmax,
                allow_extra_traversal,
                None,
                &mut rng,
            ) {
                Ok(s) => return Ok(s),
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }

    #[test]
    fn ftl_layout_picks_correct_count() {
        // The canonical FTL-style layout from the spec. The 3->1 and
        // 3->1 chokepoints need a band-2 cell that's reachable from all
        // three band-1 cells, so we give the BFS two intermediate hops.
        let layout = [1u32, 3, 1, 3, 3, 1];
        let s = try_select(&layout, 2, 32).expect("no seed satisfied layout");
        assert_eq!(s.levels.len(), layout.len());
        for (i, &want) in layout.iter().enumerate() {
            assert_eq!(s.levels[i].len(), want as usize, "stage {i}");
        }

        // Every previous-stage cell has at least one outgoing connection.
        for stage_conns in &s.connections {
            for prev_conns in stage_conns {
                assert!(!prev_conns.is_empty(), "uncovered previous-stage cell");
            }
        }
    }

    #[test]
    fn small_layout_picks_correct_count() {
        let layout = [2u32, 2, 2];
        let s = try_select(&layout, 1, 16).expect("no seed satisfied layout");
        assert_eq!(s.levels.len(), 3);
        assert_eq!(s.levels[0].len(), 2);
        assert_eq!(s.levels[1].len(), 2);
        assert_eq!(s.levels[2].len(), 2);
        for stage_conns in &s.connections {
            for prev_conns in stage_conns {
                assert!(!prev_conns.is_empty());
            }
        }
    }
}
