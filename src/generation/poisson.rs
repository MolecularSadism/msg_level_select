//! Bridson's fast Poisson-disc sampling in 2D.
//!
//! Returns points whose pairwise distance is at least `radius` and which
//! cover the input rectangle.

use bevy::prelude::*;
use rand::prelude::*;

const K: u32 = 30; // Bridson's recommended candidate count per active point.

/// Sample Poisson-disc points covering `[min, max]` (inclusive of min,
/// exclusive of max).
pub fn sample(min: Vec2, max: Vec2, radius: f32, rng: &mut impl Rng) -> Vec<Vec2> {
    assert!(radius > 0.0, "Poisson radius must be positive");
    let size = max - min;
    let cell = radius / std::f32::consts::SQRT_2;
    let cols = (size.x / cell).ceil().max(1.0) as usize;
    let rows = (size.y / cell).ceil().max(1.0) as usize;
    let mut grid: Vec<Option<usize>> = vec![None; cols * rows];

    let mut samples: Vec<Vec2> = Vec::with_capacity(cols * rows);
    let mut active: Vec<usize> = Vec::new();

    // Seed with a uniformly-random point.
    let initial = Vec2::new(
        min.x + rng.random_range(0.0..size.x),
        min.y + rng.random_range(0.0..size.y),
    );
    samples.push(initial);
    let (ix, iy) = grid_index(initial, min, cell);
    grid[iy * cols + ix] = Some(0);
    active.push(0);

    while !active.is_empty() {
        let pick = rng.random_range(0..active.len());
        let parent = samples[active[pick]];
        let mut found = false;

        for _ in 0..K {
            // Annulus sample between r and 2r.
            let r = radius * (1.0 + rng.random_range(0.0..1.0));
            let theta = rng.random_range(0.0..std::f32::consts::TAU);
            let cand = parent + Vec2::new(r * theta.cos(), r * theta.sin());

            if cand.x < min.x || cand.x >= max.x || cand.y < min.y || cand.y >= max.y {
                continue;
            }

            let (cx, cy) = grid_index(cand, min, cell);
            if neighborhood_collision(&samples, &grid, cols, rows, cx, cy, cand, radius) {
                continue;
            }

            samples.push(cand);
            let idx = samples.len() - 1;
            grid[cy * cols + cx] = Some(idx);
            active.push(idx);
            found = true;
            break;
        }

        if !found {
            active.swap_remove(pick);
        }
    }

    samples
}

#[inline]
fn grid_index(p: Vec2, min: Vec2, cell: f32) -> (usize, usize) {
    let local = p - min;
    let x = (local.x / cell).floor() as isize;
    let y = (local.y / cell).floor() as isize;
    (x.max(0) as usize, y.max(0) as usize)
}

#[inline]
fn neighborhood_collision(
    samples: &[Vec2],
    grid: &[Option<usize>],
    cols: usize,
    rows: usize,
    cx: usize,
    cy: usize,
    cand: Vec2,
    radius: f32,
) -> bool {
    let r2 = radius * radius;
    let xmin = cx.saturating_sub(2);
    let ymin = cy.saturating_sub(2);
    let xmax = (cx + 2).min(cols - 1);
    let ymax = (cy + 2).min(rows - 1);
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            if let Some(idx) = grid[y * cols + x]
                && (samples[idx] - cand).length_squared() < r2
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn pairwise_distance_respected() {
        let mut rng = StdRng::seed_from_u64(42);
        let pts = sample(Vec2::ZERO, Vec2::splat(200.0), 10.0, &mut rng);
        assert!(pts.len() > 50);
        for (i, a) in pts.iter().enumerate() {
            for b in pts.iter().skip(i + 1) {
                let d = (*a - *b).length();
                assert!(
                    d >= 10.0 - 1e-3,
                    "distance {} below radius for points {:?} {:?}",
                    d,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn determinism_from_seed() {
        let mut a = SmallRng::seed_from_u64(7);
        let mut b = SmallRng::seed_from_u64(7);
        let pa = sample(Vec2::ZERO, Vec2::splat(80.0), 8.0, &mut a);
        let pb = sample(Vec2::ZERO, Vec2::splat(80.0), 8.0, &mut b);
        assert_eq!(pa, pb);
    }
}
