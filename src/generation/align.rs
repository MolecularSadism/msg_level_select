//! Compute the rotation that aligns the entry-to-exit axis with +X
//! and column-aligns multi-entry / multi-exit groups.
//!
//! Math: minimize `F(theta) = Var(E_x'(theta)) + Var(X_x'(theta))`
//! where `x'(theta) = x*cos(theta) - y*sin(theta)`. The first
//! derivative `dF/dtheta = -a*sin(2*theta) - 2b*cos(2*theta)` has
//! `theta = 0.5 * atan2(-2b, a)`, with `a` and `b` defined below.
//! The other root (`theta + PI/2`) is the maximum; we pick whichever
//! candidate also keeps entries on the left of exits.

use bevy::prelude::*;

pub struct Alignment {
    pub rotation: f32,
}

pub fn compute(entries: &[Vec2], exits: &[Vec2]) -> Alignment {
    let entry_mean = mean(entries);
    let exit_mean = mean(exits);

    let single_axis = exit_mean - entry_mean;

    if entries.len() < 2 || exits.len() < 2 {
        return Alignment {
            rotation: align_to_x(single_axis),
        };
    }

    let var_ex = variance_x(entries, entry_mean);
    let var_ey = variance_y(entries, entry_mean);
    let var_xx = variance_x(exits, exit_mean);
    let var_xy = variance_y(exits, exit_mean);
    let cov_e = covariance(entries, entry_mean);
    let cov_x = covariance(exits, exit_mean);

    let a = (var_ex - var_ey) + (var_xx - var_xy);
    let b = cov_e + cov_x;

    if a.abs() < 1e-9 && b.abs() < 1e-9 {
        return Alignment {
            rotation: align_to_x(single_axis),
        };
    }

    let two_theta = (-2.0 * b).atan2(a);
    let theta_a = 0.5 * two_theta;
    let theta_b = theta_a + std::f32::consts::FRAC_PI_2;
    // F has period pi but the "entries on left" check has period 2*pi:
    // adding pi flips the rotated points around the origin, swapping
    // left/right. So we evaluate all four candidates.
    let pi = std::f32::consts::PI;
    let theta = [theta_a, theta_b, theta_a + pi, theta_b + pi]
        .into_iter()
        .min_by(|&t1, &t2| {
            score(t1, entries, exits)
                .partial_cmp(&score(t2, entries, exits))
                .unwrap()
        })
        .unwrap();

    Alignment { rotation: theta }
}

/// Lower is better: penalize column variance + having entries on the right.
fn score(theta: f32, entries: &[Vec2], exits: &[Vec2]) -> f32 {
    let (s, c) = theta.sin_cos();
    let rot = |p: Vec2| Vec2::new(p.x * c - p.y * s, p.x * s + p.y * c);
    let entries_rot: Vec<Vec2> = entries.iter().map(|p| rot(*p)).collect();
    let exits_rot: Vec<Vec2> = exits.iter().map(|p| rot(*p)).collect();
    let em = mean(&entries_rot);
    let xm = mean(&exits_rot);
    let var_score = variance_x(&entries_rot, em) + variance_x(&exits_rot, xm);
    let direction_penalty = if em.x <= xm.x { 0.0 } else { 1e9 };
    var_score + direction_penalty
}

fn align_to_x(direction: Vec2) -> f32 {
    if direction.length_squared() < 1e-9 {
        return 0.0;
    }
    -direction.y.atan2(direction.x)
}

fn mean(points: &[Vec2]) -> Vec2 {
    if points.is_empty() {
        return Vec2::ZERO;
    }
    points.iter().copied().sum::<Vec2>() / points.len() as f32
}

fn variance_x(points: &[Vec2], m: Vec2) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    points.iter().map(|p| (p.x - m.x).powi(2)).sum::<f32>() / points.len() as f32
}

fn variance_y(points: &[Vec2], m: Vec2) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    points.iter().map(|p| (p.y - m.y).powi(2)).sum::<f32>() / points.len() as f32
}

fn covariance(points: &[Vec2], m: Vec2) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    points
        .iter()
        .map(|p| (p.x - m.x) * (p.y - m.y))
        .sum::<f32>()
        / points.len() as f32
}

/// Apply rotation around the origin to a point.
pub fn rotate(p: Vec2, theta: f32) -> Vec2 {
    let (s, c) = theta.sin_cos();
    Vec2::new(p.x * c - p.y * s, p.x * s + p.y * c)
}

/// Compute the y-offset that minimizes the sum of squared y-deviations
/// across a set of post-rotation y-coordinates — i.e. the arithmetic
/// mean. Subtracting this offset from every world-space coordinate
/// re-centers the traversable map on `y = 0`, matching the visible
/// window (which is itself centered on the origin) and therefore fitting
/// the maximum number of levels and connecting path edges inside it.
pub fn y_centering_shift(ys: &[f32]) -> f32 {
    if ys.is_empty() {
        return 0.0;
    }
    ys.iter().sum::<f32>() / ys.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn single_entry_exit_aligns_to_x() {
        let entries = vec![Vec2::new(0.0, 0.0)];
        let exits = vec![Vec2::new(1.0, 1.0)];
        let r = compute(&entries, &exits).rotation;
        let rotated = rotate(Vec2::new(1.0, 1.0), r);
        assert!(rotated.x > 0.0);
        assert!(rotated.y.abs() < 1e-3);
    }

    #[test]
    fn y_centering_shift_is_mean() {
        let ys = [1.0, 2.0, 3.0, 10.0];
        let shift = y_centering_shift(&ys);
        // Mean = 4.0; subtracting it sums the shifted values to zero.
        let sum_shifted: f32 = ys.iter().map(|y| y - shift).sum();
        assert!(sum_shifted.abs() < 1e-4);
        // Mean minimizes sum of squared deviations — perturbing the
        // shift by any epsilon strictly increases the sum.
        let sse = |s: f32| ys.iter().map(|y| (y - s).powi(2)).sum::<f32>();
        assert!(sse(shift) < sse(shift + 0.1));
        assert!(sse(shift) < sse(shift - 0.1));
    }

    #[test]
    fn y_centering_shift_on_empty_is_zero() {
        assert_eq!(y_centering_shift(&[]), 0.0);
    }

    #[test]
    fn multi_entry_exit_recovers_rotation() {
        let entries: Vec<Vec2> = (0..3).map(|i| Vec2::new(0.0, i as f32)).collect();
        let exits: Vec<Vec2> = (0..3).map(|i| Vec2::new(10.0, i as f32)).collect();
        let theta = PI / 3.0;
        let entries_r: Vec<Vec2> = entries.iter().map(|p| rotate(*p, theta)).collect();
        let exits_r: Vec<Vec2> = exits.iter().map(|p| rotate(*p, theta)).collect();
        let recovered = compute(&entries_r, &exits_r).rotation;
        // Applying recovered to the rotated points should bring them back.
        let unrotated_entry: Vec<Vec2> = entries_r.iter().map(|p| rotate(*p, recovered)).collect();
        let unrotated_exit: Vec<Vec2> = exits_r.iter().map(|p| rotate(*p, recovered)).collect();
        let em = mean(&unrotated_entry);
        let xm = mean(&unrotated_exit);
        // Entries on the left of exits.
        assert!(em.x < xm.x);
        // Entries column-aligned (variance ~= 0).
        assert!(variance_x(&unrotated_entry, em) < 1e-3);
        assert!(variance_x(&unrotated_exit, xm) < 1e-3);
    }
}
