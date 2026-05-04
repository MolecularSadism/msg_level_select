//! Thin wrapper around `voronoice` exposing what the rest of the
//! pipeline needs: site positions, cell vertex polygons, and an
//! adjacency map.

use bevy::prelude::*;
use voronoice::{BoundingBox, Point, VoronoiBuilder};

/// Output of [`build`].
pub struct Diagram {
    pub sites: Vec<Vec2>,
    /// `cell_vertices[i]` is the polygon (CCW, world space) of cell `i`.
    pub cell_vertices: Vec<Vec<Vec2>>,
    /// Same shape as [`Diagram::cell_vertices`], but holding indices into
    /// [`Diagram::vertices`]. Used to find shared wall endpoints between
    /// two adjacent cells (= the set intersection of their index lists).
    pub cell_vertex_indices: Vec<Vec<usize>>,
    /// Global deduplicated vertex list. Voronoi vertices are shared by
    /// the (typically 3) cells meeting at that point.
    pub vertices: Vec<Vec2>,
    /// `neighbors[i]` contains the indices of cells sharing an edge
    /// with cell `i`.
    pub neighbors: Vec<Vec<usize>>,
}

pub fn build(sites: &[Vec2], min: Vec2, max: Vec2) -> Option<Diagram> {
    let center = ((min + max) * 0.5).as_dvec2();
    let half = ((max - min) * 0.5).as_dvec2();
    let bbox = BoundingBox::new(
        Point {
            x: center.x,
            y: center.y,
        },
        half.x * 2.0,
        half.y * 2.0,
    );
    let pts: Vec<Point> = sites
        .iter()
        .map(|p| Point {
            x: p.x as f64,
            y: p.y as f64,
        })
        .collect();

    let voronoi = VoronoiBuilder::default()
        .set_sites(pts)
        .set_bounding_box(bbox)
        .build()?;

    let n = sites.len();
    let mut cell_vertices: Vec<Vec<Vec2>> = Vec::with_capacity(n);
    let mut cell_vertex_indices: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n);
    let global_vertices: Vec<Vec2> = voronoi
        .vertices()
        .iter()
        .map(|p| Vec2::new(p.x as f32, p.y as f32))
        .collect();

    for i in 0..n {
        let cell = voronoi.cell(i);
        // `triangles()` gives the indices into `voronoi.vertices()`
        // for each polygon corner, in the same order as `iter_vertices`.
        // Matching indices across neighbor cells identifies the two
        // shared corner points that form their common wall.
        let tri_indices: Vec<usize> = cell.triangles().to_vec();
        let verts: Vec<Vec2> = tri_indices.iter().map(|&ix| global_vertices[ix]).collect();
        cell_vertices.push(verts);
        cell_vertex_indices.push(tri_indices);

        let n_indices: Vec<usize> = cell.iter_neighbors().collect();
        neighbors.push(n_indices);
    }

    // Drop adjacencies that don't produce a well-defined shared wall.
    // `voronoice` reports pairs of cells as neighbors whenever their
    // unclipped Voronoi regions touch, but cells on the outer ring get
    // clipped by the bounding box and may end up sharing fewer than two
    // polygon vertices. Those pairs have no `MapEdge` geometry to draw,
    // so path-finding must not route through them. Filtering here is
    // symmetric and keeps `neighbors` consistent with `shared_wall`.
    let filtered_neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let set_i: std::collections::HashSet<usize> =
                cell_vertex_indices[i].iter().copied().collect();
            neighbors[i]
                .iter()
                .copied()
                .filter(|&j| {
                    cell_vertex_indices[j]
                        .iter()
                        .filter(|v| set_i.contains(v))
                        .count()
                        >= 2
                })
                .collect()
        })
        .collect();

    Some(Diagram {
        sites: sites.to_vec(),
        cell_vertices,
        cell_vertex_indices,
        vertices: global_vertices,
        neighbors: filtered_neighbors,
    })
}

/// Find the two polygon corners shared by adjacent cells `a` and `b`
/// (the endpoints of their common wall). Returns `None` if fewer than
/// two indices are shared — e.g. for cells clipped by the bounding box.
pub fn shared_wall(diagram: &Diagram, a: usize, b: usize) -> Option<[Vec2; 2]> {
    let set_b: std::collections::HashSet<usize> =
        diagram.cell_vertex_indices[b].iter().copied().collect();
    let shared: Vec<usize> = diagram.cell_vertex_indices[a]
        .iter()
        .copied()
        .filter(|i| set_b.contains(i))
        .collect();
    if shared.len() < 2 {
        return None;
    }
    Some([diagram.vertices[shared[0]], diagram.vertices[shared[1]]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_grid(cols: usize, rows: usize, step: f32) -> Vec<Vec2> {
        let mut v = Vec::new();
        for y in 0..rows {
            for x in 0..cols {
                v.push(Vec2::new(x as f32 * step, y as f32 * step));
            }
        }
        v
    }

    #[test]
    fn neighbor_relation_is_symmetric() {
        let pts = sample_grid(6, 6, 10.0);
        let d = build(&pts, Vec2::splat(-5.0), Vec2::new(60.0, 60.0)).unwrap();
        for (i, ns) in d.neighbors.iter().enumerate() {
            for &j in ns {
                assert!(
                    d.neighbors[j].contains(&i),
                    "neighbor {} of {} did not list {} back",
                    j,
                    i,
                    i
                );
            }
        }
    }
}
