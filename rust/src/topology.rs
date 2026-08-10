// src/topology.rs
use crate::geometry::Point3D;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct TopologicalReport {
    pub num_residues: usize,
    pub max_diameter: f64,
    pub estimated_crossing_number: usize,
    pub is_entangled: bool,
}

/// Evaluates spatial backbone entanglement using projection crossing detection.
pub fn analyze_backbone_topology(points: &[Point3D]) -> TopologicalReport {
    let n = points.len();
    if n < 4 {
        return TopologicalReport {
            num_residues: n,
            max_diameter: 0.0,
            estimated_crossing_number: 0,
            is_entangled: false,
        };
    }

    // 1. Calculate Maximum Spatial Diameter
    let mut max_dist_sq = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[i].x - points[j].x;
            let dy = points[i].y - points[j].y;
            let dz = points[i].z - points[j].z;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
            }
        }
    }

    // 2. Count 2D Projected Chain Crossings (XY-Plane Projection Baseline)
    let mut crossings = 0;
    for i in 0..(n - 1) {
        for j in (i + 2)..(n - 1) {
            if i == 0 && j == n - 2 { continue; } // Skip adjacent segments
            if segments_intersect_2d(&points[i], &points[i + 1], &points[j], &points[j + 1]) {
                crossings += 1;
            }
        }
    }

    TopologicalReport {
        num_residues: n,
        max_diameter: max_dist_sq.sqrt(),
        estimated_crossing_number: crossings,
        is_entangled: crossings >= 3,
    }
}

/// Helper function to detect 2D line segment intersections on XY projection plane.
fn segments_intersect_2d(p1: &Point3D, p2: &Point3D, p3: &Point3D, p4: &Point3D) -> bool {
    fn ccw(a: &Point3D, b: &Point3D, c: &Point3D) -> bool {
        (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)
    }
    (ccw(p1, p3, p4) != ccw(p2, p3, p4)) && (ccw(p1, p2, p3) != ccw(p1, p2, p4))
}
