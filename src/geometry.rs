// src/geometry.rs
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Converts a flattened JavaScript array [x0, y0, z0, x1, y1, z1, ...] into Point3D structs.
pub fn parse_flat_coordinates(coords: &[f64]) -> Result<Vec<Point3D>, String> {
    if coords.len() % 3 != 0 {
        return Err("Coordinates array length must be a multiple of 3.".to_string());
    }

    let points = coords
        .chunks_exact(3)
        .map(|chunk| Point3D::new(chunk[0], chunk[1], chunk[2]))
        .collect();

    Ok(points)
}

/// Computes the full symmetric NxN C-alpha Euclidean distance matrix.
pub fn compute_distance_matrix(points: &[Point3D]) -> Vec<f64> {
    let n = points.len();
    let mut matrix = vec![0.0; n * n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = points[i].distance_to(&points[j]);
            matrix[i * n + j] = dist;
            matrix[j * n + i] = dist; // Symmetric matrix
        }
    }

    matrix
}
