use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    fn sub(&self, other: &Point3D) -> Point3D {
        Point3D { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
    fn add(&self, other: &Point3D) -> Point3D {
        Point3D { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
    fn scale(&self, factor: f64) -> Point3D {
        Point3D { x: self.x * factor, y: self.y * factor, z: self.z * factor }
    }
    fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    fn normalize(&self) -> Point3D {
        let n = self.norm();
        if n == 0.0 { *self } else { self.scale(1.0 / n) }
    }
}

#[derive(Serialize, Deserialize)]
pub struct KnotAnalysisResult {
    pub knot_type: String,
    pub knot_probability: f64,
    pub raw_node_count: usize,
    pub reduced_node_count: usize,
    pub is_knotted: bool,
}

fn segment_pierces_triangle(s1: Point3D, s2: Point3D, p1: Point3D, p2: Point3D, p3: Point3D) -> bool {
    let edge1 = p2.sub(&p1);
    let edge2 = p3.sub(&p1);
    let ray_dir = s2.sub(&s1);

    let pvec = Point3D {
        x: ray_dir.y * edge2.z - ray_dir.z * edge2.y,
        y: ray_dir.z * edge2.x - ray_dir.x * edge2.z,
        z: ray_dir.x * edge2.y - ray_dir.y * edge2.x,
    };

    let det = edge1.x * pvec.x + edge1.y * pvec.y + edge1.z * pvec.z;
    if det.abs() < 1e-7 { return false; }

    let inv_det = 1.0 / det;
    let tvec = s1.sub(&p1);
    let u = (tvec.x * pvec.x + tvec.y * pvec.y + tvec.z * pvec.z) * inv_det;
    if u < 0.001 || u > 0.999 { return false; }

    let qvec = Point3D {
        x: tvec.y * edge1.z - tvec.z * edge1.y,
        y: tvec.z * edge1.x - tvec.x * edge1.z,
        z: tvec.x * edge1.y - tvec.y * edge1.x,
    };

    let v = (ray_dir.x * qvec.x + ray_dir.y * qvec.y + ray_dir.z * qvec.z) * inv_det;
    if v < 0.001 || (u + v) > 0.999 { return false; }

    let t = (edge2.x * qvec.x + edge2.y * qvec.y + edge2.z * qvec.z) * inv_det;
    t > 0.001 && t < 0.999
}

fn segments_cross_2d(p1: (f64, f64), p2: (f64, f64), q1: (f64, f64), q2: (f64, f64)) -> bool {
    let det = (p2.0 - p1.0) * (q2.1 - q1.1) - (p2.1 - p1.1) * (q2.0 - q1.0);
    if det.abs() < 1e-9 { return false; }

    let t = ((q1.0 - p1.0) * (q2.1 - q1.1) - (q1.1 - p1.1) * (q2.0 - q1.0)) / det;
    let u = ((q1.0 - p1.0) * (p2.1 - p1.1) - (q1.1 - p1.1) * (p2.0 - p1.0)) / det;

    t > 0.001 && t < 0.999 && u > 0.001 && u < 0.999
}

fn closed_kmt_reduce(points: &[Point3D], dir: Point3D) -> Vec<Point3D> {
    if points.len() <= 4 { return points.to_vec(); }

    let com = Point3D {
        x: points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64,
        y: points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64,
        z: points.iter().map(|p| p.z).sum::<f64>() / points.len() as f64,
    };

    let max_r = points.iter().map(|p| p.sub(&com).norm()).fold(0.0, f64::max) * 5.0;

    let n_ext = points[0].add(&dir.scale(max_r));
    let c_ext = points[points.len() - 1].add(&dir.scale(-max_r));

    let mut loop_pts = vec![c_ext];
    loop_pts.extend_from_slice(points);
    loop_pts.push(n_ext);

    let mut modified = true;
    while modified && loop_pts.len() > 5 {
        modified = false;
        let mut i = 1;
        while i < loop_pts.len() - 2 {
            let p_prev = loop_pts[i - 1];
            let p_curr = loop_pts[i];
            let p_next = loop_pts[i + 1];

            let mut pierced = false;
            for j in 0..(loop_pts.len() - 1) {
                if j == i - 1 || j == i { continue; }
                if segment_pierces_triangle(loop_pts[j], loop_pts[j + 1], p_prev, p_curr, p_next) {
                    pierced = true;
                    break;
                }
            }

            if !pierced {
                loop_pts.remove(i);
                modified = true;
                break;
            }
            i += 1;
        }
    }
    loop_pts
}

#[wasm_bindgen]
pub fn analyze_protein_knot(coords_json: &str) -> JsValue {
    let raw_pts: Vec<Point3D> = match serde_json::from_str(coords_json) {
        Ok(pts) => pts,
        Err(_) => return JsValue::NULL,
    };

    let raw_count = raw_pts.len();
    if raw_count < 5 {
        let res = KnotAnalysisResult {
            knot_type: "Unknot (0_1)".to_string(),
            knot_probability: 0.0,
            raw_node_count: raw_count,
            reduced_node_count: raw_count,
            is_knotted: false,
        };
        return serde_wasm_bindgen::to_value(&res).unwrap();
    }

    let num_projections = 50;
    let mut trefoil_votes = 0;
    let mut unknot_votes = 0;
    let mut sample_reduced_len = 0;

    for k in 0..num_projections {
        let phi = PI * (1.0 + (5.0f64).sqrt()) * (k as f64);
        let y = 1.0 - (k as f64 / (num_projections - 1) as f64) * 2.0;
        let radius = (1.0 - y * y).sqrt();
        let x = phi.cos() * radius;
        let z = phi.sin() * radius;
        let dir = Point3D { x, y, z };

        let closed_reduced = closed_kmt_reduce(&raw_pts, dir);
        if k == 0 { sample_reduced_len = closed_reduced.len(); }

        let ez = dir.normalize();
        let ex = if ez.x.abs() < 0.9 {
            Point3D { x: 0.0, y: ez.z, z: -ez.y }.normalize()
        } else {
            Point3D { x: -ez.z, y: 0.0, z: ez.x }.normalize()
        };
        let ey = Point3D {
            x: ez.y * ex.z - ez.z * ex.y,
            y: ez.z * ex.x - ez.x * ex.z,
            z: ez.x * ex.y - ez.y * ex.x,
        };

        let proj: Vec<(f64, f64)> = closed_reduced.iter().map(|p| {
            let px = p.x * ex.x + p.y * ex.y + p.z * ex.z;
            let py = p.x * ey.x + p.y * ey.y + p.z * ey.z;
            (px, py)
        }).collect();

        let n = proj.len() - 1;
        let mut crossings = 0;

        for i in 0..n {
            let p1 = proj[i];
            let p2 = proj[i+1];
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 { continue; }
                let q1 = proj[j];
                let q2 = proj[j+1];

                if segments_cross_2d(p1, p2, q1, q2) {
                    crossings += 1;
                }
            }
        }

        if crossings >= 3 {
            trefoil_votes += 1;
        } else {
            unknot_votes += 1;
        }
    }

    let knot_probability = trefoil_votes as f64 / (trefoil_votes + unknot_votes) as f64;
    let is_knotted = knot_probability > 0.15;

    let res = KnotAnalysisResult {
        knot_type: if is_knotted { "Trefoil Knot (3_1)".to_string() } else { "Unknot (0_1)".to_string() },
        knot_probability,
        raw_node_count: raw_count,
        reduced_node_count: sample_reduced_len,
        is_knotted,
    };

    serde_wasm_bindgen::to_value(&res).unwrap()
}
