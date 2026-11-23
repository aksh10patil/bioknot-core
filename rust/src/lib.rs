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
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3D { x, y, z }
    }
    pub fn sub(&self, other: &Point3D) -> Point3D {
        Point3D { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
    pub fn add(&self, other: &Point3D) -> Point3D {
        Point3D { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
    pub fn scale(&self, factor: f64) -> Point3D {
        Point3D { x: self.x * factor, y: self.y * factor, z: self.z * factor }
    }
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn normalize(&self) -> Point3D {
        let n = self.norm();
        if n < 1e-12 { *self } else { self.scale(1.0 / n) }
    }
    pub fn dot(&self, other: &Point3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(&self, other: &Point3D) -> Point3D {
        Point3D {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    pub fn distance_to(&self, other: &Point3D) -> f64 {
        self.sub(other).norm()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnotAnalysisResult {
    pub knot_type: String,
    pub knot_probability: f64,
    pub raw_node_count: usize,
    pub reduced_node_count: usize,
    pub chain_breaks: usize,
    pub is_knotted: bool,
}

// 3D Axis-Aligned Bounding Box overlap test
fn aabb_overlap(s1: Point3D, s2: Point3D, p1: Point3D, p2: Point3D, p3: Point3D) -> bool {
    let seg_min_x = s1.x.min(s2.x);
    let seg_max_x = s1.x.max(s2.x);
    let seg_min_y = s1.y.min(s2.y);
    let seg_max_y = s1.y.max(s2.y);
    let seg_min_z = s1.z.min(s2.z);
    let seg_max_z = s1.z.max(s2.z);

    let tri_min_x = p1.x.min(p2.x).min(p3.x);
    let tri_max_x = p1.x.max(p2.x).max(p3.x);
    let tri_min_y = p1.y.min(p2.y).min(p3.y);
    let tri_max_y = p1.y.max(p2.y).max(p3.y);
    let tri_min_z = p1.z.min(p2.z).min(p3.z);
    let tri_max_z = p1.z.max(p2.z).max(p3.z);

    const TOL: f64 = 1e-6;
    if seg_max_x + TOL < tri_min_x || seg_min_x - TOL > tri_max_x { return false; }
    if seg_max_y + TOL < tri_min_y || seg_min_y - TOL > tri_max_y { return false; }
    if seg_max_z + TOL < tri_min_z || seg_min_z - TOL > tri_max_z { return false; }
    true
}

// Conservative Möller-Trumbore segment-triangle piercing check
pub fn segment_pierces_triangle(s1: Point3D, s2: Point3D, p1: Point3D, p2: Point3D, p3: Point3D) -> bool {
    if !aabb_overlap(s1, s2, p1, p2, p3) {
        return false;
    }

    let edge1 = p2.sub(&p1);
    let edge2 = p3.sub(&p1);
    let ray_dir = s2.sub(&s1);

    let pvec = ray_dir.cross(&edge2);
    let det = edge1.dot(&pvec);

    if det.abs() < 1e-9 {
        // Coplanar / nearly parallel ambiguity and AABBs overlap: fail conservatively!
        return true;
    }

    let inv_det = 1.0 / det;
    let tvec = s1.sub(&p1);
    let u = tvec.dot(&pvec) * inv_det;
    if u < -1e-6 || u > 1.0 + 1e-6 { return false; }

    let qvec = tvec.cross(&edge1);
    let v = ray_dir.dot(&qvec) * inv_det;
    if v < -1e-6 || (u + v) > 1.0 + 1e-6 { return false; }

    let t = edge2.dot(&qvec) * inv_det;
    t >= -1e-6 && t <= 1.0 + 1e-6
}

// Closed KMT (Koniaris-Muthukumar-Taylor) reduction on a closed cycle
pub fn closed_kmt_reduce(points: &[Point3D], dir: Point3D) -> Vec<Point3D> {
    if points.len() <= 4 { return points.to_vec(); }

    let com = Point3D {
        x: points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64,
        y: points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64,
        z: points.iter().map(|p| p.z).sum::<f64>() / points.len() as f64,
    };

    let max_r = points.iter().map(|p| p.sub(&com).norm()).fold(0.0, f64::max).max(1.0);
    let r_sphere = max_r * 5.0;

    let far_pt = com.add(&dir.scale(r_sphere));

    // Consistently oriented closed cycle:
    // points[0] -> ... -> points[last] -> far_pt -> (back to points[0])
    let mut loop_pts: Vec<Point3D> = Vec::with_capacity(points.len() + 1);
    loop_pts.extend_from_slice(points);
    loop_pts.push(far_pt);

    let mut is_closure = vec![false; loop_pts.len()];
    let closure_len = is_closure.len();
    is_closure[closure_len - 1] = true; // far_pt

    let mut modified = true;
    while modified && loop_pts.len() > 4 {
        modified = false;
        let mut i = 0;
        while i < loop_pts.len() {
            if loop_pts.len() <= 4 { break; }
            if is_closure[i] {
                i += 1;
                continue;
            }

            let l = loop_pts.len();
            let prev_idx = (i + l - 1) % l;
            let next_idx = (i + 1) % l;
            let p_prev = loop_pts[prev_idx];
            let p_curr = loop_pts[i];
            let p_next = loop_pts[next_idx];

            let mut pierced = false;
            for j in 0..l {
                // Skip adjacent edges: (prev_idx-1, prev_idx), (prev_idx, i), (i, next_idx), (next_idx, next_idx+1)
                if j == (i + l - 2) % l || j == prev_idx || j == i || j == next_idx {
                    continue;
                }
                let j_next = (j + 1) % l;
                if segment_pierces_triangle(loop_pts[j], loop_pts[j_next], p_prev, p_curr, p_next) {
                    pierced = true;
                    break;
                }
            }

            if !pierced {
                loop_pts.remove(i);
                is_closure.remove(i);
                modified = true;
            } else {
                i += 1;
            }
        }
    }

    loop_pts
}

#[derive(Debug, Clone)]
struct CrossingInfo {
    under_s: f64,
    over_s: f64,
    sign: f64, // +1.0 or -1.0
}

// Compute Alexander polynomial invariant from a 2D projection
fn compute_alexander_for_projection(pts: &[Point3D], proj_dir: Point3D) -> Option<(usize, usize)> {
    let n = pts.len();
    if n < 3 { return Some((1, 1)); }

    let ez = proj_dir.normalize();
    let ex = if ez.x.abs() < 0.9 {
        Point3D { x: 0.0, y: ez.z, z: -ez.y }.normalize()
    } else {
        Point3D { x: -ez.z, y: 0.0, z: ez.x }.normalize()
    };
    let ey = ez.cross(&ex).normalize();

    // Project vertices onto 2D (ex, ey) and compute z depth
    let proj: Vec<(f64, f64, f64)> = pts.iter().map(|p| {
        let x = p.dot(&ex);
        let y = p.dot(&ey);
        let z = p.dot(&ez);
        (x, y, z)
    }).collect();

    // Check for degenerate edges
    for i in 0..n {
        let next = (i + 1) % n;
        let dx = proj[next].0 - proj[i].0;
        let dy = proj[next].1 - proj[i].1;
        if (dx * dx + dy * dy).sqrt() < 1e-5 {
            return None; // Degenerate edge
        }
    }

    let mut crossings: Vec<CrossingInfo> = Vec::new();

    for i in 0..n {
        let i_next = (i + 1) % n;
        let p1 = (proj[i].0, proj[i].1);
        let p2 = (proj[i_next].0, proj[i_next].1);
        let d1 = (p2.0 - p1.0, p2.1 - p1.1);

        for j in (i + 1)..n {
            // Check adjacency in closed cycle
            if j == i || j == i_next || (j + 1) % n == i {
                continue;
            }
            let j_next = (j + 1) % n;
            let q1 = (proj[j].0, proj[j].1);
            let q2 = (proj[j_next].0, proj[j_next].1);
            let d2 = (q2.0 - q1.0, q2.1 - q1.1);

            let det = d1.0 * d2.1 - d1.1 * d2.0;
            if det.abs() < 1e-9 {
                continue; // Parallel or nearly parallel
            }

            let delta = (q1.0 - p1.0, q1.1 - p1.1);
            let t = (delta.0 * d2.1 - delta.1 * d2.0) / det;
            let u = (delta.0 * d1.1 - delta.1 * d1.0) / det;

            if t > 1e-4 && t < (1.0 - 1e-4) && u > 1e-4 && u < (1.0 - 1e-4) {
                let z1 = proj[i].2 + t * (proj[i_next].2 - proj[i].2);
                let z2 = proj[j].2 + u * (proj[j_next].2 - proj[j].2);

                if (z1 - z2).abs() < 1e-5 {
                    return None; // 3D self-intersection
                }

                let (over_s, under_s, v_over, v_under) = if z1 > z2 {
                    (i as f64 + t, j as f64 + u, d1, d2)
                } else {
                    (j as f64 + u, i as f64 + t, d2, d1)
                };

                let cross_prod = v_over.0 * v_under.1 - v_over.1 * v_under.0;
                let sign = if cross_prod >= 0.0 { 1.0 } else { -1.0 };

                crossings.push(CrossingInfo { under_s, over_s, sign });
            }
        }
    }

    let c = crossings.len();
    if c < 3 {
        return Some((1, 1));
    }

    // Sort crossings strictly by under_s
    crossings.sort_by(|a, b| a.under_s.partial_cmp(&b.under_s).unwrap());

    // Check for duplicate / near-identical under crossings (non-regular projection)
    for k in 0..(c - 1) {
        if (crossings[k + 1].under_s - crossings[k].under_s).abs() < 1e-5 {
            return None;
        }
    }

    // Identify which arc contains each over-strand
    // Arc k begins at under-crossing k and ends at under-crossing (k + 1) % c
    let get_arc_index = |s: f64| -> usize {
        if s < crossings[0].under_s || s >= crossings[c - 1].under_s {
            c - 1
        } else {
            let mut low = 0;
            let mut high = c - 1;
            while low <= high {
                let mid = (low + high) / 2;
                if mid < c - 1 && s >= crossings[mid].under_s && s < crossings[mid + 1].under_s {
                    return mid;
                } else if s < crossings[mid].under_s {
                    high = mid.saturating_sub(1);
                } else {
                    low = mid + 1;
                }
            }
            c - 1
        }
    };

    // Build Alexander matrix at t = -1 and t = -2
    let mut mat_neg1 = vec![vec![0.0f64; c]; c];
    let mut mat_neg2 = vec![vec![0.0f64; c]; c];

    for (m, cr) in crossings.iter().enumerate() {
        let in_arc = (m + c - 1) % c;
        let out_arc = m;
        let over_arc = get_arc_index(cr.over_s);

        if cr.sign > 0.0 {
            // Positive crossing: t*x_i + (1 - t)*x_k - x_j = 0
            // At t = -1: -x_i + 2*x_k - x_j = 0
            mat_neg1[m][in_arc] -= 1.0;
            mat_neg1[m][over_arc] += 2.0;
            mat_neg1[m][out_arc] -= 1.0;

            // At t = -2: -2*x_i + 3*x_k - x_j = 0
            mat_neg2[m][in_arc] -= 2.0;
            mat_neg2[m][over_arc] += 3.0;
            mat_neg2[m][out_arc] -= 1.0;
        } else {
            // Negative crossing: x_i + (t - 1)*x_k - t*x_j = 0
            // At t = -1: x_i - 2*x_k + x_j = 0
            mat_neg1[m][in_arc] += 1.0;
            mat_neg1[m][over_arc] -= 2.0;
            mat_neg1[m][out_arc] += 1.0;

            // At t = -2: x_i - 3*x_k + 2*x_j = 0
            mat_neg2[m][in_arc] += 1.0;
            mat_neg2[m][over_arc] -= 3.0;
            mat_neg2[m][out_arc] += 2.0;
        }
    }

    // Drop row 0 and column 0 for minor determinant
    let sub_neg1: Vec<Vec<f64>> = mat_neg1[1..].iter().map(|row| row[1..].to_vec()).collect();
    let sub_neg2: Vec<Vec<f64>> = mat_neg2[1..].iter().map(|row| row[1..].to_vec()).collect();

    let det1 = determinant(sub_neg1);
    let det2 = determinant(sub_neg2);

    let delta1 = det1.abs().round() as usize;
    let delta2 = det2.abs().round() as usize;

    Some((delta1, delta2))
}

// Gaussian elimination with partial pivoting in f64
fn determinant(mut mat: Vec<Vec<f64>>) -> f64 {
    let n = mat.len();
    if n == 0 { return 1.0; }
    let mut det = 1.0;

    for i in 0..n {
        let mut max_row = i;
        let mut max_val = mat[i][i].abs();
        for k in (i + 1)..n {
            if mat[k][i].abs() > max_val {
                max_val = mat[k][i].abs();
                max_row = k;
            }
        }

        if max_val < 1e-9 {
            return 0.0;
        }

        if max_row != i {
            mat.swap(i, max_row);
            det = -det;
        }

        let pivot = mat[i][i];
        det *= pivot;

        for k in (i + 1)..n {
            let factor = mat[k][i] / pivot;
            for j in (i + 1)..n {
                mat[k][j] -= factor * mat[i][j];
            }
        }
    }

    det
}

fn classify_from_alexander(delta1: usize, _delta2: usize) -> &'static str {
    match delta1 {
        0 | 1 => "Unknot (0_1)",
        3 => "Trefoil (3_1)",
        5 => "Figure-Eight (4_1)",
        7 => "5_2 Knot (5_2)",
        9 => "Stevedore (6_1)",
        _ => "Complex",
    }
}

pub fn analyze_protein_knot_native(raw_pts: &[Point3D]) -> KnotAnalysisResult {
    let raw_count = raw_pts.len();

    // 1. Detect chain breaks (> 4.5 Angstroms between consecutive C-alpha atoms)
    let mut chain_breaks = 0;
    for i in 0..raw_count.saturating_sub(1) {
        if raw_pts[i].distance_to(&raw_pts[i + 1]) > 4.5 {
            chain_breaks += 1;
        }
    }

    if raw_count < 5 {
        return KnotAnalysisResult {
            knot_type: "Unknot (0_1)".to_string(),
            knot_probability: 1.0,
            raw_node_count: raw_count,
            reduced_node_count: raw_count,
            chain_breaks,
            is_knotted: false,
        };
    }

    let num_projections = 50;
    let mut votes: std::collections::HashMap<&'static str, usize> = std::collections::HashMap::new();
    let mut total_valid_votes = 0;
    let mut sample_reduced_len = 0;

    for k in 0..num_projections {
        let phi = PI * (1.0 + (5.0f64).sqrt()) * (k as f64);
        let y = 1.0 - (k as f64 / (num_projections - 1) as f64) * 2.0;
        let radius = (1.0 - y * y).max(0.0).sqrt();
        let x = phi.cos() * radius;
        let z = phi.sin() * radius;
        let dir = Point3D { x, y, z }.normalize();

        let closed_reduced = closed_kmt_reduce(raw_pts, dir);
        if k == 0 { sample_reduced_len = closed_reduced.len(); }

        // Choose projection direction in general position (slightly perturbed from dir)
        let mut classified: Option<&'static str> = None;
        let perturbations = [
            Point3D { x: dir.y, y: dir.z, z: dir.x },
            Point3D { x: -dir.z, y: dir.x, z: dir.y },
            Point3D { x: dir.x + 0.3, y: dir.y - 0.2, z: dir.z + 0.1 },
            Point3D { x: dir.x - 0.2, y: dir.y + 0.3, z: dir.z - 0.1 },
        ];

        for pert in &perturbations {
            let proj_dir = dir.add(&pert.scale(0.35)).normalize();
            if let Some((d1, d2)) = compute_alexander_for_projection(&closed_reduced, proj_dir) {
                classified = Some(classify_from_alexander(d1, d2));
                break;
            }
        }

        let result_class = classified.unwrap_or("Unknot (0_1)");
        *votes.entry(result_class).or_insert(0) += 1;
        total_valid_votes += 1;
    }

    // Determine plurality / majority winner
    let mut winning_class = "Unknot (0_1)";
    let mut max_votes = 0;

    for (k_type, count) in &votes {
        if *count > max_votes {
            max_votes = *count;
            winning_class = k_type;
        }
    }

    let knot_probability = if total_valid_votes > 0 {
        max_votes as f64 / total_valid_votes as f64
    } else {
        1.0
    };

    let is_knotted = winning_class != "Unknot (0_1)";

    KnotAnalysisResult {
        knot_type: winning_class.to_string(),
        knot_probability,
        raw_node_count: raw_count,
        reduced_node_count: sample_reduced_len,
        chain_breaks,
        is_knotted,
    }
}

#[wasm_bindgen]
pub fn analyze_protein_knot(coords_json: &str) -> JsValue {
    let raw_pts: Vec<Point3D> = match serde_json::from_str(coords_json) {
        Ok(pts) => pts,
        Err(_) => return JsValue::NULL,
    };

    let result = analyze_protein_knot_native(&raw_pts);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Analytic Unknot (0_1) - planar circle
    #[test]
    fn test_analytic_unknot() {
        let pts: Vec<Point3D> = (0..60).map(|i| {
            let theta = 2.0 * PI * (i as f64) / 60.0;
            Point3D { x: 10.0 * theta.cos(), y: 10.0 * theta.sin(), z: 0.0 }
        }).collect();

        let res = analyze_protein_knot_native(&pts);
        assert_eq!(res.knot_type, "Unknot (0_1)");
        assert!(!res.is_knotted);
        assert!(res.knot_probability >= 0.90);
    }

    // 2. Tangled / twisted unknot with planar crossings
    #[test]
    fn test_tangled_twisted_unknot() {
        let pts: Vec<Point3D> = (0..100).map(|i| {
            let t = 2.0 * PI * (i as f64) / 100.0;
            Point3D {
                x: 10.0 * t.cos(),
                y: 10.0 * (2.0 * t).sin(),
                z: 4.0 * (3.0 * t).cos(),
            }
        }).collect();

        let res = analyze_protein_knot_native(&pts);
        assert_eq!(res.knot_type, "Unknot (0_1)");
        assert!(!res.is_knotted);
    }

    // 3. Analytic Trefoil (3_1)
    #[test]
    fn test_analytic_trefoil() {
        let pts: Vec<Point3D> = (0..120).map(|i| {
            let t = 2.0 * PI * (i as f64) / 120.0;
            Point3D {
                x: t.sin() + 2.0 * (2.0 * t).sin(),
                y: t.cos() - 2.0 * (2.0 * t).cos(),
                z: -(3.0 * t).sin() * 2.0,
            }
        }).collect();

        let res = analyze_protein_knot_native(&pts);
        assert_eq!(res.knot_type, "Trefoil (3_1)");
        assert!(res.is_knotted);
        assert!(res.knot_probability >= 0.70);
    }

    // 4. Analytic Figure-Eight (4_1)
    #[test]
    fn test_analytic_figure_eight() {
        let pts: Vec<Point3D> = (0..160).map(|i| {
            let t = 2.0 * PI * (i as f64) / 160.0;
            Point3D {
                x: (2.0 + (2.0 * t).cos()) * (3.0 * t).cos() * 5.0,
                y: (2.0 + (2.0 * t).cos()) * (3.0 * t).sin() * 5.0,
                z: (4.0 * t).sin() * 5.0,
            }
        }).collect();

        let res = analyze_protein_knot_native(&pts);
        assert_eq!(res.knot_type, "Figure-Eight (4_1)");
        assert!(res.is_knotted);
        assert!(res.knot_probability >= 0.70);
    }

    // 5. Chain break detection with 12 Angstrom gap
    #[test]
    fn test_chain_break_detection() {
        let mut pts: Vec<Point3D> = (0..20).map(|i| {
            Point3D { x: (i as f64) * 3.8, y: 0.0, z: 0.0 }
        }).collect();

        // Inject 12.0 A gap between index 9 and 10
        for i in 10..20 {
            pts[i].x += 12.0;
        }

        let res = analyze_protein_knot_native(&pts);
        assert_eq!(res.chain_breaks, 1);
    }
}
