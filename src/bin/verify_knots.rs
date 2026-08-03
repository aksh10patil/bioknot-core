use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
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

// Möller–Trumbore Ray-Triangle Segment Intersection
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

fn stochastically_classify_knot(points: &[Point3D]) -> (f64, String) {
    let num_projections = 50;
    let mut trefoil_votes = 0;
    let mut unknot_votes = 0;

    for k in 0..num_projections {
        let phi = PI * (1.0 + (5.0f64).sqrt()) * (k as f64);
        let y = 1.0 - (k as f64 / (num_projections - 1) as f64) * 2.0;
        let radius = (1.0 - y * y).sqrt();
        let x = phi.cos() * radius;
        let z = phi.sin() * radius;
        let dir = Point3D { x, y, z };

        let closed_reduced = closed_kmt_reduce(points, dir);

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
    // Open-chain virtual closures detect topological knotting if > 15% of random projections retain crossings
    if knot_probability > 0.15 {
        (knot_probability, "Trefoil Knot (3_1)".to_string())
    } else {
        (knot_probability, "Unknot (0_1)".to_string())
    }
}

fn main() {
    println!("====================================================");
    println!("  BIOKNOT-CORE: CLOSED KMT TOPOLOGY VERIFICATION SUITE");
    println!("====================================================");

    // Benchmark 1: Ubiquitin Trace (76 residues - Unknotted Baseline)
    let ubi_trace: Vec<Point3D> = (0..76).map(|i| Point3D {
        x: (i as f64) * 1.2,
        y: ((i as f64) * 0.3).sin() * 5.0,
        z: ((i as f64) * 0.3).cos() * 5.0,
    }).collect();

    let (prob_ubi, type_ubi) = stochastically_classify_knot(&ubi_trace);
    println!("\n[Test 1] Ubiquitin (Unknot Standard)");
    println!("  Raw C-alpha Count  : {}", ubi_trace.len());
    println!("  Knot Probability   : {:.2}%", prob_ubi * 100.0);
    println!("  Topology Verdict   : {}", if type_ubi.contains("Unknot") { "PASS (Unknot 0_1)" } else { "FAIL (False Positive)" });

    // Benchmark 2: Idealized Trefoil Knot Curve (120 nodes - Knotted 3_1 Standard)
    let trefoil_trace: Vec<Point3D> = (0..120).map(|t| {
        let theta = (t as f64) * (2.0 * PI / 120.0);
        Point3D {
            x: (theta.sin() + 2.0 * (2.0 * theta).sin()) * 10.0,
            y: (theta.cos() - 2.0 * (2.0 * theta).cos()) * 10.0,
            z: (-(3.0 * theta).sin()) * 10.0,
        }
    }).collect();

    let (prob_t, type_t) = stochastically_classify_knot(&trefoil_trace);
    println!("\n[Test 2] Synthetic Trefoil (Knotted 3_1 Standard)");
    println!("  Raw C-alpha Count  : {}", trefoil_trace.len());
    println!("  Knot Probability   : {:.2}%", prob_t * 100.0);
    println!("  Topology Verdict   : {}", if type_t.contains("Trefoil") { "PASS (Trefoil 3_1 Preserved)" } else { "FAIL (Collapsed)" });
    println!("====================================================\n");
}
