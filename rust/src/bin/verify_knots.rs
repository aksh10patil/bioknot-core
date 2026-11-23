use bioknot_wasm::{analyze_protein_knot_native, Point3D};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize)]
#[allow(dead_code)]
struct Fixture {
    pdb_id: String,
    chain_id: String,
    ground_truth: String,
    length: usize,
    ca_coords: Vec<Point3D>,
}

#[derive(Debug, Serialize)]
struct StructureResult {
    pdb_id: String,
    chain_id: String,
    ground_truth: String,
    predicted: String,
    knot_probability: f64,
    chain_breaks: usize,
    reduced_nodes: usize,
    elapsed_ms: f64,
    match_status: bool,
}

#[derive(Debug, Serialize)]
struct ClassStats {
    tested: usize,
    correct: usize,
    accuracy: f64,
}

#[derive(Debug, Serialize)]
struct ChainBreakStats {
    structures_with_breaks: usize,
    total_breaks: usize,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    total_tested: usize,
    exact_matches: usize,
    exact_match_rate: f64,
    unknot_false_positives: usize,
    unknot_false_positive_rate: f64,
    per_class_accuracy: HashMap<String, ClassStats>,
    chain_break_stats: ChainBreakStats,
    total_time_seconds: f64,
    throughput_structures_per_second: f64,
    results: Vec<StructureResult>,
}

fn find_fixtures_dir() -> PathBuf {
    let candidates = [
        PathBuf::from("eval/pdb_fixtures"),
        PathBuf::from("../eval/pdb_fixtures"),
        PathBuf::from("../../eval/pdb_fixtures"),
    ];

    for candidate in &candidates {
        if candidate.exists() && candidate.is_dir() {
            return candidate.clone();
        }
    }

    panic!("Could not find eval/pdb_fixtures directory. Please run eval/fetch_pdb_fixtures.py first.");
}

fn load_fixtures(fixtures_dir: &Path) -> Vec<Fixture> {
    let manifest_path = fixtures_dir.join("manifest.json");
    if manifest_path.exists() {
        let content = fs::read_to_string(&manifest_path).expect("Failed to read manifest.json");
        if let Ok(fixtures) = serde_json::from_str::<Vec<Fixture>>(&content) {
            return fixtures;
        }
    }

    let mut fixtures = Vec::new();
    let entries = fs::read_dir(fixtures_dir).expect("Failed to read fixtures dir");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if path.file_name().and_then(|s| s.to_str()) == Some("manifest.json") {
                continue;
            }
            let content = fs::read_to_string(&path).expect("Failed to read fixture file");
            if let Ok(fixture) = serde_json::from_str::<Fixture>(&content) {
                fixtures.push(fixture);
            }
        }
    }
    fixtures
}

fn main() {
    println!("==========================================================================================");
    println!("   BIOKNOT-CORE: REAL PDB GROUND-TRUTH TOPOLOGICAL BENCHMARK (ALEXANDER INVARIANT)        ");
    println!("==========================================================================================");

    let fixtures_dir = find_fixtures_dir();
    let fixtures = load_fixtures(&fixtures_dir);
    println!("Loaded {} fixtures from {}\n", fixtures.len(), fixtures_dir.display());

    let mut exact_matches = 0;
    let mut total_structures = 0;
    let mut unknot_total = 0;
    let mut unknot_false_positives = 0;
    let mut structures_with_breaks = 0;
    let mut total_breaks = 0;

    let mut class_tested: HashMap<String, usize> = HashMap::new();
    let mut class_correct: HashMap<String, usize> = HashMap::new();
    let mut detailed_results: Vec<StructureResult> = Vec::new();

    let start_all = Instant::now();

    println!("{:<6} {:<6} {:<20} {:<20} {:<8} {:<8} {:<8} {:<10} {:<6}",
        "PDB", "Chain", "Ground Truth", "Predicted", "Prob", "Breaks", "Nodes", "Time (ms)", "Status");
    println!("{:-<100}", "");

    for fixture in &fixtures {
        let t0 = Instant::now();
        let result = analyze_protein_knot_native(&fixture.ca_coords);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let is_match = result.knot_type == fixture.ground_truth;
        if is_match {
            exact_matches += 1;
            *class_correct.entry(fixture.ground_truth.clone()).or_insert(0) += 1;
        }
        *class_tested.entry(fixture.ground_truth.clone()).or_insert(0) += 1;

        let is_ground_truth_unknot = fixture.ground_truth.contains("Unknot");
        if is_ground_truth_unknot {
            unknot_total += 1;
            if result.is_knotted {
                unknot_false_positives += 1;
            }
        }

        if result.chain_breaks > 0 {
            structures_with_breaks += 1;
            total_breaks += result.chain_breaks;
        }

        total_structures += 1;
        let status_str = if is_match { "PASS" } else { "FAIL" };

        println!("{:<6} {:<6} {:<20} {:<20} {:<7.1}% {:<8} {:<8} {:<9.2} {:<6}",
            fixture.pdb_id,
            fixture.chain_id,
            fixture.ground_truth,
            result.knot_type,
            result.knot_probability * 100.0,
            result.chain_breaks,
            result.reduced_node_count,
            elapsed_ms,
            status_str,
        );

        detailed_results.push(StructureResult {
            pdb_id: fixture.pdb_id.clone(),
            chain_id: fixture.chain_id.clone(),
            ground_truth: fixture.ground_truth.clone(),
            predicted: result.knot_type,
            knot_probability: result.knot_probability,
            chain_breaks: result.chain_breaks,
            reduced_nodes: result.reduced_node_count,
            elapsed_ms,
            match_status: is_match,
        });
    }

    let total_time_sec = start_all.elapsed().as_secs_f64();
    let throughput = if total_time_sec > 0.0 {
        total_structures as f64 / total_time_sec
    } else {
        0.0
    };

    let match_rate = if total_structures > 0 {
        (exact_matches as f64 / total_structures as f64) * 100.0
    } else {
        0.0
    };

    let fp_rate = if unknot_total > 0 {
        (unknot_false_positives as f64 / unknot_total as f64) * 100.0
    } else {
        0.0
    };

    let mut per_class_accuracy: HashMap<String, ClassStats> = HashMap::new();
    for (c_name, tested_count) in &class_tested {
        let correct_count = class_correct.get(c_name).copied().unwrap_or(0);
        let acc = (correct_count as f64 / *tested_count as f64) * 100.0;
        per_class_accuracy.insert(
            c_name.clone(),
            ClassStats {
                tested: *tested_count,
                correct: correct_count,
                accuracy: acc,
            },
        );
    }

    println!("{:-<100}", "");
    println!("BENCHMARK SUMMARY:");
    println!("  Total Structures Evaluated : {}", total_structures);
    println!("  Exact Topology Matches     : {} / {}", exact_matches, total_structures);
    println!("  Exact Match Rate           : {:.2}%", match_rate);
    println!("  Unknot False Positive Rate : {:.2}% ({} / {})", fp_rate, unknot_false_positives, unknot_total);
    println!("  Structures w/ Chain Breaks : {} (Total breaks: {})", structures_with_breaks, total_breaks);
    println!("  Total Benchmark Time       : {:.3}s", total_time_sec);
    println!("  Throughput                 : {:.2} structures/sec", throughput);
    println!("==========================================================================================");

    // Export to eval/rust_benchmark.json
    let report = BenchmarkReport {
        total_tested: total_structures,
        exact_matches,
        exact_match_rate: match_rate,
        unknot_false_positives,
        unknot_false_positive_rate: fp_rate,
        per_class_accuracy,
        chain_break_stats: ChainBreakStats {
            structures_with_breaks,
            total_breaks,
        },
        total_time_seconds: total_time_sec,
        throughput_structures_per_second: throughput,
        results: detailed_results,
    };

    let report_json = serde_json::to_string_pretty(&report).expect("Failed to serialize benchmark report");
    let out_paths = [
        fixtures_dir.parent().unwrap().join("rust_benchmark.json"),
        PathBuf::from("eval/rust_benchmark.json"),
    ];

    for path in &out_paths {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(_) = fs::write(path, &report_json) {
            println!("Exported benchmark report to {}", path.display());
            break;
        }
    }

    assert!(match_rate >= 90.0, "Benchmark failed: exact match rate {:.2}% < 90.0%", match_rate);
    assert_eq!(unknot_false_positives, 0, "Benchmark failed: unknot false positive rate must be 0%");
    println!("All verification criteria met successfully!");
}
