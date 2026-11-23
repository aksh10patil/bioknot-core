use bioknot_wasm::{analyze_protein_knot_native, Point3D};
use std::io::{self, Read};

fn main() {
    let mut buffer = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut buffer) {
        eprintln!("Failed to read stdin: {}", e);
        std::process::exit(1);
    }

    let pts: Vec<Point3D> = match serde_json::from_str(&buffer) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to parse JSON coordinates from stdin: {}", e);
            std::process::exit(1);
        }
    };

    let result = analyze_protein_knot_native(&pts);
    let output = serde_json::to_string(&result).expect("Failed to serialize KnotAnalysisResult");
    println!("{}", output);
}
