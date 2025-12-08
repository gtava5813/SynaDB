//! syna Demo Runner
//!
//! This is the main entry point for running all syna demos.
//! It can run all demos sequentially or a specific demo by name.
//!
//! Usage:
//!   cargo run                    # Run all demos
//!   cargo run -- basic_crud      # Run specific demo
//!   cargo run -- --list          # List available demos
//!   cargo run -- --help          # Show help
//!
//! Or run individual demos as examples:
//!   cargo run --example basic_crud
//!   cargo run --example time_series
//!   cargo run --example compression
//!   cargo run --example concurrent
//!   cargo run --example recovery
//!   cargo run --example tensor_extraction

use std::env;
use std::process::Command;

/// Available demos with descriptions
const DEMOS: &[(&str, &str)] = &[
    ("basic_crud", "Basic CRUD operations: create, read, update, delete"),
    ("time_series", "Time-series data: append sequences, extract history"),
    ("compression", "Compression: LZ4, delta encoding, size comparison"),
    ("concurrent", "Concurrent access: multi-threaded reads and writes"),
    ("recovery", "Crash recovery: simulate crash, verify data recovery"),
    ("tensor_extraction", "Tensor extraction: ML-ready float arrays"),
];

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 {
        match args[1].as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--list" | "-l" => {
                print_list();
                return;
            }
            demo_name => {
                // Run specific demo
                if DEMOS.iter().any(|(name, _)| *name == demo_name) {
                    run_demo(demo_name);
                } else {
                    eprintln!("Unknown demo: {}", demo_name);
                    eprintln!("Use --list to see available demos");
                    std::process::exit(1);
                }
                return;
            }
        }
    }
    
    // Run all demos
    run_all_demos();
}

fn print_help() {
    println!("syna Demo Runner");
    println!();
    println!("USAGE:");
    println!("    cargo run                    Run all demos");
    println!("    cargo run -- <demo_name>     Run specific demo");
    println!("    cargo run -- --list          List available demos");
    println!("    cargo run -- --help          Show this help");
    println!();
    println!("DEMOS:");
    for (name, desc) in DEMOS {
        println!("    {:20} {}", name, desc);
    }
    println!();
    println!("EXAMPLES:");
    println!("    cargo run -- basic_crud");
    println!("    cargo run --example time_series");
}

fn print_list() {
    println!("Available demos:");
    println!();
    for (name, desc) in DEMOS {
        println!("  {:20} - {}", name, desc);
    }
}


fn run_all_demos() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           syna DATABASE - DEMO SUITE                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (name, desc) in DEMOS {
        println!("┌──────────────────────────────────────────────────────────────┐");
        println!("│ Demo: {:54} │", name);
        println!("│ {:60} │", desc);
        println!("└──────────────────────────────────────────────────────────────┘");
        println!();
        
        if run_demo(name) {
            passed += 1;
            println!();
            println!("✓ {} completed successfully", name);
        } else {
            failed += 1;
            println!();
            println!("✗ {} FAILED", name);
        }
        
        println!();
        println!("────────────────────────────────────────────────────────────────");
        println!();
    }
    
    // Summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      DEMO SUMMARY                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Passed: {:3}                                                 ║", passed);
    println!("║  Failed: {:3}                                                 ║", failed);
    println!("║  Total:  {:3}                                                 ║", passed + failed);
    println!("╚══════════════════════════════════════════════════════════════╝");
    
    if failed > 0 {
        std::process::exit(1);
    }
}

fn run_demo(name: &str) -> bool {
    // Run the demo as a cargo example
    let status = Command::new("cargo")
        .args(["run", "--example", name, "--quiet"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status();
    
    match status {
        Ok(exit_status) => exit_status.success(),
        Err(e) => {
            eprintln!("Failed to run demo {}: {}", name, e);
            false
        }
    }
}


