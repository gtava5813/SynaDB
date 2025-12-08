//! Benchmark report generation.
//!
//! This module provides comprehensive report generation for benchmark results,
//! including JSON, Markdown, and chart outputs.
//!
//! _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

use crate::{BenchmarkResult, config::SystemInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ============================================================================
// Report Data Structures
// ============================================================================

/// Complete benchmark report with all metadata and results
/// _Requirements: 11.1, 11.4, 11.5_
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub metadata: ReportMetadata,
    pub results: Vec<BenchmarkResult>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub anomalies: Vec<Anomaly>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReportSummary>,
}

/// Report metadata including system information and configuration
/// _Requirements: 11.4, 11.5_
#[derive(Debug, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub timestamp: String,
    pub SYNA_version: String,
    pub benchmark_version: String,
    pub system: SystemInfo,
    pub config: BenchmarkRunConfig,
}

/// Configuration used for the benchmark run
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkRunConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub value_sizes: Vec<usize>,
    pub thread_counts: Vec<usize>,
    pub sync_on_write: bool,
    pub databases_tested: Vec<String>,
}

impl Default for BenchmarkRunConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 1000,
            measurement_iterations: 10000,
            value_sizes: vec![64, 1024, 65536],
            thread_counts: vec![1, 4, 8],
            sync_on_write: false,
            databases_tested: vec!["Syna".to_string()],
        }
    }
}

/// Summary statistics for the benchmark report
#[derive(Debug, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_benchmarks: usize,
    pub total_duration_secs: f64,
    pub best_write_throughput: ThroughputSummary,
    pub best_read_throughput: ThroughputSummary,
    pub lowest_latency: LatencySummary,
    pub comparison_highlights: Vec<String>,
}

/// Throughput summary for a specific benchmark
#[derive(Debug, Serialize, Deserialize)]
pub struct ThroughputSummary {
    pub database: String,
    pub benchmark: String,
    pub throughput_ops_sec: f64,
}

/// Latency summary for a specific benchmark
#[derive(Debug, Serialize, Deserialize)]
pub struct LatencySummary {
    pub database: String,
    pub benchmark: String,
    pub latency_p50_us: f64,
}

/// Anomaly detected in benchmark results
/// _Requirements: 11.6_
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub benchmark: String,
    pub database: String,
    pub anomaly_type: AnomalyType,
    pub message: String,
    pub severity: AnomalySeverity,
    pub suggestion: String,
}

/// Type of anomaly detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    UnexpectedlyLow,
    UnexpectedlyHigh,
    HighVariance,
    ZeroValue,
    NegativeValue,
}

/// Severity level of the anomaly
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalySeverity {
    Warning,
    Error,
}

// ============================================================================
// Expected Ranges for Anomaly Detection
// ============================================================================

/// Expected performance ranges for anomaly detection
/// _Requirements: 11.6_
pub struct ExpectedRanges {
    /// Minimum expected write throughput (ops/sec)
    pub min_write_throughput: f64,
    /// Maximum expected write throughput (ops/sec)
    pub max_write_throughput: f64,
    /// Minimum expected read throughput (ops/sec)
    pub min_read_throughput: f64,
    /// Maximum expected read throughput (ops/sec)
    pub max_read_throughput: f64,
    /// Maximum expected p99 latency (microseconds)
    pub max_p99_latency_us: f64,
}

impl Default for ExpectedRanges {
    fn default() -> Self {
        Self {
            min_write_throughput: 1000.0,      // At least 1K ops/sec
            max_write_throughput: 10_000_000.0, // At most 10M ops/sec
            min_read_throughput: 5000.0,        // At least 5K ops/sec
            max_read_throughput: 50_000_000.0,  // At most 50M ops/sec
            max_p99_latency_us: 100_000.0,      // At most 100ms p99
        }
    }
}

// ============================================================================
// Report Generation Functions
// ============================================================================

/// Generate benchmark report in multiple formats
/// _Requirements: 11.1, 11.2_
pub fn generate_report(results: &[BenchmarkResult], output_dir: &str) {
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path).expect("Failed to create output directory");
    
    // Detect anomalies
    let anomalies = detect_anomalies(results, &ExpectedRanges::default());
    
    // Generate summary
    let summary = generate_summary(results);
    
    let report = BenchmarkReport {
        metadata: ReportMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            SYNA_version: env!("CARGO_PKG_VERSION").to_string(),
            benchmark_version: "1.0.0".to_string(),
            system: SystemInfo::collect(),
            config: extract_config_from_results(results),
        },
        results: results.to_vec(),
        anomalies: anomalies.clone(),
        summary: Some(summary),
    };
    
    // Generate JSON report
    let json_path = output_path.join("report.json");
    generate_json_report(&report, &json_path);
    println!("  Generated: {}", json_path.display());
    
    // Generate Markdown report
    let md_path = output_path.join("report.md");
    generate_markdown_report(&report, &md_path);
    println!("  Generated: {}", md_path.display());
    
    // Print anomaly warnings
    if !anomalies.is_empty() {
        println!("\n⚠️  {} anomalies detected:", anomalies.len());
        for anomaly in &anomalies {
            println!("  - [{}] {}: {}", 
                match anomaly.severity {
                    AnomalySeverity::Warning => "WARN",
                    AnomalySeverity::Error => "ERROR",
                },
                anomaly.benchmark,
                anomaly.message
            );
        }
    }
}

/// Generate JSON report file
/// _Requirements: 11.1, 11.4, 11.5_
pub fn generate_json_report(report: &BenchmarkReport, path: &Path) {
    let json = serde_json::to_string_pretty(report)
        .expect("Failed to serialize report to JSON");
    fs::write(path, json).expect("Failed to write JSON report");
}

/// Extract configuration from results
fn extract_config_from_results(results: &[BenchmarkResult]) -> BenchmarkRunConfig {
    let mut value_sizes: Vec<usize> = results.iter()
        .map(|r| r.config.value_size_bytes)
        .collect();
    value_sizes.sort();
    value_sizes.dedup();
    
    let mut thread_counts: Vec<usize> = results.iter()
        .map(|r| r.config.thread_count)
        .collect();
    thread_counts.sort();
    thread_counts.dedup();
    
    let mut databases: Vec<String> = results.iter()
        .map(|r| r.database.clone())
        .collect();
    databases.sort();
    databases.dedup();
    
    let first = results.first();
    
    BenchmarkRunConfig {
        warmup_iterations: first.map(|r| r.config.warmup_iterations).unwrap_or(1000),
        measurement_iterations: first.map(|r| r.config.measurement_iterations).unwrap_or(10000),
        value_sizes,
        thread_counts,
        sync_on_write: first.map(|r| r.config.sync_on_write).unwrap_or(false),
        databases_tested: databases,
    }
}

/// Generate summary statistics from results
fn generate_summary(results: &[BenchmarkResult]) -> ReportSummary {
    let write_results: Vec<_> = results.iter()
        .filter(|r| r.benchmark.contains("write"))
        .collect();
    
    let read_results: Vec<_> = results.iter()
        .filter(|r| r.benchmark.contains("read"))
        .collect();
    
    let best_write = write_results.iter()
        .max_by(|a, b| a.throughput_ops_sec.partial_cmp(&b.throughput_ops_sec).unwrap())
        .map(|r| ThroughputSummary {
            database: r.database.clone(),
            benchmark: r.benchmark.clone(),
            throughput_ops_sec: r.throughput_ops_sec,
        })
        .unwrap_or(ThroughputSummary {
            database: "N/A".to_string(),
            benchmark: "N/A".to_string(),
            throughput_ops_sec: 0.0,
        });
    
    let best_read = read_results.iter()
        .max_by(|a, b| a.throughput_ops_sec.partial_cmp(&b.throughput_ops_sec).unwrap())
        .map(|r| ThroughputSummary {
            database: r.database.clone(),
            benchmark: r.benchmark.clone(),
            throughput_ops_sec: r.throughput_ops_sec,
        })
        .unwrap_or(ThroughputSummary {
            database: "N/A".to_string(),
            benchmark: "N/A".to_string(),
            throughput_ops_sec: 0.0,
        });
    
    let lowest_latency = results.iter()
        .filter(|r| r.latency_p50_us > 0.0)
        .min_by(|a, b| a.latency_p50_us.partial_cmp(&b.latency_p50_us).unwrap())
        .map(|r| LatencySummary {
            database: r.database.clone(),
            benchmark: r.benchmark.clone(),
            latency_p50_us: r.latency_p50_us,
        })
        .unwrap_or(LatencySummary {
            database: "N/A".to_string(),
            benchmark: "N/A".to_string(),
            latency_p50_us: 0.0,
        });
    
    let total_duration: f64 = results.iter().map(|r| r.duration_secs).sum();
    
    // Generate comparison highlights
    let highlights = generate_comparison_highlights(results);
    
    ReportSummary {
        total_benchmarks: results.len(),
        total_duration_secs: total_duration,
        best_write_throughput: best_write,
        best_read_throughput: best_read,
        lowest_latency,
        comparison_highlights: highlights,
    }
}

/// Generate comparison highlights between databases
fn generate_comparison_highlights(results: &[BenchmarkResult]) -> Vec<String> {
    let mut highlights = Vec::new();
    
    // Group results by benchmark type
    let mut by_benchmark: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        by_benchmark.entry(result.benchmark.clone())
            .or_default()
            .push(result);
    }
    
    // Compare databases for each benchmark
    for (benchmark, bench_results) in &by_benchmark {
        if bench_results.len() > 1 {
            let Syna = bench_results.iter().find(|r| r.database == "Syna");
            let others: Vec<_> = bench_results.iter().filter(|r| r.database != "Syna").collect();
            
            if let Some(ent) = Syna {
                for other in others {
                    if other.throughput_ops_sec > 0.0 {
                        let ratio = ent.throughput_ops_sec / other.throughput_ops_sec;
                        if ratio > 1.1 {
                            highlights.push(format!(
                                "Syna is {:.1}x faster than {} for {}",
                                ratio, other.database, benchmark
                            ));
                        } else if ratio < 0.9 {
                            highlights.push(format!(
                                "{} is {:.1}x faster than Syna for {}",
                                other.database, 1.0 / ratio, benchmark
                            ));
                        }
                    }
                }
            }
        }
    }
    
    highlights
}

// ============================================================================
// Anomaly Detection
// ============================================================================

/// Detect anomalies in benchmark results
/// _Requirements: 11.6_
pub fn detect_anomalies(results: &[BenchmarkResult], ranges: &ExpectedRanges) -> Vec<Anomaly> {
    let mut anomalies = Vec::new();
    
    for result in results {
        // Check for zero or negative values
        if result.throughput_ops_sec <= 0.0 {
            anomalies.push(Anomaly {
                benchmark: result.benchmark.clone(),
                database: result.database.clone(),
                anomaly_type: AnomalyType::ZeroValue,
                message: "Throughput is zero or negative".to_string(),
                severity: AnomalySeverity::Error,
                suggestion: "Check if the benchmark ran correctly. Consider re-running.".to_string(),
            });
            continue;
        }
        
        // Check write benchmarks
        if result.benchmark.contains("write") {
            if result.throughput_ops_sec < ranges.min_write_throughput {
                anomalies.push(Anomaly {
                    benchmark: result.benchmark.clone(),
                    database: result.database.clone(),
                    anomaly_type: AnomalyType::UnexpectedlyLow,
                    message: format!(
                        "Write throughput ({:.0} ops/sec) is below expected minimum ({:.0} ops/sec)",
                        result.throughput_ops_sec, ranges.min_write_throughput
                    ),
                    severity: AnomalySeverity::Warning,
                    suggestion: "Check for disk I/O issues or system load. Consider re-running.".to_string(),
                });
            }
            
            if result.throughput_ops_sec > ranges.max_write_throughput {
                anomalies.push(Anomaly {
                    benchmark: result.benchmark.clone(),
                    database: result.database.clone(),
                    anomaly_type: AnomalyType::UnexpectedlyHigh,
                    message: format!(
                        "Write throughput ({:.0} ops/sec) exceeds expected maximum ({:.0} ops/sec)",
                        result.throughput_ops_sec, ranges.max_write_throughput
                    ),
                    severity: AnomalySeverity::Warning,
                    suggestion: "Results may be cached or measurement may be incorrect. Verify with cold start.".to_string(),
                });
            }
        }
        
        // Check read benchmarks
        if result.benchmark.contains("read") {
            if result.throughput_ops_sec < ranges.min_read_throughput {
                anomalies.push(Anomaly {
                    benchmark: result.benchmark.clone(),
                    database: result.database.clone(),
                    anomaly_type: AnomalyType::UnexpectedlyLow,
                    message: format!(
                        "Read throughput ({:.0} ops/sec) is below expected minimum ({:.0} ops/sec)",
                        result.throughput_ops_sec, ranges.min_read_throughput
                    ),
                    severity: AnomalySeverity::Warning,
                    suggestion: "Check for disk I/O issues or memory pressure. Consider re-running.".to_string(),
                });
            }
        }
        
        // Check latency
        if result.latency_p99_us > ranges.max_p99_latency_us {
            anomalies.push(Anomaly {
                benchmark: result.benchmark.clone(),
                database: result.database.clone(),
                anomaly_type: AnomalyType::UnexpectedlyHigh,
                message: format!(
                    "P99 latency ({:.0} μs) exceeds expected maximum ({:.0} μs)",
                    result.latency_p99_us, ranges.max_p99_latency_us
                ),
                severity: AnomalySeverity::Warning,
                suggestion: "High tail latency may indicate GC pauses or system interference.".to_string(),
            });
        }
        
        // Check for high variance (p99 >> p50)
        if result.latency_p50_us > 0.0 {
            let variance_ratio = result.latency_p99_us / result.latency_p50_us;
            if variance_ratio > 100.0 {
                anomalies.push(Anomaly {
                    benchmark: result.benchmark.clone(),
                    database: result.database.clone(),
                    anomaly_type: AnomalyType::HighVariance,
                    message: format!(
                        "High latency variance: p99/p50 ratio is {:.1}x",
                        variance_ratio
                    ),
                    severity: AnomalySeverity::Warning,
                    suggestion: "Consider running with more iterations or checking for system interference.".to_string(),
                });
            }
        }
    }
    
    anomalies
}

/// Check if results are consistent (for reproducibility)
/// _Requirements: 12.4_
pub fn check_consistency(results: &[BenchmarkResult], tolerance: f64) -> Vec<Anomaly> {
    let mut anomalies = Vec::new();
    
    // Group by benchmark + database
    let mut groups: HashMap<(String, String), Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        groups.entry((result.benchmark.clone(), result.database.clone()))
            .or_default()
            .push(result);
    }
    
    for ((benchmark, database), group) in groups {
        if group.len() >= 2 {
            let throughputs: Vec<f64> = group.iter().map(|r| r.throughput_ops_sec).collect();
            let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
            let variance = throughputs.iter()
                .map(|t| (t - mean).powi(2))
                .sum::<f64>() / throughputs.len() as f64;
            let stddev = variance.sqrt();
            let cv = stddev / mean; // Coefficient of variation
            
            if cv > tolerance {
                anomalies.push(Anomaly {
                    benchmark: benchmark.clone(),
                    database: database.clone(),
                    anomaly_type: AnomalyType::HighVariance,
                    message: format!(
                        "Results have high variance: CV = {:.1}% (tolerance: {:.1}%)",
                        cv * 100.0, tolerance * 100.0
                    ),
                    severity: AnomalySeverity::Warning,
                    suggestion: "Consider running more iterations or checking for system interference.".to_string(),
                });
            }
        }
    }
    
    anomalies
}


// ============================================================================
// Markdown Report Generation
// ============================================================================

/// Generate Markdown report file
/// _Requirements: 11.2_
pub fn generate_markdown_report(report: &BenchmarkReport, path: &Path) {
    let markdown = generate_markdown(report);
    fs::write(path, markdown).expect("Failed to write Markdown report");
}

/// Generate Markdown content from report
fn generate_markdown(report: &BenchmarkReport) -> String {
    let mut md = String::new();
    
    md.push_str("# Syna Benchmark Report\n\n");
    
    // System info
    md.push_str("## System Information\n\n");
    md.push_str(&format!("- **Date**: {}\n", report.metadata.timestamp));
    md.push_str(&format!("- **Syna Version**: {}\n", report.metadata.SYNA_version));
    md.push_str(&format!("- **Benchmark Version**: {}\n", report.metadata.benchmark_version));
    md.push_str(&format!("- **OS**: {}\n", report.metadata.system.os));
    md.push_str(&format!("- **CPU**: {}\n", report.metadata.system.cpu));
    md.push_str(&format!("- **Cores**: {}\n", report.metadata.system.cores));
    md.push_str(&format!("- **RAM**: {:.1} GB\n", report.metadata.system.ram_gb));
    md.push_str(&format!("- **Disk Type**: {}\n", report.metadata.system.disk_type));
    md.push_str("\n");
    
    // Configuration
    md.push_str("## Benchmark Configuration\n\n");
    md.push_str(&format!("- **Warmup Iterations**: {}\n", report.metadata.config.warmup_iterations));
    md.push_str(&format!("- **Measurement Iterations**: {}\n", report.metadata.config.measurement_iterations));
    md.push_str(&format!("- **Value Sizes**: {:?} bytes\n", report.metadata.config.value_sizes));
    md.push_str(&format!("- **Thread Counts**: {:?}\n", report.metadata.config.thread_counts));
    md.push_str(&format!("- **Databases Tested**: {}\n", report.metadata.config.databases_tested.join(", ")));
    md.push_str("\n");
    
    // Summary
    if let Some(summary) = &report.summary {
        md.push_str("## Summary\n\n");
        md.push_str(&format!("- **Total Benchmarks**: {}\n", summary.total_benchmarks));
        md.push_str(&format!("- **Total Duration**: {:.1} seconds\n", summary.total_duration_secs));
        md.push_str(&format!("- **Best Write Throughput**: {:.0} ops/sec ({} - {})\n",
            summary.best_write_throughput.throughput_ops_sec,
            summary.best_write_throughput.database,
            summary.best_write_throughput.benchmark
        ));
        md.push_str(&format!("- **Best Read Throughput**: {:.0} ops/sec ({} - {})\n",
            summary.best_read_throughput.throughput_ops_sec,
            summary.best_read_throughput.database,
            summary.best_read_throughput.benchmark
        ));
        md.push_str(&format!("- **Lowest Latency (p50)**: {:.1} μs ({} - {})\n",
            summary.lowest_latency.latency_p50_us,
            summary.lowest_latency.database,
            summary.lowest_latency.benchmark
        ));
        md.push_str("\n");
        
        // Comparison highlights
        if !summary.comparison_highlights.is_empty() {
            md.push_str("### Comparison Highlights\n\n");
            for highlight in &summary.comparison_highlights {
                md.push_str(&format!("- {}\n", highlight));
            }
            md.push_str("\n");
        }
    }
    
    // Group results by benchmark type
    let write_results: Vec<_> = report.results.iter()
        .filter(|r| r.benchmark.contains("write"))
        .collect();
    
    let read_results: Vec<_> = report.results.iter()
        .filter(|r| r.benchmark.contains("read"))
        .collect();
    
    let mixed_results: Vec<_> = report.results.iter()
        .filter(|r| r.benchmark.contains("ycsb") || r.benchmark.contains("timeseries"))
        .collect();
    
    let storage_results: Vec<_> = report.results.iter()
        .filter(|r| r.benchmark.contains("storage"))
        .collect();
    
    // Write performance table
    if !write_results.is_empty() {
        md.push_str("## Write Performance\n\n");
        md.push_str("| Database | Value Size | Throughput (ops/s) | p50 (μs) | p95 (μs) | p99 (μs) | Disk (MB) |\n");
        md.push_str("|----------|------------|-------------------|----------|----------|----------|----------|\n");
        
        for r in &write_results {
            md.push_str(&format!(
                "| {} | {} B | {:.0} | {:.1} | {:.1} | {:.1} | {:.2} |\n",
                r.database,
                r.config.value_size_bytes,
                r.throughput_ops_sec,
                r.latency_p50_us,
                r.latency_p95_us,
                r.latency_p99_us,
                r.disk_mb
            ));
        }
        md.push_str("\n");
    }
    
    // Read performance table
    if !read_results.is_empty() {
        md.push_str("## Read Performance\n\n");
        md.push_str("| Database | Benchmark | Threads | Throughput (ops/s) | p50 (μs) | p95 (μs) | p99 (μs) |\n");
        md.push_str("|----------|-----------|---------|-------------------|----------|----------|----------|\n");
        
        for r in &read_results {
            md.push_str(&format!(
                "| {} | {} | {} | {:.0} | {:.1} | {:.1} | {:.1} |\n",
                r.database,
                r.benchmark,
                r.config.thread_count,
                r.throughput_ops_sec,
                r.latency_p50_us,
                r.latency_p95_us,
                r.latency_p99_us
            ));
        }
        md.push_str("\n");
    }
    
    // Mixed workload table
    if !mixed_results.is_empty() {
        md.push_str("## Mixed Workloads\n\n");
        md.push_str("| Database | Workload | Throughput (ops/s) | p50 (μs) | p95 (μs) | Duration (s) |\n");
        md.push_str("|----------|----------|-------------------|----------|----------|-------------|\n");
        
        for r in &mixed_results {
            let workload_name = r.benchmark
                .replace("ycsb_", "YCSB-")
                .replace("timeseries", "Time-Series");
            md.push_str(&format!(
                "| {} | {} | {:.0} | {:.1} | {:.1} | {:.2} |\n",
                r.database,
                workload_name,
                r.throughput_ops_sec,
                r.latency_p50_us,
                r.latency_p95_us,
                r.duration_secs
            ));
        }
        md.push_str("\n");
    }
    
    // Storage efficiency table
    if !storage_results.is_empty() {
        md.push_str("## Storage Efficiency\n\n");
        md.push_str("| Database | Benchmark | Disk (MB) | Duration (s) |\n");
        md.push_str("|----------|-----------|-----------|-------------|\n");
        
        for r in &storage_results {
            md.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} |\n",
                r.database,
                r.benchmark,
                r.disk_mb,
                r.duration_secs
            ));
        }
        md.push_str("\n");
    }
    
    // Anomalies section
    if !report.anomalies.is_empty() {
        md.push_str("## ⚠️ Anomalies Detected\n\n");
        md.push_str("The following anomalies were detected during benchmarking:\n\n");
        md.push_str("| Severity | Benchmark | Database | Issue | Suggestion |\n");
        md.push_str("|----------|-----------|----------|-------|------------|\n");
        
        for anomaly in &report.anomalies {
            let severity_icon = match anomaly.severity {
                AnomalySeverity::Warning => "⚠️",
                AnomalySeverity::Error => "❌",
            };
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                severity_icon,
                anomaly.benchmark,
                anomaly.database,
                anomaly.message,
                anomaly.suggestion
            ));
        }
        md.push_str("\n");
    }
    
    // Footer
    md.push_str("---\n\n");
    md.push_str(&format!("*Report generated at {} by Syna Benchmark Suite v{}*\n",
        report.metadata.timestamp,
        report.metadata.benchmark_version
    ));
    
    md
}

// ============================================================================
// Chart Generation
// ============================================================================

/// Generate charts from benchmark results
/// _Requirements: 11.3_
pub fn generate_charts(report: &BenchmarkReport, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;
    
    // Generate throughput comparison chart
    generate_throughput_chart(report, output_path)?;
    
    // Generate latency distribution chart
    generate_latency_chart(report, output_path)?;
    
    Ok(())
}

/// Generate throughput comparison bar chart
/// _Requirements: 11.3_
fn generate_throughput_chart(report: &BenchmarkReport, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    
    let chart_path = output_path.join("throughput_comparison.png");
    
    // Group results by benchmark type for comparison
    let write_results: Vec<_> = report.results.iter()
        .filter(|r| r.benchmark.contains("write") && !r.benchmark.contains("sync"))
        .collect();
    
    if write_results.is_empty() {
        return Ok(());
    }
    
    // Find max throughput for y-axis scaling
    let max_throughput = write_results.iter()
        .map(|r| r.throughput_ops_sec)
        .fold(0.0f64, |a, b| a.max(b));
    
    let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Write Throughput Comparison", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(
            0..write_results.len(),
            0.0..(max_throughput * 1.1)
        )?;
    
    chart.configure_mesh()
        .x_labels(write_results.len())
        .x_label_formatter(&|idx| {
            write_results.get(*idx)
                .map(|r| format!("{}\n{}B", r.database, r.config.value_size_bytes))
                .unwrap_or_default()
        })
        .y_desc("Throughput (ops/sec)")
        .draw()?;
    
    // Draw bars
    chart.draw_series(
        write_results.iter().enumerate().map(|(idx, result)| {
            let color = match result.database.as_str() {
                "Syna" => BLUE,
                "sqlite" => GREEN,
                "duckdb" => RED,
                "leveldb" => CYAN,
                "rocksdb" => MAGENTA,
                _ => BLACK,
            };
            Rectangle::new(
                [(idx, 0.0), (idx + 1, result.throughput_ops_sec)],
                color.filled(),
            )
        })
    )?;
    
    root.present()?;
    println!("  Generated: {}", chart_path.display());
    
    Ok(())
}

/// Generate latency distribution chart
/// _Requirements: 11.3_
fn generate_latency_chart(report: &BenchmarkReport, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    
    let chart_path = output_path.join("latency_distribution.png");
    
    // Get results with valid latency data
    let results_with_latency: Vec<_> = report.results.iter()
        .filter(|r| r.latency_p50_us > 0.0 && r.latency_p99_us > 0.0)
        .take(10) // Limit to 10 for readability
        .collect();
    
    if results_with_latency.is_empty() {
        return Ok(());
    }
    
    // Find max latency for y-axis scaling
    let max_latency = results_with_latency.iter()
        .map(|r| r.latency_p99_us)
        .fold(0.0f64, |a, b| a.max(b));
    
    let root = BitMapBackend::new(&chart_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Latency Distribution (p50, p95, p99)", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(
            0..results_with_latency.len(),
            0.0..(max_latency * 1.1)
        )?;
    
    chart.configure_mesh()
        .x_labels(results_with_latency.len())
        .x_label_formatter(&|idx| {
            results_with_latency.get(*idx)
                .map(|r| format!("{}\n{}", r.database, &r.benchmark[..r.benchmark.len().min(10)]))
                .unwrap_or_default()
        })
        .y_desc("Latency (μs)")
        .draw()?;
    
    // Draw p50 bars
    chart.draw_series(
        results_with_latency.iter().enumerate().map(|(idx, result)| {
            Rectangle::new(
                [(idx, 0.0), (idx + 1, result.latency_p50_us)],
                BLUE.mix(0.8).filled(),
            )
        })
    )?.label("p50").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], BLUE.mix(0.8).filled()));
    
    // Draw p95 line
    chart.draw_series(
        results_with_latency.iter().enumerate().map(|(idx, result)| {
            Circle::new((idx, result.latency_p95_us), 5, GREEN.filled())
        })
    )?.label("p95").legend(|(x, y)| Circle::new((x + 10, y), 5, GREEN.filled()));
    
    // Draw p99 line
    chart.draw_series(
        results_with_latency.iter().enumerate().map(|(idx, result)| {
            Cross::new((idx, result.latency_p99_us), 5, RED.filled())
        })
    )?.label("p99").legend(|(x, y)| Cross::new((x + 10, y), 5, RED.filled()));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    root.present()?;
    println!("  Generated: {}", chart_path.display());
    
    Ok(())
}

/// Generate all report outputs (JSON, Markdown, Charts)
/// _Requirements: 11.1, 11.2, 11.3_
pub fn generate_full_report(results: &[BenchmarkResult], output_dir: &str) {
    println!("\n📊 Generating benchmark reports...");
    
    // Generate JSON and Markdown
    generate_report(results, output_dir);
    
    // Build report for charts
    let anomalies = detect_anomalies(results, &ExpectedRanges::default());
    let summary = generate_summary(results);
    
    let report = BenchmarkReport {
        metadata: ReportMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            SYNA_version: env!("CARGO_PKG_VERSION").to_string(),
            benchmark_version: "1.0.0".to_string(),
            system: SystemInfo::collect(),
            config: extract_config_from_results(results),
        },
        results: results.to_vec(),
        anomalies,
        summary: Some(summary),
    };
    
    // Generate charts
    match generate_charts(&report, output_dir) {
        Ok(_) => println!("  Charts generated successfully"),
        Err(e) => println!("  Warning: Failed to generate charts: {}", e),
    }
    
    println!("\n✅ Reports saved to {}/", output_dir);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchmarkConfig;
    
    fn sample_results() -> Vec<BenchmarkResult> {
        vec![
            BenchmarkResult {
                benchmark: "sequential_write".to_string(),
                database: "Syna".to_string(),
                config: BenchmarkConfig::default(),
                throughput_ops_sec: 100000.0,
                latency_p50_us: 5.0,
                latency_p95_us: 10.0,
                latency_p99_us: 20.0,
                memory_mb: 10.0,
                disk_mb: 50.0,
                duration_secs: 1.0,
            },
            BenchmarkResult {
                benchmark: "sequential_write".to_string(),
                database: "sqlite".to_string(),
                config: BenchmarkConfig::default(),
                throughput_ops_sec: 50000.0,
                latency_p50_us: 10.0,
                latency_p95_us: 20.0,
                latency_p99_us: 40.0,
                memory_mb: 15.0,
                disk_mb: 60.0,
                duration_secs: 2.0,
            },
            BenchmarkResult {
                benchmark: "random_read".to_string(),
                database: "Syna".to_string(),
                config: BenchmarkConfig::default(),
                throughput_ops_sec: 200000.0,
                latency_p50_us: 2.0,
                latency_p95_us: 5.0,
                latency_p99_us: 10.0,
                memory_mb: 10.0,
                disk_mb: 50.0,
                duration_secs: 0.5,
            },
        ]
    }
    
    #[test]
    fn test_generate_summary() {
        let results = sample_results();
        let summary = generate_summary(&results);
        
        assert_eq!(summary.total_benchmarks, 3);
        assert!(summary.total_duration_secs > 0.0);
        assert_eq!(summary.best_write_throughput.database, "Syna");
        assert_eq!(summary.best_read_throughput.database, "Syna");
    }
    
    #[test]
    fn test_detect_anomalies_zero_throughput() {
        let results = vec![
            BenchmarkResult {
                benchmark: "sequential_write".to_string(),
                database: "test".to_string(),
                config: BenchmarkConfig::default(),
                throughput_ops_sec: 0.0,
                latency_p50_us: 0.0,
                latency_p95_us: 0.0,
                latency_p99_us: 0.0,
                memory_mb: 0.0,
                disk_mb: 0.0,
                duration_secs: 0.0,
            },
        ];
        
        let anomalies = detect_anomalies(&results, &ExpectedRanges::default());
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::ZeroValue);
    }
    
    #[test]
    fn test_detect_anomalies_low_throughput() {
        let results = vec![
            BenchmarkResult {
                benchmark: "sequential_write".to_string(),
                database: "test".to_string(),
                config: BenchmarkConfig::default(),
                throughput_ops_sec: 100.0, // Very low
                latency_p50_us: 1000.0,
                latency_p95_us: 2000.0,
                latency_p99_us: 5000.0,
                memory_mb: 10.0,
                disk_mb: 50.0,
                duration_secs: 100.0,
            },
        ];
        
        let anomalies = detect_anomalies(&results, &ExpectedRanges::default());
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::UnexpectedlyLow);
    }
    
    #[test]
    fn test_generate_markdown() {
        let results = sample_results();
        let anomalies = detect_anomalies(&results, &ExpectedRanges::default());
        let summary = generate_summary(&results);
        
        let report = BenchmarkReport {
            metadata: ReportMetadata {
                timestamp: "2024-01-15T10:30:00Z".to_string(),
                SYNA_version: "0.1.0".to_string(),
                benchmark_version: "1.0.0".to_string(),
                system: SystemInfo {
                    os: "Linux".to_string(),
                    cpu: "Test CPU".to_string(),
                    cores: 8,
                    ram_gb: 16.0,
                    disk_type: "SSD".to_string(),
                },
                config: BenchmarkRunConfig::default(),
            },
            results,
            anomalies,
            summary: Some(summary),
        };
        
        let markdown = generate_markdown(&report);
        
        assert!(markdown.contains("# Syna Benchmark Report"));
        assert!(markdown.contains("## System Information"));
        assert!(markdown.contains("## Write Performance"));
        assert!(markdown.contains("## Read Performance"));
        assert!(markdown.contains("Syna"));
        assert!(markdown.contains("sqlite"));
    }
    
    #[test]
    fn test_comparison_highlights() {
        let results = sample_results();
        let highlights = generate_comparison_highlights(&results);
        
        // Syna is 2x faster than SQLite for writes
        assert!(!highlights.is_empty());
        assert!(highlights.iter().any(|h| h.contains("faster")));
    }
    
    #[test]
    fn test_extract_config() {
        let results = sample_results();
        let config = extract_config_from_results(&results);
        
        assert!(config.databases_tested.contains(&"Syna".to_string()));
        assert!(config.databases_tested.contains(&"sqlite".to_string()));
    }
}


