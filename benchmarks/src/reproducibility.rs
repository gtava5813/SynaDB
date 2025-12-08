//! Reproducibility infrastructure for benchmarks.
//!
//! This module provides tools for ensuring benchmark reproducibility:
//! - Deterministic random number generation with fixed seeds
//! - Configurable warmup and measurement phases
//! - OS cache flushing for cold-start tests
//! - Statistical analysis with multiple iterations
//!
//! _Requirements: 12.1, 12.2, 12.3, 12.4_

use crate::BenchmarkResult;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ============================================================================
// Deterministic Random Number Generation
// ============================================================================

/// Default seed for reproducible benchmarks
/// _Requirements: 12.1_
pub const DEFAULT_SEED: u64 = 0xDEADBEEF_CAFEBABE;

/// Deterministic random number generator for benchmarks.
/// Uses ChaCha8 for cryptographic quality randomness with reproducibility.
/// 
/// _Requirements: 12.1_
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    rng: ChaCha8Rng,
    seed: u64,
}

impl DeterministicRng {
    /// Create a new deterministic RNG with the given seed.
    /// 
    /// # Example
    /// ```
    /// use SYNA_benchmarks::reproducibility::DeterministicRng;
    /// 
    /// let mut rng1 = DeterministicRng::new(42);
    /// let mut rng2 = DeterministicRng::new(42);
    /// 
    /// // Same seed produces same sequence
    /// assert_eq!(rng1.next_u64(), rng2.next_u64());
    /// ```
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
        }
    }

    /// Create a new deterministic RNG with the default seed.
    pub fn with_default_seed() -> Self {
        Self::new(DEFAULT_SEED)
    }

    /// Get the seed used to initialize this RNG.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Generate the next u64 value.
    pub fn next_u64(&mut self) -> u64 {
        self.rng.gen()
    }

    /// Generate a random value in the given range.
    pub fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        self.rng.gen_range(range)
    }

    /// Generate random bytes of the specified length.
    pub fn gen_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; len];
        self.rng.fill(&mut bytes[..]);
        bytes
    }

    /// Generate a random key with the given prefix.
    pub fn gen_key(&mut self, prefix: &str) -> String {
        format!("{}/{:016x}", prefix, self.next_u64())
    }

    /// Fork this RNG to create a child RNG with a derived seed.
    /// Useful for parallel benchmarks where each thread needs its own RNG.
    pub fn fork(&mut self) -> Self {
        Self::new(self.next_u64())
    }
}

impl Default for DeterministicRng {
    fn default() -> Self {
        Self::with_default_seed()
    }
}

// ============================================================================
// Configurable Warmup and Measurement Phases
// ============================================================================

/// Configuration for benchmark phases.
/// 
/// _Requirements: 12.2_
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    /// Number of warmup iterations (not measured)
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Minimum warmup duration (warmup continues until both iteration count and duration are met)
    pub min_warmup_duration: Duration,
    /// Whether to discard outliers from measurement
    pub discard_outliers: bool,
    /// Percentile threshold for outlier detection (e.g., 0.99 means discard top 1%)
    pub outlier_percentile: f64,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 1000,
            measurement_iterations: 10000,
            min_warmup_duration: Duration::from_secs(1),
            discard_outliers: false,
            outlier_percentile: 0.99,
        }
    }
}

impl PhaseConfig {
    /// Create a quick configuration for testing.
    pub fn quick() -> Self {
        Self {
            warmup_iterations: 100,
            measurement_iterations: 1000,
            min_warmup_duration: Duration::from_millis(100),
            ..Default::default()
        }
    }

    /// Create a thorough configuration for production benchmarks.
    pub fn thorough() -> Self {
        Self {
            warmup_iterations: 5000,
            measurement_iterations: 50000,
            min_warmup_duration: Duration::from_secs(5),
            discard_outliers: true,
            outlier_percentile: 0.99,
        }
    }
}

/// A benchmark phase runner that handles warmup and measurement.
/// 
/// _Requirements: 12.2_
pub struct PhaseRunner {
    config: PhaseConfig,
    rng: DeterministicRng,
}

impl PhaseRunner {
    /// Create a new phase runner with the given configuration and seed.
    pub fn new(config: PhaseConfig, seed: u64) -> Self {
        Self {
            config,
            rng: DeterministicRng::new(seed),
        }
    }

    /// Create a new phase runner with default configuration.
    pub fn with_defaults(seed: u64) -> Self {
        Self::new(PhaseConfig::default(), seed)
    }

    /// Get a mutable reference to the RNG.
    pub fn rng(&mut self) -> &mut DeterministicRng {
        &mut self.rng
    }

    /// Run the warmup phase.
    /// 
    /// Executes the warmup function until both the iteration count and
    /// minimum duration are satisfied.
    pub fn run_warmup<F>(&mut self, mut warmup_fn: F)
    where
        F: FnMut(&mut DeterministicRng),
    {
        let start = Instant::now();
        let mut iterations = 0;

        // Run until both conditions are met
        while iterations < self.config.warmup_iterations 
            || start.elapsed() < self.config.min_warmup_duration 
        {
            warmup_fn(&mut self.rng);
            iterations += 1;
        }
    }

    /// Run the measurement phase and collect latencies.
    /// 
    /// Returns a vector of durations for each operation.
    pub fn run_measurement<F>(&mut self, mut measure_fn: F) -> Vec<Duration>
    where
        F: FnMut(&mut DeterministicRng),
    {
        let mut latencies = Vec::with_capacity(self.config.measurement_iterations);

        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            measure_fn(&mut self.rng);
            latencies.push(start.elapsed());
        }

        // Optionally discard outliers
        if self.config.discard_outliers {
            latencies = self.remove_outliers(latencies);
        }

        latencies
    }

    /// Remove outliers from latency measurements.
    fn remove_outliers(&self, mut latencies: Vec<Duration>) -> Vec<Duration> {
        if latencies.is_empty() {
            return latencies;
        }

        latencies.sort();
        let cutoff_idx = (latencies.len() as f64 * self.config.outlier_percentile) as usize;
        latencies.truncate(cutoff_idx.max(1));
        latencies
    }

    /// Run a complete benchmark with warmup and measurement phases.
    /// 
    /// Returns the collected latencies from the measurement phase.
    pub fn run_benchmark<W, M>(&mut self, warmup_fn: W, measure_fn: M) -> Vec<Duration>
    where
        W: FnMut(&mut DeterministicRng),
        M: FnMut(&mut DeterministicRng),
    {
        self.run_warmup(warmup_fn);
        self.run_measurement(measure_fn)
    }
}


// ============================================================================
// OS Cache Flushing
// ============================================================================

/// Cache flushing utilities for cold-start benchmarks.
/// 
/// _Requirements: 12.3_
pub mod cache {
    use std::path::Path;
    
    #[cfg(target_os = "macos")]
    use std::process::Command;

    /// Result type for cache operations.
    pub type CacheResult<T> = Result<T, CacheError>;

    /// Error type for cache operations.
    #[derive(Debug)]
    pub enum CacheError {
        /// Operation not supported on this platform
        NotSupported(String),
        /// Operation failed
        Failed(String),
        /// Insufficient permissions
        PermissionDenied(String),
    }

    impl std::fmt::Display for CacheError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                CacheError::NotSupported(msg) => write!(f, "Not supported: {}", msg),
                CacheError::Failed(msg) => write!(f, "Failed: {}", msg),
                CacheError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            }
        }
    }

    impl std::error::Error for CacheError {}

    /// Flush OS page cache for a specific file.
    /// 
    /// This is useful for cold-start benchmarks where you want to ensure
    /// the file is not cached in memory.
    /// 
    /// # Platform Support
    /// - Linux: Uses `posix_fadvise` with `POSIX_FADV_DONTNEED`
    /// - Windows: Uses `FlushFileBuffers` (limited effectiveness)
    /// - macOS: Uses `fcntl` with `F_NOCACHE` (limited effectiveness)
    /// 
    /// _Requirements: 12.3_
    #[cfg(target_os = "linux")]
    pub fn flush_file_cache(path: &Path) -> CacheResult<()> {
        use std::fs::File;
        use std::os::unix::io::AsRawFd;

        let file = File::open(path)
            .map_err(|e| CacheError::Failed(format!("Failed to open file: {}", e)))?;

        let fd = file.as_raw_fd();
        let file_size = file.metadata()
            .map_err(|e| CacheError::Failed(format!("Failed to get file size: {}", e)))?
            .len() as libc::off_t;

        // Use posix_fadvise to tell the kernel we don't need the cached pages
        let result = unsafe {
            libc::posix_fadvise(fd, 0, file_size, libc::POSIX_FADV_DONTNEED)
        };

        if result == 0 {
            Ok(())
        } else {
            Err(CacheError::Failed(format!(
                "posix_fadvise failed with error code: {}",
                result
            )))
        }
    }

    #[cfg(target_os = "windows")]
    pub fn flush_file_cache(path: &Path) -> CacheResult<()> {
        use std::fs::OpenOptions;
        use std::os::windows::fs::OpenOptionsExt;

        // Open with FILE_FLAG_NO_BUFFERING to bypass cache
        // Note: This is a best-effort approach on Windows
        // We just open and close the file with no buffering flag
        let _file = OpenOptions::new()
            .read(true)
            .custom_flags(0x20000000) // FILE_FLAG_NO_BUFFERING
            .open(path)
            .map_err(|e| CacheError::Failed(format!("Failed to open file: {}", e)))?;

        // On Windows, cache flushing is limited without admin privileges
        // The FILE_FLAG_NO_BUFFERING helps for future reads
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub fn flush_file_cache(path: &Path) -> CacheResult<()> {
        use std::fs::File;
        use std::os::unix::io::AsRawFd;

        let file = File::open(path)
            .map_err(|e| CacheError::Failed(format!("Failed to open file: {}", e)))?;

        let fd = file.as_raw_fd();

        // Use F_NOCACHE to disable caching for this file descriptor
        // Note: This only affects future I/O, not already cached pages
        let result = unsafe {
            libc::fcntl(fd, libc::F_NOCACHE, 1)
        };

        if result == 0 {
            Ok(())
        } else {
            Err(CacheError::Failed(format!(
                "fcntl F_NOCACHE failed with error code: {}",
                result
            )))
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    pub fn flush_file_cache(_path: &Path) -> CacheResult<()> {
        Err(CacheError::NotSupported(
            "Cache flushing not implemented for this platform".to_string()
        ))
    }

    /// Flush the entire OS page cache.
    /// 
    /// **Warning**: This requires root/administrator privileges on most systems.
    /// 
    /// # Platform Support
    /// - Linux: Writes to `/proc/sys/vm/drop_caches`
    /// - Windows: Not supported (would require system-level access)
    /// - macOS: Uses `purge` command (requires sudo)
    /// 
    /// _Requirements: 12.3_
    #[cfg(target_os = "linux")]
    pub fn flush_system_cache() -> CacheResult<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        // Sync all pending writes first
        unsafe { libc::sync() };

        // Drop caches (requires root)
        let mut file = OpenOptions::new()
            .write(true)
            .open("/proc/sys/vm/drop_caches")
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    CacheError::PermissionDenied(
                        "Root privileges required to flush system cache".to_string()
                    )
                } else {
                    CacheError::Failed(format!("Failed to open drop_caches: {}", e))
                }
            })?;

        // Write "3" to drop pagecache, dentries, and inodes
        file.write_all(b"3")
            .map_err(|e| CacheError::Failed(format!("Failed to write to drop_caches: {}", e)))?;

        Ok(())
    }

    #[cfg(target_os = "windows")]
    pub fn flush_system_cache() -> CacheResult<()> {
        Err(CacheError::NotSupported(
            "System-wide cache flushing not supported on Windows".to_string()
        ))
    }

    #[cfg(target_os = "macos")]
    pub fn flush_system_cache() -> CacheResult<()> {
        // Try to run the purge command (requires sudo)
        let output = Command::new("purge")
            .output()
            .map_err(|e| CacheError::Failed(format!("Failed to run purge: {}", e)))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("permission") || stderr.contains("Operation not permitted") {
                Err(CacheError::PermissionDenied(
                    "Root privileges required to run purge".to_string()
                ))
            } else {
                Err(CacheError::Failed(format!("purge failed: {}", stderr)))
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    pub fn flush_system_cache() -> CacheResult<()> {
        Err(CacheError::NotSupported(
            "System cache flushing not implemented for this platform".to_string()
        ))
    }

    /// Attempt to flush caches, logging warnings on failure.
    /// 
    /// This is a best-effort function that won't fail the benchmark
    /// if cache flushing is not available.
    pub fn try_flush_file_cache(path: &Path) {
        match flush_file_cache(path) {
            Ok(()) => {}
            Err(CacheError::NotSupported(msg)) => {
                eprintln!("Warning: Cache flush not supported: {}", msg);
            }
            Err(CacheError::PermissionDenied(msg)) => {
                eprintln!("Warning: Cache flush permission denied: {}", msg);
            }
            Err(CacheError::Failed(msg)) => {
                eprintln!("Warning: Cache flush failed: {}", msg);
            }
        }
    }

    /// Attempt to flush system cache, logging warnings on failure.
    pub fn try_flush_system_cache() {
        match flush_system_cache() {
            Ok(()) => {}
            Err(CacheError::NotSupported(msg)) => {
                eprintln!("Warning: System cache flush not supported: {}", msg);
            }
            Err(CacheError::PermissionDenied(msg)) => {
                eprintln!("Warning: System cache flush permission denied: {}", msg);
            }
            Err(CacheError::Failed(msg)) => {
                eprintln!("Warning: System cache flush failed: {}", msg);
            }
        }
    }
}


// ============================================================================
// Statistical Analysis
// ============================================================================

/// Statistical analysis results for benchmark runs.
/// 
/// _Requirements: 12.4_
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// Number of iterations/runs analyzed
    pub count: usize,
    /// Mean (average) value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub stddev: f64,
    /// Variance
    pub variance: f64,
    /// Coefficient of variation (stddev / mean)
    pub cv: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 50th percentile (same as median)
    pub p50: f64,
    /// 95th percentile
    pub p95: f64,
    /// 99th percentile
    pub p99: f64,
    /// Interquartile range (p75 - p25)
    pub iqr: f64,
}

impl StatisticalAnalysis {
    /// Compute statistical analysis from a slice of f64 values.
    /// 
    /// _Requirements: 12.4_
    pub fn from_values(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let variance = if count > 1 {
            values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let stddev = variance.sqrt();
        let cv = if mean != 0.0 { stddev / mean } else { 0.0 };

        let min = sorted[0];
        let max = sorted[count - 1];
        let median = percentile(&sorted, 0.50);
        let p50 = median;
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);
        let p25 = percentile(&sorted, 0.25);
        let p75 = percentile(&sorted, 0.75);
        let iqr = p75 - p25;

        Some(Self {
            count,
            mean,
            median,
            stddev,
            variance,
            cv,
            min,
            max,
            p50,
            p95,
            p99,
            iqr,
        })
    }

    /// Compute statistical analysis from Duration values.
    /// 
    /// Converts durations to microseconds for analysis.
    pub fn from_durations(durations: &[Duration]) -> Option<Self> {
        let values: Vec<f64> = durations.iter()
            .map(|d| d.as_secs_f64() * 1_000_000.0)
            .collect();
        Self::from_values(&values)
    }

    /// Check if the results are consistent within the given tolerance.
    /// 
    /// Returns true if the coefficient of variation is below the tolerance.
    pub fn is_consistent(&self, tolerance: f64) -> bool {
        self.cv <= tolerance
    }

    /// Format the analysis as a human-readable string.
    pub fn format(&self, unit: &str) -> String {
        format!(
            "n={}, mean={:.2}{}, stddev={:.2}{}, cv={:.1}%, p50={:.2}{}, p95={:.2}{}, p99={:.2}{}",
            self.count,
            self.mean, unit,
            self.stddev, unit,
            self.cv * 100.0,
            self.p50, unit,
            self.p95, unit,
            self.p99, unit
        )
    }
}

/// Calculate percentile from a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = (sorted.len() as f64 * p) as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

/// Results from running multiple iterations of a benchmark.
/// 
/// _Requirements: 12.4_
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRunResults {
    /// Individual run results
    pub runs: Vec<BenchmarkResult>,
    /// Statistical analysis of throughput
    pub throughput_stats: StatisticalAnalysis,
    /// Statistical analysis of p50 latency
    pub latency_p50_stats: StatisticalAnalysis,
    /// Statistical analysis of p99 latency
    pub latency_p99_stats: StatisticalAnalysis,
    /// Seed used for reproducibility
    pub seed: u64,
    /// Number of iterations per run
    pub iterations_per_run: usize,
}

impl MultiRunResults {
    /// Create from a vector of benchmark results.
    pub fn from_results(results: Vec<BenchmarkResult>, seed: u64) -> Option<Self> {
        if results.is_empty() {
            return None;
        }

        let throughputs: Vec<f64> = results.iter()
            .map(|r| r.throughput_ops_sec)
            .collect();
        let latencies_p50: Vec<f64> = results.iter()
            .map(|r| r.latency_p50_us)
            .collect();
        let latencies_p99: Vec<f64> = results.iter()
            .map(|r| r.latency_p99_us)
            .collect();

        let iterations_per_run = results.first()
            .map(|r| r.config.measurement_iterations)
            .unwrap_or(0);

        Some(Self {
            throughput_stats: StatisticalAnalysis::from_values(&throughputs)?,
            latency_p50_stats: StatisticalAnalysis::from_values(&latencies_p50)?,
            latency_p99_stats: StatisticalAnalysis::from_values(&latencies_p99)?,
            runs: results,
            seed,
            iterations_per_run,
        })
    }

    /// Check if results are consistent within tolerance.
    pub fn is_consistent(&self, tolerance: f64) -> bool {
        self.throughput_stats.is_consistent(tolerance)
    }

    /// Print a summary of the multi-run results.
    pub fn print_summary(&self) {
        println!("\n=== Multi-Run Statistical Summary ===");
        println!("Runs: {}", self.runs.len());
        println!("Seed: 0x{:016X}", self.seed);
        println!("Iterations per run: {}", self.iterations_per_run);
        println!();
        println!("Throughput (ops/sec):");
        println!("  {}", self.throughput_stats.format(" ops/s"));
        println!();
        println!("Latency p50 (μs):");
        println!("  {}", self.latency_p50_stats.format("μs"));
        println!();
        println!("Latency p99 (μs):");
        println!("  {}", self.latency_p99_stats.format("μs"));
        println!();
        
        let tolerance = 0.10; // 10% CV threshold
        if self.is_consistent(tolerance) {
            println!("✓ Results are consistent (CV < {:.0}%)", tolerance * 100.0);
        } else {
            println!("⚠ Results have high variance (CV = {:.1}%)", 
                self.throughput_stats.cv * 100.0);
            println!("  Consider running more iterations or checking for system interference.");
        }
    }
}

/// Run a benchmark multiple times and collect statistical analysis.
/// 
/// _Requirements: 12.4_
pub fn run_multiple_iterations<F>(
    num_runs: usize,
    seed: u64,
    mut benchmark_fn: F,
) -> MultiRunResults
where
    F: FnMut(u64) -> BenchmarkResult,
{
    let mut rng = DeterministicRng::new(seed);
    let mut results = Vec::with_capacity(num_runs);

    for run_idx in 0..num_runs {
        // Generate a unique seed for each run, but derived from the master seed
        let run_seed = rng.next_u64();
        
        println!("  Run {}/{} (seed: 0x{:016X})...", run_idx + 1, num_runs, run_seed);
        let result = benchmark_fn(run_seed);
        println!("    Throughput: {:.0} ops/sec", result.throughput_ops_sec);
        
        results.push(result);
    }

    MultiRunResults::from_results(results, seed)
        .expect("Failed to compute statistics from results")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng_reproducibility() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);

        // Same seed should produce same sequence
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_deterministic_rng_different_seeds() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(43);

        // Different seeds should produce different sequences
        let mut same_count = 0;
        for _ in 0..100 {
            if rng1.next_u64() == rng2.next_u64() {
                same_count += 1;
            }
        }
        // Statistically, we shouldn't get many matches
        assert!(same_count < 5);
    }

    #[test]
    fn test_deterministic_rng_gen_bytes() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);

        let bytes1 = rng1.gen_bytes(100);
        let bytes2 = rng2.gen_bytes(100);

        assert_eq!(bytes1, bytes2);
        assert_eq!(bytes1.len(), 100);
    }

    #[test]
    fn test_deterministic_rng_fork() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);

        let mut fork1 = rng1.fork();
        let mut fork2 = rng2.fork();

        // Forked RNGs should produce same sequence
        for _ in 0..100 {
            assert_eq!(fork1.next_u64(), fork2.next_u64());
        }
    }

    #[test]
    fn test_statistical_analysis() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = StatisticalAnalysis::from_values(&values).unwrap();

        assert_eq!(stats.count, 10);
        assert!((stats.mean - 5.5).abs() < 0.001);
        assert!((stats.min - 1.0).abs() < 0.001);
        assert!((stats.max - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_statistical_analysis_empty() {
        let values: Vec<f64> = vec![];
        let stats = StatisticalAnalysis::from_values(&values);
        assert!(stats.is_none());
    }

    #[test]
    fn test_statistical_analysis_single_value() {
        let values = vec![42.0];
        let stats = StatisticalAnalysis::from_values(&values).unwrap();

        assert_eq!(stats.count, 1);
        assert!((stats.mean - 42.0).abs() < 0.001);
        assert!((stats.variance - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_phase_config_defaults() {
        let config = PhaseConfig::default();
        assert_eq!(config.warmup_iterations, 1000);
        assert_eq!(config.measurement_iterations, 10000);
    }

    #[test]
    fn test_phase_runner_warmup() {
        let config = PhaseConfig {
            warmup_iterations: 10,
            min_warmup_duration: Duration::from_millis(1),
            ..Default::default()
        };
        let mut runner = PhaseRunner::new(config, 42);
        
        let mut count = 0;
        runner.run_warmup(|_rng| {
            count += 1;
        });

        // Should run at least warmup_iterations times
        assert!(count >= 10);
    }

    #[test]
    fn test_phase_runner_measurement() {
        let config = PhaseConfig {
            measurement_iterations: 100,
            ..Default::default()
        };
        let mut runner = PhaseRunner::new(config, 42);

        let latencies = runner.run_measurement(|_rng| {
            // Simulate some work
            std::thread::sleep(Duration::from_micros(1));
        });

        assert_eq!(latencies.len(), 100);
    }

    #[test]
    fn test_consistency_check() {
        // Low variance data should be consistent
        let values = vec![100.0, 101.0, 99.0, 100.5, 99.5];
        let stats = StatisticalAnalysis::from_values(&values).unwrap();
        assert!(stats.is_consistent(0.10)); // 10% tolerance

        // High variance data should not be consistent
        let values = vec![100.0, 200.0, 50.0, 150.0, 75.0];
        let stats = StatisticalAnalysis::from_values(&values).unwrap();
        assert!(!stats.is_consistent(0.10)); // 10% tolerance
    }
}


