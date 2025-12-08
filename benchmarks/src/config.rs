//! Benchmark configuration utilities.

use serde::{Deserialize, Serialize};
use sysinfo::System;

/// System information for benchmark reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub cpu: String,
    pub cores: usize,
    pub ram_gb: f64,
    pub disk_type: String,
}

impl SystemInfo {
    pub fn collect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        let os = format!("{} {}", 
            System::name().unwrap_or_default(),
            System::os_version().unwrap_or_default()
        );
        
        let cpu = sys.cpus().first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        
        let cores = sys.cpus().len();
        let ram_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        
        Self {
            os,
            cpu,
            cores,
            ram_gb,
            disk_type: "Unknown".to_string(), // Would need platform-specific detection
        }
    }
}

/// Generate random bytes of specified size
pub fn random_bytes(size: usize, seed: u64) -> Vec<u8> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bytes = vec![0u8; size];
    rng.fill(&mut bytes[..]);
    bytes
}

/// Generate a random key
pub fn random_key(prefix: &str, seed: u64) -> String {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    format!("{}/{:016x}", prefix, rng.gen::<u64>())
}


