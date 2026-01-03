//! Simplified LSH for Cascade Index
//!
//! Follows SynaDB physics principles:
//! - Arrow of Time: Buckets are append-only lists, no splits/rewrites
//! - The Observer: Hash computation is pure, no state mutation
//!
//! # Design
//!
//! Instead of adaptive bucket trees that split (rewrite), we use simple
//! append-only bucket lists. This trades some query efficiency for
//! much faster writes and simpler code.

use std::collections::HashMap;

/// LSH configuration
#[derive(Clone, Debug)]
pub struct SimpleLSHConfig {
    /// Number of hash bits per table
    pub num_bits: usize,
    /// Number of hash tables
    pub num_tables: usize,
    /// Vector dimensions
    pub dimensions: usize,
}

impl Default for SimpleLSHConfig {
    fn default() -> Self {
        Self {
            num_bits: 6, // 64 buckets per table
            num_tables: 10,
            dimensions: 768,
        }
    }
}

/// Simple LSH with append-only buckets
pub struct SimpleLSH {
    /// Random hyperplanes for each table: [table][bit][dimension]
    hyperplanes: Vec<Vec<Vec<f32>>>,
    /// Buckets: [table][hash] -> list of vector IDs
    buckets: Vec<HashMap<u64, Vec<u32>>>,
    /// Configuration
    config: SimpleLSHConfig,
}

impl SimpleLSH {
    /// Create new LSH with random hyperplanes
    pub fn new(config: SimpleLSHConfig) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Generate deterministic random hyperplanes
        let mut hyperplanes = Vec::with_capacity(config.num_tables);

        for table in 0..config.num_tables {
            let mut table_planes = Vec::with_capacity(config.num_bits);

            for bit in 0..config.num_bits {
                let mut plane = Vec::with_capacity(config.dimensions);

                for dim in 0..config.dimensions {
                    // Deterministic "random" based on table, bit, dim
                    let mut hasher = DefaultHasher::new();
                    (table, bit, dim, "hyperplane").hash(&mut hasher);
                    let hash = hasher.finish();

                    // Convert to [-1, 1] range
                    let val = ((hash as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
                    plane.push(val);
                }

                // Normalize
                let norm: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in plane.iter_mut() {
                        *x /= norm;
                    }
                }

                table_planes.push(plane);
            }

            hyperplanes.push(table_planes);
        }

        let buckets = (0..config.num_tables).map(|_| HashMap::new()).collect();

        Self {
            hyperplanes,
            buckets,
            config,
        }
    }

    /// Compute hash for a vector in a specific table
    #[inline]
    pub fn hash(&self, vector: &[f32], table: usize) -> u64 {
        let mut hash = 0u64;

        for (bit, plane) in self.hyperplanes[table].iter().enumerate() {
            let dot: f32 = vector.iter().zip(plane.iter()).map(|(v, p)| v * p).sum();

            if dot > 0.0 {
                hash |= 1 << bit;
            }
        }

        hash
    }

    /// Compute hashes for all tables
    pub fn hash_all(&self, vector: &[f32]) -> Vec<u64> {
        (0..self.config.num_tables)
            .map(|t| self.hash(vector, t))
            .collect()
    }

    /// Insert vector ID into buckets (Arrow of Time - append only!)
    pub fn insert(&mut self, id: u32, vector: &[f32]) {
        for table in 0..self.config.num_tables {
            let hash = self.hash(vector, table);
            self.buckets[table].entry(hash).or_default().push(id);
        }
    }

    /// Query candidates from a single table
    pub fn query_table(&self, vector: &[f32], table: usize) -> &[u32] {
        let hash = self.hash(vector, table);
        self.buckets[table]
            .get(&hash)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Query candidates with multi-probe (flip bits to check nearby buckets)
    pub fn query_multiprobe(&self, vector: &[f32], num_probes: usize) -> Vec<u32> {
        let mut candidates = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for table in 0..self.config.num_tables {
            let base_hash = self.hash(vector, table);

            // Add base bucket
            if let Some(ids) = self.buckets[table].get(&base_hash) {
                for &id in ids {
                    if seen.insert(id) {
                        candidates.push(id);
                    }
                }
            }

            // Multi-probe: flip bits to check nearby buckets
            for probe in 0..num_probes.min(self.config.num_bits) {
                let probe_hash = base_hash ^ (1 << probe);
                if let Some(ids) = self.buckets[table].get(&probe_hash) {
                    for &id in ids {
                        if seen.insert(id) {
                            candidates.push(id);
                        }
                    }
                }
            }
        }

        candidates
    }

    /// Get probe sequence for a table (base hash + flipped bits)
    pub fn get_probe_sequence(&self, vector: &[f32], table: usize, num_probes: usize) -> Vec<u64> {
        let base_hash = self.hash(vector, table);
        let mut probes = vec![base_hash];

        for bit in 0..num_probes.min(self.config.num_bits) {
            probes.push(base_hash ^ (1 << bit));
        }

        probes
    }

    /// Number of tables
    pub fn num_tables(&self) -> usize {
        self.config.num_tables
    }

    /// Total bucket count across all tables
    pub fn bucket_count(&self) -> usize {
        self.buckets.iter().map(|t| t.len()).sum()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Config
        bytes.extend(&(self.config.num_bits as u32).to_le_bytes());
        bytes.extend(&(self.config.num_tables as u32).to_le_bytes());
        bytes.extend(&(self.config.dimensions as u32).to_le_bytes());

        // Hyperplanes
        for table in &self.hyperplanes {
            for plane in table {
                for &val in plane {
                    bytes.extend(&val.to_le_bytes());
                }
            }
        }

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }

        let num_bits = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
        let num_tables = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
        let dimensions = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;

        let config = SimpleLSHConfig {
            num_bits,
            num_tables,
            dimensions,
        };

        // Read hyperplanes
        let mut offset = 12;
        let mut hyperplanes = Vec::with_capacity(num_tables);

        for _ in 0..num_tables {
            let mut table_planes = Vec::with_capacity(num_bits);

            for _ in 0..num_bits {
                let mut plane = Vec::with_capacity(dimensions);

                for _ in 0..dimensions {
                    if offset + 4 > bytes.len() {
                        return None;
                    }
                    let val = f32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
                    plane.push(val);
                    offset += 4;
                }

                table_planes.push(plane);
            }

            hyperplanes.push(table_planes);
        }

        let buckets = (0..num_tables).map(|_| HashMap::new()).collect();

        Some(Self {
            hyperplanes,
            buckets,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_lsh_hash() {
        let config = SimpleLSHConfig {
            num_bits: 6,
            num_tables: 4,
            dimensions: 64,
        };

        let lsh = SimpleLSH::new(config);

        let vec1: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let vec2: Vec<f32> = (0..64).map(|i| i as f32 * 0.01 + 0.001).collect();

        // Similar vectors should have similar hashes
        let h1 = lsh.hash_all(&vec1);
        let h2 = lsh.hash_all(&vec2);

        // At least some tables should match
        let matches = h1.iter().zip(h2.iter()).filter(|(a, b)| a == b).count();
        assert!(
            matches > 0,
            "Similar vectors should have some matching hashes"
        );
    }

    #[test]
    fn test_simple_lsh_insert_query() {
        let config = SimpleLSHConfig {
            num_bits: 4,
            num_tables: 2,
            dimensions: 32,
        };

        let mut lsh = SimpleLSH::new(config);

        // Insert some vectors
        for i in 0..10 {
            let vec: Vec<f32> = (0..32).map(|j| (i * 32 + j) as f32 * 0.01).collect();
            lsh.insert(i, &vec);
        }

        // Query should return candidates
        let query: Vec<f32> = (0..32).map(|j| j as f32 * 0.01).collect();
        let candidates = lsh.query_multiprobe(&query, 4);

        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_simple_lsh_serialization() {
        let config = SimpleLSHConfig {
            num_bits: 4,
            num_tables: 2,
            dimensions: 16,
        };

        let lsh = SimpleLSH::new(config);
        let bytes = lsh.to_bytes();

        let restored = SimpleLSH::from_bytes(&bytes).unwrap();

        // Verify same hashes
        let vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(lsh.hash_all(&vec), restored.hash_all(&vec));
    }
}
