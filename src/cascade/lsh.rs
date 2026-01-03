//! Locality-Sensitive Hashing for Cascade Index
//!
//! Uses random hyperplane LSH for cosine similarity.
//! Each hash table uses k random hyperplanes to partition the space.

use rand::prelude::*;
use rand_distr::StandardNormal;

/// Configuration for LSH
#[derive(Clone, Debug)]
pub struct LSHConfig {
    /// Number of hash bits per table
    pub num_bits: usize,

    /// Number of hash tables
    pub num_tables: usize,

    /// Vector dimensions
    pub dimensions: usize,
}

impl Default for LSHConfig {
    fn default() -> Self {
        Self {
            num_bits: 12,
            num_tables: 8,
            dimensions: 768,
        }
    }
}

/// Random hyperplane LSH for cosine similarity
///
/// For cosine similarity, we use random hyperplanes through the origin.
/// A vector is assigned bit 1 if it's on the positive side of the hyperplane,
/// and bit 0 otherwise.
pub struct HyperplaneLSH {
    config: LSHConfig,

    /// Hyperplanes stored as contiguous array: [table][bit][dimension]
    /// Total size: num_tables * num_bits * dimensions
    planes: Vec<f32>,
}

impl HyperplaneLSH {
    /// Create new LSH with random hyperplanes
    pub fn new(config: LSHConfig) -> Self {
        let mut rng = rand::thread_rng();
        let total_planes = config.num_tables * config.num_bits;
        let total_floats = total_planes * config.dimensions;

        // Generate random hyperplanes from standard normal distribution
        let mut planes = Vec::with_capacity(total_floats);
        for _ in 0..total_floats {
            planes.push(rng.sample(StandardNormal));
        }

        // Normalize each hyperplane for numerical stability
        for plane_idx in 0..total_planes {
            let start = plane_idx * config.dimensions;
            let end = start + config.dimensions;
            let plane = &mut planes[start..end];

            let norm: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in plane.iter_mut() {
                    *x /= norm;
                }
            }
        }

        Self { config, planes }
    }

    /// Create LSH with a specific seed for reproducibility
    pub fn with_seed(config: LSHConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let total_planes = config.num_tables * config.num_bits;
        let total_floats = total_planes * config.dimensions;

        let mut planes = Vec::with_capacity(total_floats);
        for _ in 0..total_floats {
            planes.push(rng.sample(StandardNormal));
        }

        // Normalize
        for plane_idx in 0..total_planes {
            let start = plane_idx * config.dimensions;
            let end = start + config.dimensions;
            let plane = &mut planes[start..end];

            let norm: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in plane.iter_mut() {
                    *x /= norm;
                }
            }
        }

        Self { config, planes }
    }

    /// Compute hash for all tables
    pub fn hash(&self, vector: &[f32]) -> Vec<u64> {
        (0..self.config.num_tables)
            .map(|table| self.hash_single_table(table, vector))
            .collect()
    }

    /// Compute hash for a single table
    pub fn hash_single_table(&self, table: usize, vector: &[f32]) -> u64 {
        let mut hash: u64 = 0;

        for bit in 0..self.config.num_bits {
            let plane_idx = table * self.config.num_bits + bit;
            let plane_start = plane_idx * self.config.dimensions;
            let plane = &self.planes[plane_start..plane_start + self.config.dimensions];

            // Compute dot product
            let dot: f32 = vector.iter().zip(plane.iter()).map(|(v, p)| v * p).sum();

            // Set bit if on positive side
            if dot > 0.0 {
                hash |= 1 << bit;
            }
        }

        hash
    }

    /// Get probe sequence for multi-probe LSH
    /// Returns hashes in order of increasing Hamming distance from original
    pub fn get_probe_sequence(&self, vector: &[f32], table: usize, num_probes: usize) -> Vec<u64> {
        let original_hash = self.hash_single_table(table, vector);
        let mut probes = Vec::with_capacity(num_probes);
        probes.push(original_hash);

        if num_probes <= 1 {
            return probes;
        }

        // Compute distances to each hyperplane for smart probing
        let mut bit_distances: Vec<(usize, f32)> = Vec::with_capacity(self.config.num_bits);

        for bit in 0..self.config.num_bits {
            let plane_idx = table * self.config.num_bits + bit;
            let plane_start = plane_idx * self.config.dimensions;
            let plane = &self.planes[plane_start..plane_start + self.config.dimensions];

            let dot: f32 = vector.iter().zip(plane.iter()).map(|(v, p)| v * p).sum();

            // Distance to hyperplane (absolute value of dot product)
            bit_distances.push((bit, dot.abs()));
        }

        // Sort by distance (closest hyperplanes first - most likely to flip)
        bit_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Generate probes by flipping bits in order of closeness
        // First: single bit flips
        for (bit, _) in bit_distances.iter() {
            if probes.len() >= num_probes {
                break;
            }
            let flipped = original_hash ^ (1 << bit);
            if !probes.contains(&flipped) {
                probes.push(flipped);
            }
        }

        // If we need more probes, do double bit flips
        if probes.len() < num_probes {
            'outer: for i in 0..bit_distances.len() {
                for j in (i + 1)..bit_distances.len() {
                    if probes.len() >= num_probes {
                        break 'outer;
                    }
                    let flipped =
                        original_hash ^ (1 << bit_distances[i].0) ^ (1 << bit_distances[j].0);
                    if !probes.contains(&flipped) {
                        probes.push(flipped);
                    }
                }
            }
        }

        probes
    }

    /// Get configuration
    pub fn config(&self) -> &LSHConfig {
        &self.config
    }

    /// Get number of tables
    pub fn num_tables(&self) -> usize {
        self.config.num_tables
    }

    /// Get number of bits per table
    pub fn num_bits(&self) -> usize {
        self.config.num_bits
    }

    /// Serialize LSH state for persistence
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header: num_tables, num_bits, dimensions
        bytes.extend_from_slice(&(self.config.num_tables as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.num_bits as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.config.dimensions as u32).to_le_bytes());

        // Planes
        for &f in &self.planes {
            bytes.extend_from_slice(&f.to_le_bytes());
        }

        bytes
    }

    /// Deserialize LSH state
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }

        let num_tables = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let num_bits = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let dimensions = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;

        let expected_floats = num_tables * num_bits * dimensions;
        let expected_bytes = 12 + expected_floats * 4;

        if bytes.len() < expected_bytes {
            return None;
        }

        let mut planes = Vec::with_capacity(expected_floats);
        for i in 0..expected_floats {
            let offset = 12 + i * 4;
            let f = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            planes.push(f);
        }

        Some(Self {
            config: LSHConfig {
                num_tables,
                num_bits,
                dimensions,
            },
            planes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_deterministic_with_seed() {
        let config = LSHConfig {
            num_bits: 8,
            num_tables: 4,
            dimensions: 128,
        };

        let lsh1 = HyperplaneLSH::with_seed(config.clone(), 42);
        let lsh2 = HyperplaneLSH::with_seed(config, 42);

        let vector: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();

        assert_eq!(lsh1.hash(&vector), lsh2.hash(&vector));
    }

    #[test]
    fn test_lsh_similar_vectors_collide() {
        let config = LSHConfig {
            num_bits: 8,
            num_tables: 8,
            dimensions: 128,
        };

        let lsh = HyperplaneLSH::with_seed(config, 42);

        // Two very similar vectors
        let v1: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let v2: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) + 0.001).collect();

        let h1 = lsh.hash(&v1);
        let h2 = lsh.hash(&v2);

        // Should have at least some matching hashes
        let matches = h1.iter().zip(h2.iter()).filter(|(a, b)| a == b).count();
        assert!(
            matches > 0,
            "Similar vectors should have some matching hashes"
        );
    }

    #[test]
    fn test_probe_sequence_includes_original() {
        let config = LSHConfig {
            num_bits: 8,
            num_tables: 4,
            dimensions: 128,
        };

        let lsh = HyperplaneLSH::with_seed(config, 42);
        let vector: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();

        let probes = lsh.get_probe_sequence(&vector, 0, 5);
        let original = lsh.hash_single_table(0, &vector);

        assert_eq!(probes[0], original, "First probe should be original hash");
        assert_eq!(probes.len(), 5, "Should return requested number of probes");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = LSHConfig {
            num_bits: 8,
            num_tables: 4,
            dimensions: 64,
        };

        let lsh = HyperplaneLSH::with_seed(config, 42);
        let bytes = lsh.to_bytes();
        let restored = HyperplaneLSH::from_bytes(&bytes).unwrap();

        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        assert_eq!(lsh.hash(&vector), restored.hash(&vector));
    }
}
