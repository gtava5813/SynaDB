//! Memory-mapped vector storage for Cascade Index
//!
//! Follows SynaDB physics principles:
//! - Arrow of Time: Append-only writes, never rewrite
//! - The Observer: Memory-mapped reads, zero-copy access
//!
//! # Storage Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes)                                            │
//! │ - magic: u32 (CMVS)                                          │
//! │ - version: u32                                               │
//! │ - dimensions: u16                                            │
//! │ - metric: u8                                                 │
//! │ - vector_count: u64                                          │
//! │ - write_offset: u64                                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Vector Entries (append-only)                                 │
//! │ [id: u32][key_len: u16][key: bytes][vector: f32 × dims]      │
//! │ [id: u32][key_len: u16][key: bytes][vector: f32 × dims]      │
//! │ ...                                                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::MmapMut;

use crate::distance::DistanceMetric;
use crate::error::{Result, SynaError};

/// Magic number for cascade mmap store
const MMAP_MAGIC: u32 = 0x434D5653; // "CMVS"

/// File format version
const MMAP_VERSION: u32 = 1;

/// Header size in bytes
const HEADER_SIZE: usize = 64;

/// Configuration for mmap vector storage
#[derive(Debug, Clone)]
pub struct MmapStoreConfig {
    /// Vector dimensions
    pub dimensions: u16,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Initial capacity (number of vectors)
    pub initial_capacity: usize,
}

impl Default for MmapStoreConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            metric: DistanceMetric::Cosine,
            initial_capacity: 100_000,
        }
    }
}

/// Memory-mapped vector storage
///
/// Provides O(1) append and zero-copy vector access via mmap.
pub struct MmapVectorStorage {
    /// File path
    #[allow(dead_code)]
    path: PathBuf,
    /// Memory-mapped region
    mmap: MmapMut,
    /// File handle
    file: File,
    /// Configuration
    config: MmapStoreConfig,
    /// Current write offset (atomic for thread safety)
    write_offset: AtomicU64,
    /// Vector count
    vector_count: AtomicU64,
    /// Key to (id, offset) mapping
    key_to_entry: HashMap<String, (u32, u64)>,
    /// ID to offset mapping (for graph lookups)
    id_to_offset: Vec<u64>,
}

impl MmapVectorStorage {
    /// Create or open mmap storage
    pub fn new<P: AsRef<Path>>(path: P, config: MmapStoreConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let exists = path.exists();

        // Calculate file size: header + vectors
        // Entry size: 4 (id) + 2 (key_len) + ~64 (avg key) + dims*4 (vector)
        let entry_size = 4 + 2 + 64 + (config.dimensions as usize * 4);
        let file_size = HEADER_SIZE + (config.initial_capacity * entry_size);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        // Ensure file is large enough
        let current_size = file.metadata()?.len() as usize;
        if current_size < file_size {
            file.set_len(file_size as u64)?;
        }

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let mut store = Self {
            path,
            mmap,
            file,
            config,
            write_offset: AtomicU64::new(HEADER_SIZE as u64),
            vector_count: AtomicU64::new(0),
            key_to_entry: HashMap::new(),
            id_to_offset: Vec::new(),
        };

        if exists {
            store.load_existing()?;
        } else {
            store.write_header()?;
        }

        Ok(store)
    }

    /// Write file header
    fn write_header(&mut self) -> Result<()> {
        let header = &mut self.mmap[0..HEADER_SIZE];

        header[0..4].copy_from_slice(&MMAP_MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&MMAP_VERSION.to_le_bytes());
        header[8..10].copy_from_slice(&self.config.dimensions.to_le_bytes());
        header[10] = self.config.metric as u8;
        header[16..24].copy_from_slice(&0u64.to_le_bytes()); // vector_count
        header[24..32].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // write_offset

        Ok(())
    }

    /// Load existing data
    fn load_existing(&mut self) -> Result<()> {
        // Validate header
        let magic = u32::from_le_bytes(self.mmap[0..4].try_into().unwrap());
        if magic != MMAP_MAGIC {
            return Err(SynaError::CorruptedIndex(
                "Invalid cascade mmap magic".into(),
            ));
        }

        let dimensions = u16::from_le_bytes(self.mmap[8..10].try_into().unwrap());
        if dimensions != self.config.dimensions {
            return Err(SynaError::DimensionMismatch {
                expected: self.config.dimensions,
                got: dimensions,
            });
        }

        let vector_count = u64::from_le_bytes(self.mmap[16..24].try_into().unwrap());
        let write_offset = u64::from_le_bytes(self.mmap[24..32].try_into().unwrap());

        self.vector_count.store(vector_count, Ordering::SeqCst);
        self.write_offset.store(write_offset, Ordering::SeqCst);

        // Rebuild index by scanning entries
        self.rebuild_index()?;

        Ok(())
    }

    /// Rebuild in-memory index from mmap
    fn rebuild_index(&mut self) -> Result<()> {
        let mut offset = HEADER_SIZE as u64;
        let write_offset = self.write_offset.load(Ordering::SeqCst);
        let dims = self.config.dimensions as usize;

        while offset < write_offset {
            let o = offset as usize;

            // Read entry: [id: u32][key_len: u16][key][vector]
            let id = u32::from_le_bytes(self.mmap[o..o + 4].try_into().unwrap());
            let key_len = u16::from_le_bytes(self.mmap[o + 4..o + 6].try_into().unwrap()) as usize;

            let key = String::from_utf8(self.mmap[o + 6..o + 6 + key_len].to_vec())
                .map_err(|_| SynaError::CorruptedIndex("Invalid UTF-8 key".into()))?;

            self.key_to_entry.insert(key, (id, offset));

            // Ensure id_to_offset is large enough
            while self.id_to_offset.len() <= id as usize {
                self.id_to_offset.push(0);
            }
            self.id_to_offset[id as usize] = offset;

            // Move to next entry
            let entry_size = 4 + 2 + key_len + (dims * 4);
            offset += entry_size as u64;
        }

        Ok(())
    }

    /// Append a vector (Arrow of Time - never rewrite!)
    /// Returns the assigned ID
    pub fn append(&mut self, key: &str, vector: &[f32]) -> Result<u32> {
        if vector.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatchUsize {
                expected: self.config.dimensions as usize,
                got: vector.len(),
            });
        }

        // Skip if key exists
        if self.key_to_entry.contains_key(key) {
            return Ok(self.key_to_entry[key].0);
        }

        let key_bytes = key.as_bytes();
        let key_len = key_bytes.len();
        let entry_size = 4 + 2 + key_len + (vector.len() * 4);

        let offset = self.write_offset.load(Ordering::SeqCst) as usize;

        // Grow if needed
        if offset + entry_size > self.mmap.len() {
            self.grow_file(entry_size)?;
        }

        let id = self.id_to_offset.len() as u32;

        // Write entry: [id][key_len][key][vector]
        self.mmap[offset..offset + 4].copy_from_slice(&id.to_le_bytes());
        self.mmap[offset + 4..offset + 6].copy_from_slice(&(key_len as u16).to_le_bytes());
        self.mmap[offset + 6..offset + 6 + key_len].copy_from_slice(key_bytes);

        // Write vector (zero-copy via pointer)
        let vec_start = offset + 6 + key_len;
        unsafe {
            let src = vector.as_ptr() as *const u8;
            let dst = self.mmap.as_mut_ptr().add(vec_start);
            std::ptr::copy_nonoverlapping(src, dst, vector.len() * 4);
        }

        // Update state
        self.write_offset
            .store((offset + entry_size) as u64, Ordering::SeqCst);
        self.vector_count.fetch_add(1, Ordering::SeqCst);
        self.key_to_entry
            .insert(key.to_string(), (id, offset as u64));
        self.id_to_offset.push(offset as u64);

        Ok(id)
    }

    /// Get vector by ID (The Observer - safe mmap read)
    ///
    /// Note: We read byte-by-byte to avoid alignment issues with mmap.
    /// This is slightly slower than zero-copy but guaranteed safe.
    #[inline]
    pub fn get_vector_by_id(&self, id: u32) -> Option<Vec<f32>> {
        let offset = *self.id_to_offset.get(id as usize)? as usize;
        if offset == 0 && id != 0 {
            return None;
        }

        let key_len =
            u16::from_le_bytes(self.mmap[offset + 4..offset + 6].try_into().ok()?) as usize;

        let vec_start = offset + 6 + key_len;
        let dims = self.config.dimensions as usize;

        // Read floats byte-by-byte to avoid alignment issues
        let mut vector = Vec::with_capacity(dims);
        for i in 0..dims {
            let float_offset = vec_start + i * 4;
            let bytes: [u8; 4] = self.mmap[float_offset..float_offset + 4].try_into().ok()?;
            vector.push(f32::from_le_bytes(bytes));
        }

        Some(vector)
    }

    /// Get vector by key
    pub fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        let (id, _) = self.key_to_entry.get(key)?;
        self.get_vector_by_id(*id)
    }

    /// Get key by ID
    pub fn get_key_by_id(&self, id: u32) -> Option<String> {
        let offset = *self.id_to_offset.get(id as usize)? as usize;
        if offset == 0 && id != 0 {
            return None;
        }

        let key_len =
            u16::from_le_bytes(self.mmap[offset + 4..offset + 6].try_into().ok()?) as usize;

        String::from_utf8(self.mmap[offset + 6..offset + 6 + key_len].to_vec()).ok()
    }

    /// Get ID by key
    pub fn get_id(&self, key: &str) -> Option<u32> {
        self.key_to_entry.get(key).map(|(id, _)| *id)
    }

    /// Check if key exists
    pub fn contains(&self, key: &str) -> bool {
        self.key_to_entry.contains_key(key)
    }

    /// Number of vectors
    pub fn len(&self) -> usize {
        self.id_to_offset.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.id_to_offset.is_empty()
    }

    /// Dimensions
    pub fn dimensions(&self) -> u16 {
        self.config.dimensions
    }

    /// Distance metric
    pub fn metric(&self) -> DistanceMetric {
        self.config.metric
    }

    /// Grow file capacity
    fn grow_file(&mut self, additional: usize) -> Result<()> {
        let current_size = self.mmap.len();
        let required = self.write_offset.load(Ordering::SeqCst) as usize + additional;
        let new_size = (current_size * 2).max(required + 1024 * 1024);

        self.mmap.flush()?;
        self.file.set_len(new_size as u64)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };

        Ok(())
    }

    /// Flush to disk
    pub fn flush(&mut self) -> Result<()> {
        // Update header
        let count = self.vector_count.load(Ordering::SeqCst);
        let offset = self.write_offset.load(Ordering::SeqCst);

        self.mmap[16..24].copy_from_slice(&count.to_le_bytes());
        self.mmap[24..32].copy_from_slice(&offset.to_le_bytes());

        self.mmap.flush()?;
        Ok(())
    }

    /// Iterate all IDs
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.id_to_offset.len() as u32)
            .filter(|&id| self.id_to_offset[id as usize] != 0 || id == 0)
    }

    /// Compute distance between query and vector at ID
    #[inline]
    pub fn distance_to_id(&self, query: &[f32], id: u32) -> Option<f32> {
        let vector = self.get_vector_by_id(id)?;
        Some(self.config.metric.distance(query, &vector))
    }
}

impl Drop for MmapVectorStorage {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_storage_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.cmvs");

        let config = MmapStoreConfig {
            dimensions: 64,
            initial_capacity: 1000,
            ..Default::default()
        };

        let mut store = MmapVectorStorage::new(&path, config).unwrap();

        let vec1: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let id = store.append("key1", &vec1).unwrap();

        assert_eq!(id, 0);
        assert_eq!(store.len(), 1);

        let retrieved = store.get_vector_by_id(0).unwrap();
        assert_eq!(retrieved.len(), 64);
        assert!((retrieved[0] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_mmap_storage_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.cmvs");

        // Create and populate
        {
            let config = MmapStoreConfig {
                dimensions: 32,
                initial_capacity: 100,
                ..Default::default()
            };

            let mut store = MmapVectorStorage::new(&path, config).unwrap();

            for i in 0..10 {
                let vec: Vec<f32> = (0..32).map(|j| (i * 32 + j) as f32).collect();
                store.append(&format!("k{}", i), &vec).unwrap();
            }

            store.flush().unwrap();
        }

        // Reopen and verify
        {
            let config = MmapStoreConfig {
                dimensions: 32,
                initial_capacity: 100,
                ..Default::default()
            };

            let store = MmapVectorStorage::new(&path, config).unwrap();
            assert_eq!(store.len(), 10);

            let vec = store.get_vector("k5").unwrap();
            assert_eq!(vec.len(), 32);
        }
    }
}
