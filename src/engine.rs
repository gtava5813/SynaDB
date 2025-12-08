//! Core database engine for Syna.
//!
//! This module contains the main database implementation:
//!
//! - [`SynaDB`] - The core database struct
//! - [`DbConfig`] - Runtime configuration options
//! - Global registry functions: [`open_db`], [`close_db`], [`with_db`]
//!
//! # Usage Patterns
//!
//! ## Direct Instance (Recommended for Rust)
//!
//! ```rust,no_run
//! use synadb::{SynaDB, Atom};
//!
//! let mut db = SynaDB::new("my.db").unwrap();
//! db.append("key", Atom::Int(42)).unwrap();
//! ```
//!
//! ## Global Registry (Used by FFI)
//!
//! ```rust,no_run
//! use synadb::{open_db, with_db, close_db, Atom};
//!
//! open_db("my.db").unwrap();
//! with_db("my.db", |db| db.append("key", Atom::Int(42))).unwrap();
//! close_db("my.db").unwrap();
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::compression::{compress, decode_delta, decompress, encode_delta, should_compress};
use crate::error::{Result, SynaError};
use crate::types::{Atom, LogHeader, HEADER_SIZE, IS_COMPRESSED, IS_DELTA, IS_TOMBSTONE};

// =============================================================================
// Global Database Registry
// =============================================================================

/// Thread-safe global registry for managing open database instances.
/// Uses canonicalized paths as keys to ensure uniqueness.
static DB_REGISTRY: Lazy<Mutex<HashMap<String, SynaDB>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Opens a database at the given path and registers it in the global registry.
///
/// If the database is already open, this function returns Ok without creating
/// a new instance.
///
/// # Arguments
/// * `path` - Path to the database file
///
/// # Returns
/// * `Ok(())` if the database was opened successfully or was already open
/// * `Err(SynaError)` if the path is invalid or the database couldn't be opened
pub fn open_db(path: &str) -> Result<()> {
    // Canonicalize path to absolute for consistent registry keys
    let canonical_path = canonicalize_path(path)?;

    let mut registry = DB_REGISTRY.lock();

    // Check if already in registry
    if registry.contains_key(&canonical_path) {
        return Ok(());
    }

    // Create new SynaDB instance
    let db = SynaDB::new(path)?;

    // Insert into registry
    registry.insert(canonical_path, db);

    Ok(())
}

/// Closes a database and removes it from the global registry.
///
/// # Arguments
/// * `path` - Path to the database file
///
/// # Returns
/// * `Ok(())` if the database was closed successfully
/// * `Err(SynaError::NotFound)` if the database is not in the registry
pub fn close_db(path: &str) -> Result<()> {
    let canonical_path = canonicalize_path(path)?;

    let mut registry = DB_REGISTRY.lock();

    // Remove from registry
    let db = registry
        .remove(&canonical_path)
        .ok_or_else(|| SynaError::NotFound(path.to_string()))?;

    // Close the database instance
    db.close()
}

/// Executes a closure with a mutable reference to the database at the given path.
///
/// This is the primary way to interact with databases in the registry, ensuring
/// thread-safe access.
///
/// # Arguments
/// * `path` - Path to the database file
/// * `f` - Closure that takes a mutable reference to SynaDB and returns a Result
///
/// # Returns
/// * The result of the closure, or an error if the database is not found
///
/// # Example
/// ```ignore
/// with_db("my.db", |db| {
///     db.append("key", Atom::Int(42))
/// })?;
/// ```
pub fn with_db<F, R>(path: &str, f: F) -> Result<R>
where
    F: FnOnce(&mut SynaDB) -> Result<R>,
{
    let canonical_path = canonicalize_path(path)?;

    let mut registry = DB_REGISTRY.lock();

    // Get mutable reference to db
    let db = registry
        .get_mut(&canonical_path)
        .ok_or_else(|| SynaError::NotFound(path.to_string()))?;

    // Call closure with db reference
    f(db)
}

/// Canonicalizes a path to an absolute path string.
///
/// If the path doesn't exist yet (for new databases), we use the parent directory's
/// canonical path combined with the filename.
fn canonicalize_path(path: &str) -> Result<String> {
    let path_buf = PathBuf::from(path);

    // Try to canonicalize directly (works if file exists)
    if let Ok(canonical) = std::fs::canonicalize(&path_buf) {
        return Ok(canonical.to_string_lossy().to_string());
    }

    // File doesn't exist yet - canonicalize parent and append filename
    let parent = path_buf.parent().unwrap_or(Path::new("."));
    let filename = path_buf
        .file_name()
        .ok_or_else(|| SynaError::InvalidPath(path.to_string()))?;

    // Canonicalize parent directory (create it if needed for the canonical path)
    let canonical_parent = if parent.as_os_str().is_empty() || parent == Path::new(".") {
        std::env::current_dir().map_err(|_| SynaError::InvalidPath(path.to_string()))?
    } else {
        std::fs::canonicalize(parent).map_err(|_| {
            SynaError::InvalidPath(format!("Parent directory not found: {}", parent.display()))
        })?
    };

    let canonical = canonical_parent.join(filename);
    Ok(canonical.to_string_lossy().to_string())
}

// =============================================================================
// Database Configuration and Core Struct
// =============================================================================

/// Runtime configuration for the database.
///
/// # Examples
///
/// ```rust,no_run
/// use synadb::{SynaDB, DbConfig};
///
/// // High-performance config for time-series data
/// let config = DbConfig {
///     enable_compression: true,   // LZ4 for large values
///     enable_delta: true,         // Delta encoding for floats
///     sync_on_write: false,       // Batch syncs for speed
/// };
///
/// let db = SynaDB::with_config("timeseries.db", config).unwrap();
/// ```
#[derive(Clone)]
pub struct DbConfig {
    /// Enable LZ4 compression for values larger than 64 bytes.
    ///
    /// When enabled, large values are compressed before writing,
    /// reducing disk usage at the cost of CPU time.
    pub enable_compression: bool,

    /// Enable delta encoding for float sequences.
    ///
    /// When enabled, consecutive float values for the same key are
    /// stored as deltas (differences), which compress better for
    /// time-series data with gradual changes.
    pub enable_delta: bool,

    /// Sync to disk after every write operation.
    ///
    /// When `true` (default), each write calls `fsync()` for durability.
    /// Set to `false` for higher throughput at the risk of data loss
    /// on crash.
    pub sync_on_write: bool,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self {
            enable_compression: false,
            enable_delta: false,
            sync_on_write: true,
        }
    }
}

/// The core database struct managing file I/O and indexing.
///
/// `SynaDB` provides an append-only, log-structured key-value store with:
/// - O(1) key lookup via in-memory index
/// - Full history tracking for time-series analysis
/// - Automatic crash recovery on open
/// - Thread-safe concurrent access
///
/// # Examples
///
/// ```rust,no_run
/// use synadb::{SynaDB, Atom};
///
/// // Create or open a database
/// let mut db = SynaDB::new("my.db").unwrap();
///
/// // Write values
/// db.append("sensor/temp", Atom::Float(23.5)).unwrap();
/// db.append("sensor/temp", Atom::Float(24.1)).unwrap();
///
/// // Read latest value
/// let temp = db.get("sensor/temp").unwrap();
/// assert!(matches!(temp, Some(Atom::Float(_))));
///
/// // Get full history for ML
/// let history = db.get_history_floats("sensor/temp").unwrap();
/// assert_eq!(history.len(), 2);
/// ```
pub struct SynaDB {
    /// Path to the database file.
    pub(crate) path: PathBuf,
    /// The append-only log file.
    pub(crate) file: File,
    /// Key -> List of offsets (history).
    pub(crate) index: HashMap<String, Vec<u64>>,
    /// Key -> Latest offset (fast lookup).
    pub(crate) latest: HashMap<String, u64>,
    /// For delta compression: previous values.
    pub(crate) previous_values: HashMap<String, Atom>,
    /// Keys with tombstone as latest entry.
    pub(crate) deleted: HashSet<String>,
    /// Cached file length.
    pub(crate) file_len: u64,
    /// Runtime configuration.
    pub(crate) config: DbConfig,
    /// Mutex for serializing writes.
    pub(crate) write_lock: Mutex<()>,
}

impl SynaDB {
    /// Opens or creates a database at the given path with default configuration.
    ///
    /// If the file exists, the index is rebuilt by scanning all entries.
    /// If the file doesn't exist, a new empty database is created.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database file
    ///
    /// # Returns
    ///
    /// * `Ok(SynaDB)` - The opened database instance
    /// * `Err(SynaError::Io)` - If the file cannot be opened/created
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::SynaDB;
    ///
    /// let db = SynaDB::new("my.db").unwrap();
    /// ```
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(path, DbConfig::default())
    }

    /// Opens or creates a database at the given path with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database file
    /// * `config` - Runtime configuration options
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::{SynaDB, DbConfig};
    ///
    /// let config = DbConfig {
    ///     enable_compression: true,
    ///     enable_delta: true,
    ///     sync_on_write: false,
    /// };
    /// let db = SynaDB::with_config("my.db", config).unwrap();
    /// ```
    pub fn with_config(path: impl AsRef<Path>, config: DbConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let file_len = file.metadata()?.len();

        let mut db = Self {
            path,
            file,
            index: HashMap::new(),
            latest: HashMap::new(),
            previous_values: HashMap::new(),
            deleted: HashSet::new(),
            file_len,
            config,
            write_lock: Mutex::new(()),
        };

        // If file is non-empty, rebuild index for recovery
        if file_len > 0 {
            let (_recovered, _skipped) = db.rebuild_index()?;
        }

        Ok(db)
    }

    /// Appends a key-value pair to the database.
    /// Returns the byte offset where the entry was written.
    ///
    /// # Arguments
    /// * `key` - A non-empty UTF-8 string (max 65535 bytes)
    /// * `value` - The Atom value to store
    ///
    /// # Errors
    /// * `SynaError::EmptyKey` - If the key is empty
    /// * `SynaError::KeyTooLong` - If the key exceeds 65535 bytes
    ///
    /// _Requirements: 5.3_
    pub fn append(&mut self, key: &str, value: Atom) -> Result<u64> {
        // Validate key: must be non-empty
        if key.is_empty() {
            return Err(SynaError::EmptyKey);
        }

        // Validate key: must fit in u16 (max 65535 bytes)
        if key.len() > u16::MAX as usize {
            return Err(SynaError::KeyTooLong(key.len()));
        }

        let _guard = self.write_lock.lock();

        // Check for delta compression opportunity
        let (value_to_store, mut flags) = if self.config.enable_delta {
            if let Atom::Float(current) = &value {
                // Check if we have a previous Float value for this key
                if let Some(Atom::Float(previous)) = self.previous_values.get(key) {
                    // Compute delta and set IS_DELTA flag
                    let delta = encode_delta(*current, *previous);
                    (Atom::Float(delta), IS_DELTA)
                } else {
                    // No previous value or previous wasn't a Float - store absolute
                    (value.clone(), 0u8)
                }
            } else {
                // Not a Float - store as-is
                (value.clone(), 0u8)
            }
        } else {
            // Delta compression disabled
            (value.clone(), 0u8)
        };

        // Serialize the value
        let value_bytes = bincode::serialize(&value_to_store)?;

        // Optionally compress (in addition to delta)
        let final_bytes = if self.config.enable_compression && should_compress(&value_bytes) {
            flags |= IS_COMPRESSED;
            compress(&value_bytes)
        } else {
            value_bytes
        };

        // Build header
        let header = LogHeader::new(key.len() as u16, final_bytes.len() as u32, flags);

        // Record offset before writing
        let offset = self.file_len;

        // Seek to end
        self.file.seek(SeekFrom::End(0))?;

        // Write header
        self.file.write_all(&header.to_bytes())?;

        // Write key
        self.file.write_all(key.as_bytes())?;

        // Write value
        self.file.write_all(&final_bytes)?;

        // Optionally sync
        if self.config.sync_on_write {
            self.file.sync_data()?;
        }

        // Update indexes
        self.latest.insert(key.to_string(), offset);
        self.index.entry(key.to_string()).or_default().push(offset);

        // Handle resurrection: remove from deleted set if key was previously deleted
        self.deleted.remove(key);

        // Update cached file length
        self.file_len += HEADER_SIZE as u64 + key.len() as u64 + final_bytes.len() as u64;

        // Store original value for delta compression (not the delta!)
        self.previous_values.insert(key.to_string(), value);

        Ok(offset)
    }

    /// Retrieves the latest value for a key.
    ///
    /// Returns `None` if the key doesn't exist or has been deleted.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Atom))` - The latest value for the key
    /// * `Ok(None)` - Key doesn't exist or was deleted
    /// * `Err(SynaError)` - I/O or deserialization error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::{SynaDB, Atom};
    ///
    /// let mut db = SynaDB::new("my.db").unwrap();
    /// db.append("key", Atom::Int(42)).unwrap();
    ///
    /// match db.get("key").unwrap() {
    ///     Some(Atom::Int(n)) => println!("Value: {}", n),
    ///     Some(_) => println!("Different type"),
    ///     None => println!("Not found"),
    /// }
    /// ```
    pub fn get(&mut self, key: &str) -> Result<Option<Atom>> {
        // Check if key is deleted (tombstone is latest entry)
        if self.deleted.contains(key) {
            return Ok(None);
        }

        // Check if key exists
        if !self.latest.contains_key(key) {
            return Ok(None);
        }

        // Get all offsets for this key to handle delta decoding
        let offsets = self.index.get(key).cloned().unwrap_or_default();
        if offsets.is_empty() {
            return Ok(None);
        }

        // Read entries and reconstruct value (handling deltas)
        let mut current_float: Option<f64> = None;
        let mut last_atom: Option<Atom> = None;

        for offset in offsets {
            let (_key, atom, flags) = self.read_entry_at(offset)?;

            if flags & IS_DELTA != 0 {
                // Delta-encoded value - need to reconstruct
                if let Atom::Float(delta) = atom {
                    if let Some(prev) = current_float {
                        let absolute = decode_delta(delta, prev);
                        current_float = Some(absolute);
                        last_atom = Some(Atom::Float(absolute));
                    } else {
                        // No previous float - this shouldn't happen, but store delta as-is
                        current_float = Some(delta);
                        last_atom = Some(Atom::Float(delta));
                    }
                } else {
                    // Non-float with delta flag - shouldn't happen, store as-is
                    last_atom = Some(atom);
                }
            } else {
                // Absolute value
                if let Atom::Float(f) = &atom {
                    current_float = Some(*f);
                }
                last_atom = Some(atom);
            }
        }

        Ok(last_atom)
    }

    /// Reads an entry at the given byte offset.
    /// Returns (key, atom, flags) where flags indicate if the value is delta-encoded.
    fn read_entry_at(&mut self, offset: u64) -> Result<(String, Atom, u8)> {
        // Seek to offset
        self.file.seek(SeekFrom::Start(offset))?;

        // Read header
        let mut header_buf = [0u8; HEADER_SIZE];
        self.file.read_exact(&mut header_buf)?;
        let header = LogHeader::from_bytes(&header_buf);

        // Validate header
        if !header.is_valid() {
            return Err(SynaError::CorruptedEntry(offset));
        }

        // Read key
        let mut key_buf = vec![0u8; header.key_len as usize];
        self.file.read_exact(&mut key_buf)?;
        let key = String::from_utf8(key_buf).map_err(|_| SynaError::CorruptedEntry(offset))?;

        // Read value
        let mut value_buf = vec![0u8; header.val_len as usize];
        self.file.read_exact(&mut value_buf)?;

        // Decompress if needed
        let value_bytes = if header.flags & IS_COMPRESSED != 0 {
            decompress(&value_buf)?
        } else {
            value_buf
        };

        // Deserialize
        let atom: Atom = bincode::deserialize(&value_bytes)?;

        Ok((key, atom, header.flags))
    }

    /// Rebuilds the in-memory index by scanning the log file.
    /// This is called on database open for crash recovery.
    ///
    /// Returns the number of entries recovered and skipped.
    pub(crate) fn rebuild_index(&mut self) -> Result<(usize, usize)> {
        self.file.seek(SeekFrom::Start(0))?;

        let mut offset = 0u64;
        let mut entries_recovered = 0usize;
        let mut entries_skipped = 0usize;

        loop {
            // Record current position as entry_offset
            let entry_offset = offset;

            // Try to read header
            let mut header_buf = [0u8; HEADER_SIZE];
            match self.file.read_exact(&mut header_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let header = LogHeader::from_bytes(&header_buf);

            // Validate header
            if !header.is_valid() {
                // Log warning and try to find next valid entry
                eprintln!(
                    "Warning: Invalid header at offset {}, attempting to find next valid entry",
                    entry_offset
                );
                entries_skipped += 1;

                // Try to scan for next valid entry
                if let Some(next_offset) = self.scan_for_next_valid_entry(entry_offset + 1)? {
                    offset = next_offset;
                    self.file.seek(SeekFrom::Start(offset))?;
                    continue;
                } else {
                    // No more valid entries found
                    break;
                }
            }

            // Read key
            let mut key_buf = vec![0u8; header.key_len as usize];
            match self.file.read_exact(&mut key_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Truncated entry at EOF
                    eprintln!("Warning: Truncated key at offset {}", entry_offset);
                    entries_skipped += 1;
                    break;
                }
                Err(e) => return Err(e.into()),
            }

            let key = match String::from_utf8(key_buf) {
                Ok(k) => k,
                Err(_) => {
                    // Corrupted key - try to find next valid entry
                    eprintln!("Warning: Invalid UTF-8 key at offset {}", entry_offset);
                    entries_skipped += 1;

                    if let Some(next_offset) =
                        self.scan_for_next_valid_entry(entry_offset + HEADER_SIZE as u64)?
                    {
                        offset = next_offset;
                        self.file.seek(SeekFrom::Start(offset))?;
                        continue;
                    } else {
                        break;
                    }
                }
            };

            // Skip value bytes (we don't need to read them for indexing)
            let value_len = header.val_len as u64;
            match self.file.seek(SeekFrom::Current(value_len as i64)) {
                Ok(new_pos) => {
                    // Check if we actually moved past the file end (truncated value)
                    if new_pos > self.file_len {
                        eprintln!("Warning: Truncated value at offset {}", entry_offset);
                        entries_skipped += 1;
                        break;
                    }
                }
                Err(_) => {
                    eprintln!("Warning: Truncated value at offset {}", entry_offset);
                    entries_skipped += 1;
                    break;
                }
            }

            // Update indexes
            self.latest.insert(key.clone(), entry_offset);
            self.index
                .entry(key.clone())
                .or_default()
                .push(entry_offset);

            // Track tombstones: if entry has IS_TOMBSTONE flag, add to deleted set
            // If subsequent non-tombstone entry for same key, remove from deleted
            if header.flags & IS_TOMBSTONE != 0 {
                self.deleted.insert(key);
            } else {
                self.deleted.remove(&key);
            }

            entries_recovered += 1;

            // Move to next entry
            offset = entry_offset + HEADER_SIZE as u64 + header.key_len as u64 + value_len;
        }

        // Log recovery stats
        if entries_skipped > 0 {
            eprintln!(
                "Recovery complete: {} entries recovered, {} entries skipped",
                entries_recovered, entries_skipped
            );
        }

        Ok((entries_recovered, entries_skipped))
    }

    /// Scans byte-by-byte looking for a valid header pattern.
    /// Returns the offset of the next valid entry or None if EOF.
    fn scan_for_next_valid_entry(&mut self, start_offset: u64) -> Result<Option<u64>> {
        // Don't scan past file end
        if start_offset >= self.file_len {
            return Ok(None);
        }

        self.file.seek(SeekFrom::Start(start_offset))?;

        let mut offset = start_offset;
        let mut header_buf = [0u8; HEADER_SIZE];

        // Scan byte-by-byte (actually we'll scan in chunks for efficiency)
        while offset + HEADER_SIZE as u64 <= self.file_len {
            self.file.seek(SeekFrom::Start(offset))?;

            match self.file.read_exact(&mut header_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(e.into()),
            }

            let header = LogHeader::from_bytes(&header_buf);

            // Check for valid header pattern
            if self.is_likely_valid_header(&header, offset) {
                return Ok(Some(offset));
            }

            offset += 1;
        }

        Ok(None)
    }

    /// Checks if a header looks like a valid entry.
    /// Uses heuristics to identify likely valid headers.
    fn is_likely_valid_header(&self, header: &LogHeader, offset: u64) -> bool {
        // Basic validation
        if !header.is_valid() {
            return false;
        }

        // Check for reasonable timestamp (not 0, not far in the future)
        // Allow timestamps from year 2000 to year 2100 (in microseconds)
        const MIN_TIMESTAMP: u64 = 946_684_800_000_000; // 2000-01-01 in microseconds
        const MAX_TIMESTAMP: u64 = 4_102_444_800_000_000; // 2100-01-01 in microseconds

        if header.timestamp != 0
            && (header.timestamp < MIN_TIMESTAMP || header.timestamp > MAX_TIMESTAMP)
        {
            return false;
        }

        // Check that key_len is reasonable (at least 1 byte for a key)
        if header.key_len == 0 {
            return false;
        }

        // Check that the entry would fit within the file
        let entry_size = HEADER_SIZE as u64 + header.key_len as u64 + header.val_len as u64;
        if offset + entry_size > self.file_len {
            return false;
        }

        // Check that flags only use valid bits
        const VALID_FLAGS_MASK: u8 = IS_DELTA | IS_COMPRESSED | IS_TOMBSTONE;
        if header.flags & !VALID_FLAGS_MASK != 0 {
            return false;
        }

        true
    }

    /// Closes the database, flushing any pending writes.
    pub fn close(self) -> Result<()> {
        self.file.sync_all()?;
        // File handle is dropped automatically
        Ok(())
    }

    /// Deletes a key by appending a tombstone entry.
    ///
    /// # Arguments
    /// * `key` - A non-empty UTF-8 string (max 65535 bytes)
    ///
    /// # Errors
    /// * `SynaError::EmptyKey` - If the key is empty
    /// * `SynaError::KeyTooLong` - If the key exceeds 65535 bytes
    ///
    /// _Requirements: 5.3_
    pub fn delete(&mut self, key: &str) -> Result<()> {
        // Validate key: must be non-empty
        if key.is_empty() {
            return Err(SynaError::EmptyKey);
        }

        // Validate key: must fit in u16 (max 65535 bytes)
        if key.len() > u16::MAX as usize {
            return Err(SynaError::KeyTooLong(key.len()));
        }

        let _guard = self.write_lock.lock();

        // Create empty value for tombstone
        let value_bytes = bincode::serialize(&Atom::Null)?;

        // Build header with IS_TOMBSTONE flag
        let header = LogHeader::new(key.len() as u16, value_bytes.len() as u32, IS_TOMBSTONE);

        // Record offset before writing
        let offset = self.file_len;

        // Seek to end
        self.file.seek(SeekFrom::End(0))?;

        // Write header
        self.file.write_all(&header.to_bytes())?;

        // Write key
        self.file.write_all(key.as_bytes())?;

        // Write value (empty/null)
        self.file.write_all(&value_bytes)?;

        // Optionally sync
        if self.config.sync_on_write {
            self.file.sync_data()?;
        }

        // Update indexes
        self.latest.insert(key.to_string(), offset);
        self.index.entry(key.to_string()).or_default().push(offset);

        // Mark key as deleted
        self.deleted.insert(key.to_string());

        // Update cached file length
        self.file_len += HEADER_SIZE as u64 + key.len() as u64 + value_bytes.len() as u64;

        Ok(())
    }

    /// Returns a list of all non-deleted keys in the database.
    ///
    /// Deleted keys (those with a tombstone as the latest entry) are excluded.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::{SynaDB, Atom};
    ///
    /// let mut db = SynaDB::new("my.db").unwrap();
    /// db.append("a", Atom::Int(1)).unwrap();
    /// db.append("b", Atom::Int(2)).unwrap();
    /// db.delete("a").unwrap();
    ///
    /// let keys = db.keys();
    /// assert!(keys.contains(&"b".to_string()));
    /// assert!(!keys.contains(&"a".to_string()));
    /// ```
    pub fn keys(&self) -> Vec<String> {
        self.latest
            .keys()
            .filter(|k| !self.deleted.contains(*k))
            .cloned()
            .collect()
    }

    /// Returns `true` if the key exists and is not deleted.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::{SynaDB, Atom};
    ///
    /// let mut db = SynaDB::new("my.db").unwrap();
    /// db.append("key", Atom::Int(42)).unwrap();
    ///
    /// assert!(db.exists("key"));
    /// assert!(!db.exists("nonexistent"));
    /// ```
    pub fn exists(&self, key: &str) -> bool {
        self.latest.contains_key(key) && !self.deleted.contains(key)
    }

    /// Retrieves the complete history of values for a key.
    /// Returns all values in chronological order (oldest first).
    ///
    /// **Note:** After compaction, history only contains the latest value for each key.
    /// Compaction removes historical entries to reclaim disk space. If you need to
    /// preserve full history, do not call `compact()` or consider implementing a
    /// "preserve history" mode in the future.
    ///
    /// **Note:** Tombstone entries (deletions) are excluded from history.
    ///
    /// # Arguments
    /// * `key` - The key to retrieve history for
    ///
    /// # Returns
    /// * `Ok(Vec<Atom>)` - All values written to this key in order (may be single value after compaction)
    /// * `Err(SynaError)` - If there was an error reading entries
    ///
    /// _Requirements: 4.1, 10.3, 11.1_
    pub fn get_history(&mut self, key: &str) -> Result<Vec<Atom>> {
        // Look up all offsets for this key
        let offsets = self.index.get(key).cloned().unwrap_or_default();

        let mut atoms = Vec::with_capacity(offsets.len());
        let mut current_float: Option<f64> = None;

        // For each offset in chronological order, read the entry and handle deltas
        for offset in offsets {
            let (_key, atom, flags) = self.read_entry_at(offset)?;

            // Skip tombstone entries - they should not appear in history
            // _Requirements: 10.3_
            if flags & IS_TOMBSTONE != 0 {
                continue;
            }

            if flags & IS_DELTA != 0 {
                // Delta-encoded value - reconstruct absolute value
                if let Atom::Float(delta) = atom {
                    if let Some(prev) = current_float {
                        let absolute = decode_delta(delta, prev);
                        current_float = Some(absolute);
                        atoms.push(Atom::Float(absolute));
                    } else {
                        // No previous float - store delta as-is (shouldn't happen)
                        current_float = Some(delta);
                        atoms.push(Atom::Float(delta));
                    }
                } else {
                    // Non-float with delta flag - shouldn't happen, store as-is
                    atoms.push(atom);
                }
            } else {
                // Absolute value
                if let Atom::Float(f) = &atom {
                    current_float = Some(*f);
                }
                atoms.push(atom);
            }
        }

        Ok(atoms)
    }

    /// Retrieves the complete history of float values for a key.
    /// Non-float atoms are filtered out.
    ///
    /// # Arguments
    /// * `key` - The key to retrieve float history for
    ///
    /// # Returns
    /// * `Ok(Vec<f64>)` - All float values written to this key in order
    /// * `Err(SynaError)` - If there was an error reading entries
    ///
    /// _Requirements: 4.1, 4.4_
    pub fn get_history_floats(&mut self, key: &str) -> Result<Vec<f64>> {
        let history = self.get_history(key)?;

        // Filter and map: keep only Float atoms, extract their values
        let floats = history
            .into_iter()
            .filter_map(|a| {
                if let Atom::Float(f) = a {
                    Some(f)
                } else {
                    None
                }
            })
            .collect();

        Ok(floats)
    }

    /// Retrieves the complete history of float values for a key as a raw pointer.
    /// This is designed for FFI use - the caller is responsible for freeing the memory
    /// using `free_tensor()`.
    ///
    /// # Arguments
    /// * `key` - The key to retrieve float history for
    ///
    /// # Returns
    /// * `Ok((ptr, len))` - Pointer to contiguous f64 array and its length
    /// * `Err(SynaError)` - If there was an error reading entries
    ///
    /// # Safety
    /// The returned pointer must be freed using `free_tensor()` to avoid memory leaks.
    ///
    /// _Requirements: 4.2_
    pub fn get_history_tensor(&mut self, key: &str) -> Result<(*mut f64, usize)> {
        let floats = self.get_history_floats(key)?;

        // Convert to boxed slice
        let boxed = floats.into_boxed_slice();

        // Get length before leaking
        let len = boxed.len();

        // Leak memory for FFI - caller must free with free_tensor()
        let ptr = Box::into_raw(boxed) as *mut f64;

        Ok((ptr, len))
    }

    /// Compacts the database by rewriting only the latest non-deleted entries.
    ///
    /// This operation:
    /// 1. Creates a temporary file with only the latest value for each non-deleted key
    /// 2. Atomically replaces the original file with the compacted file
    /// 3. Rebuilds the in-memory index
    ///
    /// **Note:** Compaction loses history - after compaction, `get_history()` will only
    /// return the latest value for each key.
    ///
    /// # Returns
    /// * `Ok(())` if compaction was successful
    /// * `Err(SynaError)` if compaction failed (original file remains intact)
    ///
    /// _Requirements: 11.1, 11.2_
    pub fn compact(&mut self) -> Result<()> {
        // Create temporary file path
        let temp_path = self.path.with_extension("compact.tmp");

        // Create temporary file
        let mut temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        // Collect keys to compact (non-deleted keys only)
        let keys_to_compact: Vec<String> = self
            .latest
            .keys()
            .filter(|k| !self.deleted.contains(*k))
            .cloned()
            .collect();

        let enable_compression = self.config.enable_compression;

        // Write each key's latest value to temp file (fresh entry, no delta encoding)
        for key in &keys_to_compact {
            // Read the latest value for this key using internal method
            let value = match self.get_value_internal(key)? {
                Some(v) => v,
                None => continue, // Skip if somehow not found
            };

            // Serialize the value (no delta encoding for compacted entries)
            let value_bytes = bincode::serialize(&value)?;

            // Optionally compress
            let (final_bytes, flags) = if enable_compression && should_compress(&value_bytes) {
                (compress(&value_bytes), IS_COMPRESSED)
            } else {
                (value_bytes, 0u8)
            };

            // Build header (fresh entry, no delta)
            let header = LogHeader::new(key.len() as u16, final_bytes.len() as u32, flags);

            // Write header
            temp_file.write_all(&header.to_bytes())?;

            // Write key
            temp_file.write_all(key.as_bytes())?;

            // Write value
            temp_file.write_all(&final_bytes)?;
        }

        // Sync temp file to disk
        temp_file.sync_all()?;

        // Drop temp file handle before rename
        drop(temp_file);

        // Close current file handle by dropping it
        // We need to reopen after rename anyway
        let original_path = self.path.clone();

        // Atomically rename temp file to original path
        if let Err(e) = atomic_replace(&temp_path, &original_path) {
            // Rename failed - clean up temp file and return error
            let _ = std::fs::remove_file(&temp_path);
            return Err(SynaError::Io(e));
        }

        // Reopen file handle
        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&original_path)?;

        // Update file length
        self.file_len = self.file.metadata()?.len();

        // Clear and rebuild index
        self.index.clear();
        self.latest.clear();
        self.deleted.clear();
        self.previous_values.clear();

        // Rebuild index from compacted file
        self.rebuild_index()?;

        Ok(())
    }

    /// Internal method to get a value without acquiring the write lock.
    /// Used by compact() to read values during compaction.
    fn get_value_internal(&mut self, key: &str) -> Result<Option<Atom>> {
        // Check if key is deleted (tombstone is latest entry)
        if self.deleted.contains(key) {
            return Ok(None);
        }

        // Check if key exists
        if !self.latest.contains_key(key) {
            return Ok(None);
        }

        // Get all offsets for this key to handle delta decoding
        let offsets = self.index.get(key).cloned().unwrap_or_default();
        if offsets.is_empty() {
            return Ok(None);
        }

        // Read entries and reconstruct value (handling deltas)
        let mut current_float: Option<f64> = None;
        let mut last_atom: Option<Atom> = None;

        for offset in offsets {
            let (_key, atom, flags) = self.read_entry_at(offset)?;

            if flags & IS_DELTA != 0 {
                // Delta-encoded value - need to reconstruct
                if let Atom::Float(delta) = atom {
                    if let Some(prev) = current_float {
                        let absolute = decode_delta(delta, prev);
                        current_float = Some(absolute);
                        last_atom = Some(Atom::Float(absolute));
                    } else {
                        // No previous float - this shouldn't happen, but store delta as-is
                        current_float = Some(delta);
                        last_atom = Some(Atom::Float(delta));
                    }
                } else {
                    // Non-float with delta flag - shouldn't happen, store as-is
                    last_atom = Some(atom);
                }
            } else {
                // Absolute value
                if let Atom::Float(f) = &atom {
                    current_float = Some(*f);
                }
                last_atom = Some(atom);
            }
        }

        Ok(last_atom)
    }
}

/// Frees memory allocated by `get_history_tensor()`.
///
/// # Safety
/// * `ptr` must have been returned by `get_history_tensor()`
/// * `len` must be the length returned alongside the pointer
/// * This function must only be called once per pointer
///
/// _Requirements: 4.3, 6.5_
#[allow(clippy::cast_slice_from_raw_parts)]
pub unsafe fn free_tensor(ptr: *mut f64, len: usize) {
    if !ptr.is_null() && len > 0 {
        // Reconstruct the box from the raw pointer
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
        // Box is dropped here, memory is freed
    }
}

/// Atomically replaces a file with another file.
/// On Unix, this uses rename() which is atomic.
/// On Windows, this uses a rename with retry strategy.
///
/// # Arguments
/// * `src` - Source file path (the new file)
/// * `dst` - Destination file path (the file to replace)
///
/// # Returns
/// * `Ok(())` if the replacement was successful
/// * `Err(std::io::Error)` if the replacement failed
#[cfg(unix)]
fn atomic_replace(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::rename(src, dst)
}

#[cfg(windows)]
fn atomic_replace(src: &Path, dst: &Path) -> std::io::Result<()> {
    // On Windows, rename can fail if the destination exists
    // We use a retry strategy with remove + rename
    const MAX_RETRIES: u32 = 3;

    for attempt in 0..MAX_RETRIES {
        // Try direct rename first (works if dst doesn't exist or is unlocked)
        match std::fs::rename(src, dst) {
            Ok(()) => return Ok(()),
            Err(_) if attempt < MAX_RETRIES - 1 => {
                // Try removing destination first, then rename
                let _ = std::fs::remove_file(dst);
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    // Final attempt
    std::fs::rename(src, dst)
}
