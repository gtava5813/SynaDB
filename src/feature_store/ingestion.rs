// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Ingestion pipeline and write-ahead buffer.
//!
//! The [`WriteAheadBuffer`] batches feature writes before flushing to the
//! append-only log. This amortizes I/O overhead and enables atomic batch
//! ingestion.

use std::time::Instant;

use super::schema::FeatureValue;

/// A single buffered entry waiting to be flushed.
#[derive(Debug, Clone)]
pub struct BufferedEntry {
    /// Feature group name.
    pub group: String,
    /// Entity key value.
    pub entity_key: String,
    /// Feature name.
    pub feature: String,
    /// Feature value.
    pub value: FeatureValue,
    /// Event timestamp (when the event occurred).
    pub event_ts: u64,
    /// Ingestion timestamp (when the value was ingested).
    pub ingestion_ts: u64,
}

/// Batching buffer that accumulates writes before flushing to the log.
///
/// Flushes when either:
/// - The buffer exceeds `max_entries` entries
/// - The buffer age exceeds `max_age_micros` microseconds
///
/// This reduces I/O overhead for high-frequency ingestion.
pub struct WriteAheadBuffer {
    /// Buffered entries.
    buffer: Vec<BufferedEntry>,
    /// Maximum entries before auto-flush.
    max_entries: usize,
    /// Maximum age in microseconds before auto-flush.
    max_age_micros: u64,
    /// When the buffer was last flushed (or created).
    last_flush: Instant,
}

impl WriteAheadBuffer {
    /// Create a new write-ahead buffer with the given thresholds.
    pub fn new(max_entries: usize, max_age_micros: u64) -> Self {
        Self {
            buffer: Vec::with_capacity(max_entries),
            max_entries,
            max_age_micros,
            last_flush: Instant::now(),
        }
    }

    /// Add an entry to the buffer.
    ///
    /// Returns `Some(entries)` if the buffer should be flushed (threshold exceeded),
    /// otherwise returns `None`.
    pub fn push(&mut self, entry: BufferedEntry) -> Option<Vec<BufferedEntry>> {
        self.buffer.push(entry);

        if self.buffer.len() >= self.max_entries {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Force flush all buffered entries, returning them.
    ///
    /// Resets the buffer and the flush timer.
    pub fn flush(&mut self) -> Vec<BufferedEntry> {
        self.last_flush = Instant::now();
        std::mem::take(&mut self.buffer)
    }

    /// Check if the buffer should be flushed based on age.
    pub fn should_flush(&self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }
        let elapsed_micros = self.last_flush.elapsed().as_micros() as u64;
        elapsed_micros >= self.max_age_micros
    }

    /// Returns the number of entries currently buffered.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_entry(group: &str, entity: &str, feature: &str) -> BufferedEntry {
        BufferedEntry {
            group: group.to_string(),
            entity_key: entity.to_string(),
            feature: feature.to_string(),
            value: FeatureValue::Float64(1.0),
            event_ts: 1_000_000,
            ingestion_ts: 1_000_001,
        }
    }

    #[test]
    fn test_buffer_push_no_flush() {
        let mut buffer = WriteAheadBuffer::new(10, 10_000);
        let result = buffer.push(test_entry("g", "e", "f"));
        assert!(result.is_none());
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_buffer_auto_flush_on_threshold() {
        let mut buffer = WriteAheadBuffer::new(3, 10_000);
        assert!(buffer.push(test_entry("g", "e", "f1")).is_none());
        assert!(buffer.push(test_entry("g", "e", "f2")).is_none());
        let result = buffer.push(test_entry("g", "e", "f3"));
        assert!(result.is_some());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 3);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_manual_flush() {
        let mut buffer = WriteAheadBuffer::new(100, 10_000);
        buffer.push(test_entry("g", "e", "f1"));
        buffer.push(test_entry("g", "e", "f2"));

        let entries = buffer.flush();
        assert_eq!(entries.len(), 2);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_should_flush_empty() {
        let buffer = WriteAheadBuffer::new(100, 10_000);
        assert!(!buffer.should_flush());
    }

    #[test]
    fn test_buffer_should_flush_age() {
        let mut buffer = WriteAheadBuffer::new(100, 0); // 0 micros = immediate
        buffer.push(test_entry("g", "e", "f"));
        // With max_age_micros = 0, should always flush
        assert!(buffer.should_flush());
    }
}
