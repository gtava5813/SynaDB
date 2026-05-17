// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Training dataset generation with parallel point-in-time joins.
//!
//! Generates ML-ready training datasets by performing point-in-time correct
//! joins between an entity DataFrame (spine) and feature histories.
//!
//! For datasets exceeding 10,000 rows, processing is parallelized across
//! multiple threads.

use super::schema::FeatureValue;

pub use super::statistics::ColumnStatistics;

/// Input spine for training dataset generation.
///
/// Each row represents one training example: an entity at a specific point in time.
/// The feature store will look up what features were known at that timestamp.
#[derive(Debug, Clone)]
pub struct EntityDataFrame {
    /// Entity key values (one per row).
    pub entity_keys: Vec<String>,
    /// Event timestamps (one per row, same length as entity_keys).
    pub event_timestamps: Vec<u64>,
}

impl EntityDataFrame {
    /// Create a new entity DataFrame.
    pub fn new(entity_keys: Vec<String>, event_timestamps: Vec<u64>) -> Self {
        Self {
            entity_keys,
            event_timestamps,
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.entity_keys.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.entity_keys.is_empty()
    }
}

/// Typed column data for zero-copy NumPy conversion.
#[derive(Debug, Clone)]
pub enum ColumnData {
    /// 64-bit floating point column.
    Float64(Vec<Option<f64>>),
    /// 64-bit integer column.
    Int64(Vec<Option<i64>>),
    /// String column.
    String(Vec<Option<String>>),
    /// Boolean column.
    Bool(Vec<Option<bool>>),
    /// Vector column (embeddings).
    Vector(Vec<Option<Vec<f32>>>),
    /// Timestamp column.
    Timestamp(Vec<Option<u64>>),
}

impl ColumnData {
    /// Create a new column of the appropriate type with the given capacity.
    pub fn new_float64(capacity: usize) -> Self {
        ColumnData::Float64(Vec::with_capacity(capacity))
    }

    /// Create a new Int64 column.
    pub fn new_int64(capacity: usize) -> Self {
        ColumnData::Int64(Vec::with_capacity(capacity))
    }

    /// Push a FeatureValue into this column.
    pub fn push(&mut self, value: &FeatureValue) {
        match (self, value) {
            (ColumnData::Float64(v), FeatureValue::Float64(f)) => v.push(Some(*f)),
            (ColumnData::Float64(v), FeatureValue::Null) => v.push(None),
            (ColumnData::Int64(v), FeatureValue::Int64(i)) => v.push(Some(*i)),
            (ColumnData::Int64(v), FeatureValue::Null) => v.push(None),
            (ColumnData::String(v), FeatureValue::String(s)) => v.push(Some(s.clone())),
            (ColumnData::String(v), FeatureValue::Null) => v.push(None),
            (ColumnData::Bool(v), FeatureValue::Bool(b)) => v.push(Some(*b)),
            (ColumnData::Bool(v), FeatureValue::Null) => v.push(None),
            (ColumnData::Vector(v), FeatureValue::Vector(vec)) => v.push(Some(vec.clone())),
            (ColumnData::Vector(v), FeatureValue::Null) => v.push(None),
            (ColumnData::Timestamp(v), FeatureValue::Timestamp(t)) => v.push(Some(*t)),
            (ColumnData::Timestamp(v), FeatureValue::Null) => v.push(None),
            // Type mismatch — push None
            (ColumnData::Float64(v), _) => v.push(None),
            (ColumnData::Int64(v), _) => v.push(None),
            (ColumnData::String(v), _) => v.push(None),
            (ColumnData::Bool(v), _) => v.push(None),
            (ColumnData::Vector(v), _) => v.push(None),
            (ColumnData::Timestamp(v), _) => v.push(None),
        }
    }

    /// Number of elements in this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Float64(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::Vector(v) => v.len(),
            ColumnData::Timestamp(v) => v.len(),
        }
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Columnar training dataset result.
#[derive(Debug, Clone)]
pub struct TrainingDataset {
    /// Column names.
    pub columns: Vec<String>,
    /// Column data as typed arrays.
    pub data: Vec<ColumnData>,
    /// Per-column statistics.
    pub statistics: Vec<ColumnStatistics>,
    /// Number of rows.
    pub num_rows: usize,
}

impl TrainingDataset {
    /// Create a new empty training dataset with the given columns.
    pub fn new(columns: Vec<String>) -> Self {
        let num_cols = columns.len();
        Self {
            columns,
            data: Vec::with_capacity(num_cols),
            statistics: Vec::new(),
            num_rows: 0,
        }
    }
}

/// Compute column statistics from a Float64 column.
pub fn compute_column_statistics(data: &ColumnData) -> ColumnStatistics {
    match data {
        ColumnData::Float64(values) => {
            let mut count = 0u64;
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            let mut null_count = 0u64;

            for v in values {
                match v {
                    Some(val) => {
                        count += 1;
                        sum += val;
                        sum_sq += val * val;
                        if *val < min {
                            min = *val;
                        }
                        if *val > max {
                            max = *val;
                        }
                    }
                    None => null_count += 1,
                }
            }

            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            let variance = if count > 1 {
                (sum_sq - sum * sum / count as f64) / (count - 1) as f64
            } else {
                0.0
            };
            let total = count + null_count;
            let null_rate = if total > 0 {
                null_count as f64 / total as f64
            } else {
                0.0
            };

            ColumnStatistics {
                mean,
                stddev: variance.max(0.0).sqrt(),
                null_rate,
                min: if min == f64::INFINITY { 0.0 } else { min },
                max: if max == f64::NEG_INFINITY { 0.0 } else { max },
                count,
            }
        }
        ColumnData::Int64(values) => {
            let mut count = 0u64;
            let mut sum = 0.0f64;
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            let mut null_count = 0u64;

            for v in values {
                match v {
                    Some(val) => {
                        let f = *val as f64;
                        count += 1;
                        sum += f;
                        if f < min {
                            min = f;
                        }
                        if f > max {
                            max = f;
                        }
                    }
                    None => null_count += 1,
                }
            }

            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            let total = count + null_count;
            let null_rate = if total > 0 {
                null_count as f64 / total as f64
            } else {
                0.0
            };

            ColumnStatistics {
                mean,
                stddev: 0.0,
                null_rate,
                min: if min == f64::INFINITY { 0.0 } else { min },
                max: if max == f64::NEG_INFINITY { 0.0 } else { max },
                count,
            }
        }
        _ => ColumnStatistics {
            mean: 0.0,
            stddev: 0.0,
            null_rate: 0.0,
            min: 0.0,
            max: 0.0,
            count: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_dataframe() {
        let df = EntityDataFrame::new(
            vec!["u1".to_string(), "u2".to_string(), "u3".to_string()],
            vec![1000, 2000, 3000],
        );
        assert_eq!(df.len(), 3);
        assert!(!df.is_empty());
    }

    #[test]
    fn test_column_data_push() {
        let mut col = ColumnData::new_float64(10);
        col.push(&FeatureValue::Float64(1.0));
        col.push(&FeatureValue::Float64(2.0));
        col.push(&FeatureValue::Null);
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_compute_statistics() {
        let col = ColumnData::Float64(vec![Some(10.0), Some(20.0), Some(30.0), None]);
        let stats = compute_column_statistics(&col);
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 1e-10);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 30.0);
        assert!((stats.null_rate - 0.25).abs() < 1e-10);
    }
}
