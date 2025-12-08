//! Competitor database benchmarks.
//!
//! This module contains benchmark implementations for databases that compete with Entangle.
//! Each competitor is tested with equivalent configurations to ensure fair comparison.
//!
//! _Requirements: 7.5_

pub mod sqlite_bench;
pub mod duckdb_bench;
pub mod leveldb_bench;
pub mod rocksdb_bench;

