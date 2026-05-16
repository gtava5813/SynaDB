//! Syna Query — SQL-like (EQL) and MongoDB-like (EMQ) query layer for SynaDB.
//!
//! # Overview
//!
//! The `query` module provides a declarative query interface on top of the
//! core SynaDB engine. It supports two syntactic flavours:
//!
//! - **EQL** — SQL-like syntax (`SELECT * FROM "sensor/*" WHERE ...`)
//! - **EMQ** — MongoDB-like JSON documents (`{ filter: {...}, sort: {...} }`)
//!
//! Both parse into a shared [`ast::QueryAst`], then go through a planner,
//! optimizer, and executor to produce a [`QueryResult`].
//!
//! # Status
//!
//! **Experimental, under active development.** This is the scaffolding for
//! Task 1 of the `syna-query` spec — AST types and error taxonomy only.
//! The parser, planner, optimizer, and executor ship in later tasks.

pub mod aggregation;
pub mod anomaly;
pub mod ast;
pub mod correlation;
pub mod emq_parser;
pub mod error;
pub mod executor;
pub mod explain;
pub mod ffi;
pub mod freshness_query;
pub mod lineage;
pub mod macros;
pub mod optimizer;
pub mod parser;
pub mod pattern;
pub mod planner;
pub mod predict;
pub mod prepared;
pub mod streaming;
pub mod temporal_join;
pub mod timeseries;
pub mod vector_query;

// Future submodules (added as tasks land)
// pub mod parser;
// pub mod emq_parser;
// pub mod planner;
// pub mod optimizer;
// pub mod executor;
// pub mod aggregation;
// pub mod timeseries;
// pub mod temporal_join;
// pub mod anomaly;
// pub mod pattern;
// pub mod predict;
// pub mod correlation;
// pub mod streaming;
// pub mod explain;
// pub mod macros;
// pub mod lineage;
// pub mod prepared;
// pub mod ffi;
// pub mod vector_query;
// pub mod freshness_query;

// Re-exports for downstream consumers (Task 1.1)
pub use ast::QueryAst;
pub use error::{ParseError, QueryError};

/// A single row in a query result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResultRow {
    /// The key for this row.
    pub key: String,
    /// The stored value.
    pub value: crate::types::Atom,
    /// Unix microseconds when the value was written.
    pub timestamp: u64,
}

/// Metadata attached to every [`QueryResult`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryMetadata {
    /// Total query execution time in microseconds.
    pub execution_time_us: u64,
    /// How many rows the executor had to read.
    pub rows_scanned: u64,
    /// How many rows made it into the final result.
    pub rows_returned: u64,
    /// `true` if an index scan was used (not a full scan).
    pub index_used: bool,
}

/// Final result of executing a query.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryResult {
    /// Result rows in execution order.
    pub rows: Vec<ResultRow>,
    /// Execution metadata.
    pub metadata: QueryMetadata,
}
