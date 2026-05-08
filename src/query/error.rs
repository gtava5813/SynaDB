//! Error taxonomy for Syna Query.
//!
//! Maps parser, planner, executor, and type errors to a single
//! [`QueryError`] enum and provides FFI-friendly integer error codes.

use crate::error::SynaError;
use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════════
//  FFI error codes
// ═══════════════════════════════════════════════════════════════════════

/// Query succeeded.
pub const QUERY_SUCCESS: i32 = 1;
/// Syntactic parse failure.
pub const QUERY_ERR_PARSE: i32 = -10;
/// Semantic type error.
pub const QUERY_ERR_TYPE: i32 = -11;
/// Unknown function name.
pub const QUERY_ERR_UNKNOWN_FUNC: i32 = -12;
/// Query exceeded configured timeout.
pub const QUERY_ERR_TIMEOUT: i32 = -13;
/// Invalid regex pattern.
pub const QUERY_ERR_INVALID_REGEX: i32 = -14;
/// Aggregation attempted on a non-numeric column.
pub const QUERY_ERR_NON_NUMERIC: i32 = -15;
/// Not enough data for the requested operation.
pub const QUERY_ERR_INSUFFICIENT_DATA: i32 = -16;
/// Underlying database error (returns the code from [`SynaError`]).
pub const QUERY_ERR_DATABASE: i32 = -17;
/// Internal panic caught by `catch_unwind`.
pub const QUERY_ERR_INTERNAL: i32 = -100;

// ═══════════════════════════════════════════════════════════════════════
//  ParseError — carries location for good error reporting
// ═══════════════════════════════════════════════════════════════════════

/// A syntactic parse failure with location information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Human-readable message.
    pub message: String,
    /// 1-based line number.
    pub line: usize,
    /// 1-based column number.
    pub column: usize,
    /// Source snippet around the offending character.
    pub snippet: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parse error at {}:{}: {}\n  {}",
            self.line, self.column, self.message, self.snippet
        )
    }
}

impl std::error::Error for ParseError {}

// ═══════════════════════════════════════════════════════════════════════
//  QueryError — top-level error type
// ═══════════════════════════════════════════════════════════════════════

/// Top-level error type returned from parse, plan, and execute.
#[derive(Debug, Error)]
pub enum QueryError {
    /// Syntactic parse failure.
    #[error("Parse error at {line}:{column}: {message}")]
    Parse {
        /// Human-readable message.
        message: String,
        /// 1-based line.
        line: usize,
        /// 1-based column.
        column: usize,
    },

    /// Function name not known to the query engine.
    #[error("Unknown function: {0}")]
    UnknownFunction(String),

    /// Operand types don't match the operator.
    #[error("Type error: expected {expected}, got {actual}")]
    TypeError {
        /// Expected type name.
        expected: String,
        /// Actual type name.
        actual: String,
    },

    /// Query exceeded its timeout.
    #[error("Query timeout after {0}ms")]
    Timeout(u64),

    /// Regex failed to compile.
    #[error("Invalid regex: {0}")]
    InvalidRegex(String),

    /// An aggregation was attempted on a non-numeric column.
    #[error("Non-numeric aggregation on type: {0}")]
    NonNumericAggregation(String),

    /// An operation needed at least `required` rows but only `actual` were available.
    #[error("Insufficient data: need {required}, have {actual}")]
    InsufficientData {
        /// Required count.
        required: usize,
        /// Actual count.
        actual: usize,
    },

    /// An underlying database error.
    #[error("Database error: {0}")]
    Database(#[from] SynaError),

    /// An internal logic error (bug).
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<ParseError> for QueryError {
    fn from(err: ParseError) -> Self {
        QueryError::Parse {
            message: err.message,
            line: err.line,
            column: err.column,
        }
    }
}

impl From<&QueryError> for i32 {
    fn from(err: &QueryError) -> Self {
        match err {
            QueryError::Parse { .. } => QUERY_ERR_PARSE,
            QueryError::TypeError { .. } => QUERY_ERR_TYPE,
            QueryError::UnknownFunction(_) => QUERY_ERR_UNKNOWN_FUNC,
            QueryError::Timeout(_) => QUERY_ERR_TIMEOUT,
            QueryError::InvalidRegex(_) => QUERY_ERR_INVALID_REGEX,
            QueryError::NonNumericAggregation(_) => QUERY_ERR_NON_NUMERIC,
            QueryError::InsufficientData { .. } => QUERY_ERR_INSUFFICIENT_DATA,
            QueryError::Database(_) => QUERY_ERR_DATABASE,
            QueryError::Internal(_) => QUERY_ERR_INTERNAL,
        }
    }
}

impl From<QueryError> for i32 {
    fn from(err: QueryError) -> Self {
        (&err).into()
    }
}

/// Convenience `Result` alias.
pub type QueryResultT<T> = Result<T, QueryError>;

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_error_codes_are_stable() {
        let e = QueryError::Timeout(500);
        let code: i32 = (&e).into();
        assert_eq!(code, QUERY_ERR_TIMEOUT);
    }

    #[test]
    fn parse_error_into_query_error() {
        let pe = ParseError {
            message: "expected SELECT".into(),
            line: 1,
            column: 1,
            snippet: "SELEC * FROM ...".into(),
        };
        let qe: QueryError = pe.into();
        let code: i32 = qe.into();
        assert_eq!(code, QUERY_ERR_PARSE);
    }

    #[test]
    fn parse_error_display_includes_location() {
        let pe = ParseError {
            message: "expected keyword".into(),
            line: 3,
            column: 7,
            snippet: "...".into(),
        };
        let s = format!("{}", pe);
        assert!(s.contains("3:7"));
        assert!(s.contains("expected keyword"));
    }
}
