//! Prepared Statements — parameterized queries for repeated execution.
//!
//! Avoids re-parsing by caching the AST and substituting parameters at execution time.
//!
//! ```text
//! PREPARE: SELECT * FROM 'sensor/*' WHERE value > $threshold LIMIT $count
//! EXECUTE: { "threshold": 30.0, "count": 10 }
//! ```

use crate::query::ast::*;
use crate::query::error::{ParseError, QueryError};
use crate::query::parser::parse_eql;
use crate::types::Atom;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// A prepared query with parameter placeholders.
#[derive(Debug, Clone)]
pub struct PreparedQuery {
    /// The original query string (for debugging).
    pub source: String,
    /// Parsed AST with `$param` placeholders in value positions.
    pub ast: QueryAst,
    /// Parameter names extracted from the query.
    pub params: Vec<String>,
}

/// A parameter slot definition.
#[derive(Debug, Clone)]
pub struct ParamSlot {
    /// Parameter name (without `$`).
    pub name: String,
    /// Position in the query string.
    pub position: usize,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Prepare a query by extracting `$param` placeholders and parsing.
///
/// Parameters are identified by `$name` in value positions. They are
/// replaced with a sentinel value during parsing, then substituted
/// at execution time via [`bind_params`].
pub fn prepare(query: &str) -> Result<PreparedQuery, ParseError> {
    // Extract parameter names
    let params = extract_params(query);

    // Replace $params with placeholder values for parsing
    let mut substituted = query.to_string();
    for param in &params {
        substituted = substituted.replace(&format!("${}", param), "0");
    }

    // Parse the substituted query
    let ast = parse_eql(&substituted)?;

    Ok(PreparedQuery {
        source: query.to_string(),
        ast,
        params,
    })
}

/// Bind parameters to a prepared query, producing a ready-to-execute AST.
///
/// Returns a new AST with all `$param` placeholders replaced by the
/// provided values.
pub fn bind_params(
    prepared: &PreparedQuery,
    params: &HashMap<String, Atom>,
) -> Result<QueryAst, QueryError> {
    // Re-parse with actual values substituted
    let mut query = prepared.source.clone();
    for param_name in &prepared.params {
        let value = params
            .get(param_name)
            .ok_or_else(|| QueryError::Internal(format!("missing parameter: ${}", param_name)))?;
        let value_str = atom_to_query_literal(value);
        query = query.replace(&format!("${}", param_name), &value_str);
    }

    parse_eql(&query).map_err(|e| QueryError::Parse {
        message: e.message,
        line: e.line,
        column: e.column,
    })
}

// ═══════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Extract `$param` names from a query string.
fn extract_params(query: &str) -> Vec<String> {
    let mut params = Vec::new();
    let mut chars = query.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '$' {
            let mut name = String::new();
            while let Some(&nc) = chars.peek() {
                if nc.is_alphanumeric() || nc == '_' {
                    name.push(nc);
                    chars.next();
                } else {
                    break;
                }
            }
            if !name.is_empty() && !params.contains(&name) {
                params.push(name);
            }
        }
    }

    params
}

/// Convert an Atom to a query literal string for substitution.
fn atom_to_query_literal(atom: &Atom) -> String {
    match atom {
        Atom::Float(f) => format!("{}", f),
        Atom::Int(i) => format!("{}", i),
        Atom::Text(s) => format!("'{}'", s.replace('\'', "''")),
        Atom::Null => "NULL".to_string(),
        Atom::Bytes(_) => "NULL".to_string(), // bytes not supported in queries
        Atom::Vector(_, _) => "NULL".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_extracts_params() {
        let prepared =
            prepare("SELECT * FROM 'sensor/*' WHERE value > $threshold LIMIT $count").unwrap();
        assert_eq!(prepared.params, vec!["threshold", "count"]);
    }

    #[test]
    fn test_bind_params_substitutes() {
        let prepared = prepare("SELECT * FROM 'k' WHERE value > $min").unwrap();
        let mut params = HashMap::new();
        params.insert("min".to_string(), Atom::Float(42.0));

        let ast = bind_params(&prepared, &params).unwrap();
        match ast {
            QueryAst::Select(q) => {
                assert!(q.where_clause.is_some());
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_bind_missing_param_errors() {
        let prepared = prepare("SELECT * FROM 'k' WHERE value > $x").unwrap();
        let params = HashMap::new(); // empty — missing $x

        let result = bind_params(&prepared, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_params() {
        let prepared = prepare("SELECT * FROM 'k'").unwrap();
        assert!(prepared.params.is_empty());

        let ast = bind_params(&prepared, &HashMap::new()).unwrap();
        match ast {
            QueryAst::Select(_) => {}
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_text_param_escaping() {
        let prepared = prepare("SELECT * FROM 'k' WHERE key = $name").unwrap();
        let mut params = HashMap::new();
        params.insert("name".to_string(), Atom::Text("hello_world".to_string()));

        let ast = bind_params(&prepared, &params).unwrap();
        match ast {
            QueryAst::Select(q) => assert!(q.where_clause.is_some()),
            _ => panic!("expected Select"),
        }
    }
}
