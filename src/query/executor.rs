//! Query Executor — runs a [`QueryPlan`] against a [`SynaDB`] instance.
//!
//! Produces a [`QueryResult`] with rows and execution metadata.

use crate::engine::SynaDB;
use crate::query::ast::*;
use crate::query::error::QueryError;
use crate::query::planner::{AggregationPlan, FilterStep, QueryPlan, ScanType};
use crate::query::{QueryMetadata, QueryResult, ResultRow};
use crate::types::Atom;
use std::cmp::Ordering;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════
//  QueryExecutor
// ═══════════════════════════════════════════════════════════════════════

/// Executes a query plan against a database.
pub struct QueryExecutor<'a> {
    db: &'a mut SynaDB,
    start_time: Instant,
    rows_scanned: u64,
}

impl<'a> QueryExecutor<'a> {
    /// Create a new executor bound to a database.
    pub fn new(db: &'a mut SynaDB) -> Self {
        Self {
            db,
            start_time: Instant::now(),
            rows_scanned: 0,
        }
    }

    /// Execute a query plan and return results.
    pub fn execute(&mut self, plan: QueryPlan) -> Result<QueryResult, QueryError> {
        self.start_time = Instant::now();
        self.rows_scanned = 0;

        // 1. Scan for candidate keys
        let candidates = self.execute_scan(&plan.scan)?;

        // 2. Load values and build rows
        let mut rows = self.load_rows(&candidates)?;

        // 3. Apply filters
        rows = self.apply_filters(rows, &plan.filters);

        // 4. Apply aggregation (if any)
        if let Some(agg) = &plan.aggregation {
            rows = self.execute_aggregation(rows, agg)?;
        }

        // 5. Apply ordering
        if let Some(order) = &plan.ordering {
            self.sort_rows(&mut rows, order);
        }

        // 6. Apply pagination
        if let Some(pag) = &plan.pagination {
            let offset = pag.offset as usize;
            let limit = pag.limit as usize;
            rows = rows.into_iter().skip(offset).take(limit).collect();
        }

        let rows_returned = rows.len() as u64;
        let execution_time_us = self.start_time.elapsed().as_micros() as u64;
        let index_used = !matches!(plan.scan, ScanType::FullScan);

        Ok(QueryResult {
            rows,
            metadata: QueryMetadata {
                execution_time_us,
                rows_scanned: self.rows_scanned,
                rows_returned,
                index_used,
            },
        })
    }

    // ─── Scan ────────────────────────────────────────────────────────

    fn execute_scan(&self, scan: &ScanType) -> Result<Vec<String>, QueryError> {
        let all_keys = self.db.keys();
        let keys = match scan {
            ScanType::IndexExact(key) => {
                if self.db.exists(key) {
                    vec![key.clone()]
                } else {
                    vec![]
                }
            }
            ScanType::IndexPrefix(prefix) => all_keys
                .into_iter()
                .filter(|k| k.starts_with(prefix.as_str()))
                .collect(),
            ScanType::PatternScan(pattern) => all_keys
                .into_iter()
                .filter(|k| matches_key(k, pattern))
                .collect(),
            ScanType::FullScan => all_keys,
        };
        Ok(keys)
    }

    // ─── Load rows ───────────────────────────────────────────────────

    fn load_rows(&mut self, keys: &[String]) -> Result<Vec<ResultRow>, QueryError> {
        let mut rows = Vec::with_capacity(keys.len());
        for key in keys {
            self.rows_scanned += 1;
            if let Some(atom) = self.db.get(key)? {
                // Use 0 as timestamp placeholder — full timestamp requires
                // reading the log header which is an optimization for later.
                rows.push(ResultRow {
                    key: key.clone(),
                    value: atom,
                    timestamp: 0,
                });
            }
        }
        Ok(rows)
    }

    // ─── Filters ─────────────────────────────────────────────────────

    fn apply_filters(&self, rows: Vec<ResultRow>, filters: &[FilterStep]) -> Vec<ResultRow> {
        if filters.is_empty() {
            return rows;
        }
        rows.into_iter()
            .filter(|row| filters.iter().all(|f| evaluate_filter(row, f)))
            .collect()
    }

    // ─── Aggregation ─────────────────────────────────────────────────

    fn execute_aggregation(
        &self,
        rows: Vec<ResultRow>,
        agg: &AggregationPlan,
    ) -> Result<Vec<ResultRow>, QueryError> {
        // For now: compute aggregates over all rows (no GROUP BY splitting yet)
        let mut result_rows = Vec::new();

        for func in &agg.functions {
            let value = compute_aggregate(func, &rows)?;
            result_rows.push(ResultRow {
                key: format!("{:?}", func),
                value,
                timestamp: 0,
            });
        }

        Ok(result_rows)
    }

    // ─── Ordering ────────────────────────────────────────────────────

    fn sort_rows(&self, rows: &mut [ResultRow], order: &OrderBy) {
        rows.sort_by(|a, b| {
            let cmp = match order.field {
                OrderField::Timestamp => a.timestamp.cmp(&b.timestamp),
                OrderField::Key => a.key.cmp(&b.key),
                OrderField::Value => compare_atoms(&a.value, &b.value),
            };
            match order.direction {
                Direction::Asc => cmp,
                Direction::Desc => cmp.reverse(),
            }
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Filter evaluation
// ═══════════════════════════════════════════════════════════════════════

fn evaluate_filter(row: &ResultRow, filter: &FilterStep) -> bool {
    match filter {
        FilterStep::KeyPattern(kp) => matches_key(&row.key, kp),
        FilterStep::ValueComparison { field, op, value } => {
            let left = match field {
                OrderField::Key => &Atom::Text(row.key.clone()),
                OrderField::Value => &row.value,
                OrderField::Timestamp => &Atom::Int(row.timestamp as i64),
            };
            compare_with_filter(left, op, value)
        }
        FilterStep::TimeRange(tr) => {
            let ts = row.timestamp;
            let after = tr.start.map_or(true, |s| ts >= s);
            let before = tr.end.map_or(true, |e| ts <= e);
            after && before
        }
        FilterStep::Boolean { op, children } => match op {
            crate::query::planner::BoolOp::And => children.iter().all(|c| evaluate_filter(row, c)),
            crate::query::planner::BoolOp::Or => children.iter().any(|c| evaluate_filter(row, c)),
            crate::query::planner::BoolOp::Not => {
                children.first().map_or(true, |c| !evaluate_filter(row, c))
            }
        },
    }
}

fn compare_with_filter(left: &Atom, op: &ComparisonOp, rhs: &ValueFilter) -> bool {
    match rhs {
        ValueFilter::Single(right) => compare_atoms_op(left, op, right),
        ValueFilter::List(list) => match op {
            ComparisonOp::In => list.iter().any(|v| left == v),
            ComparisonOp::Nin => !list.iter().any(|v| left == v),
            _ => false,
        },
    }
}

fn compare_atoms_op(left: &Atom, op: &ComparisonOp, right: &Atom) -> bool {
    match op {
        ComparisonOp::Eq => left == right,
        ComparisonOp::Ne => left != right,
        ComparisonOp::Gt => compare_atoms(left, right) == Ordering::Greater,
        ComparisonOp::Gte => compare_atoms(left, right) != Ordering::Less,
        ComparisonOp::Lt => compare_atoms(left, right) == Ordering::Less,
        ComparisonOp::Lte => compare_atoms(left, right) != Ordering::Greater,
        ComparisonOp::In | ComparisonOp::Nin => false, // handled by List branch
        ComparisonOp::Like | ComparisonOp::Regex => {
            // Simple LIKE: convert % to .* and match
            if let (Atom::Text(l), Atom::Text(r)) = (left, right) {
                let pattern = r.replace('%', ".*").replace('_', ".");
                regex::Regex::new(&format!("^{}$", pattern))
                    .map(|re| re.is_match(l))
                    .unwrap_or(false)
            } else {
                false
            }
        }
    }
}

fn compare_atoms(a: &Atom, b: &Atom) -> Ordering {
    match (a, b) {
        (Atom::Float(l), Atom::Float(r)) => l.partial_cmp(r).unwrap_or(Ordering::Equal),
        (Atom::Int(l), Atom::Int(r)) => l.cmp(r),
        (Atom::Text(l), Atom::Text(r)) => l.cmp(r),
        (Atom::Float(l), Atom::Int(r)) => l.partial_cmp(&(*r as f64)).unwrap_or(Ordering::Equal),
        (Atom::Int(l), Atom::Float(r)) => (*l as f64).partial_cmp(r).unwrap_or(Ordering::Equal),
        _ => Ordering::Equal,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Key pattern matching
// ═══════════════════════════════════════════════════════════════════════

fn matches_key(key: &str, pattern: &KeyPattern) -> bool {
    match pattern {
        KeyPattern::Exact(p) => key == p,
        KeyPattern::Prefix(p) => key.starts_with(p.as_str()),
        KeyPattern::Glob(g) => {
            let re_pattern = g.replace('*', ".*").replace('?', ".");
            regex::Regex::new(&format!("^{}$", re_pattern))
                .map(|re| re.is_match(key))
                .unwrap_or(false)
        }
        KeyPattern::Regex(r) => regex::Regex::new(r)
            .map(|re| re.is_match(key))
            .unwrap_or(false),
        KeyPattern::Union(parts) => parts.iter().any(|p| matches_key(key, p)),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Aggregation
// ═══════════════════════════════════════════════════════════════════════

fn compute_aggregate(func: &AggregateFunction, rows: &[ResultRow]) -> Result<Atom, QueryError> {
    match func {
        AggregateFunction::Count => Ok(Atom::Int(rows.len() as i64)),
        AggregateFunction::Sum => {
            let sum = extract_floats(rows)?.iter().sum::<f64>();
            Ok(Atom::Float(sum))
        }
        AggregateFunction::Avg => {
            let floats = extract_floats(rows)?;
            if floats.is_empty() {
                return Ok(Atom::Null);
            }
            Ok(Atom::Float(
                floats.iter().sum::<f64>() / floats.len() as f64,
            ))
        }
        AggregateFunction::Min => {
            let floats = extract_floats(rows)?;
            Ok(floats
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(Atom::Float)
                .unwrap_or(Atom::Null))
        }
        AggregateFunction::Max => {
            let floats = extract_floats(rows)?;
            Ok(floats
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(Atom::Float)
                .unwrap_or(Atom::Null))
        }
        AggregateFunction::First => Ok(rows.first().map(|r| r.value.clone()).unwrap_or(Atom::Null)),
        AggregateFunction::Last => Ok(rows.last().map(|r| r.value.clone()).unwrap_or(Atom::Null)),
    }
}

fn extract_floats(rows: &[ResultRow]) -> Result<Vec<f64>, QueryError> {
    let mut floats = Vec::new();
    for row in rows {
        match &row.value {
            Atom::Float(f) => floats.push(*f),
            Atom::Int(i) => floats.push(*i as f64),
            _ => {} // Skip non-numeric (per spec: "skip non-numeric values")
        }
    }
    Ok(floats)
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::optimizer::optimize;
    use crate::query::parser::parse_eql;
    use crate::query::planner::QueryPlan;
    use tempfile::tempdir;

    fn setup_db() -> (SynaDB, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let mut db = SynaDB::new(path).unwrap();
        db.append("sensor/temp", Atom::Float(23.5)).unwrap();
        db.append("sensor/temp", Atom::Float(24.1)).unwrap();
        db.append("sensor/humidity", Atom::Float(45.0)).unwrap();
        db.append("config/name", Atom::Text("test".into())).unwrap();
        db.append("counter", Atom::Int(42)).unwrap();
        (db, dir)
    }

    #[test]
    fn execute_select_star() {
        let (mut db, _dir) = setup_db();
        let ast = parse_eql("SELECT * FROM \"sensor/*\"").unwrap();
        let mut plan = QueryPlan::from_ast(&ast, db.keys().len() as u64).unwrap();
        optimize(&mut plan);

        let mut executor = QueryExecutor::new(&mut db);
        let result = executor.execute(plan).unwrap();

        // Should find sensor/temp and sensor/humidity
        assert_eq!(result.rows.len(), 2);
        assert!(result.metadata.index_used);
    }

    #[test]
    fn execute_select_with_value_filter() {
        let (mut db, _dir) = setup_db();
        let ast = parse_eql("SELECT * FROM \"sensor/*\" WHERE value > 24").unwrap();
        let mut plan = QueryPlan::from_ast(&ast, db.keys().len() as u64).unwrap();
        optimize(&mut plan);

        let mut executor = QueryExecutor::new(&mut db);
        let result = executor.execute(plan).unwrap();

        // Only sensor/humidity (45.0) passes value > 24
        // sensor/temp latest is 24.1 which also passes
        assert!(result.rows.len() >= 1);
        for row in &result.rows {
            match &row.value {
                Atom::Float(f) => assert!(*f > 24.0),
                _ => panic!("expected float"),
            }
        }
    }

    #[test]
    fn execute_aggregate_count() {
        let (mut db, _dir) = setup_db();
        let ast = parse_eql("SELECT COUNT(*) FROM \"sensor/*\"").unwrap();
        let mut plan = QueryPlan::from_ast(&ast, db.keys().len() as u64).unwrap();
        optimize(&mut plan);

        let mut executor = QueryExecutor::new(&mut db);
        let result = executor.execute(plan).unwrap();

        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].value, Atom::Int(2)); // 2 sensor keys
    }

    #[test]
    fn execute_aggregate_avg() {
        let (mut db, _dir) = setup_db();
        let ast = parse_eql("SELECT AVG(value) FROM \"sensor/*\"").unwrap();
        let mut plan = QueryPlan::from_ast(&ast, db.keys().len() as u64).unwrap();
        optimize(&mut plan);

        let mut executor = QueryExecutor::new(&mut db);
        let result = executor.execute(plan).unwrap();

        assert_eq!(result.rows.len(), 1);
        match &result.rows[0].value {
            Atom::Float(avg) => {
                // avg of 24.1 and 45.0 = 34.55
                assert!((*avg - 34.55).abs() < 0.01);
            }
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn execute_with_limit() {
        let (mut db, _dir) = setup_db();
        let ast = parse_eql("SELECT * FROM \"sensor/*\" LIMIT 1").unwrap();
        let mut plan = QueryPlan::from_ast(&ast, db.keys().len() as u64).unwrap();
        optimize(&mut plan);

        let mut executor = QueryExecutor::new(&mut db);
        let result = executor.execute(plan).unwrap();

        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn execute_metadata_populated() {
        let (mut db, _dir) = setup_db();
        let ast = parse_eql("SELECT * FROM \"sensor/*\"").unwrap();
        let mut plan = QueryPlan::from_ast(&ast, db.keys().len() as u64).unwrap();
        optimize(&mut plan);

        let mut executor = QueryExecutor::new(&mut db);
        let result = executor.execute(plan).unwrap();

        assert!(result.metadata.execution_time_us > 0 || result.metadata.rows_scanned > 0);
        assert!(result.metadata.rows_scanned >= result.metadata.rows_returned);
        assert!(result.metadata.index_used);
    }
}
