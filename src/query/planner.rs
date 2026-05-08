//! Query Planner — converts AST into an executable query plan.
//!
//! Responsibilities:
//! - Select optimal scan type (index exact, prefix, pattern, full)
//! - Build filter pipeline from WHERE clause
//! - Type-check aggregation functions
//! - Estimate query cost

use crate::query::ast::*;
use crate::query::error::QueryError;
use crate::types::Atom;

// ═══════════════════════════════════════════════════════════════════════
//  Query Plan
// ═══════════════════════════════════════════════════════════════════════

/// A fully resolved, executable query plan.
#[derive(Debug)]
pub struct QueryPlan {
    /// How to scan the database for candidate keys.
    pub scan: ScanType,
    /// Filters to apply after scanning.
    pub filters: Vec<FilterStep>,
    /// Aggregation to apply (if any).
    pub aggregation: Option<AggregationPlan>,
    /// Projections to apply to each result row.
    pub projections: Vec<Projection>,
    /// Ordering to apply after filtering.
    pub ordering: Option<OrderBy>,
    /// Pagination (limit + offset).
    pub pagination: Option<PaginationStep>,
    /// Estimated cost of executing this plan.
    pub estimated_cost: CostEstimate,
}

/// How to scan the database for candidate keys.
#[derive(Debug, Clone)]
pub enum ScanType {
    /// O(1) — exact key lookup from the in-memory index.
    IndexExact(String),
    /// O(k) — prefix scan from the in-memory index.
    IndexPrefix(String),
    /// O(n) filtered — scan all keys, apply glob/regex filter.
    PatternScan(KeyPattern),
    /// O(n) — read every key.
    FullScan,
}

/// A single filter step in the pipeline.
#[derive(Debug, Clone)]
pub enum FilterStep {
    /// Filter by key pattern.
    KeyPattern(KeyPattern),
    /// Filter by value comparison.
    ValueComparison {
        field: OrderField,
        op: ComparisonOp,
        value: ValueFilter,
    },
    /// Filter by time range.
    TimeRange(TimeRange),
    /// Boolean combination of sub-filters.
    Boolean {
        op: BoolOp,
        children: Vec<FilterStep>,
    },
}

/// Simplified boolean op for filter steps.
#[derive(Debug, Clone)]
pub enum BoolOp {
    And,
    Or,
    Not,
}

/// Aggregation plan.
#[derive(Debug, Clone)]
pub struct AggregationPlan {
    /// Functions to compute.
    pub functions: Vec<AggregateFunction>,
    /// Group-by specification.
    pub group_by: Option<GroupBy>,
}

/// Pagination step.
#[derive(Debug, Clone, Copy)]
pub struct PaginationStep {
    pub limit: u64,
    pub offset: u64,
}

/// Estimated cost of a query plan.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CostEstimate {
    /// Estimated number of disk/index reads.
    pub io_cost: f64,
    /// Estimated CPU operations.
    pub cpu_cost: f64,
    /// Estimated memory usage in bytes.
    pub memory_bytes: u64,
    /// Estimated result row count.
    pub estimated_rows: u64,
}

impl Default for CostEstimate {
    fn default() -> Self {
        Self {
            io_cost: 0.0,
            cpu_cost: 0.0,
            memory_bytes: 0,
            estimated_rows: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Plan Builder
// ═══════════════════════════════════════════════════════════════════════

impl QueryPlan {
    /// Build a query plan from a parsed AST.
    ///
    /// `total_keys` is the current number of keys in the database,
    /// used for cost estimation.
    pub fn from_ast(ast: &QueryAst, total_keys: u64) -> Result<Self, QueryError> {
        match ast {
            QueryAst::Select(q) => Self::plan_select(q, total_keys),
            QueryAst::Aggregate(q) => Self::plan_aggregate(q, total_keys),
            QueryAst::Explain(inner) => Self::from_ast(inner, total_keys),
            _ => Err(QueryError::Internal(
                "unsupported query type for planning".into(),
            )),
        }
    }

    fn plan_select(q: &SelectQuery, total_keys: u64) -> Result<Self, QueryError> {
        let scan = select_scan_type(q.from.as_ref(), total_keys);
        let filters = build_filters(q.where_clause.as_ref());
        let pagination = match (q.limit, q.offset) {
            (Some(limit), offset) => Some(PaginationStep {
                limit,
                offset: offset.unwrap_or(0),
            }),
            (None, Some(offset)) => Some(PaginationStep {
                limit: u64::MAX,
                offset,
            }),
            _ => None,
        };
        let estimated_cost = estimate_cost(&scan, &filters, total_keys);

        Ok(QueryPlan {
            scan,
            filters,
            aggregation: None,
            projections: q.projections.clone(),
            ordering: q.order_by.clone(),
            pagination,
            estimated_cost,
        })
    }

    fn plan_aggregate(q: &AggregateQuery, total_keys: u64) -> Result<Self, QueryError> {
        // Type-check: SUM and AVG require numeric
        for func in &q.aggregations {
            if matches!(func, AggregateFunction::Sum | AggregateFunction::Avg) {
                // We can't fully type-check without data, but we note the constraint
                // The executor will return NonNumericAggregation if it encounters non-numeric
            }
        }

        let scan = select_scan_type(q.from.as_ref(), total_keys);
        let filters = build_filters(q.where_clause.as_ref());
        let estimated_cost = estimate_cost(&scan, &filters, total_keys);

        Ok(QueryPlan {
            scan,
            filters,
            aggregation: Some(AggregationPlan {
                functions: q.aggregations.clone(),
                group_by: q.group_by.clone(),
            }),
            projections: vec![],
            ordering: q.order_by.clone(),
            pagination: q.limit.map(|limit| PaginationStep { limit, offset: 0 }),
            estimated_cost,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Scan Type Selection
// ═══════════════════════════════════════════════════════════════════════

/// Select the optimal scan type based on the FROM key pattern.
fn select_scan_type(from: Option<&KeyPattern>, _total_keys: u64) -> ScanType {
    match from {
        None => ScanType::FullScan,
        Some(KeyPattern::Exact(key)) => ScanType::IndexExact(key.clone()),
        Some(KeyPattern::Prefix(prefix)) => ScanType::IndexPrefix(prefix.clone()),
        Some(KeyPattern::Glob(glob)) => {
            // Try to extract a literal prefix from the glob
            if let Some(prefix) = extract_prefix_from_glob(glob) {
                ScanType::IndexPrefix(prefix)
            } else {
                ScanType::PatternScan(KeyPattern::Glob(glob.clone()))
            }
        }
        Some(KeyPattern::Regex(re)) => ScanType::PatternScan(KeyPattern::Regex(re.clone())),
        Some(KeyPattern::Union(parts)) => {
            // If all parts are exact, we could do multiple index lookups
            // For now, treat as pattern scan
            ScanType::PatternScan(KeyPattern::Union(parts.clone()))
        }
    }
}

/// Extract a literal prefix from a glob pattern.
///
/// `"sensor/*"` → `Some("sensor/")`
/// `"*"` → `None`
/// `"s?nsor/*"` → `None` (wildcard before end)
fn extract_prefix_from_glob(glob: &str) -> Option<String> {
    if let Some(star_pos) = glob.find('*') {
        let prefix = &glob[..star_pos];
        // Only valid if no other wildcards in the prefix
        if !prefix.contains('?') && !prefix.is_empty() {
            return Some(prefix.to_string());
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════
//  Filter Building
// ═══════════════════════════════════════════════════════════════════════

/// Convert a WHERE clause into a list of filter steps.
fn build_filters(where_clause: Option<&WhereClause>) -> Vec<FilterStep> {
    match where_clause {
        None => vec![],
        Some(wc) => vec![condition_to_filter(&wc.root)],
    }
}

fn condition_to_filter(cond: &Condition) -> FilterStep {
    match cond {
        Condition::Comparison { field, op, rhs } => FilterStep::ValueComparison {
            field: field.clone(),
            op: *op,
            value: rhs.clone(),
        },
        Condition::Key(kp) => FilterStep::KeyPattern(kp.clone()),
        Condition::TimeRange(tr) => FilterStep::TimeRange(*tr),
        Condition::Boolean(BooleanOp::And(cs)) => FilterStep::Boolean {
            op: BoolOp::And,
            children: cs.iter().map(condition_to_filter).collect(),
        },
        Condition::Boolean(BooleanOp::Or(cs)) => FilterStep::Boolean {
            op: BoolOp::Or,
            children: cs.iter().map(condition_to_filter).collect(),
        },
        Condition::Boolean(BooleanOp::Not(c)) => FilterStep::Boolean {
            op: BoolOp::Not,
            children: vec![condition_to_filter(c)],
        },
        // For now, advanced conditions (anomaly, pattern, freshness, similarity)
        // pass through as value comparisons with a sentinel
        _ => FilterStep::ValueComparison {
            field: OrderField::Value,
            op: ComparisonOp::Eq,
            value: ValueFilter::Single(Atom::Null),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Cost Estimation
// ═══════════════════════════════════════════════════════════════════════

fn estimate_cost(scan: &ScanType, filters: &[FilterStep], total_keys: u64) -> CostEstimate {
    let (io_cost, estimated_rows) = match scan {
        ScanType::IndexExact(_) => (1.0, 1),
        ScanType::IndexPrefix(_) => {
            // Estimate 10% of keys match a prefix
            let est = (total_keys as f64 * 0.1).max(1.0) as u64;
            (est as f64, est)
        }
        ScanType::PatternScan(_) => (total_keys as f64, total_keys / 2),
        ScanType::FullScan => (total_keys as f64, total_keys),
    };

    // Each filter reduces rows by ~50% (rough heuristic)
    let selectivity = 0.5_f64.powi(filters.len() as i32);
    let final_rows = (estimated_rows as f64 * selectivity).max(1.0) as u64;

    CostEstimate {
        io_cost,
        cpu_cost: io_cost * 2.0,        // CPU is ~2x IO for in-memory index
        memory_bytes: final_rows * 128, // ~128 bytes per row estimate
        estimated_rows: final_rows,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::parser::parse_eql;

    #[test]
    fn plan_simple_select() {
        let ast = parse_eql("SELECT * FROM \"sensor/*\"").unwrap();
        let plan = QueryPlan::from_ast(&ast, 1000).unwrap();
        match &plan.scan {
            ScanType::IndexPrefix(p) => assert_eq!(p, "sensor/"),
            other => panic!("expected IndexPrefix, got {:?}", other),
        }
    }

    #[test]
    fn plan_exact_key() {
        let ast = parse_eql("SELECT * FROM \"config/name\"").unwrap();
        let plan = QueryPlan::from_ast(&ast, 1000).unwrap();
        match &plan.scan {
            ScanType::IndexExact(k) => assert_eq!(k, "config/name"),
            other => panic!("expected IndexExact, got {:?}", other),
        }
    }

    #[test]
    fn plan_with_limit() {
        let ast = parse_eql("SELECT * FROM \"k\" LIMIT 10 OFFSET 5").unwrap();
        let plan = QueryPlan::from_ast(&ast, 100).unwrap();
        let pag = plan.pagination.unwrap();
        assert_eq!(pag.limit, 10);
        assert_eq!(pag.offset, 5);
    }

    #[test]
    fn plan_aggregate_has_aggregation() {
        let ast = parse_eql("SELECT AVG(value) FROM \"s\" GROUP BY HOUR").unwrap();
        let plan = QueryPlan::from_ast(&ast, 500).unwrap();
        let agg = plan.aggregation.unwrap();
        assert_eq!(agg.functions, vec![AggregateFunction::Avg]);
        assert_eq!(agg.group_by, Some(GroupBy::TimeBucket(TimeBucket::Hour)));
    }

    #[test]
    fn plan_cost_estimate_index_cheaper_than_full() {
        let ast_prefix = parse_eql("SELECT * FROM \"sensor/*\"").unwrap();
        let ast_full = parse_eql("SELECT * FROM \"*\"").unwrap();

        let plan_prefix = QueryPlan::from_ast(&ast_prefix, 10000).unwrap();
        let plan_full = QueryPlan::from_ast(&ast_full, 10000).unwrap();

        assert!(plan_prefix.estimated_cost.io_cost < plan_full.estimated_cost.io_cost);
    }

    #[test]
    fn extract_prefix_from_glob_works() {
        assert_eq!(extract_prefix_from_glob("sensor/*"), Some("sensor/".into()));
        assert_eq!(
            extract_prefix_from_glob("data/temp/*"),
            Some("data/temp/".into())
        );
        assert_eq!(extract_prefix_from_glob("*"), None);
        assert_eq!(extract_prefix_from_glob("s?nsor/*"), None);
    }
}
