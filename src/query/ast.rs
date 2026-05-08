//! Abstract syntax tree for Syna Query.
//!
//! Covers both EQL (SQL-like) and EMQ (MongoDB-like) syntaxes. The parsers
//! in later tasks produce [`QueryAst`] values that the planner/executor
//! consume.

use crate::types::Atom;
use serde::{Deserialize, Serialize};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
//  Root AST
// ═══════════════════════════════════════════════════════════════════════

/// Root AST node for every query type Syna Query understands.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryAst {
    /// SQL-like SELECT statement.
    Select(SelectQuery),
    /// MongoDB-like find document.
    Find(FindQuery),
    /// Aggregation query (`SELECT AVG(value) FROM ...`).
    Aggregate(AggregateQuery),
    /// Temporal join query.
    TemporalJoin(TemporalJoinQuery),
    /// Streaming (continuous) query.
    Stream(StreamQuery),
    /// EXPLAIN or EXPLAIN ANALYZE wrapping another query.
    Explain(Box<QueryAst>),
    /// Macro definition (CREATE MACRO ...).
    Macro(MacroDefinition),
    /// Data lineage query (LINEAGE() / DERIVED_FROM()).
    Lineage(LineageQuery),
}

// ═══════════════════════════════════════════════════════════════════════
//  EQL SELECT
// ═══════════════════════════════════════════════════════════════════════

/// A SQL-like SELECT query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectQuery {
    /// Columns/expressions to project.
    pub projections: Vec<Projection>,
    /// FROM clause — which keys to scan.
    pub from: Option<KeyPattern>,
    /// WHERE clause.
    pub where_clause: Option<WhereClause>,
    /// ORDER BY clause.
    pub order_by: Option<OrderBy>,
    /// LIMIT N.
    pub limit: Option<u64>,
    /// OFFSET N.
    pub offset: Option<u64>,
}

// ═══════════════════════════════════════════════════════════════════════
//  EMQ Find
// ═══════════════════════════════════════════════════════════════════════

/// A MongoDB-like find query built from a JSON filter document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FindQuery {
    /// Filter document (`{ value: { $gt: 10 } }`).
    pub filter: FilterDocument,
    /// Optional projection document (`{ key: 1, value: 1 }`).
    pub projection: Option<ProjectionDocument>,
    /// Optional sort document (`{ timestamp: -1 }`).
    pub sort: Option<SortDocument>,
    /// Upper bound on results.
    pub limit: Option<u64>,
    /// How many matching documents to skip.
    pub skip: Option<u64>,
}

/// Placeholder for EMQ filter documents. Populated by the EMQ parser.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FilterDocument {
    /// Raw JSON representation of the filter.
    pub raw: serde_json::Value,
}

/// Placeholder for EMQ projection documents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProjectionDocument {
    /// Raw JSON representation of the projection.
    pub raw: serde_json::Value,
}

/// Placeholder for EMQ sort documents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SortDocument {
    /// Raw JSON representation of the sort spec.
    pub raw: serde_json::Value,
}

// ═══════════════════════════════════════════════════════════════════════
//  Aggregate
// ═══════════════════════════════════════════════════════════════════════

/// An aggregation query (`SELECT AVG(value) FROM ... GROUP BY ...`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregateQuery {
    /// Aggregate functions to compute.
    pub aggregations: Vec<AggregateFunction>,
    /// FROM clause.
    pub from: Option<KeyPattern>,
    /// WHERE clause applied before aggregation.
    pub where_clause: Option<WhereClause>,
    /// Optional GROUP BY.
    pub group_by: Option<GroupBy>,
    /// Optional HAVING clause.
    pub having: Option<WhereClause>,
    /// ORDER BY.
    pub order_by: Option<OrderBy>,
    /// LIMIT.
    pub limit: Option<u64>,
}

/// Available aggregate functions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AggregateFunction {
    /// `COUNT(*)`.
    Count,
    /// `SUM(value)` — numeric only.
    Sum,
    /// `AVG(value)` — numeric only.
    Avg,
    /// `MIN(value)`.
    Min,
    /// `MAX(value)`.
    Max,
    /// `FIRST(value)` — earliest value.
    First,
    /// `LAST(value)` — latest value.
    Last,
}

/// GROUP BY specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GroupBy {
    /// Group by key (each distinct key is a group).
    Key,
    /// Group by time bucket.
    TimeBucket(TimeBucket),
}

/// Time bucket sizes for `GROUP BY TIME_BUCKET(...)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TimeBucket {
    /// 1 minute.
    Minute,
    /// 1 hour.
    Hour,
    /// 1 day.
    Day,
    /// 1 week.
    Week,
    /// 1 month.
    Month,
}

// ═══════════════════════════════════════════════════════════════════════
//  Temporal Join / Stream / Macro / Lineage (placeholders)
// ═══════════════════════════════════════════════════════════════════════

/// Temporal join query (`A TEMPORAL JOIN B ASOF WITHIN 5m`).
///
/// Placeholder populated by Task 12.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalJoinQuery {
    /// Left side key pattern.
    pub left: KeyPattern,
    /// Right side key pattern.
    pub right: KeyPattern,
    /// Join type (e.g., "ASOF", "INTERPOLATED", "FORWARD_FILL").
    pub join_type: String,
    /// Maximum time gap accepted for the match, in microseconds.
    pub within_micros: Option<u64>,
}

/// Streaming (continuous) query.
///
/// Placeholder populated by Task 19.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamQuery {
    /// Stream name.
    pub name: String,
    /// Underlying query definition.
    pub body: Box<QueryAst>,
}

/// Macro definition.
///
/// Placeholder populated by Task 21.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MacroDefinition {
    /// Macro name.
    pub name: String,
    /// Parameter names.
    pub params: Vec<String>,
    /// Body template (not yet parsed until expansion).
    pub body: String,
}

/// Data lineage query.
///
/// Placeholder populated by Task 22.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LineageQuery {
    /// Key whose lineage is being queried.
    pub key: String,
}

// ═══════════════════════════════════════════════════════════════════════
//  Shared AST pieces — key patterns, conditions, projections
// ═══════════════════════════════════════════════════════════════════════

/// Key matching patterns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyPattern {
    /// Exact key match (`"sensor/temp"`).
    Exact(String),
    /// Prefix match (`"sensor/"`).
    Prefix(String),
    /// Glob pattern (`"sensor/*"`).
    Glob(String),
    /// Regex pattern.
    Regex(String),
    /// Union of multiple patterns (`A OR B`).
    Union(Vec<KeyPattern>),
}

/// Comparison operators used in value filters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOp {
    /// `=`, `==`, `$eq`.
    Eq,
    /// `!=`, `<>`, `$ne`.
    Ne,
    /// `>`, `$gt`.
    Gt,
    /// `>=`, `$gte`.
    Gte,
    /// `<`, `$lt`.
    Lt,
    /// `<=`, `$lte`.
    Lte,
    /// `IN`, `$in`.
    In,
    /// `NOT IN`, `$nin`.
    Nin,
    /// `LIKE`.
    Like,
    /// `REGEX`, `$regex`.
    Regex,
}

/// Boolean composition of conditions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BooleanOp {
    /// Conjunction.
    And(Vec<Condition>),
    /// Disjunction.
    Or(Vec<Condition>),
    /// Negation.
    Not(Box<Condition>),
}

/// Which field of a row a condition or ordering applies to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderField {
    /// The row key.
    Key,
    /// The stored value.
    Value,
    /// The write timestamp.
    Timestamp,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Direction {
    /// Ascending.
    Asc,
    /// Descending.
    Desc,
}

/// An ORDER BY clause.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OrderBy {
    /// Field to sort by.
    pub field: OrderField,
    /// Direction.
    pub direction: Direction,
}

/// A time range filter for WHERE clauses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TimeRange {
    /// Inclusive start, Unix microseconds.
    pub start: Option<u64>,
    /// Inclusive end, Unix microseconds.
    pub end: Option<u64>,
}

/// One or more values used on the right-hand side of a comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValueFilter {
    /// A single atom.
    Single(Atom),
    /// A list of atoms (used for `IN` / `NOT IN`).
    List(Vec<Atom>),
}

/// The WHERE clause root — a tree of conditions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WhereClause {
    /// Root condition.
    pub root: Condition,
}

/// A single predicate inside a WHERE clause.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Condition {
    /// `<field> <op> <value>`.
    Comparison {
        /// Left operand (field or timestamp reference).
        field: OrderField,
        /// Operator.
        op: ComparisonOp,
        /// Right-hand operand.
        rhs: ValueFilter,
    },
    /// Key-pattern predicate (`FROM "sensor/*"` may also appear here in OR unions).
    Key(KeyPattern),
    /// `timestamp BETWEEN a AND b`.
    TimeRange(TimeRange),
    /// `ANOMALY(value, <method>)`.
    Anomaly(AnomalyMethod),
    /// `MATCHES_PATTERN(value, '<pattern>')`.
    Pattern(TimeSeriesPattern),
    /// `FRESHNESS <op> <num>` / `STALE` / `FRESH` (DAVO).
    Freshness(FreshnessCondition),
    /// `SIMILAR_TO(<vec>, <k>)`.
    Similarity(SimilarityCondition),
    /// `AND` / `OR` / `NOT`.
    Boolean(BooleanOp),
}

// ─── WHERE-clause sub-types (placeholders for later tasks) ─────────────

/// Anomaly-detection method used inside `ANOMALY(...)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AnomalyMethod {
    /// Z-score, with threshold.
    ZScore(f64),
    /// Interquartile range, with multiplier.
    Iqr(f64),
    /// Moving-average deviation.
    MovingAverage {
        /// Rolling window.
        window: u64,
        /// Threshold in std-devs.
        threshold: f64,
    },
}

/// Named time-series shape for `MATCHES_PATTERN`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimeSeriesPattern {
    /// Sudden rise then fall.
    Spike {
        /// Minimum amplitude.
        threshold: f64,
        /// Maximum duration in microseconds.
        max_duration_micros: Option<u64>,
    },
    /// Sudden drop then recovery.
    Dip {
        /// Minimum amplitude.
        threshold: f64,
        /// Maximum duration in microseconds.
        max_duration_micros: Option<u64>,
    },
    /// Sustained rising trend.
    Rising {
        /// Minimum duration in microseconds.
        min_duration_micros: u64,
    },
    /// Sustained falling trend.
    Falling {
        /// Minimum duration in microseconds.
        min_duration_micros: u64,
    },
    /// Flat region.
    Plateau {
        /// Minimum duration in microseconds.
        min_duration_micros: u64,
        /// Maximum variance allowed.
        tolerance: f64,
    },
}

/// DAVO freshness condition (Task 30).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FreshnessCondition {
    /// `FRESHNESS <op> <value>`.
    Compare {
        /// Operator.
        op: ComparisonOp,
        /// Threshold in \[0, 1\].
        value: f64,
    },
    /// `STALE` — freshness below configured threshold.
    Stale,
    /// `FRESH` — freshness at or above configured threshold.
    Fresh,
}

/// Vector similarity condition (Task 28).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimilarityCondition {
    /// Query vector.
    pub query_vector: Vec<f32>,
    /// Number of neighbours requested.
    pub k: usize,
    /// Optional index hint (e.g., `"HNSW"`, `"GWI"`, `"CASCADE"`).
    pub index_hint: Option<String>,
}

// ─── Projection & function calls ────────────────────────────────────────

/// What to return in each result row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Projection {
    /// `*`.
    All,
    /// `key`.
    Key,
    /// `value`.
    Value,
    /// `timestamp`.
    Timestamp,
    /// A function call like `AVG(value)` or `MOVING_AVG(value, 10)`.
    Function(FunctionCall),
    /// A prediction expression like `PREDICT(value, HOLT_WINTERS(24), horizon=24)`.
    Predict(PredictExpr),
    /// An aliased projection (`<inner> AS <alias>`).
    Aliased {
        /// Inner projection.
        inner: Box<Projection>,
        /// Alias name.
        alias: String,
    },
}

/// A generic function call used as a projection (or inside HAVING).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    /// Function name (e.g., `"MOVING_AVG"`, `"RATE"`).
    pub name: String,
    /// Arguments as AST expressions (stringified for now).
    pub args: Vec<String>,
}

/// `PREDICT(...)` expression.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictExpr {
    /// Method name (e.g., `"HOLT_WINTERS"`, `"EXP_SMOOTHING"`).
    pub method: String,
    /// Forecast horizon in number of points.
    pub horizon: u64,
    /// Optional sampling interval (e.g., `"1h"`).
    pub interval: Option<String>,
    /// Additional keyword arguments.
    pub args: Vec<(String, String)>,
}

// ═══════════════════════════════════════════════════════════════════════
//  Display implementations — pretty-print AST
// ═══════════════════════════════════════════════════════════════════════

impl fmt::Display for QueryAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryAst::Select(q) => write!(f, "{}", q),
            QueryAst::Find(q) => write!(f, "FIND {:?}", q.filter.raw),
            QueryAst::Aggregate(q) => write!(f, "{}", q),
            QueryAst::TemporalJoin(q) => {
                write!(f, "{} TEMPORAL JOIN {} {}", q.left, q.right, q.join_type)
            }
            QueryAst::Stream(q) => write!(f, "CREATE STREAM {} AS {}", q.name, q.body),
            QueryAst::Explain(inner) => write!(f, "EXPLAIN {}", inner),
            QueryAst::Macro(m) => {
                write!(
                    f,
                    "CREATE MACRO {}({}) AS {}",
                    m.name,
                    m.params.join(", "),
                    m.body
                )
            }
            QueryAst::Lineage(l) => write!(f, "LINEAGE({})", l.key),
        }
    }
}

impl fmt::Display for SelectQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SELECT ")?;
        if self.projections.is_empty() {
            write!(f, "*")?;
        } else {
            for (i, p) in self.projections.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", p)?;
            }
        }
        if let Some(from) = &self.from {
            write!(f, " FROM {}", from)?;
        }
        if let Some(w) = &self.where_clause {
            write!(f, " WHERE {}", w.root)?;
        }
        if let Some(order) = &self.order_by {
            write!(f, " ORDER BY {:?} {:?}", order.field, order.direction)?;
        }
        if let Some(lim) = self.limit {
            write!(f, " LIMIT {}", lim)?;
        }
        if let Some(off) = self.offset {
            write!(f, " OFFSET {}", off)?;
        }
        Ok(())
    }
}

impl fmt::Display for AggregateQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SELECT ")?;
        for (i, a) in self.aggregations.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", a)?;
        }
        if let Some(from) = &self.from {
            write!(f, " FROM {}", from)?;
        }
        if let Some(w) = &self.where_clause {
            write!(f, " WHERE {}", w.root)?;
        }
        if let Some(g) = &self.group_by {
            write!(f, " GROUP BY {:?}", g)?;
        }
        Ok(())
    }
}

impl fmt::Display for KeyPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyPattern::Exact(s) => write!(f, "\"{}\"", s),
            KeyPattern::Prefix(s) => write!(f, "\"{}*\"", s),
            KeyPattern::Glob(s) => write!(f, "\"{}\"", s),
            KeyPattern::Regex(s) => write!(f, "REGEX(\"{}\")", s),
            KeyPattern::Union(parts) => {
                for (i, p) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " OR ")?;
                    }
                    write!(f, "{}", p)?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Condition::Comparison { field, op, rhs } => {
                write!(f, "{:?} {:?} {:?}", field, op, rhs)
            }
            Condition::Key(kp) => write!(f, "key MATCHES {}", kp),
            Condition::TimeRange(tr) => {
                write!(f, "timestamp BETWEEN {:?} AND {:?}", tr.start, tr.end)
            }
            Condition::Anomaly(m) => write!(f, "ANOMALY(value, {:?})", m),
            Condition::Pattern(p) => write!(f, "MATCHES_PATTERN(value, {:?})", p),
            Condition::Freshness(fc) => write!(f, "{:?}", fc),
            Condition::Similarity(s) => write!(f, "SIMILAR_TO(<vec>, {})", s.k),
            Condition::Boolean(BooleanOp::And(cs)) => {
                for (i, c) in cs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " AND ")?;
                    }
                    write!(f, "({})", c)?;
                }
                Ok(())
            }
            Condition::Boolean(BooleanOp::Or(cs)) => {
                for (i, c) in cs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " OR ")?;
                    }
                    write!(f, "({})", c)?;
                }
                Ok(())
            }
            Condition::Boolean(BooleanOp::Not(c)) => write!(f, "NOT ({})", c),
        }
    }
}

impl fmt::Display for Projection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Projection::All => write!(f, "*"),
            Projection::Key => write!(f, "key"),
            Projection::Value => write!(f, "value"),
            Projection::Timestamp => write!(f, "timestamp"),
            Projection::Function(fc) => write!(f, "{}({})", fc.name, fc.args.join(", ")),
            Projection::Predict(p) => write!(f, "PREDICT({}, horizon={})", p.method, p.horizon),
            Projection::Aliased { inner, alias } => write!(f, "{} AS {}", inner, alias),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Unit tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip_simple_select() {
        let ast = QueryAst::Select(SelectQuery {
            projections: vec![Projection::All],
            from: Some(KeyPattern::Glob("sensor/*".into())),
            where_clause: None,
            order_by: Some(OrderBy {
                field: OrderField::Timestamp,
                direction: Direction::Asc,
            }),
            limit: Some(100),
            offset: None,
        });

        let bytes = bincode::serialize(&ast).unwrap();
        let decoded: QueryAst = bincode::deserialize(&bytes).unwrap();
        assert_eq!(ast, decoded);
    }

    #[test]
    fn display_pretty_prints_select() {
        let ast = QueryAst::Select(SelectQuery {
            projections: vec![Projection::Key, Projection::Value],
            from: Some(KeyPattern::Prefix("sensor/".into())),
            where_clause: None,
            order_by: None,
            limit: Some(10),
            offset: None,
        });
        let s = format!("{}", ast);
        assert!(s.contains("SELECT"));
        assert!(s.contains("FROM"));
        assert!(s.contains("LIMIT 10"));
    }
}
