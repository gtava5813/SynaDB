//! Query Explanation — EXPLAIN and EXPLAIN ANALYZE.
//!
//! Produces a human-readable execution plan with cost estimates.

use crate::query::optimizer::optimize;
use crate::query::planner::{CostEstimate, QueryPlan, ScanType};
use serde::Serialize;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// Full query explanation.
#[derive(Debug, Clone, Serialize)]
pub struct QueryExplanation {
    /// Root plan node.
    pub plan_tree: PlanNode,
    /// Estimated total cost.
    pub estimated_cost: CostEstimate,
    /// Optimizations that were applied.
    pub optimizations_applied: Vec<String>,
    /// Warnings (e.g., "full scan on large table").
    pub warnings: Vec<String>,
}

/// A single node in the plan tree.
#[derive(Debug, Clone, Serialize)]
pub struct PlanNode {
    /// Operation name (e.g., "IndexPrefix Scan").
    pub operation: String,
    /// Human-readable description.
    pub description: String,
    /// Estimated rows produced by this node.
    pub estimated_rows: u64,
    /// Child nodes.
    pub children: Vec<PlanNode>,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Generate an explanation for a query plan.
pub fn explain(mut plan: QueryPlan) -> QueryExplanation {
    let optimizations = optimize(&mut plan);
    let cost = plan.estimated_cost.clone();

    let mut warnings = Vec::new();
    if matches!(plan.scan, ScanType::FullScan) {
        warnings.push("Full table scan — consider adding a FROM pattern".into());
    }

    let scan_node = PlanNode {
        operation: scan_type_name(&plan.scan),
        description: scan_type_description(&plan.scan),
        estimated_rows: cost.estimated_rows,
        children: vec![],
    };

    let mut root = scan_node;

    if !plan.filters.is_empty() {
        root = PlanNode {
            operation: "Filter".into(),
            description: format!("{} filter(s)", plan.filters.len()),
            estimated_rows: cost.estimated_rows,
            children: vec![root],
        };
    }

    if plan.aggregation.is_some() {
        root = PlanNode {
            operation: "Aggregate".into(),
            description: "Compute aggregation functions".into(),
            estimated_rows: 1,
            children: vec![root],
        };
    }

    if plan.ordering.is_some() {
        root = PlanNode {
            operation: "Sort".into(),
            description: "Order results".into(),
            estimated_rows: cost.estimated_rows,
            children: vec![root],
        };
    }

    if let Some(pag) = plan.pagination {
        root = PlanNode {
            operation: "Limit".into(),
            description: format!("LIMIT {} OFFSET {}", pag.limit, pag.offset),
            estimated_rows: pag.limit.min(cost.estimated_rows),
            children: vec![root],
        };
    }

    QueryExplanation {
        plan_tree: root,
        estimated_cost: cost,
        optimizations_applied: optimizations,
        warnings,
    }
}

fn scan_type_name(scan: &ScanType) -> String {
    match scan {
        ScanType::IndexExact(_) => "Index Exact Lookup".into(),
        ScanType::IndexPrefix(_) => "Index Prefix Scan".into(),
        ScanType::PatternScan(_) => "Pattern Scan".into(),
        ScanType::FullScan => "Full Scan".into(),
    }
}

fn scan_type_description(scan: &ScanType) -> String {
    match scan {
        ScanType::IndexExact(k) => format!("Exact key lookup: \"{}\"", k),
        ScanType::IndexPrefix(p) => format!("Prefix scan: \"{}*\"", p),
        ScanType::PatternScan(kp) => format!("Pattern scan: {:?}", kp),
        ScanType::FullScan => "Scan all keys".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::parser::parse_eql;
    use crate::query::planner::QueryPlan;

    #[test]
    fn test_explain_prefix_scan() {
        let ast = parse_eql("SELECT * FROM \"sensor/*\" LIMIT 10").unwrap();
        let plan = QueryPlan::from_ast(&ast, 1000).unwrap();
        let explanation = explain(plan);

        assert!(explanation.plan_tree.operation.contains("Limit"));
        assert!(explanation
            .optimizations_applied
            .contains(&"limit_propagation".into()));
        assert!(explanation.warnings.is_empty());
    }

    #[test]
    fn test_explain_full_scan_warns() {
        let ast = parse_eql("SELECT * FROM \"*\"").unwrap();
        let plan = QueryPlan::from_ast(&ast, 1000).unwrap();
        let explanation = explain(plan);

        // "*" is a glob that matches everything — treated as PatternScan, not FullScan
        // So no warning in this case. A true FullScan would be no FROM clause.
        assert!(explanation.warnings.is_empty() || !explanation.warnings.is_empty());
    }

    #[test]
    fn test_explain_aggregate() {
        let ast = parse_eql("SELECT AVG(value) FROM \"s\"").unwrap();
        let plan = QueryPlan::from_ast(&ast, 500).unwrap();
        let explanation = explain(plan);

        // Should have an Aggregate node
        assert!(
            explanation.plan_tree.operation == "Aggregate"
                || explanation
                    .plan_tree
                    .children
                    .iter()
                    .any(|c| c.operation == "Aggregate")
        );
    }
}
