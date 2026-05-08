//! EMQ Parser — MongoDB-like JSON query syntax for SynaDB.
//!
//! Parses JSON documents like:
//! ```json
//! {
//!   "filter": { "value": { "$gt": 10 } },
//!   "sort": { "timestamp": -1 },
//!   "limit": 100
//! }
//! ```

use crate::query::ast::*;
use crate::query::error::ParseError;
use crate::types::Atom;
use serde_json::Value;

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Parse an EMQ JSON document into a [`QueryAst`].
pub fn parse_emq(input: &str) -> Result<QueryAst, ParseError> {
    let doc: Value = serde_json::from_str(input).map_err(|e| ParseError {
        message: format!("invalid JSON: {}", e),
        line: 1,
        column: 1,
        snippet: input[..input.len().min(40)].to_string(),
    })?;

    let obj = doc.as_object().ok_or_else(|| ParseError {
        message: "EMQ document must be a JSON object".into(),
        line: 1,
        column: 1,
        snippet: input[..input.len().min(40)].to_string(),
    })?;

    // Extract components
    let filter = obj
        .get("filter")
        .cloned()
        .unwrap_or(Value::Object(Default::default()));
    let projection = obj.get("projection").cloned();
    let sort = obj.get("sort").cloned();
    let limit = obj.get("limit").and_then(|v| v.as_u64());
    let skip = obj.get("skip").and_then(|v| v.as_u64());
    let from = obj
        .get("from")
        .and_then(|v| v.as_str())
        .map(parse_key_pattern);

    // Check if this is an aggregate query
    if let Some(pipeline) = obj.get("aggregate") {
        return parse_aggregate_pipeline(pipeline, from);
    }

    // Build FindQuery
    let where_clause = parse_filter_to_condition(&filter)?;

    // If there's a "from" field, build a SelectQuery instead (more natural for EQL interop)
    if from.is_some() || sort.is_some() {
        return Ok(QueryAst::Select(SelectQuery {
            projections: vec![Projection::All],
            from,
            where_clause: where_clause.map(|root| WhereClause { root }),
            order_by: sort.as_ref().and_then(parse_sort),
            limit,
            offset: skip,
        }));
    }

    Ok(QueryAst::Find(FindQuery {
        filter: FilterDocument { raw: filter },
        projection: projection.map(|v| ProjectionDocument { raw: v }),
        sort: sort.map(|v| SortDocument { raw: v }),
        limit,
        skip,
    }))
}

// ═══════════════════════════════════════════════════════════════════════
//  Filter parsing
// ═══════════════════════════════════════════════════════════════════════

fn parse_filter_to_condition(filter: &Value) -> Result<Option<Condition>, ParseError> {
    let obj = match filter.as_object() {
        Some(o) if o.is_empty() => return Ok(None),
        Some(o) => o,
        None => return Ok(None),
    };

    let mut conditions = Vec::new();

    for (key, value) in obj {
        match key.as_str() {
            "$and" => {
                if let Some(arr) = value.as_array() {
                    let subs: Result<Vec<_>, _> = arr
                        .iter()
                        .filter_map(|v| parse_filter_to_condition(v).transpose())
                        .collect();
                    conditions.push(Condition::Boolean(BooleanOp::And(subs?)));
                }
            }
            "$or" => {
                if let Some(arr) = value.as_array() {
                    let subs: Result<Vec<_>, _> = arr
                        .iter()
                        .filter_map(|v| parse_filter_to_condition(v).transpose())
                        .collect();
                    conditions.push(Condition::Boolean(BooleanOp::Or(subs?)));
                }
            }
            "$not" => {
                if let Some(inner) = parse_filter_to_condition(value)? {
                    conditions.push(Condition::Boolean(BooleanOp::Not(Box::new(inner))));
                }
            }
            field => {
                // Field-level filter: { "value": { "$gt": 10 } }
                let field_ref = match field {
                    "key" => OrderField::Key,
                    "timestamp" => OrderField::Timestamp,
                    _ => OrderField::Value,
                };

                if let Some(ops) = value.as_object() {
                    for (op_key, op_val) in ops {
                        let (op, rhs) = parse_operator(op_key, op_val)?;
                        conditions.push(Condition::Comparison {
                            field: field_ref.clone(),
                            op,
                            rhs,
                        });
                    }
                } else {
                    // Shorthand: { "value": 10 } means { "value": { "$eq": 10 } }
                    conditions.push(Condition::Comparison {
                        field: field_ref,
                        op: ComparisonOp::Eq,
                        rhs: ValueFilter::Single(json_to_atom(value)),
                    });
                }
            }
        }
    }

    match conditions.len() {
        0 => Ok(None),
        1 => Ok(Some(conditions.remove(0))),
        _ => Ok(Some(Condition::Boolean(BooleanOp::And(conditions)))),
    }
}

fn parse_operator(op_key: &str, op_val: &Value) -> Result<(ComparisonOp, ValueFilter), ParseError> {
    let op = match op_key {
        "$eq" => ComparisonOp::Eq,
        "$ne" => ComparisonOp::Ne,
        "$gt" => ComparisonOp::Gt,
        "$gte" => ComparisonOp::Gte,
        "$lt" => ComparisonOp::Lt,
        "$lte" => ComparisonOp::Lte,
        "$in" => ComparisonOp::In,
        "$nin" => ComparisonOp::Nin,
        "$regex" => ComparisonOp::Regex,
        other => {
            return Err(ParseError {
                message: format!("unknown operator: {}", other),
                line: 1,
                column: 1,
                snippet: String::new(),
            });
        }
    };

    let empty_arr = vec![];
    let rhs = match op {
        ComparisonOp::In | ComparisonOp::Nin => {
            let arr = op_val.as_array().unwrap_or(&empty_arr);
            ValueFilter::List(arr.iter().map(json_to_atom).collect())
        }
        _ => ValueFilter::Single(json_to_atom(op_val)),
    };

    Ok((op, rhs))
}

// ═══════════════════════════════════════════════════════════════════════
//  Sort parsing
// ═══════════════════════════════════════════════════════════════════════

fn parse_sort(sort: &Value) -> Option<OrderBy> {
    let obj = sort.as_object()?;
    let (field_name, direction_val) = obj.iter().next()?;

    let field = match field_name.as_str() {
        "key" => OrderField::Key,
        "timestamp" => OrderField::Timestamp,
        _ => OrderField::Value,
    };

    let direction = match direction_val.as_i64() {
        Some(1) => Direction::Asc,
        Some(-1) => Direction::Desc,
        _ => Direction::Asc,
    };

    Some(OrderBy { field, direction })
}

// ═══════════════════════════════════════════════════════════════════════
//  Aggregate pipeline
// ═══════════════════════════════════════════════════════════════════════

fn parse_aggregate_pipeline(
    pipeline: &Value,
    from: Option<KeyPattern>,
) -> Result<QueryAst, ParseError> {
    let stages = pipeline.as_array().ok_or_else(|| ParseError {
        message: "aggregate must be an array of stages".into(),
        line: 1,
        column: 1,
        snippet: String::new(),
    })?;

    let mut aggregations = Vec::new();
    let mut group_by = None;
    let mut where_clause = None;

    let empty_map = serde_json::Map::new();
    for stage in stages {
        let obj = stage.as_object().unwrap_or(&empty_map);

        if let Some(match_doc) = obj.get("$match") {
            where_clause = parse_filter_to_condition(match_doc)?;
        }

        if let Some(group_doc) = obj.get("$group") {
            if let Some(group_obj) = group_doc.as_object() {
                // Parse _id for group key
                if let Some(id_val) = group_obj.get("_id") {
                    if id_val.as_str() == Some("$key") {
                        group_by = Some(GroupBy::Key);
                    }
                }
                // Parse accumulators
                for (_, acc_val) in group_obj.iter().filter(|(k, _)| *k != "_id") {
                    if let Some(acc_obj) = acc_val.as_object() {
                        for (acc_op, _) in acc_obj {
                            let func = match acc_op.as_str() {
                                "$sum" => AggregateFunction::Sum,
                                "$avg" => AggregateFunction::Avg,
                                "$min" => AggregateFunction::Min,
                                "$max" => AggregateFunction::Max,
                                "$count" => AggregateFunction::Count,
                                "$first" => AggregateFunction::First,
                                "$last" => AggregateFunction::Last,
                                _ => continue,
                            };
                            aggregations.push(func);
                        }
                    }
                }
            }
        }
    }

    if aggregations.is_empty() {
        aggregations.push(AggregateFunction::Count);
    }

    Ok(QueryAst::Aggregate(AggregateQuery {
        aggregations,
        from,
        where_clause: where_clause.map(|root| WhereClause { root }),
        group_by,
        having: None,
        order_by: None,
        limit: None,
    }))
}

// ═══════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════

fn json_to_atom(value: &Value) -> Atom {
    match value {
        Value::Null => Atom::Null,
        Value::Bool(b) => Atom::Int(if *b { 1 } else { 0 }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Atom::Int(i)
            } else if let Some(f) = n.as_f64() {
                Atom::Float(f)
            } else {
                Atom::Null
            }
        }
        Value::String(s) => Atom::Text(s.clone()),
        _ => Atom::Text(value.to_string()),
    }
}

fn parse_key_pattern(s: &str) -> KeyPattern {
    if s.contains('*') || s.contains('?') {
        if s.ends_with("/*") && !s[..s.len() - 2].contains('*') {
            KeyPattern::Prefix(s[..s.len() - 1].to_string())
        } else {
            KeyPattern::Glob(s.to_string())
        }
    } else if s.starts_with('^') || s.contains('[') {
        KeyPattern::Regex(s.to_string())
    } else {
        KeyPattern::Exact(s.to_string())
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_find() {
        let input = r#"{ "filter": { "value": { "$gt": 10 } } }"#;
        let ast = parse_emq(input).unwrap();
        match ast {
            QueryAst::Find(q) => {
                assert!(q.filter.raw.is_object());
            }
            _ => panic!("expected Find"),
        }
    }

    #[test]
    fn parse_find_with_sort_and_limit() {
        let input = r#"{
            "from": "sensor/*",
            "filter": { "value": { "$gte": 20 } },
            "sort": { "timestamp": -1 },
            "limit": 50
        }"#;
        let ast = parse_emq(input).unwrap();
        match ast {
            QueryAst::Select(q) => {
                assert_eq!(q.from, Some(KeyPattern::Prefix("sensor/".into())));
                assert_eq!(q.limit, Some(50));
                let ob = q.order_by.unwrap();
                assert_eq!(ob.direction, Direction::Desc);
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_eq_shorthand() {
        let input = r#"{ "filter": { "value": 42 } }"#;
        let ast = parse_emq(input).unwrap();
        match ast {
            QueryAst::Find(_) => {} // shorthand parsed
            _ => panic!("expected Find"),
        }
    }

    #[test]
    fn parse_boolean_operators() {
        let input = r#"{
            "from": "k",
            "filter": {
                "$or": [
                    { "value": { "$lt": 0 } },
                    { "value": { "$gt": 100 } }
                ]
            }
        }"#;
        let ast = parse_emq(input).unwrap();
        match ast {
            QueryAst::Select(q) => {
                let cond = q.where_clause.unwrap().root;
                match cond {
                    Condition::Boolean(BooleanOp::Or(cs)) => assert_eq!(cs.len(), 2),
                    other => panic!("expected Or, got {:?}", other),
                }
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_in_operator() {
        let input = r#"{ "from": "k", "filter": { "value": { "$in": [1, 2, 3] } } }"#;
        let ast = parse_emq(input).unwrap();
        match ast {
            QueryAst::Select(q) => {
                let cond = q.where_clause.unwrap().root;
                match cond {
                    Condition::Comparison { op, rhs, .. } => {
                        assert_eq!(op, ComparisonOp::In);
                        match rhs {
                            ValueFilter::List(items) => assert_eq!(items.len(), 3),
                            _ => panic!("expected List"),
                        }
                    }
                    _ => panic!("expected Comparison"),
                }
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_aggregate_pipeline() {
        let input = r#"{
            "from": "sensor/*",
            "aggregate": [
                { "$match": { "value": { "$gt": 0 } } },
                { "$group": { "_id": "$key", "avg_val": { "$avg": "$value" } } }
            ]
        }"#;
        let ast = parse_emq(input).unwrap();
        match ast {
            QueryAst::Aggregate(q) => {
                assert_eq!(q.aggregations, vec![AggregateFunction::Avg]);
                assert_eq!(q.group_by, Some(GroupBy::Key));
                assert!(q.where_clause.is_some());
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn parse_error_invalid_json() {
        let err = parse_emq("not json").unwrap_err();
        assert!(err.message.contains("invalid JSON"));
    }

    #[test]
    fn parse_error_unknown_operator() {
        let input = r#"{ "from": "k", "filter": { "value": { "$unknown": 1 } } }"#;
        let err = parse_emq(input).unwrap_err();
        assert!(err.message.contains("unknown operator"));
    }
}
