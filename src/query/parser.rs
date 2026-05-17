//! EQL Parser — SQL-like query syntax for SynaDB.
//!
//! Parses strings like:
//! ```text
//! SELECT * FROM "sensor/*" WHERE value > 10 ORDER BY timestamp DESC LIMIT 100
//! SELECT AVG(value) FROM "sensor/*" GROUP BY HOUR
//! ```
//!
//! Uses the `nom` combinator library for zero-copy parsing.

use crate::query::ast::*;
use crate::query::error::ParseError;
use crate::types::Atom;
use nom::{
    branch::alt,
    bytes::complete::{tag, tag_no_case, take_while1},
    character::complete::{char, multispace0, multispace1},
    combinator::{map, opt, value},
    multi::separated_list1,
    number::complete::double,
    sequence::{delimited, preceded, tuple},
    IResult,
};

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Parse an EQL query string into a [`QueryAst`].
///
/// Supports SELECT, aggregate SELECT with GROUP BY, and EXPLAIN.
pub fn parse_eql(input: &str) -> Result<QueryAst, ParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParseError {
            message: "empty query".into(),
            line: 1,
            column: 1,
            snippet: String::new(),
        });
    }

    // Try EXPLAIN wrapper
    if trimmed.len() > 7
        && trimmed[..7].eq_ignore_ascii_case("EXPLAIN")
        && trimmed
            .as_bytes()
            .get(7)
            .is_some_and(|b| b.is_ascii_whitespace())
    {
        let inner = &trimmed[7..].trim_start();
        let inner_ast = parse_eql(inner)?;
        return Ok(QueryAst::Explain(Box::new(inner_ast)));
    }

    match parse_statement(trimmed) {
        Ok(("", ast)) => Ok(ast),
        Ok((rest, _)) => Err(make_error(
            input,
            rest,
            format!("unexpected trailing input: '{}'", truncate(rest, 30)),
        )),
        Err(nom::Err::Error(e) | nom::Err::Failure(e)) => Err(make_error(
            input,
            e.input,
            format!("parse error: {:?}", e.code),
        )),
        Err(nom::Err::Incomplete(_)) => Err(ParseError {
            message: "incomplete input".into(),
            line: 1,
            column: input.len(),
            snippet: truncate(input, 40).to_string(),
        }),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Top-level statement
// ═══════════════════════════════════════════════════════════════════════

fn parse_statement(input: &str) -> IResult<&str, QueryAst> {
    alt((
        map(parse_aggregate_select, QueryAst::Aggregate),
        map(parse_select, QueryAst::Select),
    ))(input)
}

// ═══════════════════════════════════════════════════════════════════════
//  SELECT
// ═══════════════════════════════════════════════════════════════════════

fn parse_select(input: &str) -> IResult<&str, SelectQuery> {
    let (input, _) = tag_no_case("SELECT")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, projections) = parse_projections(input)?;
    let (input, from) = opt(parse_from_clause)(input)?;
    let (input, where_clause) = opt(parse_where_clause)(input)?;
    let (input, order_by) = opt(parse_order_by)(input)?;
    let (input, limit) = opt(parse_limit)(input)?;
    let (input, offset) = opt(parse_offset)(input)?;
    let (input, _) = multispace0(input)?;

    Ok((
        input,
        SelectQuery {
            projections,
            from,
            where_clause,
            order_by,
            limit,
            offset,
        },
    ))
}

// ═══════════════════════════════════════════════════════════════════════
//  Aggregate SELECT (detects aggregate functions in projections)
// ═══════════════════════════════════════════════════════════════════════

fn parse_aggregate_select(input: &str) -> IResult<&str, AggregateQuery> {
    let (input, _) = tag_no_case("SELECT")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, aggregations) = parse_aggregate_functions(input)?;
    let (input, from) = opt(parse_from_clause)(input)?;
    let (input, where_clause) = opt(parse_where_clause)(input)?;
    let (input, group_by) = opt(parse_group_by)(input)?;
    let (input, order_by) = opt(parse_order_by)(input)?;
    let (input, limit) = opt(parse_limit)(input)?;
    let (input, _) = multispace0(input)?;

    Ok((
        input,
        AggregateQuery {
            aggregations,
            from,
            where_clause,
            group_by,
            having: None,
            order_by,
            limit,
        },
    ))
}

fn parse_aggregate_functions(input: &str) -> IResult<&str, Vec<AggregateFunction>> {
    separated_list1(
        tuple((multispace0, char(','), multispace0)),
        parse_single_aggregate,
    )(input)
}

fn parse_single_aggregate(input: &str) -> IResult<&str, AggregateFunction> {
    let (input, func) = alt((
        value(AggregateFunction::Count, tag_no_case("COUNT")),
        value(AggregateFunction::Sum, tag_no_case("SUM")),
        value(AggregateFunction::Avg, tag_no_case("AVG")),
        value(AggregateFunction::Min, tag_no_case("MIN")),
        value(AggregateFunction::Max, tag_no_case("MAX")),
        value(AggregateFunction::First, tag_no_case("FIRST")),
        value(AggregateFunction::Last, tag_no_case("LAST")),
    ))(input)?;
    // Consume optional parenthesized argument like COUNT(*) or SUM(value)
    let (input, _) = opt(delimited(
        tuple((multispace0, char('('))),
        take_while1(|c| c != ')'),
        char(')'),
    ))(input)?;
    Ok((input, func))
}

// ═══════════════════════════════════════════════════════════════════════
//  Projections
// ═══════════════════════════════════════════════════════════════════════

fn parse_projections(input: &str) -> IResult<&str, Vec<Projection>> {
    separated_list1(
        tuple((multispace0, char(','), multispace0)),
        parse_single_projection,
    )(input)
}

fn parse_single_projection(input: &str) -> IResult<&str, Projection> {
    alt((
        value(Projection::All, char('*')),
        value(Projection::Timestamp, tag_no_case("timestamp")),
        value(Projection::Key, tag_no_case("key")),
        value(Projection::Value, tag_no_case("value")),
    ))(input)
}

// ═══════════════════════════════════════════════════════════════════════
//  FROM clause
// ═══════════════════════════════════════════════════════════════════════

fn parse_from_clause(input: &str) -> IResult<&str, KeyPattern> {
    preceded(
        tuple((multispace1, tag_no_case("FROM"), multispace1)),
        parse_key_pattern,
    )(input)
}

fn parse_key_pattern(input: &str) -> IResult<&str, KeyPattern> {
    // Quoted string: "sensor/*" or 'sensor/*'
    let (input, pattern_str) = alt((
        delimited(char('"'), take_while1(|c| c != '"'), char('"')),
        delimited(char('\''), take_while1(|c| c != '\''), char('\'')),
    ))(input)?;

    // Classify the pattern
    let kp = if pattern_str.contains('*') || pattern_str.contains('?') {
        // Check if it's a simple prefix glob like "sensor/*"
        if pattern_str.ends_with("/*") && !pattern_str[..pattern_str.len() - 2].contains('*') {
            KeyPattern::Prefix(pattern_str[..pattern_str.len() - 1].to_string())
        } else {
            KeyPattern::Glob(pattern_str.to_string())
        }
    } else if pattern_str.starts_with('^') || pattern_str.contains('[') {
        KeyPattern::Regex(pattern_str.to_string())
    } else {
        KeyPattern::Exact(pattern_str.to_string())
    };

    Ok((input, kp))
}

// ═══════════════════════════════════════════════════════════════════════
//  WHERE clause
// ═══════════════════════════════════════════════════════════════════════

fn parse_where_clause(input: &str) -> IResult<&str, WhereClause> {
    preceded(
        tuple((multispace1, tag_no_case("WHERE"), multispace1)),
        map(parse_condition, |root| WhereClause { root }),
    )(input)
}

fn parse_condition(input: &str) -> IResult<&str, Condition> {
    // Try OR-separated conditions (lowest precedence)
    let (input, first) = parse_and_condition(input)?;
    let (input, rest) = nom::multi::many0(preceded(
        tuple((multispace1, tag_no_case("OR"), multispace1)),
        parse_and_condition,
    ))(input)?;

    if rest.is_empty() {
        Ok((input, first))
    } else {
        let mut all = vec![first];
        all.extend(rest);
        Ok((input, Condition::Boolean(BooleanOp::Or(all))))
    }
}

fn parse_and_condition(input: &str) -> IResult<&str, Condition> {
    let (input, first) = parse_unary_condition(input)?;
    let (input, rest) = nom::multi::many0(preceded(
        tuple((multispace1, tag_no_case("AND"), multispace1)),
        parse_unary_condition,
    ))(input)?;

    if rest.is_empty() {
        Ok((input, first))
    } else {
        let mut all = vec![first];
        all.extend(rest);
        Ok((input, Condition::Boolean(BooleanOp::And(all))))
    }
}

fn parse_unary_condition(input: &str) -> IResult<&str, Condition> {
    alt((
        // NOT <condition>
        map(
            preceded(
                tuple((tag_no_case("NOT"), multispace1)),
                parse_unary_condition,
            ),
            |c| Condition::Boolean(BooleanOp::Not(Box::new(c))),
        ),
        // Parenthesized group
        delimited(
            tuple((char('('), multispace0)),
            parse_condition,
            tuple((multispace0, char(')'))),
        ),
        // FRESHNESS <op> <value> (must come before FRESH/STALE to avoid prefix match)
        parse_freshness_comparison,
        // FRESH / STALE keywords
        value(
            Condition::Freshness(FreshnessCondition::Fresh),
            tag_no_case("FRESH"),
        ),
        value(
            Condition::Freshness(FreshnessCondition::Stale),
            tag_no_case("STALE"),
        ),
        // ANOMALY(value, ZSCORE(3.0))
        parse_anomaly_condition,
        // BETWEEN
        parse_between_condition,
        // Simple comparison: field op value
        parse_comparison_condition,
    ))(input)
}

fn parse_anomaly_condition(input: &str) -> IResult<&str, Condition> {
    let (input, _) = tag_no_case("ANOMALY")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('(')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag_no_case("value")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(',')(input)?;
    let (input, _) = multispace0(input)?;
    // Parse method: ZSCORE(3.0) or IQR(1.5)
    let (input, method) = alt((
        map(
            preceded(
                tuple((tag_no_case("ZSCORE"), multispace0, char('('))),
                nom::number::complete::double,
            ),
            AnomalyMethod::ZScore,
        ),
        map(
            preceded(
                tuple((tag_no_case("IQR"), multispace0, char('('))),
                nom::number::complete::double,
            ),
            AnomalyMethod::Iqr,
        ),
    ))(input)?;
    let (input, _) = char(')')(input)?; // close method parens
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?; // close ANOMALY parens

    Ok((input, Condition::Anomaly(method)))
}

fn parse_freshness_comparison(input: &str) -> IResult<&str, Condition> {
    let (input, _) = tag_no_case("FRESHNESS")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, op) = parse_comparison_op(input)?;
    let (input, _) = multispace0(input)?;
    let (input, threshold) = nom::number::complete::double(input)?;

    Ok((
        input,
        Condition::Freshness(FreshnessCondition::Compare {
            op,
            value: threshold,
        }),
    ))
}

fn parse_between_condition(input: &str) -> IResult<&str, Condition> {
    let (input, _) = tag_no_case("timestamp")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("BETWEEN")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, start) = parse_integer(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("AND")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, end) = parse_integer(input)?;

    Ok((
        input,
        Condition::TimeRange(TimeRange {
            start: Some(start as u64),
            end: Some(end as u64),
        }),
    ))
}

fn parse_comparison_condition(input: &str) -> IResult<&str, Condition> {
    let (input, field) = parse_field_ref(input)?;
    let (input, _) = multispace0(input)?;
    let (input, op) = parse_comparison_op(input)?;
    let (input, _) = multispace0(input)?;
    let (input, rhs) = parse_value_expr(input)?;

    Ok((input, Condition::Comparison { field, op, rhs }))
}

fn parse_field_ref(input: &str) -> IResult<&str, OrderField> {
    alt((
        value(OrderField::Timestamp, tag_no_case("timestamp")),
        value(OrderField::Key, tag_no_case("key")),
        value(OrderField::Value, tag_no_case("value")),
    ))(input)
}

fn parse_comparison_op(input: &str) -> IResult<&str, ComparisonOp> {
    alt((
        value(ComparisonOp::Gte, tag(">=")),
        value(ComparisonOp::Lte, tag("<=")),
        value(ComparisonOp::Ne, tag("!=")),
        value(ComparisonOp::Ne, tag("<>")),
        value(ComparisonOp::Eq, tag("==")),
        value(ComparisonOp::Eq, tag("=")),
        value(ComparisonOp::Gt, tag(">")),
        value(ComparisonOp::Lt, tag("<")),
        value(ComparisonOp::Nin, tag_no_case("NOT IN")),
        value(ComparisonOp::In, tag_no_case("IN")),
        value(ComparisonOp::Like, tag_no_case("LIKE")),
    ))(input)
}

fn parse_value_expr(input: &str) -> IResult<&str, ValueFilter> {
    alt((
        // Quoted string
        map(
            alt((
                delimited(char('"'), take_while1(|c| c != '"'), char('"')),
                delimited(char('\''), take_while1(|c| c != '\''), char('\'')),
            )),
            |s: &str| ValueFilter::Single(Atom::Text(s.to_string())),
        ),
        // Number (try float first)
        map(double, |v| ValueFilter::Single(Atom::Float(v))),
        // NULL
        value(ValueFilter::Single(Atom::Null), tag_no_case("NULL")),
    ))(input)
}

// ═══════════════════════════════════════════════════════════════════════
//  ORDER BY
// ═══════════════════════════════════════════════════════════════════════

fn parse_order_by(input: &str) -> IResult<&str, OrderBy> {
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("ORDER")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag_no_case("BY")(input)?;
    let (input, _) = multispace1(input)?;
    let (input, field) = parse_field_ref(input)?;
    let (input, direction) = opt(preceded(
        multispace1,
        alt((
            value(Direction::Asc, tag_no_case("ASC")),
            value(Direction::Desc, tag_no_case("DESC")),
        )),
    ))(input)?;

    Ok((
        input,
        OrderBy {
            field,
            direction: direction.unwrap_or(Direction::Asc),
        },
    ))
}

// ═══════════════════════════════════════════════════════════════════════
//  LIMIT / OFFSET
// ═══════════════════════════════════════════════════════════════════════

fn parse_limit(input: &str) -> IResult<&str, u64> {
    preceded(
        tuple((multispace1, tag_no_case("LIMIT"), multispace1)),
        map(parse_integer, |v| v as u64),
    )(input)
}

fn parse_offset(input: &str) -> IResult<&str, u64> {
    preceded(
        tuple((multispace1, tag_no_case("OFFSET"), multispace1)),
        map(parse_integer, |v| v as u64),
    )(input)
}

// ═══════════════════════════════════════════════════════════════════════
//  GROUP BY
// ═══════════════════════════════════════════════════════════════════════

fn parse_group_by(input: &str) -> IResult<&str, GroupBy> {
    preceded(
        tuple((
            multispace1,
            tag_no_case("GROUP"),
            multispace1,
            tag_no_case("BY"),
            multispace1,
        )),
        alt((
            value(GroupBy::Key, tag_no_case("key")),
            value(
                GroupBy::TimeBucket(TimeBucket::Minute),
                tag_no_case("MINUTE"),
            ),
            value(GroupBy::TimeBucket(TimeBucket::Hour), tag_no_case("HOUR")),
            value(GroupBy::TimeBucket(TimeBucket::Day), tag_no_case("DAY")),
            value(GroupBy::TimeBucket(TimeBucket::Week), tag_no_case("WEEK")),
            value(GroupBy::TimeBucket(TimeBucket::Month), tag_no_case("MONTH")),
        )),
    )(input)
}

// ═══════════════════════════════════════════════════════════════════════
//  Primitives
// ═══════════════════════════════════════════════════════════════════════

fn parse_integer(input: &str) -> IResult<&str, i64> {
    let (input, neg) = opt(char('-'))(input)?;
    let (input, digits) = take_while1(|c: char| c.is_ascii_digit())(input)?;
    let val: i64 = digits.parse().unwrap_or(0);
    Ok((input, if neg.is_some() { -val } else { val }))
}

// ═══════════════════════════════════════════════════════════════════════
//  Error helpers
// ═══════════════════════════════════════════════════════════════════════

fn make_error(full_input: &str, remaining: &str, message: String) -> ParseError {
    let consumed = full_input.len() - remaining.len();
    let (line, column) = line_col(full_input, consumed);
    ParseError {
        message,
        line,
        column,
        snippet: truncate(&full_input[consumed.saturating_sub(10)..], 40).to_string(),
    }
}

fn line_col(input: &str, offset: usize) -> (usize, usize) {
    let prefix = &input[..offset.min(input.len())];
    let line = prefix.chars().filter(|c| *c == '\n').count() + 1;
    let last_newline = prefix.rfind('\n').map_or(0, |p| p + 1);
    let column = offset - last_newline + 1;
    (line, column)
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Unit tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_select_star() {
        let ast = parse_eql("SELECT * FROM \"sensor/*\"").unwrap();
        match ast {
            QueryAst::Select(q) => {
                assert_eq!(q.projections, vec![Projection::All]);
                assert_eq!(q.from, Some(KeyPattern::Prefix("sensor/".into())));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_with_where_and_limit() {
        let ast = parse_eql("SELECT key, value FROM \"data\" WHERE value > 10 LIMIT 50").unwrap();
        match ast {
            QueryAst::Select(q) => {
                assert_eq!(q.projections.len(), 2);
                assert_eq!(q.from, Some(KeyPattern::Exact("data".into())));
                assert!(q.where_clause.is_some());
                assert_eq!(q.limit, Some(50));
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_select_order_by_desc() {
        let ast = parse_eql("SELECT * FROM \"k\" ORDER BY timestamp DESC").unwrap();
        match ast {
            QueryAst::Select(q) => {
                let ob = q.order_by.unwrap();
                assert_eq!(ob.field, OrderField::Timestamp);
                assert_eq!(ob.direction, Direction::Desc);
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_aggregate_avg_group_by_hour() {
        let ast = parse_eql("SELECT AVG(value) FROM \"sensor/*\" GROUP BY HOUR").unwrap();
        match ast {
            QueryAst::Aggregate(q) => {
                assert_eq!(q.aggregations, vec![AggregateFunction::Avg]);
                assert_eq!(q.group_by, Some(GroupBy::TimeBucket(TimeBucket::Hour)));
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn parse_between_condition() {
        let ast = parse_eql("SELECT * FROM \"k\" WHERE timestamp BETWEEN 1000 AND 2000").unwrap();
        match ast {
            QueryAst::Select(q) => {
                let cond = q.where_clause.unwrap().root;
                match cond {
                    Condition::TimeRange(tr) => {
                        assert_eq!(tr.start, Some(1000));
                        assert_eq!(tr.end, Some(2000));
                    }
                    _ => panic!("expected TimeRange, got {:?}", cond),
                }
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_explain_wraps_inner() {
        let ast = parse_eql("EXPLAIN SELECT * FROM \"k\"").unwrap();
        match ast {
            QueryAst::Explain(inner) => match *inner {
                QueryAst::Select(_) => {}
                _ => panic!("expected Select inside Explain"),
            },
            _ => panic!("expected Explain"),
        }
    }

    #[test]
    fn parse_boolean_and_or() {
        let ast = parse_eql("SELECT * FROM \"k\" WHERE value > 1 AND value < 100").unwrap();
        match ast {
            QueryAst::Select(q) => match q.where_clause.unwrap().root {
                Condition::Boolean(BooleanOp::And(cs)) => assert_eq!(cs.len(), 2),
                other => panic!("expected AND, got {:?}", other),
            },
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_error_on_empty() {
        let err = parse_eql("").unwrap_err();
        assert_eq!(err.line, 1);
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn parse_error_on_garbage() {
        let err = parse_eql("GARBAGE QUERY").unwrap_err();
        assert_eq!(err.line, 1);
    }

    #[test]
    fn parse_multiple_aggregates() {
        let ast = parse_eql("SELECT COUNT(*), AVG(value), MAX(value) FROM \"s\"").unwrap();
        match ast {
            QueryAst::Aggregate(q) => {
                assert_eq!(q.aggregations.len(), 3);
                assert_eq!(q.aggregations[0], AggregateFunction::Count);
                assert_eq!(q.aggregations[1], AggregateFunction::Avg);
                assert_eq!(q.aggregations[2], AggregateFunction::Max);
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn parse_anomaly_zscore() {
        let ast = parse_eql("SELECT * FROM 'k' WHERE ANOMALY(value, ZSCORE(3.0))").unwrap();
        match ast {
            QueryAst::Select(q) => match q.where_clause.unwrap().root {
                Condition::Anomaly(AnomalyMethod::ZScore(t)) => {
                    assert!((t - 3.0).abs() < 1e-10);
                }
                other => panic!("expected Anomaly ZScore, got {:?}", other),
            },
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_anomaly_iqr() {
        let ast = parse_eql("SELECT * FROM 'k' WHERE ANOMALY(value, IQR(1.5))").unwrap();
        match ast {
            QueryAst::Select(q) => match q.where_clause.unwrap().root {
                Condition::Anomaly(AnomalyMethod::Iqr(m)) => {
                    assert!((m - 1.5).abs() < 1e-10);
                }
                other => panic!("expected Anomaly IQR, got {:?}", other),
            },
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_fresh_keyword() {
        let ast = parse_eql("SELECT * FROM 'k' WHERE FRESH").unwrap();
        match ast {
            QueryAst::Select(q) => match q.where_clause.unwrap().root {
                Condition::Freshness(FreshnessCondition::Fresh) => {}
                other => panic!("expected Fresh, got {:?}", other),
            },
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_stale_keyword() {
        let ast = parse_eql("SELECT * FROM 'k' WHERE STALE").unwrap();
        match ast {
            QueryAst::Select(q) => match q.where_clause.unwrap().root {
                Condition::Freshness(FreshnessCondition::Stale) => {}
                other => panic!("expected Stale, got {:?}", other),
            },
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn parse_freshness_comparison() {
        let ast = parse_eql("SELECT * FROM 'k' WHERE FRESHNESS > 0.7").unwrap();
        match ast {
            QueryAst::Select(q) => match q.where_clause.unwrap().root {
                Condition::Freshness(FreshnessCondition::Compare { op, value }) => {
                    assert_eq!(op, ComparisonOp::Gt);
                    assert!((value - 0.7).abs() < 1e-10);
                }
                other => panic!("expected Freshness Compare, got {:?}", other),
            },
            _ => panic!("expected Select"),
        }
    }
}
