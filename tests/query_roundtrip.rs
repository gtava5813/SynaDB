//! Property 1: Query AST Serialization Round-Trip
//!
//! For any randomly generated `QueryAst`, `serialize` → `deserialize` must
//! yield a value equal to the original.

use proptest::prelude::*;
use synadb::query::ast::*;
use synadb::types::Atom;

// ═══════════════════════════════════════════════════════════════════════
//  Strategies — generate random AST nodes
// ═══════════════════════════════════════════════════════════════════════

fn arb_key_pattern() -> impl Strategy<Value = KeyPattern> {
    let leaf = prop_oneof![
        "[a-z]{1,10}".prop_map(KeyPattern::Exact),
        "[a-z]{1,10}/".prop_map(KeyPattern::Prefix),
        "[a-z]{1,5}/\\*".prop_map(KeyPattern::Glob),
        "\\^[a-z]+".prop_map(KeyPattern::Regex),
    ];
    leaf.prop_recursive(2, 6, 3, |inner| {
        prop::collection::vec(inner, 1..3).prop_map(KeyPattern::Union)
    })
}

fn arb_comparison_op() -> impl Strategy<Value = ComparisonOp> {
    prop_oneof![
        Just(ComparisonOp::Eq),
        Just(ComparisonOp::Ne),
        Just(ComparisonOp::Gt),
        Just(ComparisonOp::Gte),
        Just(ComparisonOp::Lt),
        Just(ComparisonOp::Lte),
        Just(ComparisonOp::In),
        Just(ComparisonOp::Nin),
        Just(ComparisonOp::Like),
        Just(ComparisonOp::Regex),
    ]
}

fn arb_order_field() -> impl Strategy<Value = OrderField> {
    prop_oneof![
        Just(OrderField::Key),
        Just(OrderField::Value),
        Just(OrderField::Timestamp),
    ]
}

fn arb_direction() -> impl Strategy<Value = Direction> {
    prop_oneof![Just(Direction::Asc), Just(Direction::Desc)]
}

fn arb_atom() -> impl Strategy<Value = Atom> {
    prop_oneof![
        any::<f64>()
            .prop_filter("finite", |v| v.is_finite())
            .prop_map(Atom::Float),
        any::<i64>().prop_map(Atom::Int),
        "[a-z]{0,20}".prop_map(Atom::Text),
        Just(Atom::Null),
    ]
}

fn arb_value_filter() -> impl Strategy<Value = ValueFilter> {
    prop_oneof![
        arb_atom().prop_map(ValueFilter::Single),
        prop::collection::vec(arb_atom(), 1..5).prop_map(ValueFilter::List),
    ]
}

fn arb_condition() -> impl Strategy<Value = Condition> {
    let leaf = prop_oneof![
        (arb_order_field(), arb_comparison_op(), arb_value_filter())
            .prop_map(|(field, op, rhs)| Condition::Comparison { field, op, rhs }),
        arb_key_pattern().prop_map(Condition::Key),
        any::<Option<u64>>()
            .prop_flat_map(|s| any::<Option<u64>>().prop_map(move |e| (s, e)))
            .prop_map(|(start, end)| Condition::TimeRange(TimeRange { start, end })),
    ];
    leaf.prop_recursive(2, 8, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 1..3)
                .prop_map(|cs| Condition::Boolean(BooleanOp::And(cs))),
            prop::collection::vec(inner.clone(), 1..3)
                .prop_map(|cs| Condition::Boolean(BooleanOp::Or(cs))),
            inner.prop_map(|c| Condition::Boolean(BooleanOp::Not(Box::new(c)))),
        ]
    })
}

fn arb_projection() -> impl Strategy<Value = Projection> {
    prop_oneof![
        Just(Projection::All),
        Just(Projection::Key),
        Just(Projection::Value),
        Just(Projection::Timestamp),
    ]
}

fn arb_order_by() -> impl Strategy<Value = OrderBy> {
    (arb_order_field(), arb_direction()).prop_map(|(field, direction)| OrderBy { field, direction })
}

fn arb_select_query() -> impl Strategy<Value = SelectQuery> {
    (
        prop::collection::vec(arb_projection(), 1..4),
        prop::option::of(arb_key_pattern()),
        prop::option::of(arb_condition().prop_map(|root| WhereClause { root })),
        prop::option::of(arb_order_by()),
        prop::option::of(any::<u64>()),
        prop::option::of(any::<u64>()),
    )
        .prop_map(
            |(projections, from, where_clause, order_by, limit, offset)| SelectQuery {
                projections,
                from,
                where_clause,
                order_by,
                limit,
                offset,
            },
        )
}

fn arb_aggregate_function() -> impl Strategy<Value = AggregateFunction> {
    prop_oneof![
        Just(AggregateFunction::Count),
        Just(AggregateFunction::Sum),
        Just(AggregateFunction::Avg),
        Just(AggregateFunction::Min),
        Just(AggregateFunction::Max),
        Just(AggregateFunction::First),
        Just(AggregateFunction::Last),
    ]
}

fn arb_group_by() -> impl Strategy<Value = GroupBy> {
    prop_oneof![
        Just(GroupBy::Key),
        Just(GroupBy::TimeBucket(TimeBucket::Minute)),
        Just(GroupBy::TimeBucket(TimeBucket::Hour)),
        Just(GroupBy::TimeBucket(TimeBucket::Day)),
        Just(GroupBy::TimeBucket(TimeBucket::Week)),
        Just(GroupBy::TimeBucket(TimeBucket::Month)),
    ]
}

fn arb_aggregate_query() -> impl Strategy<Value = AggregateQuery> {
    (
        prop::collection::vec(arb_aggregate_function(), 1..3),
        prop::option::of(arb_key_pattern()),
        prop::option::of(arb_condition().prop_map(|root| WhereClause { root })),
        prop::option::of(arb_group_by()),
        prop::option::of(arb_order_by()),
        prop::option::of(any::<u64>()),
    )
        .prop_map(
            |(aggregations, from, where_clause, group_by, order_by, limit)| AggregateQuery {
                aggregations,
                from,
                where_clause,
                group_by,
                having: None,
                order_by,
                limit,
            },
        )
}

fn arb_query_ast() -> impl Strategy<Value = QueryAst> {
    prop_oneof![
        arb_select_query().prop_map(QueryAst::Select),
        arb_aggregate_query().prop_map(QueryAst::Aggregate),
    ]
}

// ═══════════════════════════════════════════════════════════════════════
//  Property 1: Serialization round-trip (bincode)
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// **Property 1: Query AST Serialization Round-Trip (bincode)**
    ///
    /// For any randomly generated QueryAst, bincode encode → decode must
    /// produce a value equal to the original.
    ///
    /// _Validates: Requirements 8.1, 8.5_
    #[test]
    fn prop_query_ast_bincode_roundtrip(ast in arb_query_ast()) {
        let bytes = bincode::serialize(&ast).unwrap();
        let decoded: QueryAst = bincode::deserialize(&bytes).unwrap();
        prop_assert_eq!(ast, decoded);
    }
}
