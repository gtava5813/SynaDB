//! DAVO Freshness Queries — filter and sort by data freshness.
//!
//! Enables queries like:
//! ```sql
//! SELECT * FROM 'sensor/*' WHERE FRESH
//! SELECT * FROM 'cache/*' WHERE STALE
//! SELECT * FROM 'data/*' WHERE FRESHNESS > 0.7 ORDER BY freshness DESC
//! ```

use crate::query::ast::FreshnessCondition;
use crate::query::ResultRow;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for freshness evaluation.
#[derive(Debug, Clone)]
pub struct FreshnessConfig {
    /// Threshold below which data is considered stale (default: 0.5).
    pub threshold: f32,
    /// Decay rates per key pattern (key prefix → λ).
    pub decay_rates: Vec<(String, f32)>,
    /// Default decay rate if no pattern matches.
    pub default_decay_rate: f32,
}

impl Default for FreshnessConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            decay_rates: Vec::new(),
            default_decay_rate: 0.01,
        }
    }
}

/// A row annotated with its freshness score.
#[derive(Debug, Clone)]
pub struct FreshnessAnnotatedRow {
    /// The original row.
    pub row: ResultRow,
    /// Computed freshness (0.0–1.0).
    pub freshness: f32,
    /// Whether this row is considered stale.
    pub is_stale: bool,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Compute freshness for a set of rows based on their timestamps and decay rates.
///
/// `now_micros` is the current time in microseconds since epoch.
pub fn annotate_freshness(
    rows: &[ResultRow],
    config: &FreshnessConfig,
    now_micros: u64,
) -> Vec<FreshnessAnnotatedRow> {
    rows.iter()
        .map(|row| {
            let decay_rate = find_decay_rate(&row.key, config);
            let age_micros = now_micros.saturating_sub(row.timestamp);
            let age_secs = age_micros as f32 / 1_000_000.0;
            let freshness = (-decay_rate * age_secs).exp();
            let is_stale = freshness < config.threshold;

            FreshnessAnnotatedRow {
                row: row.clone(),
                freshness,
                is_stale,
            }
        })
        .collect()
}

/// Filter rows by a freshness condition.
pub fn filter_by_freshness(
    rows: Vec<ResultRow>,
    condition: &FreshnessCondition,
    config: &FreshnessConfig,
    now_micros: u64,
) -> Vec<ResultRow> {
    let annotated = annotate_freshness(&rows, config, now_micros);

    annotated
        .into_iter()
        .filter(|a| matches_freshness_condition(a, condition, config))
        .map(|a| a.row)
        .collect()
}

/// Sort rows by freshness (descending — freshest first).
pub fn sort_by_freshness(
    rows: &mut [ResultRow],
    config: &FreshnessConfig,
    now_micros: u64,
    descending: bool,
) {
    rows.sort_by(|a, b| {
        let fa = compute_freshness(a, config, now_micros);
        let fb = compute_freshness(b, config, now_micros);
        if descending {
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
}

/// Count stale entries.
pub fn count_stale(rows: &[ResultRow], config: &FreshnessConfig, now_micros: u64) -> usize {
    annotate_freshness(rows, config, now_micros)
        .iter()
        .filter(|a| a.is_stale)
        .count()
}

// ═══════════════════════════════════════════════════════════════════════
//  Internals
// ═══════════════════════════════════════════════════════════════════════

fn compute_freshness(row: &ResultRow, config: &FreshnessConfig, now_micros: u64) -> f32 {
    let decay_rate = find_decay_rate(&row.key, config);
    let age_micros = now_micros.saturating_sub(row.timestamp);
    let age_secs = age_micros as f32 / 1_000_000.0;
    (-decay_rate * age_secs).exp()
}

fn find_decay_rate(key: &str, config: &FreshnessConfig) -> f32 {
    for (prefix, rate) in &config.decay_rates {
        if key.starts_with(prefix.as_str()) {
            return *rate;
        }
    }
    config.default_decay_rate
}

fn matches_freshness_condition(
    annotated: &FreshnessAnnotatedRow,
    condition: &FreshnessCondition,
    _config: &FreshnessConfig,
) -> bool {
    match condition {
        FreshnessCondition::Fresh => !annotated.is_stale,
        FreshnessCondition::Stale => annotated.is_stale,
        FreshnessCondition::Compare { op, value } => {
            use crate::query::ast::ComparisonOp;
            let f = annotated.freshness as f64;
            match op {
                ComparisonOp::Gt => f > *value,
                ComparisonOp::Gte => f >= *value,
                ComparisonOp::Lt => f < *value,
                ComparisonOp::Lte => f <= *value,
                ComparisonOp::Eq => (f - *value).abs() < 0.001,
                ComparisonOp::Ne => (f - *value).abs() >= 0.001,
                _ => true,
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::ast::ComparisonOp;
    use crate::types::Atom;

    fn make_row(key: &str, ts: u64) -> ResultRow {
        ResultRow {
            key: key.to_string(),
            value: Atom::Float(1.0),
            timestamp: ts,
        }
    }

    #[test]
    fn test_fresh_data_has_high_freshness() {
        let now = 1_000_000_000u64; // 1000 seconds
        let row = make_row("sensor/temp", now - 1_000_000); // 1 second ago
        let config = FreshnessConfig {
            default_decay_rate: 0.01,
            ..Default::default()
        };

        let annotated = annotate_freshness(&[row], &config, now);
        assert_eq!(annotated.len(), 1);
        assert!(annotated[0].freshness > 0.99); // e^(-0.01 * 1) ≈ 0.99
        assert!(!annotated[0].is_stale);
    }

    #[test]
    fn test_old_data_is_stale() {
        let now = 1_000_000_000u64;
        let row = make_row("sensor/temp", now - 100_000_000); // 100 seconds ago
        let config = FreshnessConfig {
            default_decay_rate: 0.01,
            threshold: 0.5,
            ..Default::default()
        };

        let annotated = annotate_freshness(&[row], &config, now);
        // e^(-0.01 * 100) = e^(-1) ≈ 0.368 < 0.5 → stale
        assert!(annotated[0].freshness < 0.5);
        assert!(annotated[0].is_stale);
    }

    #[test]
    fn test_filter_fresh_only() {
        let now = 1_000_000_000u64;
        let rows = vec![
            make_row("fresh", now - 1_000_000),   // 1s ago → fresh
            make_row("stale", now - 200_000_000), // 200s ago → stale
        ];
        let config = FreshnessConfig {
            default_decay_rate: 0.01,
            threshold: 0.5,
            ..Default::default()
        };

        let filtered = filter_by_freshness(rows, &FreshnessCondition::Fresh, &config, now);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].key, "fresh");
    }

    #[test]
    fn test_filter_stale_only() {
        let now = 1_000_000_000u64;
        let rows = vec![
            make_row("fresh", now - 1_000_000),
            make_row("stale", now - 200_000_000),
        ];
        let config = FreshnessConfig::default();

        let filtered = filter_by_freshness(rows, &FreshnessCondition::Stale, &config, now);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].key, "stale");
    }

    #[test]
    fn test_filter_by_threshold() {
        let now = 1_000_000_000u64;
        let rows = vec![
            make_row("high", now - 1_000_000),    // freshness ~0.99
            make_row("medium", now - 50_000_000), // freshness ~0.61
            make_row("low", now - 200_000_000),   // freshness ~0.14
        ];
        let config = FreshnessConfig::default();

        let condition = FreshnessCondition::Compare {
            op: ComparisonOp::Gt,
            value: 0.5,
        };
        let filtered = filter_by_freshness(rows, &condition, &config, now);
        assert_eq!(filtered.len(), 2); // high and medium
    }

    #[test]
    fn test_sort_by_freshness() {
        let now = 1_000_000_000u64;
        let mut rows = vec![
            make_row("old", now - 100_000_000),
            make_row("new", now - 1_000_000),
            make_row("mid", now - 50_000_000),
        ];
        let config = FreshnessConfig::default();

        sort_by_freshness(&mut rows, &config, now, true); // descending
        assert_eq!(rows[0].key, "new"); // freshest first
        assert_eq!(rows[2].key, "old"); // stalest last
    }

    #[test]
    fn test_custom_decay_rate_per_prefix() {
        let now = 1_000_000_000u64;
        let rows = vec![
            make_row("sensor/temp", now - 10_000_000), // 10s ago
            make_row("config/name", now - 10_000_000), // 10s ago
        ];
        let config = FreshnessConfig {
            threshold: 0.5,
            decay_rates: vec![
                ("sensor/".to_string(), 0.1),   // fast decay
                ("config/".to_string(), 0.001), // slow decay
            ],
            default_decay_rate: 0.01,
        };

        let annotated = annotate_freshness(&rows, &config, now);
        // sensor: e^(-0.1 * 10) = e^(-1) ≈ 0.37 → stale
        assert!(annotated[0].is_stale);
        // config: e^(-0.001 * 10) = e^(-0.01) ≈ 0.99 → fresh
        assert!(!annotated[1].is_stale);
    }

    #[test]
    fn test_count_stale() {
        let now = 1_000_000_000u64;
        let rows = vec![
            make_row("a", now - 1_000_000),
            make_row("b", now - 200_000_000),
            make_row("c", now - 300_000_000),
        ];
        let config = FreshnessConfig::default();

        let stale = count_stale(&rows, &config, now);
        assert_eq!(stale, 2); // b and c are stale
    }
}
