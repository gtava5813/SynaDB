//! Streaming Queries — continuous queries with windowing.
//!
//! Supports tumbling, sliding, and session windows.

use crate::query::ResultRow;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// Window type for streaming queries.
#[derive(Debug, Clone)]
pub enum StreamWindow {
    /// Non-overlapping fixed-size windows.
    Tumbling { duration_micros: u64 },
    /// Overlapping windows that slide by a fixed interval.
    Sliding {
        duration_micros: u64,
        slide_micros: u64,
    },
    /// Windows defined by gaps in data arrival.
    Session { gap_micros: u64 },
    /// Count-based window.
    Count { size: usize },
}

/// A single window of data ready for aggregation.
#[derive(Debug, Clone)]
pub struct Window {
    /// Window start timestamp (inclusive).
    pub start: u64,
    /// Window end timestamp (exclusive).
    pub end: u64,
    /// Rows in this window.
    pub rows: Vec<ResultRow>,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Split a sorted stream of rows into windows.
///
/// `rows` must be sorted by timestamp.
pub fn windowize(rows: Vec<ResultRow>, window: &StreamWindow) -> Vec<Window> {
    match window {
        StreamWindow::Tumbling { duration_micros } => tumbling_windows(rows, *duration_micros),
        StreamWindow::Sliding {
            duration_micros,
            slide_micros,
        } => sliding_windows(rows, *duration_micros, *slide_micros),
        StreamWindow::Session { gap_micros } => session_windows(rows, *gap_micros),
        StreamWindow::Count { size } => count_windows(rows, *size),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Implementations
// ═══════════════════════════════════════════════════════════════════════

fn tumbling_windows(rows: Vec<ResultRow>, duration: u64) -> Vec<Window> {
    if rows.is_empty() || duration == 0 {
        return vec![];
    }

    let first_ts = rows[0].timestamp;
    let mut windows: Vec<Window> = Vec::new();
    let mut current_start = (first_ts / duration) * duration;

    let mut current_rows: Vec<ResultRow> = Vec::new();

    for row in rows {
        while row.timestamp >= current_start + duration {
            if !current_rows.is_empty() {
                windows.push(Window {
                    start: current_start,
                    end: current_start + duration,
                    rows: std::mem::take(&mut current_rows),
                });
            }
            current_start += duration;
        }
        current_rows.push(row);
    }

    if !current_rows.is_empty() {
        windows.push(Window {
            start: current_start,
            end: current_start + duration,
            rows: current_rows,
        });
    }

    windows
}

fn sliding_windows(rows: Vec<ResultRow>, duration: u64, slide: u64) -> Vec<Window> {
    if rows.is_empty() || duration == 0 || slide == 0 {
        return vec![];
    }

    let first_ts = rows[0].timestamp;
    let last_ts = rows[rows.len() - 1].timestamp;
    let mut windows = Vec::new();
    let mut window_start = (first_ts / slide) * slide;

    while window_start <= last_ts {
        let window_end = window_start + duration;
        let window_rows: Vec<ResultRow> = rows
            .iter()
            .filter(|r| r.timestamp >= window_start && r.timestamp < window_end)
            .cloned()
            .collect();

        if !window_rows.is_empty() {
            windows.push(Window {
                start: window_start,
                end: window_end,
                rows: window_rows,
            });
        }
        window_start += slide;
    }

    windows
}

fn session_windows(rows: Vec<ResultRow>, gap: u64) -> Vec<Window> {
    if rows.is_empty() {
        return vec![];
    }

    let mut windows = Vec::new();
    let mut session_start = rows[0].timestamp;
    let mut session_rows: Vec<ResultRow> = vec![rows[0].clone()];

    for row in rows.into_iter().skip(1) {
        let last_ts = session_rows.last().unwrap().timestamp;
        if row.timestamp - last_ts > gap {
            // Gap exceeded — close current session
            let end = session_rows.last().unwrap().timestamp + 1;
            windows.push(Window {
                start: session_start,
                end,
                rows: std::mem::take(&mut session_rows),
            });
            session_start = row.timestamp;
        }
        session_rows.push(row);
    }

    if !session_rows.is_empty() {
        let end = session_rows.last().unwrap().timestamp + 1;
        windows.push(Window {
            start: session_start,
            end,
            rows: session_rows,
        });
    }

    windows
}

fn count_windows(rows: Vec<ResultRow>, size: usize) -> Vec<Window> {
    if rows.is_empty() || size == 0 {
        return vec![];
    }

    rows.chunks(size)
        .map(|chunk| {
            let start = chunk[0].timestamp;
            let end = chunk.last().unwrap().timestamp + 1;
            Window {
                start,
                end,
                rows: chunk.to_vec(),
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Atom;

    fn make_rows(timestamps: &[u64]) -> Vec<ResultRow> {
        timestamps
            .iter()
            .map(|ts| ResultRow {
                key: "k".into(),
                value: Atom::Float(*ts as f64),
                timestamp: *ts,
            })
            .collect()
    }

    #[test]
    fn test_tumbling_windows() {
        let rows = make_rows(&[0, 5, 10, 15, 20, 25]);
        let windows = windowize(
            rows,
            &StreamWindow::Tumbling {
                duration_micros: 10,
            },
        );
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].rows.len(), 2); // 0, 5
        assert_eq!(windows[1].rows.len(), 2); // 10, 15
        assert_eq!(windows[2].rows.len(), 2); // 20, 25
    }

    #[test]
    fn test_sliding_windows() {
        let rows = make_rows(&[0, 5, 10, 15, 20]);
        let windows = windowize(
            rows,
            &StreamWindow::Sliding {
                duration_micros: 10,
                slide_micros: 5,
            },
        );
        // Windows: [0,10), [5,15), [10,20), [15,25)
        assert!(windows.len() >= 3);
    }

    #[test]
    fn test_session_windows() {
        // Two sessions separated by a gap > 100
        let rows = make_rows(&[0, 10, 20, 200, 210, 220]);
        let windows = windowize(rows, &StreamWindow::Session { gap_micros: 100 });
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].rows.len(), 3); // 0, 10, 20
        assert_eq!(windows[1].rows.len(), 3); // 200, 210, 220
    }

    #[test]
    fn test_count_windows() {
        let rows = make_rows(&[0, 1, 2, 3, 4, 5, 6]);
        let windows = windowize(rows, &StreamWindow::Count { size: 3 });
        assert_eq!(windows.len(), 3); // [0,1,2], [3,4,5], [6]
        assert_eq!(windows[0].rows.len(), 3);
        assert_eq!(windows[2].rows.len(), 1);
    }

    #[test]
    fn test_empty_input() {
        let windows = windowize(
            vec![],
            &StreamWindow::Tumbling {
                duration_micros: 10,
            },
        );
        assert!(windows.is_empty());
    }
}
