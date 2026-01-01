//! SynaDB Command-Line Interface
//!
//! A CLI tool for inspecting, querying, and managing SynaDB databases.
//!
//! # Commands
//!
//! - `info` - Display database information
//! - `get` - Get a value by key
//! - `put` - Set a value
//! - `keys` - List all keys
//! - `search` - Search vectors (requires vector store)
//! - `export` - Export to file (JSON, JSONL, CSV, MessagePack, CBOR)
//! - `compact` - Compact database

use clap::{Parser, Subcommand};
use serde::Serialize;
use std::path::Path;
use synadb::{Atom, SynaDB};

/// SynaDB CLI - Command-line interface for SynaDB database
#[derive(Parser, Debug)]
#[command(name = "syna")]
#[command(author, version, about = "SynaDB database CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Show database info
    Info {
        /// Path to database
        path: String,
    },
    /// Get a value by key
    Get {
        /// Path to database
        path: String,
        /// Key to retrieve
        key: String,
    },
    /// Set a value
    Put {
        /// Path to database
        path: String,
        /// Key to set
        key: String,
        /// Value to store
        value: String,
        /// Value type: float, int, text (default: text)
        #[arg(short, long, default_value = "text")]
        r#type: String,
    },
    /// List all keys
    Keys {
        /// Path to database
        path: String,
        /// Filter keys by pattern (prefix match)
        #[arg(short, long)]
        pattern: Option<String>,
    },
    /// Search vectors (requires vector data)
    Search {
        /// Path to database
        path: String,
        /// Query vector as comma-separated floats
        #[arg(short, long)]
        query: String,
        /// Number of results to return
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Vector dimensions (required for search)
        #[arg(short, long, default_value = "768")]
        dimensions: u16,
    },
    /// Export to file
    Export {
        /// Path to database
        path: String,
        /// Output format: json, jsonl, csv, msgpack, cbor
        #[arg(short, long)]
        format: String,
        /// Output file path
        #[arg(short, long)]
        output: String,
    },
    /// Compact database (remove old entries, keep latest values)
    Compact {
        /// Path to database
        path: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Info { path } => cmd_info(&path),
        Commands::Get { path, key } => cmd_get(&path, &key),
        Commands::Put {
            path,
            key,
            value,
            r#type,
        } => cmd_put(&path, &key, &value, &r#type),
        Commands::Keys { path, pattern } => cmd_keys(&path, pattern),
        Commands::Search {
            path,
            query,
            k,
            dimensions,
        } => cmd_search(&path, &query, k, dimensions),
        Commands::Export {
            path,
            format,
            output,
        } => cmd_export(&path, &format, &output),
        Commands::Compact { path } => cmd_compact(&path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Display database information
fn cmd_info(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Database not found: {}", path).into());
    }

    let db = SynaDB::new(path)?;
    let keys = db.keys();
    let file_size = std::fs::metadata(path)?.len();

    println!("Database: {}", path);
    println!(
        "File size: {} bytes ({:.2} KB)",
        file_size,
        file_size as f64 / 1024.0
    );
    println!("Total keys: {}", keys.len());

    // Count keys by type prefix
    let mut prefixes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for key in &keys {
        let prefix = key.split('/').next().unwrap_or(key);
        *prefixes.entry(prefix.to_string()).or_insert(0) += 1;
    }

    if !prefixes.is_empty() {
        println!("\nKey prefixes:");
        let mut sorted_prefixes: Vec<_> = prefixes.iter().collect();
        sorted_prefixes.sort_by(|a, b| b.1.cmp(a.1));
        for (prefix, count) in sorted_prefixes.iter().take(10) {
            println!("  {}: {} keys", prefix, count);
        }
        if sorted_prefixes.len() > 10 {
            println!("  ... and {} more prefixes", sorted_prefixes.len() - 10);
        }
    }

    Ok(())
}

/// Get a value by key
fn cmd_get(path: &str, key: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Database not found: {}", path).into());
    }

    let mut db = SynaDB::new(path)?;

    match db.get(key)? {
        Some(atom) => {
            println!("{}", format_atom(&atom));
        }
        None => {
            println!("Key not found: {}", key);
        }
    }

    Ok(())
}

/// Set a value
fn cmd_put(
    path: &str,
    key: &str,
    value: &str,
    value_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut db = SynaDB::new(path)?;

    let atom = match value_type.to_lowercase().as_str() {
        "float" | "f" | "f64" => {
            let f: f64 = value
                .parse()
                .map_err(|_| format!("Invalid float: {}", value))?;
            Atom::Float(f)
        }
        "int" | "i" | "i64" => {
            let i: i64 = value
                .parse()
                .map_err(|_| format!("Invalid integer: {}", value))?;
            Atom::Int(i)
        }
        "text" | "t" | "string" | "str" => Atom::Text(value.to_string()),
        "bytes" | "b" => {
            // Interpret as hex string
            let bytes = hex_to_bytes(value)?;
            Atom::Bytes(bytes)
        }
        _ => {
            return Err(
                format!("Unknown type: {}. Use: float, int, text, bytes", value_type).into(),
            );
        }
    };

    let offset = db.append(key, atom)?;
    println!("Stored at offset: {}", offset);

    Ok(())
}

/// List all keys
fn cmd_keys(path: &str, pattern: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Database not found: {}", path).into());
    }

    let db = SynaDB::new(path)?;
    let mut keys = db.keys();
    keys.sort();

    // Filter by pattern if provided
    if let Some(ref pat) = pattern {
        keys.retain(|k| k.starts_with(pat) || k.contains(pat));
    }

    if keys.is_empty() {
        if pattern.is_some() {
            println!("No keys matching pattern");
        } else {
            println!("Database is empty");
        }
    } else {
        println!("Keys ({}):", keys.len());
        for key in &keys {
            println!("  {}", key);
        }
    }

    Ok(())
}

/// Search vectors
fn cmd_search(
    path: &str,
    query_str: &str,
    k: usize,
    dimensions: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    use synadb::distance::DistanceMetric;
    use synadb::vector::{VectorConfig, VectorStore};

    if !Path::new(path).exists() {
        return Err(format!("Database not found: {}", path).into());
    }

    // Parse query vector from comma-separated string
    let query: Vec<f32> = query_str
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "Invalid query vector format. Use comma-separated floats.")?;

    if query.len() != dimensions as usize {
        return Err(format!(
            "Query vector has {} dimensions, expected {}",
            query.len(),
            dimensions
        )
        .into());
    }

    let config = VectorConfig {
        dimensions,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let mut store = VectorStore::new(path, config)?;

    if store.is_empty() {
        println!("No vectors in database");
        return Ok(());
    }

    let results = store.search(&query, k)?;

    if results.is_empty() {
        println!("No results found");
    } else {
        println!("Search results ({}):", results.len());
        println!("{:<30} {:<12}", "Key", "Score");
        println!("{}", "-".repeat(44));
        for result in &results {
            println!("{:<30} {:.6}", result.key, result.score);
        }
    }

    Ok(())
}

/// Export database to file
fn cmd_export(path: &str, format: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Database not found: {}", path).into());
    }

    let mut db = SynaDB::new(path)?;
    let keys = db.keys();

    let format_name = match format.to_lowercase().as_str() {
        "json" => {
            export_json(&mut db, &keys, output)?;
            "JSON"
        }
        "jsonl" | "jsonlines" | "ndjson" => {
            export_jsonl(&mut db, &keys, output)?;
            "JSONL"
        }
        "csv" => {
            export_csv(&mut db, &keys, output)?;
            "CSV"
        }
        "msgpack" | "messagepack" | "mp" => {
            export_msgpack(&mut db, &keys, output)?;
            "MessagePack"
        }
        "cbor" => {
            export_cbor(&mut db, &keys, output)?;
            "CBOR"
        }
        _ => {
            return Err(format!(
                "Unsupported format: {}. Use: json, jsonl, csv, msgpack, cbor",
                format
            )
            .into());
        }
    };

    println!(
        "Exported {} keys to {} ({})",
        keys.len(),
        output,
        format_name
    );
    Ok(())
}

/// Export to JSON format
fn export_json(
    db: &mut SynaDB,
    keys: &[String],
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = std::fs::File::create(output)?;
    writeln!(file, "{{")?;

    for (i, key) in keys.iter().enumerate() {
        if let Some(atom) = db.get(key)? {
            let json_value = atom_to_json(&atom);
            let comma = if i < keys.len() - 1 { "," } else { "" };
            writeln!(
                file,
                "  \"{}\": {}{}",
                escape_json_string(key),
                json_value,
                comma
            )?;
        }
    }

    writeln!(file, "}}")?;
    Ok(())
}

/// Export to CSV format
fn export_csv(
    db: &mut SynaDB,
    keys: &[String],
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = std::fs::File::create(output)?;
    writeln!(file, "key,type,value")?;

    for key in keys {
        if let Some(atom) = db.get(key)? {
            let (type_name, value_str) = atom_to_csv(&atom);
            writeln!(
                file,
                "{},{},{}",
                escape_csv(key),
                type_name,
                escape_csv(&value_str)
            )?;
        }
    }

    Ok(())
}

/// Export to JSONL (JSON Lines) format - one JSON object per line
fn export_jsonl(
    db: &mut SynaDB,
    keys: &[String],
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = std::fs::File::create(output)?;

    for key in keys {
        if let Some(atom) = db.get(key)? {
            let record = ExportRecord {
                key: key.clone(),
                value: atom_to_serde_value(&atom),
            };
            let json = serde_json::to_string(&record)?;
            writeln!(file, "{}", json)?;
        }
    }

    Ok(())
}

/// Export to MessagePack format
fn export_msgpack(
    db: &mut SynaDB,
    keys: &[String],
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufWriter;

    let file = std::fs::File::create(output)?;
    let mut writer = BufWriter::new(file);

    // Collect all records
    let mut records: Vec<ExportRecord> = Vec::new();
    for key in keys {
        if let Some(atom) = db.get(key)? {
            records.push(ExportRecord {
                key: key.clone(),
                value: atom_to_serde_value(&atom),
            });
        }
    }

    // Write as MessagePack
    rmp_serde::encode::write(&mut writer, &records)?;

    Ok(())
}

/// Export to CBOR format
fn export_cbor(
    db: &mut SynaDB,
    keys: &[String],
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufWriter;

    let file = std::fs::File::create(output)?;
    let writer = BufWriter::new(file);

    // Collect all records
    let mut records: Vec<ExportRecord> = Vec::new();
    for key in keys {
        if let Some(atom) = db.get(key)? {
            records.push(ExportRecord {
                key: key.clone(),
                value: atom_to_serde_value(&atom),
            });
        }
    }

    // Write as CBOR
    ciborium::into_writer(&records, writer)?;

    Ok(())
}

/// Record structure for serialization
#[derive(Serialize)]
struct ExportRecord {
    key: String,
    value: serde_json::Value,
}

/// Convert Atom to serde_json::Value for serialization
fn atom_to_serde_value(atom: &Atom) -> serde_json::Value {
    match atom {
        Atom::Null => serde_json::Value::Null,
        Atom::Float(f) => {
            if f.is_nan() || f.is_infinite() {
                serde_json::Value::String(format!("{}", f))
            } else {
                serde_json::json!(f)
            }
        }
        Atom::Int(i) => serde_json::json!(i),
        Atom::Text(s) => serde_json::Value::String(s.clone()),
        Atom::Bytes(b) => {
            use std::fmt::Write;
            let hex = b.iter().fold(String::new(), |mut acc, byte| {
                let _ = write!(acc, "{:02x}", byte);
                acc
            });
            serde_json::json!({
                "type": "bytes",
                "hex": hex
            })
        }
        Atom::Vector(v, dims) => {
            serde_json::json!({
                "type": "vector",
                "dimensions": dims,
                "data": v
            })
        }
    }
}

/// Compact database
fn cmd_compact(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Database not found: {}", path).into());
    }

    let size_before = std::fs::metadata(path)?.len();

    let mut db = SynaDB::new(path)?;
    db.compact()?;

    let size_after = std::fs::metadata(path)?.len();
    let saved = size_before.saturating_sub(size_after);

    println!("Compaction complete");
    println!("  Before: {} bytes", size_before);
    println!("  After:  {} bytes", size_after);
    println!(
        "  Saved:  {} bytes ({:.1}%)",
        saved,
        (saved as f64 / size_before as f64) * 100.0
    );

    Ok(())
}

// =============================================================================
// Helper functions
// =============================================================================

/// Format an Atom for display
fn format_atom(atom: &Atom) -> String {
    match atom {
        Atom::Null => "null".to_string(),
        Atom::Float(f) => format!("{} (float)", f),
        Atom::Int(i) => format!("{} (int)", i),
        Atom::Text(s) => format!("\"{}\" (text)", s),
        Atom::Bytes(b) => format!("[{} bytes] (bytes)", b.len()),
        Atom::Vector(v, dims) => format!("[{} floats, {} dims] (vector)", v.len(), dims),
    }
}

/// Convert Atom to JSON value string
fn atom_to_json(atom: &Atom) -> String {
    match atom {
        Atom::Null => "null".to_string(),
        Atom::Float(f) => {
            if f.is_nan() {
                "\"NaN\"".to_string()
            } else if f.is_infinite() {
                if *f > 0.0 {
                    "\"Infinity\"".to_string()
                } else {
                    "\"-Infinity\"".to_string()
                }
            } else {
                format!("{}", f)
            }
        }
        Atom::Int(i) => format!("{}", i),
        Atom::Text(s) => format!("\"{}\"", escape_json_string(s)),
        Atom::Bytes(b) => {
            use std::fmt::Write;
            let hex = b.iter().fold(String::new(), |mut acc, byte| {
                let _ = write!(acc, "{:02x}", byte);
                acc
            });
            format!("\"{}\"", hex)
        }
        Atom::Vector(v, _) => {
            let values: Vec<String> = v.iter().map(|f| format!("{}", f)).collect();
            format!("[{}]", values.join(", "))
        }
    }
}

/// Convert Atom to CSV (type, value) pair
fn atom_to_csv(atom: &Atom) -> (&'static str, String) {
    match atom {
        Atom::Null => ("null", "".to_string()),
        Atom::Float(f) => ("float", format!("{}", f)),
        Atom::Int(i) => ("int", format!("{}", i)),
        Atom::Text(s) => ("text", s.clone()),
        Atom::Bytes(b) => {
            use std::fmt::Write;
            let hex = b.iter().fold(String::new(), |mut acc, byte| {
                let _ = write!(acc, "{:02x}", byte);
                acc
            });
            ("bytes", hex)
        }
        Atom::Vector(v, dims) => {
            let values: Vec<String> = v.iter().map(|f| format!("{}", f)).collect();
            ("vector", format!("dims={}:{}", dims, values.join(";")))
        }
    }
}

/// Escape a string for JSON
fn escape_json_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Escape a string for CSV
fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Convert hex string to bytes
fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let hex = hex.trim_start_matches("0x").trim_start_matches("0X");
    if hex.len() % 2 != 0 {
        return Err("Hex string must have even length".into());
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|_| format!("Invalid hex at position {}", i).into())
        })
        .collect()
}
