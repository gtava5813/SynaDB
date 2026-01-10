//! Real-World Example: Hybrid RAG Document Store
//!
//! Simulates a production RAG system where:
//! - Documents stream in continuously (news articles, support tickets, etc.)
//! - Hot layer handles real-time ingestion with immediate searchability
//! - Cold layer stores historical documents with optimized search
//! - Periodic promotion moves aged data from hot to cold
//!
//! Use case: Customer support system ingesting tickets in real-time
//! while maintaining searchable history of resolved tickets.

use std::time::Instant;
use synadb::arch::{HybridConfig, HybridVectorStore, ResultSource};
use synadb::cascade::CascadeConfig;
use synadb::gwi::GwiConfig;

/// Simulated document with embedding
struct Document {
    id: String,
    title: String,
    category: String,
    embedding: Vec<f32>,
}

/// Generate a pseudo-random embedding based on category
/// In production, you'd use a real embedding model (OpenAI, Sentence-Transformers, etc.)
fn generate_embedding(text: &str, dims: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut embedding = Vec::with_capacity(dims);
    for i in 0..dims {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        i.hash(&mut hasher);
        let h = hasher.finish();
        embedding.push(((h % 2000) as f32 / 1000.0) - 1.0);
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}

/// Generate sample support tickets
fn generate_tickets(count: usize, dims: usize) -> Vec<Document> {
    let categories = [
        "billing",
        "technical",
        "account",
        "shipping",
        "returns",
        "product",
    ];
    let issues = [
        "payment failed",
        "login issue",
        "password reset",
        "order delayed",
        "refund request",
        "product defect",
        "subscription cancel",
        "upgrade plan",
        "api error",
        "integration help",
    ];

    (0..count)
        .map(|i| {
            let category = categories[i % categories.len()];
            let issue = issues[i % issues.len()];
            let title = format!("Ticket #{}: {} - {}", i, category, issue);

            // Embedding is based on category + issue for clustering
            let embed_text = format!("{} {}", category, issue);

            Document {
                id: format!("ticket_{}", i),
                title,
                category: category.to_string(),
                embedding: generate_embedding(&embed_text, dims),
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Hybrid RAG Document Store - Real-World Example           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Configuration
    const DIMS: usize = 384; // MiniLM-L6 dimensions
    const HISTORICAL_TICKETS: usize = 10_000; // Resolved tickets (cold)
    const STREAMING_TICKETS: usize = 1_000; // New tickets (hot)
    const SEARCH_K: usize = 5;

    // Use temp directory for demo
    let temp_dir = std::env::temp_dir().join("synadb_rag_demo");
    std::fs::create_dir_all(&temp_dir)?;
    let hot_path = temp_dir.join("tickets_hot.gwi");
    let cold_path = temp_dir.join("tickets_cold.cascade");

    // Clean up any existing files
    let _ = std::fs::remove_file(&hot_path);
    let _ = std::fs::remove_file(&cold_path);

    println!("📁 Storage: {}", temp_dir.display());
    println!("📊 Dimensions: {} (MiniLM-L6 compatible)", DIMS);
    println!("📚 Historical tickets: {}", HISTORICAL_TICKETS);
    println!("🔥 Streaming tickets: {}", STREAMING_TICKETS);
    println!();

    // =========================================================================
    // Step 1: Initialize Hybrid Store
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Initialize Hybrid Store");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let config = HybridConfig {
        hot: GwiConfig {
            dimensions: DIMS as u16,
            branching_factor: 16,
            num_levels: 3,
            nprobe: 20,
            initial_capacity: STREAMING_TICKETS * 2,
            ..Default::default()
        },
        cold: CascadeConfig {
            dimensions: DIMS as u16,
            ..CascadeConfig::large() // Optimized for 100K+ vectors
        },
    };

    let start = Instant::now();
    let mut store = HybridVectorStore::new(&hot_path, &cold_path, config)?;
    println!("✓ Store created in {:?}", start.elapsed());

    // Generate sample data for attractor initialization
    let sample_tickets = generate_tickets(2000, DIMS);
    let sample_refs: Vec<&[f32]> = sample_tickets.iter().map(|d| d.embedding.as_slice()).collect();

    let start = Instant::now();
    store.initialize_hot(&sample_refs)?;
    println!("✓ Attractors initialized in {:?}", start.elapsed());
    println!();

    // =========================================================================
    // Step 2: Load Historical Data (Cold Layer)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 2: Load Historical Tickets (simulating cold data import)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let historical = generate_tickets(HISTORICAL_TICKETS, DIMS);

    // In production, you'd bulk-load directly to cold layer
    // Here we ingest to hot then promote (simulating migration)
    let start = Instant::now();
    for doc in &historical {
        store.ingest(&doc.id, &doc.embedding)?;
    }
    let ingest_time = start.elapsed();
    println!(
        "✓ Ingested {} historical tickets in {:?} ({:.0} tickets/sec)",
        HISTORICAL_TICKETS,
        ingest_time,
        HISTORICAL_TICKETS as f64 / ingest_time.as_secs_f64()
    );

    // Promote to cold layer
    let start = Instant::now();
    let promoted = store.promote_to_cold()?;
    println!(
        "✓ Promoted {} tickets to cold layer in {:?}",
        promoted,
        start.elapsed()
    );
    println!("  Hot count: {}, Cold count: {}", store.hot_count(), store.cold_count());
    println!();

    // =========================================================================
    // Step 3: Stream New Tickets (Hot Layer)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 3: Stream New Tickets (real-time ingestion)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let streaming = generate_tickets(STREAMING_TICKETS, DIMS);

    let start = Instant::now();
    for (i, doc) in streaming.iter().enumerate() {
        let key = format!("new_{}", doc.id);
        store.ingest(&key, &doc.embedding)?;

        // Progress indicator
        if (i + 1) % 250 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush()?;
        }
    }
    println!();
    let stream_time = start.elapsed();
    println!(
        "✓ Streamed {} new tickets in {:?} ({:.0} tickets/sec)",
        STREAMING_TICKETS,
        stream_time,
        STREAMING_TICKETS as f64 / stream_time.as_secs_f64()
    );
    println!("  Hot count: {}, Cold count: {}", store.hot_count(), store.cold_count());
    println!();

    // =========================================================================
    // Step 4: Search Scenarios
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 4: Search Scenarios");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Scenario A: Customer asks about billing issue
    println!("\n🔍 Scenario A: Customer asks about billing issue");
    let query_billing = generate_embedding("billing payment failed charge", DIMS);

    let start = Instant::now();
    let results = store.search(&query_billing, SEARCH_K)?;
    let search_time = start.elapsed();

    println!("   Query: \"billing payment failed charge\"");
    println!("   Found {} results in {:?}:", results.len(), search_time);
    for (i, r) in results.iter().enumerate() {
        let source = match r.source {
            ResultSource::Hot => "🔥 HOT",
            ResultSource::Cold => "❄️ COLD",
        };
        println!("   {}. {} [{}] score={:.4}", i + 1, r.key, source, r.score);
    }

    // Scenario B: Technical support query
    println!("\n🔍 Scenario B: Technical support query");
    let query_tech = generate_embedding("technical api error integration", DIMS);

    let start = Instant::now();
    let results = store.search(&query_tech, SEARCH_K)?;
    let search_time = start.elapsed();

    println!("   Query: \"technical api error integration\"");
    println!("   Found {} results in {:?}:", results.len(), search_time);
    for (i, r) in results.iter().enumerate() {
        let source = match r.source {
            ResultSource::Hot => "🔥 HOT",
            ResultSource::Cold => "❄️ COLD",
        };
        println!("   {}. {} [{}] score={:.4}", i + 1, r.key, source, r.score);
    }

    // Scenario C: Search only recent tickets (hot layer)
    println!("\n🔍 Scenario C: Search only recent tickets (hot layer)");
    let start = Instant::now();
    let hot_results = store.search_hot(&query_billing, SEARCH_K)?;
    let hot_time = start.elapsed();
    println!(
        "   Hot-only search: {} results in {:?}",
        hot_results.len(),
        hot_time
    );

    // Scenario D: Search only historical tickets (cold layer)
    println!("\n🔍 Scenario D: Search only historical tickets (cold layer)");
    let start = Instant::now();
    let cold_results = store.search_cold(&query_billing, SEARCH_K)?;
    let cold_time = start.elapsed();
    println!(
        "   Cold-only search: {} results in {:?}",
        cold_results.len(),
        cold_time
    );

    // =========================================================================
    // Step 5: Performance Summary
    // =========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Performance Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Total vectors: {}", store.len());
    println!("  - Hot layer (recent): {}", store.hot_count());
    println!("  - Cold layer (historical): {}", store.cold_count());
    println!();
    println!("Ingestion rates:");
    println!(
        "  - Historical bulk: {:.0} vectors/sec",
        HISTORICAL_TICKETS as f64 / ingest_time.as_secs_f64()
    );
    println!(
        "  - Streaming: {:.0} vectors/sec",
        STREAMING_TICKETS as f64 / stream_time.as_secs_f64()
    );
    println!();
    println!("Search latency:");
    println!("  - Hot-only: {:?}", hot_time);
    println!("  - Cold-only: {:?}", cold_time);
    println!("  - Unified (both): {:?}", search_time);

    // Cleanup
    let _ = std::fs::remove_file(&hot_path);
    let _ = std::fs::remove_file(&cold_path);

    println!("\n✅ Demo complete!");
    Ok(())
}
