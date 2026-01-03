#!/usr/bin/env python3
"""
RAG Example: Amazon Reviews Semantic Search

This example demonstrates SynaDB's edge AI capabilities by building a local
semantic search system over Amazon product reviews. No cloud APIs needed!

## Why SynaDB for Edge RAG?

Traditional RAG requires:
- Cloud vector DB (Pinecone, Weaviate) → Network latency, API costs
- External embedding API (OpenAI) → Privacy concerns, rate limits

SynaDB approach:
- Single file database → Works offline, no server
- Local embeddings → sentence-transformers runs on CPU/GPU
- Multiple index options → Choose speed vs recall tradeoff

## Index Comparison

| Index | Build Time | Search | Recall | Best For |
|-------|------------|--------|--------|----------|
| VectorStore (HNSW) | Slow | Fast | 95%+ | Production, high recall |
| GravityWellIndex | Fast | Fast | 90%+ | Large datasets, fast build |
| CascadeIndex | Medium | Medium | 95% | Balanced |

## Usage

```bash
# Install dependencies
pip install datasets sentence-transformers

# Run with scale presets (100K, 1M, 10M, 50M, or full 87.9M)
python rag_amazon_reviews.py --scale 100k
python rag_amazon_reviews.py --scale 1m --index gwi
python rag_amazon_reviews.py --scale full --index cascade  # 87.9M reviews!

# Custom number of reviews
python rag_amazon_reviews.py --num-reviews 5000

# Test all indices
python rag_amazon_reviews.py --index all --scale 100k --benchmark

# Interactive search
python rag_amazon_reviews.py --interactive --scale 100k
```

## Scale Presets

| Scale | Reviews | Memory (384d) | Use Case |
|-------|---------|---------------|----------|
| 100k  | 100,000 | ~140-150 MB | Quick testing |
| 1m    | 1,000,000 | ~2 GB | Development |
| 10m   | 10,000,000 | ~20 GB | Production testing |
| 50m   | 50,000,000 | ~100 GB | Large scale |
| full  | 87,900,000 | ~160 GB | Full dataset |

## Dataset

Uses `sentence-transformers/amazon-reviews` from HuggingFace.
Data is downloaded locally - no cloud APIs needed for inference!
"""

import argparse
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Add synadb to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Scale presets: name -> number of reviews
SCALE_PRESETS = {
    "100k": 100_000,
    "1m": 1_000_000,
    "10m": 10_000_000,
    "50m": 50_000_000,
    "full": None,  # Full dataset (87.9M)
}

# Check for required dependencies
HAS_DATASETS = False
HAS_SENTENCE_TRANSFORMERS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    print("⚠ datasets not installed. Using synthetic data.")
    print("  Install with: pip install datasets")

try:
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except (ImportError, ValueError) as e:
    print("⚠ sentence-transformers not available. Using random embeddings.")
    print(f"  Error: {e}")
    print("  Install with: pip install sentence-transformers")


@dataclass
class SearchResult:
    """Search result with review text and metadata."""
    text: str
    score: float
    rating: Optional[int] = None
    title: Optional[str] = None


@dataclass 
class BenchmarkResult:
    """Benchmark results for an index."""
    index_name: str
    build_time: float
    search_time_avg: float
    search_time_p50: float
    search_time_p99: float
    recall_at_10: float
    num_vectors: int


def download_amazon_reviews(num_reviews: int = 10000, streaming: bool = False, cache_dir: Path = None) -> Tuple[List[str], np.ndarray, List[dict]]:
    """
    Download Amazon reviews from HuggingFace and generate embeddings.
    Caches embeddings to disk for reuse.
    
    Args:
        num_reviews: Number of reviews to load (None for full dataset)
        streaming: Use streaming mode for large datasets
        cache_dir: Directory to cache embeddings
    
    Returns:
        texts: List of review texts
        embeddings: numpy array of shape (n, 384)
        metadata: List of dicts with title, etc.
    """
    if not HAS_DATASETS:
        raise ImportError("Please install datasets: pip install datasets")
    
    # Check for cached embeddings
    if cache_dir:
        cache_dir = Path(cache_dir)
        scale_name = "full" if num_reviews is None else f"{num_reviews}"
        emb_cache = cache_dir / f"embeddings_{scale_name}.npy"
        text_cache = cache_dir / f"texts_{scale_name}.npy"
        
        if emb_cache.exists() and text_cache.exists():
            print(f"Loading cached embeddings from {emb_cache}...")
            embeddings = np.load(emb_cache)
            texts = np.load(text_cache, allow_pickle=True).tolist()
            metadata = [{"title": ""} for _ in texts]  # Metadata not cached
            print(f"  Loaded {len(texts):,} cached embeddings")
            print(f"  Embedding shape: {embeddings.shape}")
            return texts, embeddings, metadata
    
    if num_reviews is None:
        print("Loading FULL Amazon reviews dataset...")
        print("⚠ This will take a while and require significant memory!")
        streaming = True  # Force streaming for full dataset
        split = "train"
    else:
        print(f"Downloading {num_reviews:,} Amazon reviews from HuggingFace...")
        split = f"train[:{num_reviews}]"
    
    # Load a sample first to detect field names
    print("  Detecting dataset schema...")
    sample = load_dataset("sentence-transformers/amazon-reviews", split="train[:1]")
    available_fields = list(sample[0].keys())
    print(f"  Available fields: {available_fields}")
    
    # Detect text field
    text_field = None
    for candidate in ["review", "text", "review_body", "content", "sentence"]:
        if candidate in available_fields:
            text_field = candidate
            break
    
    if text_field is None:
        raise ValueError(f"Could not find text field in dataset. Available fields: {available_fields}")
    
    print(f"  Using '{text_field}' as text field")
    
    # Load dataset - use streaming for large datasets
    if streaming or (num_reviews and num_reviews > 1_000_000):
        print("Using streaming mode for large dataset...")
        dataset = load_dataset(
            "sentence-transformers/amazon-reviews",
            split="train",
            streaming=True
        )
        
        texts = []
        metadata = []
        count = 0
        target = num_reviews or float('inf')
        
        for item in dataset:
            if count >= target:
                break
            text = item.get(text_field, "")
            if not text:  # Skip empty reviews
                continue
            texts.append(text)
            metadata.append({
                "title": item.get("title", ""),
            })
            count += 1
            if count % 100_000 == 0:
                print(f"  Loaded {count:,} reviews...")
        
        print(f"Loaded {len(texts):,} reviews")
    else:
        dataset = load_dataset(
            "sentence-transformers/amazon-reviews",
            split=split
        )
        
        texts = []
        metadata = []
        
        for item in dataset:
            text = item.get(text_field, "")
            if not text:  # Skip empty reviews
                continue
            texts.append(text)
            metadata.append({
                "title": item.get("title", ""),
            })
        
        print(f"Loaded {len(texts):,} reviews")
    
    if len(texts) == 0:
        raise ValueError("No reviews loaded! Check dataset and field names.")
    
    # Generate embeddings
    print("Generating embeddings with MiniLM...")
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Batch encode for large datasets
    batch_size = 10000
    if len(texts) > batch_size:
        print(f"Encoding in batches of {batch_size:,}...")
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings_list.append(batch_emb)
            print(f"  Encoded {min(i+batch_size, len(texts)):,}/{len(texts):,}")
        embeddings = np.vstack(embeddings_list)
    else:
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Validate embeddings
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding norm (first): {np.linalg.norm(embeddings[0]):.4f}")
    
    if embeddings.shape[0] != len(texts):
        raise ValueError(f"Embedding count mismatch: {embeddings.shape[0]} vs {len(texts)} texts")
    
    embeddings = embeddings.astype(np.float32)
    
    # Cache embeddings
    if cache_dir:
        print(f"Caching embeddings to {emb_cache}...")
        np.save(emb_cache, embeddings)
        np.save(text_cache, np.array(texts, dtype=object))
    
    return texts, embeddings, metadata


def create_vectorstore_index(path: str, texts: List[str], embeddings: np.ndarray, metadata: List[dict]):
    """Create VectorStore (HNSW) index."""
    from synadb import VectorStore
    
    dims = embeddings.shape[1]
    # Disable sync_on_write for bulk loading (456x faster)
    store = VectorStore(path, dimensions=dims, metric="cosine", sync_on_write=False)
    
    start = time.time()
    
    # Use batch insert for speed (skips index updates during insert)
    keys = [f"review_{i}" for i in range(len(texts))]
    batch_size = 10000
    
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i+batch_size]
        batch_vecs = embeddings[i:i+batch_size]
        store.insert_batch_fast(batch_keys, batch_vecs)
        print(f"  Inserted {min(i+batch_size, len(keys)):,}/{len(keys):,}")
    
    # Build HNSW index once at the end
    print("  Building HNSW index...")
    store.build_index()
    build_time = time.time() - start
    
    store.flush()
    return store, build_time


def create_gwi_index(path: str, texts: List[str], embeddings: np.ndarray, metadata: List[dict]):
    """Create GravityWellIndex."""
    from synadb import GravityWellIndex
    
    dims = embeddings.shape[1]
    gwi = GravityWellIndex(path, dimensions=dims)
    
    start = time.time()
    
    # Initialize with sample
    sample_size = min(500, len(embeddings))
    sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
    print(f"  Initializing with {sample_size} sample vectors...")
    gwi.initialize(embeddings[sample_idx])
    
    # Insert all
    keys = [f"review_{i}" for i in range(len(texts))]
    print(f"  Inserting {len(keys):,} vectors...")
    gwi.insert_batch(keys, embeddings)
    
    build_time = time.time() - start
    
    # Flush to disk
    gwi.flush()
    
    return gwi, build_time


def create_cascade_index(path: str, texts: List[str], embeddings: np.ndarray, metadata: List[dict]):
    """Create CascadeIndex (original)."""
    from synadb import CascadeIndex
    
    dims = embeddings.shape[1]
    index = CascadeIndex(path, dimensions=dims)
    
    start = time.time()
    
    keys = [f"review_{i}" for i in range(len(texts))]
    print(f"  Inserting {len(keys):,} vectors...")
    index.insert_batch(keys, embeddings)
    
    build_time = time.time() - start
    
    index.flush()
    return index, build_time


def create_mmap_cascade_index(path: str, texts: List[str], embeddings: np.ndarray, metadata: List[dict]):
    """Create MmapCascadeIndex (optimized)."""
    # MmapCascadeIndex doesn't have Python wrapper yet, skip
    return None, 0.0


def search_vectorstore(store, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """Search VectorStore."""
    results = store.search(query_embedding, k=k)
    return [(r.key, r.score) for r in results]


def search_gwi(gwi, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """Search GWI."""
    results = gwi.search(query_embedding, k=k, nprobe=100)
    return [(r.key, r.score) for r in results]


def search_cascade(index, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """Search CascadeIndex."""
    results = index.search(query_embedding, k=k)
    return [(r.key, r.score) for r in results]


def compute_recall(results: List[Tuple[str, float]], ground_truth: List[str], k: int = 10) -> float:
    """Compute recall@k."""
    result_keys = set(key for key, _ in results[:k])
    gt_keys = set(ground_truth[:k])
    return len(result_keys & gt_keys) / k


def brute_force_search(embeddings: np.ndarray, query: np.ndarray, k: int = 10) -> List[str]:
    """Brute force search for ground truth using cosine similarity."""
    # sentence-transformers embeddings are already normalized
    # Just use dot product directly
    similarities = np.dot(embeddings, query)
    top_k = np.argsort(-similarities)[:k]
    return [f"review_{i}" for i in top_k]


def benchmark_index(
    index_name: str,
    index,
    search_fn,
    embeddings: np.ndarray,
    build_time: float,
    num_queries: int = 100,
    k: int = 10,
) -> BenchmarkResult:
    """Benchmark an index."""
    
    # Random query vectors
    query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
    
    search_times = []
    recalls = []
    
    for i, idx in enumerate(query_indices):
        query = embeddings[idx]
        
        # Ground truth
        gt = brute_force_search(embeddings, query, k)
        
        # Timed search
        start = time.time()
        results = search_fn(index, query, k)
        search_times.append(time.time() - start)
        
        # Recall
        recall = compute_recall(results, gt, k)
        recalls.append(recall)
        
        # Debug first query only
        if i == 0:
            print(f"  DEBUG: Query idx={idx}, GT[0]={gt[0]}, Result[0]={results[0] if results else 'empty'}, Recall={recall:.1%}")
    
    search_times_ms = [t * 1000 for t in search_times]
    
    return BenchmarkResult(
        index_name=index_name,
        build_time=build_time,
        search_time_avg=np.mean(search_times_ms),
        search_time_p50=np.percentile(search_times_ms, 50),
        search_time_p99=np.percentile(search_times_ms, 99),
        recall_at_10=np.mean(recalls),
        num_vectors=len(embeddings),
    )


def print_benchmark_results(results: List[BenchmarkResult]):
    """Print benchmark results as a table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Index':<20} {'Build (s)':<12} {'Search p50':<12} {'Search p99':<12} {'Recall@10':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.index_name:<20} {r.build_time:<12.2f} {r.search_time_p50:<12.2f}ms {r.search_time_p99:<12.2f}ms {r.recall_at_10*100:<10.1f}%")
    
    print("=" * 80)


def interactive_search(
    texts: List[str],
    embeddings: np.ndarray,
    metadata: List[dict],
    index,
    search_fn,
    model,
):
    """Interactive search loop."""
    print("\n" + "=" * 60)
    print("INTERACTIVE SEARCH")
    print("=" * 60)
    print("Enter a query to search Amazon reviews.")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("Query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        
        if not query:
            continue
        
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        
        # Search
        start = time.time()
        results = search_fn(index, query_embedding, k=5)
        search_time = (time.time() - start) * 1000
        
        print(f"\nResults ({search_time:.1f}ms):")
        print("-" * 60)
        
        for i, (key, score) in enumerate(results, 1):
            idx = int(key.split("_")[1])
            text = texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx]
            rating = metadata[idx].get("rating", "?")
            print(f"{i}. [★{rating}] (score: {score:.4f})")
            print(f"   {text}")
            print()
        
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Example: Amazon Reviews Semantic Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scale presets:
  100k   100,000 reviews (~150 MB)
  1m     1,000,000 reviews (~1.5 GB)
  10m    10,000,000 reviews (~15 GB)
  50m    50,000,000 reviews (~75 GB)
  full   87,900,000 reviews (~130 GB)

Examples:
  python rag_amazon_reviews.py --scale 100k --benchmark
  python rag_amazon_reviews.py --scale 1m --index gwi
  python rag_amazon_reviews.py --index all --scale 100k
  python rag_amazon_reviews.py --interactive --scale 100k
        """
    )
    parser.add_argument("--scale", choices=list(SCALE_PRESETS.keys()), 
                        help="Scale preset (100k, 1m, 10m, 50m, full)")
    parser.add_argument("--num-reviews", type=int, default=10000, 
                        help="Number of reviews (overridden by --scale)")
    parser.add_argument("--index", choices=["all", "vectorstore", "gwi", "cascade"], 
                        default="all", help="Index type to use")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark comparison")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run interactive search")
    parser.add_argument("--data-dir", type=str, default="./data", 
                        help="Directory for data files")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode (for very large datasets)")
    args = parser.parse_args()
    
    # Resolve scale preset
    if args.scale:
        num_reviews = SCALE_PRESETS[args.scale]
        scale_name = args.scale
    else:
        num_reviews = args.num_reviews
        scale_name = f"{num_reviews:,}"
    
    print("=" * 60)
    print("SynaDB RAG Example: Amazon Reviews Semantic Search")
    print("=" * 60)
    print("\nThis example shows how to build a local RAG system with SynaDB.")
    print("No cloud APIs needed - everything runs on your machine!\n")
    
    if num_reviews is None:
        print(f"Scale: FULL DATASET (87.9M reviews)")
    else:
        print(f"Scale: {scale_name} ({num_reviews:,} reviews)")
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Download data (with caching)
    use_streaming = args.streaming or (num_reviews and num_reviews > 1_000_000)
    texts, embeddings, metadata = download_amazon_reviews(num_reviews, streaming=use_streaming, cache_dir=data_dir)
    
    print(f"\nDataset: {len(texts):,} reviews, {embeddings.shape[1]} dimensions")
    print(f"Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB\n")
    
    # Create indices
    results = []
    indices = {}
    
    if args.index in ("all", "vectorstore"):
        print("\n--- Creating VectorStore (HNSW) ---")
        scale_suffix = args.scale if args.scale else f"{num_reviews}"
        path = str(data_dir / f"reviews_vectorstore_{scale_suffix}.db")
        store, build_time = create_vectorstore_index(path, texts, embeddings, metadata)
        indices["vectorstore"] = (store, search_vectorstore)
        print(f"Build time: {build_time:.2f}s")
        
        if args.benchmark:
            result = benchmark_index("VectorStore", store, search_vectorstore, embeddings, build_time)
            results.append(result)
    
    if args.index in ("all", "gwi"):
        print("\n--- Creating GravityWellIndex ---")
        scale_suffix = args.scale if args.scale else f"{num_reviews}"
        path = str(data_dir / f"reviews_gwi_{scale_suffix}.gwi")
        gwi, build_time = create_gwi_index(path, texts, embeddings, metadata)
        indices["gwi"] = (gwi, search_gwi)
        print(f"Build time: {build_time:.2f}s")
        
        if args.benchmark:
            result = benchmark_index("GWI", gwi, search_gwi, embeddings, build_time)
            results.append(result)
    
    if args.index in ("all", "cascade"):
        print("\n--- Creating CascadeIndex ---")
        scale_suffix = args.scale if args.scale else f"{num_reviews}"
        path = str(data_dir / f"reviews_cascade_{scale_suffix}.cascade")
        cascade, build_time = create_cascade_index(path, texts, embeddings, metadata)
        indices["cascade"] = (cascade, search_cascade)
        print(f"Build time: {build_time:.2f}s")
        
        if args.benchmark:
            result = benchmark_index("CascadeIndex", cascade, search_cascade, embeddings, build_time)
            results.append(result)
    
    # Print benchmark results
    if args.benchmark and results:
        print_benchmark_results(results)
    
    # Interactive search
    if args.interactive:
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Interactive mode requires sentence-transformers")
            return
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Use first available index
        if indices:
            index_name = list(indices.keys())[0]
            index, search_fn = indices[index_name]
            print(f"\nUsing {index_name} for interactive search")
            interactive_search(texts, embeddings, metadata, index, search_fn, model)
    
    # Demo search if not interactive
    if not args.interactive and not args.benchmark:
        print("\n--- Demo Search ---")
        
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Demo requires sentence-transformers for query encoding")
            return
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        demo_queries = [
            "great battery life lasts all day",
            "terrible quality broke after one week",
            "perfect gift for kids",
            "best purchase I ever made",
        ]
        
        for index_name, (index, search_fn) in indices.items():
            print(f"\n[{index_name}]")
            
            for query in demo_queries[:2]:
                query_emb = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
                
                start = time.time()
                results = search_fn(index, query_emb, k=3)
                search_time = (time.time() - start) * 1000
                
                print(f"\nQuery: '{query}' ({search_time:.1f}ms)")
                for key, score in results[:2]:
                    idx = int(key.split("_")[1])
                    text = texts[idx][:100] + "..."
                    print(f"  [{score:.3f}] {text}")
    
    print("\n✓ Done! Data saved to:", data_dir)
    print(f"\nTo run interactive search:")
    print(f"  python {__file__} --interactive --index cascade")
    if args.scale:
        print(f"  python {__file__} --interactive --scale {args.scale}")
    print(f"\nTo benchmark all indices:")
    print(f"  python {__file__} --benchmark --index all")


if __name__ == "__main__":
    main()
