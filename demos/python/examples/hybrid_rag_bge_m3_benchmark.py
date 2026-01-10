#!/usr/bin/env python3
"""
SynaDB Hybrid Vector Store Benchmark
====================================

Pure benchmark comparing insert speeds across SynaDB vector stores:
- GWI (Gravity Well Index) - Append-only, O(1) insert
- Cascade Index - Three-stage hybrid index
- SparseVectorStore - Inverted index for lexical search

Uses Amazon ESCI dataset (US locale) with BGE-M3 embeddings.
Pre-embeds all documents first, then measures pure SynaDB insert speed.

Usage:
    python hybrid_rag_bge_m3_benchmark.py --num-docs 100000
    python hybrid_rag_bge_m3_benchmark.py --num-docs 1000000 --skip-search
    python hybrid_rag_bge_m3_benchmark.py --scale-test  # Test at 100K, 200K, ..., 500K
    python hybrid_rag_bge_m3_benchmark.py --iterations 10  # Run 10 times for consistency
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install datasets tqdm numpy")
    sys.exit(1)

try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_FLAG_EMBEDDING = True
except ImportError:
    HAS_FLAG_EMBEDDING = False
    print("FlagEmbedding not installed. Install with: pip install FlagEmbedding")
    sys.exit(1)

from synadb import GravityWellIndex, CascadeIndex, SparseVectorStore


@dataclass
class BenchmarkResult:
    """Benchmark result for a single index."""
    name: str
    num_vectors: int
    total_time: float
    vectors_per_sec: float
    
    def __str__(self):
        return f"{self.name}: {self.num_vectors:,} vectors in {self.total_time:.2f}s ({self.vectors_per_sec:,.0f} vec/sec)"


def load_esci_products(num_docs: int) -> List[str]:
    """Load US product texts from Amazon ESCI dataset."""
    print(f"Loading Amazon ESCI products (US locale, max {num_docs:,})...")
    
    products_ds = load_dataset("milistu/amazon-esci-data", "products", split="train")
    
    texts = []
    for item in tqdm(products_ds, desc="Loading products"):
        if len(texts) >= num_docs:
            break
        
        if item.get("product_locale", "") != "us":
            continue
        
        title = item.get("product_title", "") or ""
        description = item.get("product_description", "") or ""
        brand = item.get("product_brand", "") or ""
        bullet_points = item.get("product_bullet_point", "") or ""
        
        parts = [title]
        if brand:
            parts.append(f"Brand: {brand}")
        if bullet_points:
            parts.append(str(bullet_points)[:500])
        if description:
            parts.append(str(description)[:500])
        
        text = " | ".join(filter(None, parts))
        if text.strip():
            texts.append(text)
    
    print(f"✓ Loaded {len(texts):,} US products")
    return texts


def encode_documents(texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, List[Dict[int, float]]]:
    """Encode documents with BGE-M3 (dense + sparse)."""
    print(f"Loading BGE-M3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    
    print(f"Encoding {len(texts):,} documents...")
    start = time.time()
    
    output = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
    )
    
    dense_vecs = output['dense_vecs']
    
    # Convert sparse to {term_id: weight} format
    vocab = {}
    next_id = 0
    sparse_vecs = []
    
    for lexical_weights in output['lexical_weights']:
        sparse_vec = {}
        for token, weight in lexical_weights.items():
            if token not in vocab:
                vocab[token] = next_id
                next_id += 1
            sparse_vec[vocab[token]] = float(weight)
        sparse_vecs.append(sparse_vec)
    
    elapsed = time.time() - start
    print(f"✓ Encoded in {elapsed:.1f}s ({len(texts)/elapsed:.0f} docs/sec)")
    
    return dense_vecs, sparse_vecs


def benchmark_gwi(dense_vecs: np.ndarray, data_dir: Path) -> BenchmarkResult:
    """Benchmark GWI insert speed."""
    gwi_path = str(data_dir / "bench_gwi.gwi")
    if os.path.exists(gwi_path):
        os.remove(gwi_path)
    
    # Initialize with sample vectors
    gwi = GravityWellIndex(
        gwi_path,
        dimensions=1024,
        branching_factor=16,
        num_levels=3,
        initial_capacity=len(dense_vecs) + 1000
    )
    gwi.initialize(dense_vecs[:min(2000, len(dense_vecs))])
    
    # Benchmark inserts
    print(f"\nBenchmarking GWI ({len(dense_vecs):,} vectors)...")
    start = time.time()
    
    for i, vec in enumerate(tqdm(dense_vecs, desc="GWI insert")):
        gwi.insert(f"doc_{i}", vec)
    
    elapsed = time.time() - start
    gwi.close()
    
    return BenchmarkResult(
        name="GWI",
        num_vectors=len(dense_vecs),
        total_time=elapsed,
        vectors_per_sec=len(dense_vecs) / elapsed
    )


def benchmark_cascade(dense_vecs: np.ndarray, data_dir: Path) -> BenchmarkResult:
    """Benchmark Cascade insert speed."""
    cascade_path = str(data_dir / "bench_cascade.cascade")
    if os.path.exists(cascade_path):
        os.remove(cascade_path)
    
    cascade = CascadeIndex(
        cascade_path,
        dimensions=1024,
        num_probes=20,
        ef_search=100
    )
    
    # Benchmark inserts
    print(f"\nBenchmarking Cascade ({len(dense_vecs):,} vectors)...")
    start = time.time()
    
    for i, vec in enumerate(tqdm(dense_vecs, desc="Cascade insert")):
        cascade.insert(f"doc_{i}", vec)
    
    elapsed = time.time() - start
    cascade.close()
    
    return BenchmarkResult(
        name="Cascade",
        num_vectors=len(dense_vecs),
        total_time=elapsed,
        vectors_per_sec=len(dense_vecs) / elapsed
    )


def benchmark_svs(sparse_vecs: List[Dict[int, float]], data_dir: Path) -> BenchmarkResult:
    """Benchmark SparseVectorStore insert speed."""
    svs_path = str(data_dir / "bench_svs.svs")
    if os.path.exists(svs_path):
        os.remove(svs_path)
    
    svs = SparseVectorStore(svs_path)
    
    # Calculate average NNZ
    avg_nnz = sum(len(v) for v in sparse_vecs) / len(sparse_vecs) if sparse_vecs else 0
    
    # Benchmark inserts
    print(f"\nBenchmarking SVS ({len(sparse_vecs):,} vectors, avg {avg_nnz:.1f} NNZ)...")
    start = time.time()
    
    for i, vec in enumerate(tqdm(sparse_vecs, desc="SVS insert")):
        if vec:
            svs.index(f"doc_{i}", vec)
    
    elapsed = time.time() - start
    svs.close()
    
    return BenchmarkResult(
        name="SVS",
        num_vectors=len(sparse_vecs),
        total_time=elapsed,
        vectors_per_sec=len(sparse_vecs) / elapsed
    )


def benchmark_search(dense_vecs: np.ndarray, sparse_vecs: List[Dict[int, float]], 
                     data_dir: Path, num_queries: int = 100) -> Dict[str, float]:
    """Benchmark search latency."""
    print(f"\n{'='*70}")
    print(f"  Search Latency Benchmark ({num_queries} queries)")
    print(f"{'='*70}")
    
    # Sample query vectors
    query_indices = np.random.choice(len(dense_vecs), min(num_queries, len(dense_vecs)), replace=False)
    
    results = {}
    
    # GWI search
    gwi_path = str(data_dir / "bench_gwi.gwi")
    if os.path.exists(gwi_path):
        gwi = GravityWellIndex(gwi_path, dimensions=1024)
        
        latencies = []
        for idx in tqdm(query_indices, desc="GWI search"):
            start = time.time()
            gwi.search(dense_vecs[idx], k=10)
            latencies.append((time.time() - start) * 1000)
        
        results["GWI p50 (ms)"] = np.percentile(latencies, 50)
        results["GWI p99 (ms)"] = np.percentile(latencies, 99)
        gwi.close()
    
    # Cascade search
    cascade_path = str(data_dir / "bench_cascade.cascade")
    if os.path.exists(cascade_path):
        cascade = CascadeIndex(cascade_path, dimensions=1024)
        
        latencies = []
        for idx in tqdm(query_indices, desc="Cascade search"):
            start = time.time()
            cascade.search(dense_vecs[idx], k=10)
            latencies.append((time.time() - start) * 1000)
        
        results["Cascade p50 (ms)"] = np.percentile(latencies, 50)
        results["Cascade p99 (ms)"] = np.percentile(latencies, 99)
        cascade.close()
    
    # SVS search
    svs_path = str(data_dir / "bench_svs.svs")
    if os.path.exists(svs_path):
        svs = SparseVectorStore(svs_path)
        
        latencies = []
        for idx in tqdm(query_indices, desc="SVS search"):
            if sparse_vecs[idx]:
                start = time.time()
                svs.search(sparse_vecs[idx], k=10)
                latencies.append((time.time() - start) * 1000)
        
        if latencies:
            results["SVS p50 (ms)"] = np.percentile(latencies, 50)
            results["SVS p99 (ms)"] = np.percentile(latencies, 99)
        svs.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SynaDB Hybrid Vector Store Benchmark")
    parser.add_argument("--num-docs", type=int, default=100000, help="Number of documents")
    parser.add_argument("--data-dir", type=str, default="./benchmark_data", help="Data directory")
    parser.add_argument("--skip-search", action="store_true", help="Skip search benchmark")
    parser.add_argument("--skip-cascade", action="store_true", help="Skip Cascade (slow at scale)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations for consistency testing")
    parser.add_argument("--scale-test", action="store_true", help="Test at 100K, 200K, 300K, 400K, 500K increments")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  SynaDB Hybrid Vector Store Benchmark")
    print("  GWI vs Cascade vs SparseVectorStore")
    print("=" * 70)
    print()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.scale_test:
        # Scale test: run at multiple sizes
        run_scale_test(data_dir, args)
    elif args.iterations > 1:
        # Consistency test: run multiple iterations
        run_consistency_test(data_dir, args)
    else:
        # Single run
        run_single_benchmark(data_dir, args)


def run_single_benchmark(data_dir: Path, args):
    """Run a single benchmark."""
    texts = load_esci_products(args.num_docs)
    dense_vecs, sparse_vecs = encode_documents(texts)
    
    print(f"\n{'='*70}")
    print(f"  Insert Benchmarks ({len(dense_vecs):,} vectors)")
    print(f"{'='*70}")
    
    results = []
    
    # GWI benchmark
    gwi_result = benchmark_gwi(dense_vecs, data_dir)
    results.append(gwi_result)
    print(f"✓ {gwi_result}")
    
    # Cascade benchmark (optional - slow at scale)
    if not args.skip_cascade:
        cascade_result = benchmark_cascade(dense_vecs, data_dir)
        results.append(cascade_result)
        print(f"✓ {cascade_result}")
    else:
        print("⏭️  Skipping Cascade (--skip-cascade)")
    
    # SVS benchmark
    svs_result = benchmark_svs(sparse_vecs, data_dir)
    results.append(svs_result)
    print(f"✓ {svs_result}")
    
    # Search benchmark
    search_results = {}
    if not args.skip_search:
        search_results = benchmark_search(dense_vecs, sparse_vecs, data_dir)
    
    # Summary
    print_summary(results, search_results, len(dense_vecs), data_dir)


def run_consistency_test(data_dir: Path, args):
    """Run multiple iterations to test consistency."""
    print(f"Running {args.iterations} iterations for consistency testing...")
    print()
    
    texts = load_esci_products(args.num_docs)
    dense_vecs, sparse_vecs = encode_documents(texts)
    
    gwi_speeds = []
    cascade_speeds = []
    svs_speeds = []
    
    for i in range(args.iterations):
        print(f"\n{'='*70}")
        print(f"  Iteration {i+1}/{args.iterations} ({len(dense_vecs):,} vectors)")
        print(f"{'='*70}")
        
        # GWI
        gwi_result = benchmark_gwi(dense_vecs, data_dir)
        gwi_speeds.append(gwi_result.vectors_per_sec)
        print(f"✓ {gwi_result}")
        
        # Cascade
        if not args.skip_cascade:
            cascade_result = benchmark_cascade(dense_vecs, data_dir)
            cascade_speeds.append(cascade_result.vectors_per_sec)
            print(f"✓ {cascade_result}")
        
        # SVS
        svs_result = benchmark_svs(sparse_vecs, data_dir)
        svs_speeds.append(svs_result.vectors_per_sec)
        print(f"✓ {svs_result}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"  CONSISTENCY TEST RESULTS ({args.iterations} iterations)")
    print(f"{'='*70}")
    
    print(f"\n📊 GWI Insert Speed (vec/sec):")
    print(f"   Mean:   {np.mean(gwi_speeds):,.0f}")
    print(f"   Std:    {np.std(gwi_speeds):,.0f}")
    print(f"   Min:    {np.min(gwi_speeds):,.0f}")
    print(f"   Max:    {np.max(gwi_speeds):,.0f}")
    print(f"   CV:     {np.std(gwi_speeds)/np.mean(gwi_speeds)*100:.1f}%")
    
    if cascade_speeds:
        print(f"\n📊 Cascade Insert Speed (vec/sec):")
        print(f"   Mean:   {np.mean(cascade_speeds):,.0f}")
        print(f"   Std:    {np.std(cascade_speeds):,.0f}")
        print(f"   Min:    {np.min(cascade_speeds):,.0f}")
        print(f"   Max:    {np.max(cascade_speeds):,.0f}")
        print(f"   CV:     {np.std(cascade_speeds)/np.mean(cascade_speeds)*100:.1f}%")
        
        speedups = [g/c for g, c in zip(gwi_speeds, cascade_speeds)]
        print(f"\n🚀 GWI vs Cascade Speedup:")
        print(f"   Mean:   {np.mean(speedups):.1f}x")
        print(f"   Std:    {np.std(speedups):.2f}x")
        print(f"   Min:    {np.min(speedups):.1f}x")
        print(f"   Max:    {np.max(speedups):.1f}x")
    
    print(f"\n📊 SVS Insert Speed (vec/sec):")
    print(f"   Mean:   {np.mean(svs_speeds):,.0f}")
    print(f"   Std:    {np.std(svs_speeds):,.0f}")
    print(f"   Min:    {np.min(svs_speeds):,.0f}")
    print(f"   Max:    {np.max(svs_speeds):,.0f}")
    print(f"   CV:     {np.std(svs_speeds)/np.mean(svs_speeds)*100:.1f}%")
    
    print(f"\n✅ Consistency test complete!")


def run_scale_test(data_dir: Path, args):
    """Run benchmarks at multiple scales (100K increments)."""
    scales = [100000, 200000, 300000, 400000, 500000]
    
    print(f"Running scale test at: {', '.join(f'{s//1000}K' for s in scales)}")
    print()
    
    # Load max docs needed
    max_docs = max(scales)
    texts = load_esci_products(max_docs)
    dense_vecs, sparse_vecs = encode_documents(texts)
    
    # Results storage
    scale_results = {
        "scale": [],
        "gwi_speed": [],
        "cascade_speed": [],
        "svs_speed": [],
        "gwi_cascade_ratio": [],
    }
    
    for scale in scales:
        print(f"\n{'='*70}")
        print(f"  Scale Test: {scale:,} vectors")
        print(f"{'='*70}")
        
        # Slice to current scale
        dense_slice = dense_vecs[:scale]
        sparse_slice = sparse_vecs[:scale]
        
        scale_results["scale"].append(scale)
        
        # GWI
        gwi_result = benchmark_gwi(dense_slice, data_dir)
        scale_results["gwi_speed"].append(gwi_result.vectors_per_sec)
        print(f"✓ {gwi_result}")
        
        # Cascade
        if not args.skip_cascade:
            cascade_result = benchmark_cascade(dense_slice, data_dir)
            scale_results["cascade_speed"].append(cascade_result.vectors_per_sec)
            scale_results["gwi_cascade_ratio"].append(gwi_result.vectors_per_sec / cascade_result.vectors_per_sec)
            print(f"✓ {cascade_result}")
        else:
            scale_results["cascade_speed"].append(0)
            scale_results["gwi_cascade_ratio"].append(0)
        
        # SVS
        svs_result = benchmark_svs(sparse_slice, data_dir)
        scale_results["svs_speed"].append(svs_result.vectors_per_sec)
        print(f"✓ {svs_result}")
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"  SCALE TEST RESULTS")
    print(f"{'='*70}")
    
    print(f"\n{'Scale':<12} {'GWI (vec/s)':<15} {'Cascade (vec/s)':<18} {'SVS (vec/s)':<15} {'GWI/Cascade':<12}")
    print("-" * 72)
    
    for i, scale in enumerate(scale_results["scale"]):
        gwi = scale_results["gwi_speed"][i]
        cascade = scale_results["cascade_speed"][i]
        svs = scale_results["svs_speed"][i]
        ratio = scale_results["gwi_cascade_ratio"][i]
        
        cascade_str = f"{cascade:,.0f}" if cascade > 0 else "skipped"
        ratio_str = f"{ratio:.1f}x" if ratio > 0 else "-"
        
        print(f"{scale//1000}K{'':<9} {gwi:>12,.0f}   {cascade_str:>15}   {svs:>12,.0f}   {ratio_str:>10}")
    
    # Trend analysis
    if not args.skip_cascade and len(scale_results["gwi_cascade_ratio"]) > 1:
        ratios = [r for r in scale_results["gwi_cascade_ratio"] if r > 0]
        print(f"\n📈 GWI vs Cascade Speedup Trend:")
        print(f"   Average: {np.mean(ratios):.1f}x")
        print(f"   Range:   {np.min(ratios):.1f}x - {np.max(ratios):.1f}x")
    
    print(f"\n✅ Scale test complete!")


def print_summary(results, search_results, num_vectors, data_dir):
    """Print benchmark summary."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"\n📊 Insert Speed (same {num_vectors:,} vectors to each):")
    print("-" * 50)
    
    for r in results:
        bar_len = int(r.vectors_per_sec / max(x.vectors_per_sec for x in results) * 30)
        bar = "█" * bar_len
        print(f"  {r.name:<10} {r.vectors_per_sec:>10,.0f} vec/sec  {bar}")
    
    if search_results:
        print(f"\n⚡ Search Latency (k=10):")
        print("-" * 50)
        for metric, value in search_results.items():
            print(f"  {metric:<20} {value:>8.2f}")
    
    # Speedup comparison
    if len(results) >= 2:
        fastest = max(results, key=lambda x: x.vectors_per_sec)
        slowest = min(results, key=lambda x: x.vectors_per_sec)
        speedup = fastest.vectors_per_sec / slowest.vectors_per_sec
        print(f"\n🚀 {fastest.name} is {speedup:.1f}x faster than {slowest.name} for inserts")
    
    print(f"\n✅ Benchmark complete! Data in: {data_dir}")


if __name__ == "__main__":
    main()
