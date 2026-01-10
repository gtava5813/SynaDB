#!/usr/bin/env python3
"""
Hybrid RAG System with BGE-M3 Embeddings (Dense + Sparse)
=========================================================

A production-ready RAG (Retrieval-Augmented Generation) system using:
- BAAI/bge-m3: State-of-the-art multilingual embeddings (1024 dims dense + sparse)
- SynaDB: GWI for real-time dense vectors, Cascade for historical, SVS for sparse
- Amazon ESCI dataset: Product search with mixed keyword/semantic queries

Features:
- Dense vector search with GWI (real-time) and Cascade (historical)
- Sparse vector search with SparseVectorStore (lexical matching)
- Hybrid fusion of dense + sparse results
- Evaluation metrics (MRR, Recall@K)

Requirements:
    pip install sentence-transformers datasets tqdm FlagEmbedding

Usage:
    python hybrid_rag_bge_m3.py
    python hybrid_rag_bge_m3.py --num-docs 5000 --num-queries 100
    python hybrid_rag_bge_m3.py --sparse-only  # Test sparse search only
    python hybrid_rag_bge_m3.py --batch-mode   # Measure pure SynaDB speed
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Add parent directory for synadb import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install sentence-transformers datasets tqdm numpy")
    sys.exit(1)

# Try to import FlagEmbedding for sparse embeddings
try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_FLAG_EMBEDDING = True
except ImportError:
    HAS_FLAG_EMBEDDING = False
    print("Note: FlagEmbedding not installed. Sparse search will use simulated BM25-style vectors.")
    print("      Install with: pip install FlagEmbedding")

from synadb import GravityWellIndex, CascadeIndex, SparseVectorStore


@dataclass
class Document:
    """A document with its embeddings."""
    id: str
    title: str
    context: str
    dense_embedding: Optional[np.ndarray] = None
    sparse_embedding: Optional[Dict[int, float]] = None


@dataclass
class Query:
    """A query with ground truth."""
    id: str
    question: str
    answer: str
    context_id: str
    dense_embedding: Optional[np.ndarray] = None
    sparse_embedding: Optional[Dict[int, float]] = None


@dataclass
class SearchResult:
    """A search result."""
    doc_id: str
    score: float
    source: str  # "gwi", "cascade", "sparse", or "hybrid"


class HybridRAGSystem:
    """
    Hybrid RAG system combining:
    - GWI: Fast dense vector ingestion for recent/streaming documents
    - Cascade: Optimized dense vector search for historical data
    - SparseVectorStore: Lexical/sparse vector search for keyword matching
    
    This enables true hybrid search combining semantic (dense) and lexical (sparse) retrieval.
    """
    
    DIMS = 1024  # BGE-M3 dense output dimensions
    
    def __init__(self, data_dir: str, model_name: str = "BAAI/bge-m3", use_sparse: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.gwi_path = str(self.data_dir / "realtime.gwi")
        self.cascade_path = str(self.data_dir / "historical.cascade")
        self.sparse_path = str(self.data_dir / "lexical.svs")
        
        self.use_sparse = use_sparse and HAS_FLAG_EMBEDDING
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        if self.use_sparse:
            # Use FlagEmbedding for both dense and sparse
            self.model = BGEM3FlagModel(model_name, use_fp16=True)
            self.dense_model = None
        else:
            # Fall back to sentence-transformers for dense only
            self.dense_model = SentenceTransformer(model_name)
            self.model = None
        
        # Initialize indices
        self.gwi_index: Optional[GravityWellIndex] = None
        self.cascade_index: Optional[CascadeIndex] = None
        self.sparse_store: Optional[SparseVectorStore] = None
        
        # Document metadata store (in-memory for demo)
        self.doc_metadata: dict = {}
        
        # Track which docs are in which index
        self.gwi_docs: set = set()
        self.cascade_docs: set = set()
        self.sparse_docs: set = set()
        
        # Vocabulary for sparse vectors (term -> id mapping)
        self.vocab: Dict[str, int] = {}
        self.next_term_id = 0
        
    def _get_term_id(self, term: str) -> int:
        """Get or create term ID for vocabulary."""
        if term not in self.vocab:
            self.vocab[term] = self.next_term_id
            self.next_term_id += 1
        return self.vocab[term]
    
    def _text_to_sparse_bm25(self, text: str) -> Dict[int, float]:
        """Convert text to BM25-style sparse vector (fallback when FlagEmbedding not available)."""
        # Simple tokenization
        words = text.lower().split()
        # Count term frequencies
        tf = {}
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if word and len(word) > 2:
                term_id = self._get_term_id(word)
                tf[term_id] = tf.get(term_id, 0) + 1
        # Apply log(1 + tf) weighting
        return {k: np.log1p(v) for k, v in tf.items()}
        
    def initialize(self, sample_texts: List[str]):
        """Initialize the hybrid system with sample texts for GWI attractor training."""
        print("Generating sample embeddings for GWI attractor initialization...")
        
        if self.use_sparse:
            # Use FlagEmbedding
            output = self.model.encode(
                sample_texts[:2000],
                return_dense=True,
                return_sparse=False,
            )
            sample_embeddings = output['dense_vecs']
        else:
            # Use sentence-transformers
            sample_embeddings = self.dense_model.encode(
                sample_texts[:2000],
                show_progress_bar=True,
                normalize_embeddings=True
            )
        
        # Clean up existing files
        for path in [self.gwi_path, self.cascade_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # Initialize GWI (real-time dense layer)
        print("Initializing GWI (real-time dense index)...")
        self.gwi_index = GravityWellIndex(
            self.gwi_path,
            dimensions=self.DIMS,
            branching_factor=16,
            num_levels=3,
            initial_capacity=50000
        )
        self.gwi_index.initialize(sample_embeddings)
        
        # Initialize Cascade (historical dense layer)
        print("Initializing Cascade (historical dense index)...")
        self.cascade_index = CascadeIndex(
            self.cascade_path,
            dimensions=self.DIMS,
            num_probes=20,
            ef_search=100
        )
        
        # Initialize SparseVectorStore (lexical layer)
        print("Initializing SparseVectorStore (lexical index)...")
        self.sparse_store = SparseVectorStore(self.sparse_path)
        
        print("✓ Hybrid system initialized (dense + sparse)")
    
    def encode_documents(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        """Encode documents to both dense and sparse vectors."""
        if self.use_sparse:
            # Use FlagEmbedding for both
            output = self.model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=True,
            )
            dense_vecs = output['dense_vecs']
            # Convert lexical weights to our format
            sparse_vecs = []
            for lexical_weights in output['lexical_weights']:
                # lexical_weights is a dict of {token: weight}
                sparse_vec = {}
                for token, weight in lexical_weights.items():
                    term_id = self._get_term_id(token)
                    sparse_vec[term_id] = float(weight)
                sparse_vecs.append(sparse_vec)
            return dense_vecs, sparse_vecs
        else:
            # Use sentence-transformers for dense, BM25 for sparse
            dense_vecs = self.dense_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            sparse_vecs = [self._text_to_sparse_bm25(t) for t in texts]
            return dense_vecs, sparse_vecs
    
    def encode_query(self, text: str) -> Tuple[np.ndarray, Dict[int, float]]:
        """Encode a query to both dense and sparse vectors."""
        if self.use_sparse:
            output = self.model.encode(
                [text],
                return_dense=True,
                return_sparse=True,
            )
            dense_vec = output['dense_vecs'][0]
            lexical_weights = output['lexical_weights'][0]
            sparse_vec = {self._get_term_id(t): float(w) for t, w in lexical_weights.items()}
            return dense_vec, sparse_vec
        else:
            dense_vec = self.dense_model.encode(text, normalize_embeddings=True)
            sparse_vec = self._text_to_sparse_bm25(text)
            return dense_vec, sparse_vec
            
    def ingest_batch_to_gwi(self, docs: List[Document], batch_size: int = 32):
        """Batch ingest documents to GWI (dense) and SparseVectorStore (sparse)."""
        total = len(docs)
        
        for i in tqdm(range(0, total, batch_size), desc="Ingesting to GWI + Sparse"):
            batch = docs[i:i + batch_size]
            texts = [d.context for d in batch]
            
            # Batch encode (dense + sparse)
            dense_vecs, sparse_vecs = self.encode_documents(texts, batch_size)
            
            # Insert to GWI (dense)
            for doc, dense_emb, sparse_emb in zip(batch, dense_vecs, sparse_vecs):
                self.gwi_index.insert(doc.id, dense_emb)
                self.gwi_docs.add(doc.id)
                
                # Insert to SparseVectorStore
                if sparse_emb:
                    self.sparse_store.index(doc.id, sparse_emb)
                    self.sparse_docs.add(doc.id)
                
                self.doc_metadata[doc.id] = {
                    "title": doc.title,
                    "context": doc.context[:200] + "..."
                }
                
    def ingest_batch_to_cascade(self, docs: List[Document], batch_size: int = 32):
        """Batch ingest documents to Cascade (dense) and SparseVectorStore (sparse)."""
        total = len(docs)
        
        for i in tqdm(range(0, total, batch_size), desc="Ingesting to Cascade + Sparse"):
            batch = docs[i:i + batch_size]
            texts = [d.context for d in batch]
            
            # Batch encode (dense + sparse)
            dense_vecs, sparse_vecs = self.encode_documents(texts, batch_size)
            
            # Insert to Cascade (dense)
            for doc, dense_emb, sparse_emb in zip(batch, dense_vecs, sparse_vecs):
                self.cascade_index.insert(doc.id, dense_emb)
                self.cascade_docs.add(doc.id)
                
                # Insert to SparseVectorStore
                if sparse_emb:
                    self.sparse_store.index(doc.id, sparse_emb)
                    self.sparse_docs.add(doc.id)
                
                self.doc_metadata[doc.id] = {
                    "title": doc.title,
                    "context": doc.context[:200] + "..."
                }
    
    def search_dense(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search only dense indices (GWI + Cascade)."""
        dense_emb, _ = self.encode_query(query)
        
        # Search both dense indices
        gwi_results = self.gwi_index.search(dense_emb, k)
        cascade_results = self.cascade_index.search(dense_emb, k)
        
        # Merge and deduplicate
        seen = {}
        for r in gwi_results:
            seen[r.key] = SearchResult(doc_id=r.key, score=r.score, source="gwi")
        for r in cascade_results:
            if r.key not in seen or r.score < seen[r.key].score:
                seen[r.key] = SearchResult(doc_id=r.key, score=r.score, source="cascade")
        
        results = sorted(seen.values(), key=lambda x: x.score)
        return results[:k]
    
    def search_sparse(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search only sparse index (SparseVectorStore)."""
        _, sparse_emb = self.encode_query(query)
        
        if not sparse_emb:
            return []
        
        results = self.sparse_store.search(sparse_emb, k=k)
        return [
            SearchResult(doc_id=r.key, score=r.score, source="sparse")
            for r in results
        ]
    
    def search_hybrid(self, query: str, k: int = 10, dense_weight: float = 0.7) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse results.
        
        Uses reciprocal rank fusion (RRF) to combine results from:
        - Dense search (GWI + Cascade)
        - Sparse search (SparseVectorStore)
        
        Args:
            query: Search query
            k: Number of results
            dense_weight: Weight for dense results (0-1), sparse gets (1 - dense_weight)
        """
        dense_emb, sparse_emb = self.encode_query(query)
        
        # Get dense results
        gwi_results = self.gwi_index.search(dense_emb, k * 2)
        cascade_results = self.cascade_index.search(dense_emb, k * 2)
        
        # Merge dense results
        dense_seen = {}
        for r in gwi_results:
            dense_seen[r.key] = r.score
        for r in cascade_results:
            if r.key not in dense_seen or r.score < dense_seen[r.key]:
                dense_seen[r.key] = r.score
        
        # Get sparse results
        sparse_results = []
        if sparse_emb:
            sparse_results = self.sparse_store.search(sparse_emb, k=k * 2)
        
        # Reciprocal Rank Fusion
        rrf_scores = {}
        rrf_k = 60  # RRF constant
        
        # Add dense scores (lower distance = better, so use negative rank)
        dense_ranked = sorted(dense_seen.items(), key=lambda x: x[1])
        for rank, (doc_id, _) in enumerate(dense_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + dense_weight / (rrf_k + rank + 1)
        
        # Add sparse scores (higher score = better)
        sparse_weight = 1.0 - dense_weight
        for rank, r in enumerate(sparse_results):
            rrf_scores[r.key] = rrf_scores.get(r.key, 0) + sparse_weight / (rrf_k + rank + 1)
        
        # Sort by RRF score (higher = better)
        sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
        
        return [
            SearchResult(doc_id=doc_id, score=score, source="hybrid")
            for doc_id, score in sorted_results[:k]
        ]
        
    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> List[SearchResult]:
        """
        Search with specified mode.
        
        Args:
            query: Search query
            k: Number of results
            mode: "dense", "sparse", or "hybrid"
        """
        if mode == "dense":
            return self.search_dense(query, k)
        elif mode == "sparse":
            return self.search_sparse(query, k)
        else:
            return self.search_hybrid(query, k)
        
    def get_stats(self) -> dict:
        """Get system statistics."""
        sparse_stats = self.sparse_store.stats() if self.sparse_store else None
        return {
            "gwi_count": len(self.gwi_index) if self.gwi_index else 0,
            "cascade_count": len(self.cascade_index) if self.cascade_index else 0,
            "sparse_count": len(self.sparse_store) if self.sparse_store else 0,
            "sparse_terms": sparse_stats.num_terms if sparse_stats else 0,
            "sparse_postings": sparse_stats.num_postings if sparse_stats else 0,
            "total_docs": len(self.doc_metadata),
        }
        
    def close(self):
        """Close and cleanup."""
        if self.gwi_index:
            self.gwi_index.close()
        if self.cascade_index:
            self.cascade_index.flush()
            self.cascade_index.close()
        if self.sparse_store:
            self.sparse_store.close()


def load_esci_data(num_docs: int = 1000, num_queries: int = 100) -> Tuple[List[Document], List[Query]]:
    """
    Load Amazon ESCI dataset for product search evaluation.
    
    ESCI (E-commerce Search Challenge for Improving Product Search) contains:
    - Products with titles, descriptions, brand, color, etc.
    - Real user queries from Amazon search
    - Relevance labels: E (Exact), S (Substitute), C (Complement), I (Irrelevant)
    
    This dataset is ideal for hybrid search because queries include:
    - Exact product names/model numbers (favors sparse/lexical)
    - Semantic queries like "comfortable shoes" (favors dense)
    - Mixed queries like "iPhone 15 case" (benefits from hybrid)
    """
    print(f"Loading Amazon ESCI dataset (docs={num_docs}, queries={num_queries}, locale='us')...")
    
    # Load products and queries from milistu/amazon-esci-data
    # Dataset has 3 configs: 'products', 'queries', 'sources'
    products_ds = load_dataset("milistu/amazon-esci-data", "products", split="train")
    queries_ds = load_dataset("milistu/amazon-esci-data", "queries", split="train")
    
    # Build product lookup and documents (US locale only for English)
    # Products columns: product_id, product_title, product_description, product_bullet_point, 
    #                   product_brand, product_color, product_locale, split
    product_lookup = {}  # product_id -> doc_id
    documents = []
    
    for item in tqdm(products_ds, desc="Loading products (US only)"):
        if len(documents) >= num_docs:
            break
        
        # Filter to US locale for English products
        locale = item.get("product_locale", "")
        if locale != "us":
            continue
            
        product_id = item.get("product_id", "")
        if not product_id or product_id in product_lookup:
            continue
            
        # Build product text from available fields
        title = item.get("product_title", "") or ""
        description = item.get("product_description", "") or ""
        brand = item.get("product_brand", "") or ""
        color = item.get("product_color", "") or ""
        bullet_points = item.get("product_bullet_point", "") or ""
        
        # Combine into searchable text
        context_parts = [title]
        if brand:
            context_parts.append(f"Brand: {brand}")
        if color:
            context_parts.append(f"Color: {color}")
        if bullet_points:
            context_parts.append(str(bullet_points)[:500])
        if description:
            context_parts.append(str(description)[:500])
            
        context = " | ".join(filter(None, context_parts))
        
        if not context.strip():
            continue
        
        doc_id = f"product_{len(documents)}"
        documents.append(Document(
            id=doc_id,
            title=title[:100] if title else "Unknown Product",
            context=context
        ))
        product_lookup[product_id] = doc_id
    
    # Build queries from queries dataset (E and S are relevant, US locale only)
    # Queries columns: query, product_id, esci_label, product_locale
    queries = []
    query_seen = set()
    
    for item in tqdm(queries_ds, desc="Loading queries (US only, E/S labels)"):
        if len(queries) >= num_queries:
            break
        
        # Filter to US locale
        locale = item.get("product_locale", "")
        if locale != "us":
            continue
            
        query_text = item.get("query", "") or ""
        product_id = item.get("product_id", "") or ""
        label = item.get("esci_label", "") or ""
        
        # Only use queries where we have the product and it's relevant (E or S)
        if not query_text or not product_id:
            continue
        if product_id not in product_lookup:
            continue
        if label not in ["E", "S"]:
            continue
        if query_text in query_seen:
            continue
            
        query_seen.add(query_text)
        queries.append(Query(
            id=f"q_{len(queries)}",
            question=query_text,
            answer="",  # ESCI doesn't have answer text
            context_id=product_lookup[product_id]
        ))
    
    print(f"✓ Loaded {len(documents)} US products and {len(queries)} queries")
    print(f"   Sample product: {documents[0].title[:50]}..." if documents else "")
    print(f"   Sample query: \"{queries[0].question}\"" if queries else "")
    
    return documents, queries
    
    return documents, queries


def evaluate_retrieval(
    system: HybridRAGSystem,
    queries: List[Query],
    k_values: List[int] = [1, 5, 10],
    mode: str = "hybrid"
) -> dict:
    """Evaluate retrieval performance."""
    print(f"\nEvaluating {mode} retrieval on {len(queries)} queries...")
    
    metrics = {f"recall@{k}": 0.0 for k in k_values}
    metrics["mrr"] = 0.0
    
    for query in tqdm(queries, desc=f"Evaluating ({mode})"):
        results = system.search(query.question, k=max(k_values), mode=mode)
        result_ids = [r.doc_id for r in results]
        
        # Check if ground truth is in results
        for k in k_values:
            if query.context_id in result_ids[:k]:
                metrics[f"recall@{k}"] += 1
                
        # MRR
        if query.context_id in result_ids:
            rank = result_ids.index(query.context_id) + 1
            metrics["mrr"] += 1.0 / rank
            
    # Normalize
    n = len(queries)
    for key in metrics:
        metrics[key] /= n
        
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG System with BGE-M3 (Dense + Sparse)")
    parser.add_argument("--num-docs", type=int, default=2000, help="Number of documents")
    parser.add_argument("--num-queries", type=int, default=200, help="Number of queries")
    parser.add_argument("--data-dir", type=str, default="./hybrid_rag_data", help="Data directory")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--sparse-only", action="store_true", help="Test sparse search only")
    parser.add_argument("--dense-only", action="store_true", help="Test dense search only")
    parser.add_argument("--batch-mode", action="store_true", help="Pre-embed all docs, then measure pure SynaDB insert speed")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Hybrid RAG System with BGE-M3 (Dense + Sparse)")
    print("  Using: GWI (real-time) + Cascade (historical) + SVS (lexical)")
    if args.batch_mode:
        print("  [BATCH MODE: Pre-embed → Measure Pure SynaDB Speed]")
    print("=" * 70)
    print()
    
    if not HAS_FLAG_EMBEDDING:
        print("⚠️  FlagEmbedding not installed - using simulated BM25 for sparse vectors")
        print("   For true BGE-M3 sparse embeddings: pip install FlagEmbedding")
        print()
    
    # Load data
    documents, queries = load_esci_data(args.num_docs, args.num_queries)
    
    # Initialize system
    system = HybridRAGSystem(args.data_dir, use_sparse=True)
    
    # Use document texts for attractor initialization
    sample_texts = [d.context for d in documents[:2000]]
    system.initialize(sample_texts)
    
    # Split documents
    historical_docs = documents[:int(len(documents) * 0.8)]
    new_docs = documents[int(len(documents) * 0.8):]
    
    if args.batch_mode:
        # =====================================================================
        # BATCH MODE: Pre-embed everything, then measure pure SynaDB speed
        # =====================================================================
        print("\n" + "=" * 70)
        print("  Phase 1: Batch Embedding (GPU-accelerated)")
        print("=" * 70)
        
        all_texts = [d.context for d in documents]
        print(f"Encoding {len(all_texts)} documents (dense + sparse)...")
        
        embed_start = time.time()
        all_dense, all_sparse = system.encode_documents(all_texts, batch_size=32)
        embed_time = time.time() - embed_start
        embed_rate = len(all_texts) / embed_time
        print(f"✓ Embedded {len(all_texts)} docs in {embed_time:.2f}s ({embed_rate:.0f} docs/sec)")
        
        # Split embeddings
        hist_dense = all_dense[:len(historical_docs)]
        hist_sparse = all_sparse[:len(historical_docs)]
        new_dense = all_dense[len(historical_docs):]
        new_sparse = all_sparse[len(historical_docs):]
        
        print("\n" + "=" * 70)
        print("  Phase 2: Pure SynaDB Insert Speed (Cascade)")
        print("=" * 70)
        
        cascade_start = time.time()
        for doc, dense_emb in zip(historical_docs, hist_dense):
            system.cascade_index.insert(doc.id, dense_emb)
            system.cascade_docs.add(doc.id)
            system.doc_metadata[doc.id] = {"title": doc.title, "context": doc.context[:200] + "..."}
        cascade_time = time.time() - cascade_start
        cascade_rate = len(historical_docs) / cascade_time
        print(f"✓ Cascade: {len(historical_docs):,} vectors in {cascade_time:.3f}s ({cascade_rate:,.0f} vectors/sec)")
        
        print("\n" + "=" * 70)
        print("  Phase 3: Pure SynaDB Insert Speed (GWI)")
        print("=" * 70)
        
        gwi_start = time.time()
        for doc, dense_emb in zip(new_docs, new_dense):
            system.gwi_index.insert(doc.id, dense_emb)
            system.gwi_docs.add(doc.id)
            system.doc_metadata[doc.id] = {"title": doc.title, "context": doc.context[:200] + "..."}
        gwi_time = time.time() - gwi_start
        gwi_rate = len(new_docs) / gwi_time
        print(f"✓ GWI: {len(new_docs):,} vectors in {gwi_time:.3f}s ({gwi_rate:,.0f} vectors/sec)")
        
        print("\n" + "=" * 70)
        print("  Phase 4: Pure SynaDB Insert Speed (SparseVectorStore)")
        print("=" * 70)
        
        svs_start = time.time()
        for doc, sparse_emb in zip(documents, all_sparse):
            if sparse_emb:
                system.sparse_store.index(doc.id, sparse_emb)
                system.sparse_docs.add(doc.id)
        svs_time = time.time() - svs_start
        svs_rate = len(documents) / svs_time
        print(f"✓ SVS: {len(documents):,} sparse vectors in {svs_time:.3f}s ({svs_rate:,.0f} vectors/sec)")
        
        # Calculate average sparse vector stats
        avg_nnz = sum(len(s) for s in all_sparse) / len(all_sparse) if all_sparse else 0
        print(f"   Average sparse vector NNZ: {avg_nnz:.1f} terms")
        
    else:
        # =====================================================================
        # STANDARD MODE: Encode + insert interleaved (realistic pipeline)
        # =====================================================================
        
        # Phase 1: Historical Data Ingestion (Cascade + Sparse)
        print("\n" + "=" * 70)
        print("  Phase 1: Historical Data Ingestion (Cascade + Sparse)")
        print("=" * 70)
        
        start = time.time()
        system.ingest_batch_to_cascade(historical_docs, batch_size=32)
        cascade_time = time.time() - start
        print(f"✓ Ingested {len(historical_docs)} docs to Cascade + Sparse in {cascade_time:.2f}s "
              f"({len(historical_docs)/cascade_time:.0f} docs/sec)")
        
        # Phase 2: Real-time Document Streaming (GWI + Sparse)
        print("\n" + "=" * 70)
        print("  Phase 2: Real-time Document Streaming (GWI + Sparse)")
        print("=" * 70)
        
        start = time.time()
        system.ingest_batch_to_gwi(new_docs, batch_size=32)
        gwi_time = time.time() - start
        print(f"✓ Streamed {len(new_docs)} new docs to GWI + Sparse in {gwi_time:.2f}s "
              f"({len(new_docs)/gwi_time:.0f} docs/sec)")
    
    # =========================================================================
    # Search Demo
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Search Demo (Dense vs Sparse vs Hybrid)")
    print("=" * 70)
    
    demo_queries = [
        "iPhone 15 Pro Max case",
        "wireless bluetooth headphones noise cancelling",
        "running shoes comfortable",
        "laptop stand adjustable",
        "USB-C hub multiport adapter",
    ]
    
    for q in demo_queries:
        print(f"\n🔍 Query: \"{q}\"")
        
        # Dense search
        if not args.sparse_only:
            start = time.time()
            dense_results = system.search(q, k=3, mode="dense")
            dense_time = (time.time() - start) * 1000
            print(f"\n   📊 Dense Search ({dense_time:.1f}ms):")
            for i, r in enumerate(dense_results, 1):
                source_icon = "🔥" if r.source == "gwi" else "❄️"
                meta = system.doc_metadata.get(r.doc_id, {})
                title = meta.get("title", "Unknown")[:30]
                print(f"      {i}. [{source_icon}] {r.doc_id} - {title}... (score={r.score:.4f})")
        
        # Sparse search
        if not args.dense_only:
            start = time.time()
            sparse_results = system.search(q, k=3, mode="sparse")
            sparse_time = (time.time() - start) * 1000
            print(f"\n   📝 Sparse Search ({sparse_time:.1f}ms):")
            for i, r in enumerate(sparse_results, 1):
                meta = system.doc_metadata.get(r.doc_id, {})
                title = meta.get("title", "Unknown")[:30]
                print(f"      {i}. [📝] {r.doc_id} - {title}... (score={r.score:.4f})")
        
        # Hybrid search
        if not args.sparse_only and not args.dense_only:
            start = time.time()
            hybrid_results = system.search(q, k=3, mode="hybrid")
            hybrid_time = (time.time() - start) * 1000
            print(f"\n   🔀 Hybrid Search ({hybrid_time:.1f}ms):")
            for i, r in enumerate(hybrid_results, 1):
                meta = system.doc_metadata.get(r.doc_id, {})
                title = meta.get("title", "Unknown")[:30]
                print(f"      {i}. [🔀] {r.doc_id} - {title}... (RRF={r.score:.4f})")
    
    # =========================================================================
    # Phase 3: Evaluation
    # =========================================================================
    if not args.skip_eval:
        print("\n" + "=" * 70)
        print("  Phase 3: Retrieval Evaluation")
        print("=" * 70)
        
        modes_to_eval = []
        if not args.sparse_only:
            modes_to_eval.append("dense")
        if not args.dense_only:
            modes_to_eval.append("sparse")
        if not args.sparse_only and not args.dense_only:
            modes_to_eval.append("hybrid")
        
        all_metrics = {}
        for mode in modes_to_eval:
            metrics = evaluate_retrieval(system, queries, k_values=[1, 5, 10, 20], mode=mode)
            all_metrics[mode] = metrics
        
        print("\n📊 Retrieval Metrics Comparison:")
        print("-" * 60)
        print(f"{'Metric':<15}", end="")
        for mode in modes_to_eval:
            print(f"{mode.capitalize():<15}", end="")
        print()
        print("-" * 60)
        
        for metric in ["mrr", "recall@1", "recall@5", "recall@10", "recall@20"]:
            print(f"{metric:<15}", end="")
            for mode in modes_to_eval:
                print(f"{all_metrics[mode][metric]:<15.4f}", end="")
            print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    
    stats = system.get_stats()
    print(f"\n📈 System Statistics:")
    print(f"   GWI (real-time dense):    {stats['gwi_count']:,} vectors")
    print(f"   Cascade (historical):     {stats['cascade_count']:,} vectors")
    print(f"   Sparse (lexical):         {stats['sparse_count']:,} documents")
    print(f"   Sparse vocabulary:        {stats['sparse_terms']:,} terms")
    print(f"   Sparse postings:          {stats['sparse_postings']:,} entries")
    print(f"   Total documents:          {stats['total_docs']:,}")
    
    print(f"\n⚡ Performance:")
    if args.batch_mode:
        print(f"   Embedding (BGE-M3):       {embed_rate:.0f} docs/sec")
        print(f"   Cascade insert (pure):    {cascade_rate:,.0f} vectors/sec")
        print(f"   GWI insert (pure):        {gwi_rate:,.0f} vectors/sec")
        print(f"   SVS insert (pure):        {svs_rate:,.0f} sparse vectors/sec")
    else:
        print(f"   Cascade + Sparse ingestion: {len(historical_docs)/cascade_time:.0f} docs/sec")
        print(f"   GWI + Sparse ingestion:     {len(new_docs)/gwi_time:.0f} docs/sec")
    
    # Cleanup
    system.close()
    
    print("\n✅ Demo complete!")
    print(f"   Data stored in: {args.data_dir}")


if __name__ == "__main__":
    main()
