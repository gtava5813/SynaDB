from typing import List, Optional, Literal
import numpy as np

class VectorStore:
    def __init__(
        self,
        path: str,
        dimensions: int,
        metric: Literal["cosine", "euclidean", "dot_product"] = "cosine",
        backend: Literal["hnsw", "faiss", "auto"] = "auto",
        faiss_index_type: str = "IVF1024,Flat",  # FAISS index factory string
        faiss_nprobe: int = 10,
        use_gpu: bool = False,
    ):
        """
        Create a vector store with configurable backend. 
        
        Args:
            path: Database file path
            dimensions: Vector dimensionality
            metric: Distance metric
            backend: Index backend ("hnsw", "faiss", or "auto")
            faiss_index_type: FAISS index factory string (if using FAISS)
            faiss_nprobe: Number of probes for IVF search
            use_gpu: Use GPU acceleration (FAISS only)
        
        Examples:
            # Use built-in HNSW
            store = VectorStore("vectors.db", 768, backend="hnsw")
            
            # Use FAISS with IVF+PQ for billion-scale
            store = VectorStore(
                "vectors.db", 768, 
                backend="faiss",
                faiss_index_type="IVF4096,PQ32",
                use_gpu=True
            )
            
            # Use FAISS HNSW (different implementation than built-in)
            store = VectorStore(
                "vectors.db", 768,
                backend="faiss", 
                faiss_index_type="HNSW32"
            )
        """
        pass  # Implementation calls Rust FFI