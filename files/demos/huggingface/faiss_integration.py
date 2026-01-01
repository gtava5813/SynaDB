"""
FAISS integration demo:  Use SynaDB for storage, FAISS for search. 
"""
import faiss
import numpy as np
from synadb import SynaDB, Atom

class HybridVectorStore:
    """
    Hybrid store using SynaDB for persistence and FAISS for search. 
    
    - SynaDB handles:  persistence, crash recovery, metadata storage
    - FAISS handles:  fast similarity search, GPU acceleration
    """
    
    def __init__(self, db_path: str, dimensions: int, use_gpu: bool = False):
        self.db = SynaDB(db_path)
        self.dimensions = dimensions
        self.use_gpu = use_gpu
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(dimensions)  # Inner product
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.keys:  List[str] = []
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild FAISS index from SynaDB on startup."""
        vectors = []
        for key in self.db.keys():
            if key.startswith("vec/"):
                atom = self.db.get(key)
                if isinstance(atom, Atom) and atom.is_vector():
                    vectors.append(atom.as_vector())
                    self.keys.append(key. replace("vec/", ""))
        
        if vectors:
            matrix = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(matrix)  # For cosine similarity
            self.index. add(matrix)
    
    def insert(self, key: str, vector: np.ndarray):
        """Insert vector into both SynaDB and FAISS."""
        # Persist to SynaDB
        self.db.put_vector(f"vec/{key}", vector. tolist())
        
        # Add to FAISS index
        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.keys.append(key)
    
    def search(self, query: np.ndarray, k: int = 10) -> List[tuple]:
        """Search using FAISS, return keys from SynaDB."""
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        
        distances, indices = self.index.search(q, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.keys):
                results.append((self.keys[idx], float(dist)))
        
        return results
    
    def save(self):
        """Save FAISS index to disk alongside SynaDB."""
        faiss.write_index(
            faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index,
            self.db.path. replace(". db", ". faiss")
        )


# Usage example
if __name__ == "__main__":
    store = HybridVectorStore("embeddings.db", dimensions=768, use_gpu=True)
    
    # Insert embeddings
    for i in range(100000):
        embedding = np.random.randn(768).astype(np.float32)
        store.insert(f"doc{i}", embedding)
    
    # Fast search with GPU
    query = np.random.randn(768).astype(np.float32)
    results = store. search(query, k=10)
    for key, score in results:
        print(f"{key}: {score:. 4f}")