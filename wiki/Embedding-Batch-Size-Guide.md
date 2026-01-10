# Embedding Batch Size Guide

When using SynaDB with embedding models, understanding batch sizes is critical for performance.

## The Two Bottlenecks

| Operation | Speed | Bottleneck |
|-----------|-------|------------|
| Embedding generation | 100-500 docs/sec | GPU/CPU compute |
| SynaDB vector insertion | 3,000-32,000+ vectors/sec | Disk I/O |

**Key insight:** SynaDB is rarely the bottleneck. Embedding models are 10-100x slower.

## What is Batch Size?

`batch_size` tells the embedding model how many texts to process in a single forward pass.

```python
# Process 32 texts at once
embeddings = model.encode(texts, batch_size=32)
```

## Recommended Batch Sizes

| Model | Dimensions | Parameters | Recommended Batch Size |
|-------|------------|------------|------------------------|
| all-MiniLM-L6-v2 | 384 | 22M | 128-256 |
| all-mpnet-base-v2 | 768 | 110M | 64-128 |
| BAAI/bge-base-en | 768 | 110M | 64-128 |
| BAAI/bge-large-en | 1024 | 335M | 32-64 |
| **BAAI/bge-m3** | 1024 | 560M | **32** |
| OpenAI ada-002 | 1536 | API | N/A (API handles it) |

## GPU Memory Requirements

| Batch Size | ~VRAM for 768-dim model | ~VRAM for 1024-dim model |
|------------|-------------------------|--------------------------|
| 32 | 2-4 GB | 4-6 GB |
| 64 | 4-8 GB | 8-12 GB |
| 128 | 8-16 GB | 16-24 GB |
| 256 | 16-32 GB | 32-48 GB |

## What Happens When Batch Size is Too Large?

```
# Too large for GPU memory
batch_size=256 with BGE-M3 (1024 dims)

Result:
- GPU runs out of memory
- Falls back to CPU or memory swapping
- 67 seconds per batch instead of 0.3 seconds
- 83 minutes total instead of 3 minutes
```

## Best Practices

### 1. Start Small, Scale Up
```python
# Start with 32, increase if GPU has headroom
batch_size = 32
```

### 2. Monitor GPU Memory
```bash
# Linux/Mac
nvidia-smi -l 1

# Windows
nvidia-smi -l 1
```

### 3. Separate Embedding from Insertion

For maximum throughput, pre-embed everything first:

```python
# Phase 1: Batch embed (GPU-bound)
embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=True)

# Phase 2: Bulk insert (SynaDB speed)
for doc, emb in zip(docs, embeddings):
    gwi.insert(doc.id, emb)  # 32,000+ vectors/sec
```

### 4. Use Appropriate Index for Your Use Case

| Index | Insert Speed | Search Speed | Best For |
|-------|--------------|--------------|----------|
| GWI | 32,000+ vec/sec | Fast | Real-time streaming |
| Cascade | 3,000+ vec/sec | Very fast | Historical data |
| VectorStore | 500-1,000 vec/sec | Medium | Small datasets |

## Real-World Benchmark

From our BGE-M3 + SQuAD benchmark (18,891 documents):

```
Phase 1: Embedding
  - Model: BAAI/bge-m3 (1024 dims)
  - Batch size: 32
  - Time: 116.75s
  - Rate: 162 docs/sec

Phase 2: Cascade Insert (pre-embedded)
  - Vectors: 15,112
  - Time: 4.6s
  - Rate: 3,272 vectors/sec

Phase 3: GWI Insert (pre-embedded)
  - Vectors: 3,779
  - Time: 0.117s
  - Rate: 32,332 vectors/sec
```

**Conclusion:** Embedding is 200x slower than GWI insertion. Optimize your embedding pipeline first.

## Common Mistakes

❌ **Using batch_size=256 with large models**
```python
# BAD: Will cause memory issues with BGE-M3
model.encode(texts, batch_size=256)
```

✅ **Match batch size to model size**
```python
# GOOD: Appropriate for BGE-M3
model.encode(texts, batch_size=32)
```

❌ **Embedding one at a time**
```python
# BAD: 10x slower than batching
for text in texts:
    emb = model.encode(text)
    index.insert(key, emb)
```

✅ **Batch embed, then bulk insert**
```python
# GOOD: Maximize both GPU and SynaDB throughput
embeddings = model.encode(texts, batch_size=32)
for key, emb in zip(keys, embeddings):
    index.insert(key, emb)
```

## See Also

- [Getting Started](Getting-Started.md)
- [API Reference](API-Reference.md)
- [Architecture](Architecture.md)
