# SynaDB PythonAnywhere Demo

Live interactive demo of SynaDB running on PythonAnywhere.

**Live Demo:** https://gtava5813.pythonanywhere.com/

## Benchmarks

This demo uses **relative benchmarks** that are hardware-independent:

### Relative Comparisons

| Benchmark | Comparison | Claim |
|-----------|------------|-------|
| Mmap vs VectorStore | MmapVectorStore / VectorStore | Batch insert faster |
| GWI vs HNSW | HNSW build / GWI build | GWI builds faster |
| HNSW vs Brute Force | Brute force / HNSW search | HNSW search faster |

### Functional Tests

| Test | Claim |
|------|-------|
| Schema-Free | Store Float, Int, Text, Bytes without migrations |
| Crash Recovery | Full data recovery after reopen |
| Tensor Extraction | Direct NumPy tensor from history |
| Compression | Delta + LZ4 reduces storage |

## Deployment

### Files to Upload

1. `flask_app.py` - Main Flask application
2. `templates/index.html` - Demo page template
3. `requirements.txt` - Python dependencies

### Setup on PythonAnywhere

1. Create a new web app (Flask, Python 3.10+)
2. Upload files to your web app directory
3. Install dependencies: `pip install -r requirements.txt`
4. Set the WSGI file to point to `flask_app.py`
5. Reload the web app

### Requirements

```
flask>=2.0
numpy>=1.20
synadb>=1.0.4
```

## Local Development

```bash
cd demos/pythonanywhere
pip install -r requirements.txt
python flask_app.py
```

Open http://localhost:5000 in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Demo page |
| `/api/status` | GET | Check SynaDB installation |
| `/api/benchmark/mmap_vs_vector` | POST | MmapVectorStore vs VectorStore |
| `/api/benchmark/gwi_vs_hnsw` | POST | GWI vs HNSW build time |
| `/api/benchmark/hnsw_vs_brute` | POST | HNSW vs brute force search |
| `/api/benchmark/cascade_search` | POST | Cascade Index search |
| `/api/benchmark/schema` | POST | Schema-free storage test |
| `/api/benchmark/recovery` | POST | Crash recovery test |
| `/api/benchmark/tensor` | POST | Tensor extraction test |
| `/api/benchmark/compression` | POST | Compression test |
| `/api/summary` | GET | Get all results |
| `/api/reset` | GET | Clear results |
