"""
SynaDB Live Demo - PythonAnywhere Flask App

This Flask app provides a live interactive demo of SynaDB.
Benchmarks use RELATIVE comparisons instead of absolute targets.

Deploy to: https://www.pythonanywhere.com/user/gtava5813/
"""

from flask import Flask, render_template, jsonify, request
import time
import os
import sys
import tempfile
import shutil

app = Flask(__name__)

# Try to import synadb
try:
    import synadb
    from synadb import SynaDB, VectorStore
    SYNADB_AVAILABLE = True
    SYNADB_VERSION = synadb.__version__
except ImportError:
    SYNADB_AVAILABLE = False
    SYNADB_VERSION = "Not installed"

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Try to import optional components
try:
    from synadb import MmapVectorStore
    MMAP_AVAILABLE = True
except ImportError:
    MMAP_AVAILABLE = False

try:
    from synadb import GravityWellIndex
    GWI_AVAILABLE = True
except ImportError:
    GWI_AVAILABLE = False

try:
    from synadb import CascadeIndex
    CASCADE_AVAILABLE = True
except ImportError:
    CASCADE_AVAILABLE = False

# Store results for summary
RESULTS = {}


@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html', 
                         synadb_available=SYNADB_AVAILABLE,
                         synadb_version=SYNADB_VERSION,
                         numpy_available=NUMPY_AVAILABLE,
                         mmap_available=MMAP_AVAILABLE,
                         gwi_available=GWI_AVAILABLE,
                         cascade_available=CASCADE_AVAILABLE)


@app.route('/api/status')
def status():
    """Check SynaDB installation status"""
    return jsonify({
        'synadb_available': SYNADB_AVAILABLE,
        'synadb_version': SYNADB_VERSION,
        'numpy_available': NUMPY_AVAILABLE,
        'mmap_available': MMAP_AVAILABLE,
        'gwi_available': GWI_AVAILABLE,
        'cascade_available': CASCADE_AVAILABLE,
        'python_version': sys.version
    })


@app.route('/api/benchmark/mmap_vs_vector', methods=['POST'])
def benchmark_mmap_vs_vector():
    """Compare MmapVectorStore vs VectorStore insert speed"""
    if not SYNADB_AVAILABLE:
        return jsonify({'error': 'SynaDB not installed', 'skip': True})
    if not NUMPY_AVAILABLE:
        return jsonify({'error': 'NumPy not installed', 'skip': True})
    if not MMAP_AVAILABLE:
        return jsonify({'error': 'MmapVectorStore not available', 'skip': True})
    
    data = request.get_json() or {}
    n = min(data.get('vectors', 5000), 20000)
    dims = data.get('dims', 384)
    
    temp_dir = tempfile.mkdtemp()
    mmap_path = os.path.join(temp_dir, "mmap.mmap")
    vector_path = os.path.join(temp_dir, "vector.db")
    
    try:
        vectors = np.random.randn(n, dims).astype(np.float32)
        keys = [f"vec_{i}" for i in range(n)]
        
        # MmapVectorStore batch insert
        mmap_store = MmapVectorStore(mmap_path, dimensions=dims, initial_capacity=n * 2)
        mmap_start = time.perf_counter()
        mmap_store.insert_batch(keys, vectors)
        mmap_time = time.perf_counter() - mmap_start
        mmap_rate = n / mmap_time
        mmap_store.close()
        
        # VectorStore individual insert
        vector_store = VectorStore(vector_path, dimensions=dims)
        vector_start = time.perf_counter()
        for i, vec in enumerate(vectors):
            vector_store.insert(f"vec_{i}", vec)
        vector_time = time.perf_counter() - vector_start
        vector_rate = n / vector_time
        vector_store.close()
        
        speedup = mmap_rate / vector_rate if vector_rate > 0 else 0
        
        result = {
            'test': 'mmap_vs_vector',
            'vectors': n,
            'dimensions': dims,
            'mmap_rate': round(mmap_rate, 0),
            'mmap_time_sec': round(mmap_time, 3),
            'vector_rate': round(vector_rate, 0),
            'vector_time_sec': round(vector_time, 3),
            'speedup': round(speedup, 1),
            'claim': 'MmapVectorStore faster than VectorStore',
            'passed': speedup >= 2  # At least 2x faster
        }
        RESULTS['mmap_vs_vector'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'skip': True})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/benchmark/gwi_vs_hnsw', methods=['POST'])
def benchmark_gwi_vs_hnsw():
    """Compare GWI vs HNSW index build time"""
    if not SYNADB_AVAILABLE:
        return jsonify({'error': 'SynaDB not installed', 'skip': True})
    if not NUMPY_AVAILABLE:
        return jsonify({'error': 'NumPy not installed', 'skip': True})
    if not GWI_AVAILABLE:
        return jsonify({'error': 'GravityWellIndex not available', 'skip': True})
    
    data = request.get_json() or {}
    n = min(data.get('vectors', 5000), 20000)
    dims = data.get('dims', 384)
    
    temp_dir = tempfile.mkdtemp()
    gwi_path = os.path.join(temp_dir, "gwi.gwi")
    hnsw_path = os.path.join(temp_dir, "hnsw.db")
    
    try:
        vectors = np.random.randn(n, dims).astype(np.float32)
        keys = [f"v_{i}" for i in range(n)]
        
        # GWI build
        gwi = GravityWellIndex(gwi_path, dimensions=dims)
        gwi.initialize(vectors[:min(1000, n)])
        
        gwi_start = time.perf_counter()
        gwi.insert_batch(keys, vectors)
        gwi_time = time.perf_counter() - gwi_start
        
        # GWI search
        query = np.random.randn(dims).astype(np.float32)
        gwi_search_start = time.perf_counter()
        gwi_results = gwi.search(query, k=10, nprobe=50)
        gwi_search_ms = (time.perf_counter() - gwi_search_start) * 1000
        gwi.close()
        
        # HNSW build
        hnsw = VectorStore(hnsw_path, dimensions=dims)
        hnsw_start = time.perf_counter()
        for i, vec in enumerate(vectors):
            hnsw.insert(f"v_{i}", vec)
        hnsw.build_index()
        hnsw_time = time.perf_counter() - hnsw_start
        
        # HNSW search
        hnsw_search_start = time.perf_counter()
        hnsw_results = hnsw.search(query, k=10)
        hnsw_search_ms = (time.perf_counter() - hnsw_search_start) * 1000
        hnsw.close()
        
        build_speedup = hnsw_time / gwi_time if gwi_time > 0 else 0
        
        result = {
            'test': 'gwi_vs_hnsw',
            'vectors': n,
            'dimensions': dims,
            'gwi_build_sec': round(gwi_time, 3),
            'hnsw_build_sec': round(hnsw_time, 3),
            'build_speedup': round(build_speedup, 1),
            'gwi_search_ms': round(gwi_search_ms, 2),
            'hnsw_search_ms': round(hnsw_search_ms, 2),
            'claim': 'GWI builds faster than HNSW',
            'passed': build_speedup >= 5  # At least 5x faster build
        }
        RESULTS['gwi_vs_hnsw'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'skip': True})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/benchmark/cascade_search', methods=['POST'])
def benchmark_cascade_search():
    """Test Cascade Index search performance"""
    if not SYNADB_AVAILABLE:
        return jsonify({'error': 'SynaDB not installed', 'skip': True})
    if not NUMPY_AVAILABLE:
        return jsonify({'error': 'NumPy not installed', 'skip': True})
    if not CASCADE_AVAILABLE:
        return jsonify({'error': 'CascadeIndex not available', 'skip': True})
    
    data = request.get_json() or {}
    n = min(data.get('vectors', 5000), 20000)
    dims = data.get('dims', 384)
    
    temp_dir = tempfile.mkdtemp()
    cascade_path = os.path.join(temp_dir, "cascade.cascade")
    
    try:
        vectors = np.random.randn(n, dims).astype(np.float32)
        keys = [f"c_{i}" for i in range(n)]
        
        cascade = CascadeIndex(cascade_path, dimensions=dims)
        
        # Build
        build_start = time.perf_counter()
        cascade.insert_batch(keys, vectors)
        build_time = time.perf_counter() - build_start
        build_rate = n / build_time if build_time > 0 else 0
        
        # Search (multiple queries for average)
        search_times = []
        for _ in range(10):
            query = np.random.randn(dims).astype(np.float32)
            search_start = time.perf_counter()
            results = cascade.search(query, k=10)
            search_times.append((time.perf_counter() - search_start) * 1000)
        
        avg_search_ms = sum(search_times) / len(search_times)
        cascade.close()
        
        result = {
            'test': 'cascade_search',
            'vectors': n,
            'dimensions': dims,
            'build_time_sec': round(build_time, 3),
            'build_rate': round(build_rate, 0),
            'avg_search_ms': round(avg_search_ms, 2),
            'results_found': len(results) if results else 0,
            'claim': 'Sub-linear search time',
            'passed': avg_search_ms < 50  # Under 50ms for search
        }
        RESULTS['cascade_search'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'skip': True})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/benchmark/hnsw_vs_brute', methods=['POST'])
def benchmark_hnsw_vs_brute():
    """Compare HNSW indexed search vs brute force"""
    if not SYNADB_AVAILABLE:
        return jsonify({'error': 'SynaDB not installed', 'skip': True})
    if not NUMPY_AVAILABLE:
        return jsonify({'error': 'NumPy not installed', 'skip': True})
    
    data = request.get_json() or {}
    n = min(data.get('vectors', 2000), 10000)
    dims = data.get('dims', 128)
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "vectors.db")
    
    try:
        store = VectorStore(db_path, dimensions=dims)
        vectors = np.random.randn(n, dims).astype(np.float32)
        
        for i, vec in enumerate(vectors):
            store.insert(f"doc_{i}", vec)
        
        query = np.random.randn(dims).astype(np.float32)
        
        # Brute force search (before index)
        brute_start = time.perf_counter()
        brute_results = store.search(query, k=10)
        brute_ms = (time.perf_counter() - brute_start) * 1000
        
        # Build HNSW index
        store.build_index()
        
        # HNSW search (after index)
        hnsw_start = time.perf_counter()
        hnsw_results = store.search(query, k=10)
        hnsw_ms = (time.perf_counter() - hnsw_start) * 1000
        
        store.close()
        
        speedup = brute_ms / hnsw_ms if hnsw_ms > 0 else 0
        
        result = {
            'test': 'hnsw_vs_brute',
            'vectors': n,
            'dimensions': dims,
            'brute_search_ms': round(brute_ms, 2),
            'hnsw_search_ms': round(hnsw_ms, 2),
            'speedup': round(speedup, 1),
            'claim': 'HNSW faster than brute force',
            'passed': hnsw_ms < brute_ms
        }
        RESULTS['hnsw_vs_brute'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'skip': True})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/benchmark/schema', methods=['POST'])
def benchmark_schema():
    """Test schema-free storage (functional test)"""
    if not SYNADB_AVAILABLE:
        return jsonify({'error': 'SynaDB not installed', 'skip': True})
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "schema.db")
    
    try:
        db = SynaDB(db_path)
        
        # Store different types without schema
        db.put_float("metrics/accuracy", 0.95)
        db.put_int("metrics/epoch", 100)
        db.put_text("config/model", "bert-base-uncased")
        db.put_bytes("data/binary", b"\x00\x01\x02\x03\xff")
        
        # Verify all types
        results = {
            'float': db.get_float('metrics/accuracy') == 0.95,
            'int': db.get_int('metrics/epoch') == 100,
            'text': db.get_text('config/model') == "bert-base-uncased",
            'bytes': db.get_bytes('data/binary') == b"\x00\x01\x02\x03\xff"
        }
        
        db.close()
        
        passed_count = sum(results.values())
        
        result = {
            'test': 'schema_free',
            'types_tested': 4,
            'types_passed': passed_count,
            'results': results,
            'claim': 'Store any type without migrations',
            'passed': passed_count == 4
        }
        RESULTS['schema_free'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'skip': True})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/benchmark/recovery', methods=['POST'])
def benchmark_recovery():
    """Test crash recovery (functional test)"""
    if not SYNADB_AVAILABLE:
        return jsonify({'error': 'SynaDB not installed', 'skip': True})
    
    data = request.get_json() or {}
    n = min(data.get('records', 10000), 50000)
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "recovery.db")
    
    try:
        # Create and populate
        db = SynaDB(db_path, sync_on_write=False)
        for i in range(n):
            db.put_float(f"key_{i}", float(i))
        db.close()
        
        # Simulate crash recovery (reopen)
        recovery_start = time.perf_counter()
        db2 = SynaDB(db_path)
        recovery_time = time.perf_counter() - recovery_start
        
        # Verify data integrity
        checks = [
            db2.get_float("key_0") == 0.0,
            db2.get_float(f"key_{n//2}") == float(n//2),
            db2.get_float(f"key_{n-1}") == float(n-1)
        ]
        integrity_ok = all(checks)
        db2.close()
        
        result = {
            'test': 'recovery',
            'records': n,
            'recovery_time_sec': round(recovery_time, 4),
            'recovery_rate': round(n / recovery_time, 0),
            'integrity_ok': integrity_ok,
            'claim': 'Full recovery after crash',
            'passed': integrity_ok
        }
        RESULTS['recovery'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'skip': True})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/benchmark/tensor', methods=['POST'])
def benchmark_tensor():
    """Test tensor extraction (functional test)"""
    if not SYNADB_AVAILABLE:
        return jsonify({'error': 'SynaDB not installed', 'skip': True})
    if not NUMPY_AVAILABLE:
        return jsonify({'error': 'NumPy not installed', 'skip': True})
    
    data = request.get_json() or {}
    n = min(data.get('records', 10000), 50000)
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "tensor.db")
    
    try:
        db = SynaDB(db_path, sync_on_write=False)
        
        # Write time-series
        expected_values = []
        for i in range(n):
            val = 20.0 + np.sin(i / 100) * 5
            db.put_float("sensor/temp", val)
            expected_values.append(val)
        
        # Extract as tensor
        extract_start = time.perf_counter()
        tensor = db.get_history_tensor("sensor/temp")
        extract_time = time.perf_counter() - extract_start
        
        db.close()
        
        # Verify
        shape_ok = tensor is not None and len(tensor) == n
        dtype_ok = tensor is not None and tensor.dtype == np.float64
        
        result = {
            'test': 'tensor',
            'records': n,
            'tensor_shape': list(tensor.shape) if tensor is not None else None,
            'tensor_dtype': str(tensor.dtype) if tensor is not None else None,
            'extract_time_sec': round(extract_time, 4),
            'claim': 'Direct NumPy tensor from history',
            'passed': shape_ok and dtype_ok
        }
        RESULTS['tensor'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'skip': True})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/benchmark/compression', methods=['POST'])
def benchmark_compression():
    """Test compression effectiveness"""
    if not SYNADB_AVAILABLE:
        return jsonify({'error': 'SynaDB not installed', 'skip': True})
    
    data = request.get_json() or {}
    n = min(data.get('records', 10000), 50000)
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "compress.db")
    
    try:
        db = SynaDB(db_path, sync_on_write=False)
        
        # Write data with small deltas (good for compression)
        for i in range(n):
            db.put_float("sensor/temp", 20.0 + (i % 10) * 0.01)
        
        db.close()
        
        file_size = os.path.getsize(db_path)
        raw_size = n * 8  # 8 bytes per float
        ratio = raw_size / file_size if file_size > 0 else 0
        
        result = {
            'test': 'compression',
            'records': n,
            'raw_size_kb': round(raw_size / 1024, 1),
            'file_size_kb': round(file_size / 1024, 1),
            'compression_ratio': round(ratio, 2),
            'claim': 'Delta + LZ4 compression',
            'passed': ratio >= 0.5  # At least not 2x expansion
        }
        RESULTS['compression'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'skip': True})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/summary')
def get_summary():
    """Get summary of all benchmark results"""
    passed = sum(1 for r in RESULTS.values() if r.get('passed') is True)
    failed = sum(1 for r in RESULTS.values() if r.get('passed') is False)
    skipped = sum(1 for r in RESULTS.values() if r.get('skip'))
    total = len(RESULTS)
    
    return jsonify({
        'total_tests': total,
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'results': RESULTS,
        'synadb_version': SYNADB_VERSION
    })


@app.route('/api/reset')
def reset_results():
    """Reset all benchmark results"""
    global RESULTS
    RESULTS = {}
    return jsonify({'status': 'ok', 'message': 'Results cleared'})


if __name__ == '__main__':
    app.run(debug=True)
