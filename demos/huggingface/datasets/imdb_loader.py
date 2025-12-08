#!/usr/bin/env python3
"""
IMDb Dataset Loader Demo

This demo shows how to:
- Download IMDb reviews from HuggingFace datasets
- Store text reviews as Text atoms
- Store sentiment labels as Int
- Show text search patterns

Run with: python imdb_loader.py
"""

import os
import sys
import time

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB


SENTIMENT_LABELS = ['negative', 'positive']


def main():
    print("=== IMDb Dataset Loader Demo ===\n")
    
    # Check if datasets library is available
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return 1
    
    db_path = "imdb_Syna.db"
    
    # Clean up existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        # 1. Load IMDb from HuggingFace
        print("1. Loading IMDb from HuggingFace...")
        start = time.time()
        dataset = load_dataset("imdb", split="train[:500]")  # First 500 for demo
        load_time = time.time() - start
        print(f"   ✓ Loaded {len(dataset)} reviews in {load_time:.2f}s\n")
        
        # 2. Store in Syna
        print("2. Storing in Syna database...")
        start = time.time()
        
        total_text_bytes = 0
        with SynaDB(db_path) as db:
            for i, sample in enumerate(dataset):
                text = sample['text']
                label = sample['label']
                
                # Store review text
                db.put_text(f"train/review/{i}", text)
                total_text_bytes += len(text.encode('utf-8'))
                
                # Store sentiment label (0=negative, 1=positive)
                db.put_int(f"train/label/{i}", label)
                
                if (i + 1) % 100 == 0:
                    print(f"   Stored {i + 1}/{len(dataset)} reviews...")
        
        store_time = time.time() - start
        print(f"   ✓ Stored {len(dataset)} reviews in {store_time:.2f}s\n")

        # 3. Check storage statistics
        file_size = os.path.getsize(db_path)
        print("3. Storage statistics:")
        print(f"   Database size: {file_size / 1024 / 1024:.2f} MB")
        print(f"   Raw text size: {total_text_bytes / 1024 / 1024:.2f} MB")
        print(f"   Avg review length: {total_text_bytes / len(dataset):.0f} bytes")
        print(f"   Storage overhead: {(file_size - total_text_bytes) / total_text_bytes * 100:.1f}%\n")
        
        # 4. Retrieve and display sample reviews
        print("4. Sample reviews:")
        
        with SynaDB(db_path) as db:
            for idx in [0, 1, 2]:
                text = db.get_text(f"train/review/{idx}")
                label = db.get_int(f"train/label/{idx}")
                
                # Truncate for display
                preview = text[:150] + "..." if len(text) > 150 else text
                preview = preview.replace('\n', ' ')
                
                print(f"   [{idx}] {SENTIMENT_LABELS[label].upper()}")
                print(f"       \"{preview}\"\n")
        
        # 5. Text search patterns demo
        print("5. Text search patterns...")
        
        with SynaDB(db_path) as db:
            # Search for reviews containing specific words
            search_terms = ['excellent', 'terrible', 'boring']
            
            for term in search_terms:
                matches = []
                for i in range(len(dataset)):
                    text = db.get_text(f"train/review/{i}")
                    if text and term.lower() in text.lower():
                        label = db.get_int(f"train/label/{i}")
                        matches.append((i, label))
                
                pos_count = sum(1 for _, l in matches if l == 1)
                neg_count = len(matches) - pos_count
                
                print(f"   '{term}': {len(matches)} matches ({pos_count} positive, {neg_count} negative)")
        print()
        
        # 6. Sentiment distribution
        print("6. Sentiment distribution:")
        
        with SynaDB(db_path) as db:
            positive = 0
            negative = 0
            
            for i in range(len(dataset)):
                label = db.get_int(f"train/label/{i}")
                if label == 1:
                    positive += 1
                else:
                    negative += 1
            
            print(f"   Positive reviews: {positive} ({positive/len(dataset)*100:.1f}%)")
            print(f"   Negative reviews: {negative} ({negative/len(dataset)*100:.1f}%)\n")
        
        # 7. Benchmark text retrieval
        print("7. Text retrieval benchmark...")
        
        with SynaDB(db_path) as db:
            # Measure retrieval time
            start = time.time()
            for i in range(100):
                db.get_text(f"train/review/{i}")
            read_time = time.time() - start
            
            print(f"   100 text reads: {read_time * 1000:.2f}ms")
            print(f"   Throughput: {100 / read_time:.0f} reads/sec\n")
        
        # 8. Verify data integrity
        print("8. Verifying data integrity...")
        
        with SynaDB(db_path) as db:
            test_indices = [0, 42, 100, 250, 499]
            all_match = True
            
            for idx in test_indices:
                stored_text = db.get_text(f"train/review/{idx}")
                stored_label = db.get_int(f"train/label/{idx}")
                
                original_text = dataset[idx]['text']
                original_label = dataset[idx]['label']
                
                if stored_text != original_text:
                    print(f"   ✗ Text mismatch at index {idx}")
                    all_match = False
                if stored_label != original_label:
                    print(f"   ✗ Label mismatch at index {idx}")
                    all_match = False
            
            if all_match:
                print(f"   ✓ All {len(test_indices)} samples verified correctly\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
    
    print("=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

