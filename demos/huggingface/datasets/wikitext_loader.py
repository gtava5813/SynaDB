#!/usr/bin/env python3
"""
WikiText Dataset Loader Demo

This demo shows how to:
- Download WikiText-2 from HuggingFace datasets
- Tokenize using HuggingFace tokenizer
- Store token sequences in Syna

Run with: python wikitext_loader.py
"""

import os
import sys
import time
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB


def main():
    print("=== WikiText Dataset Loader Demo ===\n")
    
    # Check dependencies
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return 1
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: 'transformers' library not installed.")
        print("Install with: pip install transformers")
        return 1
    
    db_path = "wikitext_Syna.db"
    
    # Clean up existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        # 1. Load WikiText-2 from HuggingFace
        print("1. Loading WikiText-2 from HuggingFace...")
        start = time.time()
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]")
        load_time = time.time() - start
        print(f"   ✓ Loaded {len(dataset)} text samples in {load_time:.2f}s\n")
        
        # 2. Load tokenizer
        print("2. Loading GPT-2 tokenizer...")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer_time = time.time() - start
        print(f"   ✓ Tokenizer loaded in {tokenizer_time:.2f}s")
        print(f"   Vocabulary size: {tokenizer.vocab_size}\n")
        
        # 3. Tokenize and store
        print("3. Tokenizing and storing in Syna...")
        start = time.time()
        
        total_tokens = 0
        stored_count = 0
        
        with SynaDB(db_path) as db:
            for i, sample in enumerate(dataset):
                text = sample['text'].strip()
                
                # Skip empty lines
                if not text:
                    continue
                
                # Tokenize
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                if len(tokens) == 0:
                    continue
                
                # Store token sequence as bytes (int32 array)
                token_array = np.array(tokens, dtype=np.int32)
                db.put_bytes(f"train/tokens/{stored_count}", token_array.tobytes())
                
                # Store original text for reference
                db.put_text(f"train/text/{stored_count}", text)
                
                # Store sequence length
                db.put_int(f"train/length/{stored_count}", len(tokens))
                
                total_tokens += len(tokens)
                stored_count += 1
                
                if (stored_count) % 100 == 0:
                    print(f"   Stored {stored_count} sequences...")
        
        store_time = time.time() - start
        print(f"   ✓ Stored {stored_count} sequences ({total_tokens} tokens) in {store_time:.2f}s\n")

        # 4. Storage statistics
        file_size = os.path.getsize(db_path)
        print("4. Storage statistics:")
        print(f"   Database size: {file_size / 1024:.2f} KB")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Sequences stored: {stored_count}")
        print(f"   Avg tokens per sequence: {total_tokens / stored_count:.1f}")
        print(f"   Bytes per token: {file_size / total_tokens:.2f}\n")
        
        # 5. Retrieve and decode sample sequences
        print("5. Sample sequences:")
        
        with SynaDB(db_path) as db:
            for idx in range(min(3, stored_count)):
                # Get tokens
                token_bytes = db.get_bytes(f"train/tokens/{idx}")
                tokens = np.frombuffer(token_bytes, dtype=np.int32).tolist()
                
                # Get original text
                original = db.get_text(f"train/text/{idx}")
                
                # Decode tokens back to text
                decoded = tokenizer.decode(tokens)
                
                # Truncate for display
                original_preview = original[:80] + "..." if len(original) > 80 else original
                decoded_preview = decoded[:80] + "..." if len(decoded) > 80 else decoded
                
                print(f"   [{idx}] {len(tokens)} tokens")
                print(f"       Original: \"{original_preview}\"")
                print(f"       Decoded:  \"{decoded_preview}\"")
                print(f"       Match: {original.strip() == decoded.strip()}\n")
        
        # 6. Token distribution analysis
        print("6. Token distribution analysis...")
        
        with SynaDB(db_path) as db:
            all_tokens = []
            sequence_lengths = []
            
            for i in range(min(100, stored_count)):
                token_bytes = db.get_bytes(f"train/tokens/{i}")
                if token_bytes:
                    tokens = np.frombuffer(token_bytes, dtype=np.int32).tolist()
                    all_tokens.extend(tokens)
                    sequence_lengths.append(len(tokens))
            
            # Most common tokens
            from collections import Counter
            token_counts = Counter(all_tokens)
            most_common = token_counts.most_common(10)
            
            print("   Most common tokens:")
            for token_id, count in most_common:
                token_str = tokenizer.decode([token_id])
                print(f"     {token_id:5d} ({repr(token_str):15s}): {count}")
            
            print(f"\n   Sequence length stats:")
            print(f"     Min: {min(sequence_lengths)}")
            print(f"     Max: {max(sequence_lengths)}")
            print(f"     Mean: {np.mean(sequence_lengths):.1f}")
            print(f"     Median: {np.median(sequence_lengths):.1f}\n")
        
        # 7. Batch retrieval for language modeling
        print("7. Batch retrieval demo...")
        
        with SynaDB(db_path) as db:
            batch_size = 32
            max_length = 128
            
            start = time.time()
            
            # Collect sequences
            sequences = []
            for i in range(min(batch_size, stored_count)):
                token_bytes = db.get_bytes(f"train/tokens/{i}")
                if token_bytes:
                    tokens = np.frombuffer(token_bytes, dtype=np.int32)
                    # Truncate or pad to max_length
                    if len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    sequences.append(tokens)
            
            # Pad sequences to same length
            padded = np.zeros((len(sequences), max_length), dtype=np.int32)
            for i, seq in enumerate(sequences):
                padded[i, :len(seq)] = seq
            
            batch_time = time.time() - start
            
            print(f"   Batch shape: {padded.shape}")
            print(f"   Batch load time: {batch_time * 1000:.2f}ms")
            print(f"   Throughput: {len(sequences) / batch_time:.0f} sequences/sec\n")
        
        # 8. Verify round-trip integrity
        print("8. Verifying tokenization round-trip...")
        
        with SynaDB(db_path) as db:
            test_indices = [0, 10, 50, 100]
            all_match = True
            
            for idx in test_indices:
                if idx >= stored_count:
                    continue
                    
                # Get stored tokens
                token_bytes = db.get_bytes(f"train/tokens/{idx}")
                stored_tokens = np.frombuffer(token_bytes, dtype=np.int32).tolist()
                
                # Get original text and re-tokenize
                original = db.get_text(f"train/text/{idx}")
                fresh_tokens = tokenizer.encode(original, add_special_tokens=False)
                
                if stored_tokens != fresh_tokens:
                    print(f"   ✗ Token mismatch at index {idx}")
                    all_match = False
            
            if all_match:
                print(f"   ✓ All tested sequences verified correctly\n")
        
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

