#!/usr/bin/env python3
"""
Common Voice Dataset Loader Demo

This demo shows how to:
- Download audio samples from Common Voice (subset)
- Extract MFCC features using librosa
- Store features as float tensors in Syna

Note: This demo uses a small subset due to Common Voice's size.
      For full dataset, you'll need to accept the license on HuggingFace.

Run with: python common_voice.py
"""

import os
import sys
import time
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB


def extract_mfcc_features(audio_array, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from audio.
    
    Args:
        audio_array: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        MFCC features as numpy array (n_mfcc, time_frames)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for MFCC extraction. Install with: pip install librosa")
    
    # Ensure audio is float32
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio_array,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return mfccs


def main():
    print("=== Common Voice Dataset Loader Demo ===\n")
    
    # Check dependencies
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return 1
    
    try:
        import librosa
    except ImportError:
        print("Error: 'librosa' library not installed.")
        print("Install with: pip install librosa")
        return 1
    
    db_path = "common_voice_Syna.db"
    
    # Clean up existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        # 1. Load Common Voice subset
        # Using a smaller, freely available speech dataset as alternative
        print("1. Loading speech dataset from HuggingFace...")
        print("   (Using 'speech_commands' as a freely available alternative)")
        start = time.time()
        
        # speech_commands is smaller and doesn't require authentication
        dataset = load_dataset(
            "speech_commands",
            "v0.02",
            split="train[:100]"  # Small subset for demo
        )
        
        load_time = time.time() - start
        print(f"   ✓ Loaded {len(dataset)} audio samples in {load_time:.2f}s\n")

        # 2. Extract MFCC features and store
        print("2. Extracting MFCC features and storing...")
        start = time.time()
        
        total_features = 0
        feature_shapes = []
        
        with SynaDB(db_path) as db:
            for i, sample in enumerate(dataset):
                # Get audio data
                audio = sample['audio']
                audio_array = np.array(audio['array'], dtype=np.float32)
                sample_rate = audio['sampling_rate']
                
                # Extract MFCC features
                try:
                    mfccs = extract_mfcc_features(audio_array, sample_rate)
                    feature_shapes.append(mfccs.shape)
                    
                    # Store MFCC features as bytes (flattened float32)
                    mfcc_bytes = mfccs.astype(np.float32).tobytes()
                    db.put_bytes(f"train/mfcc/{i}", mfcc_bytes)
                    
                    # Store shape info for reconstruction
                    db.put_int(f"train/mfcc_rows/{i}", mfccs.shape[0])
                    db.put_int(f"train/mfcc_cols/{i}", mfccs.shape[1])
                    
                    # Store label
                    label = sample.get('label', 0)
                    db.put_int(f"train/label/{i}", label)
                    
                    total_features += mfccs.size
                    
                except Exception as e:
                    print(f"   Warning: Failed to process sample {i}: {e}")
                    continue
                
                if (i + 1) % 20 == 0:
                    print(f"   Processed {i + 1}/{len(dataset)} samples...")
        
        store_time = time.time() - start
        print(f"   ✓ Stored {len(feature_shapes)} samples in {store_time:.2f}s\n")
        
        # 3. Storage statistics
        file_size = os.path.getsize(db_path)
        print("3. Storage statistics:")
        print(f"   Database size: {file_size / 1024:.2f} KB")
        print(f"   Total MFCC values: {total_features}")
        print(f"   Avg features per sample: {total_features / len(feature_shapes):.0f}")
        if feature_shapes:
            avg_shape = (
                sum(s[0] for s in feature_shapes) / len(feature_shapes),
                sum(s[1] for s in feature_shapes) / len(feature_shapes)
            )
            print(f"   Avg MFCC shape: ({avg_shape[0]:.0f}, {avg_shape[1]:.0f})\n")
        
        # 4. Retrieve and verify features
        print("4. Retrieving and verifying features...")
        
        with SynaDB(db_path) as db:
            # Get a sample
            idx = 0
            
            mfcc_bytes = db.get_bytes(f"train/mfcc/{idx}")
            rows = db.get_int(f"train/mfcc_rows/{idx}")
            cols = db.get_int(f"train/mfcc_cols/{idx}")
            label = db.get_int(f"train/label/{idx}")
            
            if mfcc_bytes and rows and cols:
                # Reconstruct MFCC array
                mfccs = np.frombuffer(mfcc_bytes, dtype=np.float32).reshape(rows, cols)
                
                print(f"   Sample {idx}:")
                print(f"     MFCC shape: {mfccs.shape}")
                print(f"     Label: {label}")
                print(f"     MFCC range: [{mfccs.min():.2f}, {mfccs.max():.2f}]")
                print(f"     MFCC mean: {mfccs.mean():.2f}\n")
        
        # 5. Batch retrieval for ML
        print("5. Batch retrieval demo...")
        
        with SynaDB(db_path) as db:
            batch_size = 16
            start = time.time()
            
            batch_features = []
            batch_labels = []
            
            for i in range(min(batch_size, len(feature_shapes))):
                mfcc_bytes = db.get_bytes(f"train/mfcc/{i}")
                rows = db.get_int(f"train/mfcc_rows/{i}")
                cols = db.get_int(f"train/mfcc_cols/{i}")
                label = db.get_int(f"train/label/{i}")
                
                if mfcc_bytes and rows and cols:
                    mfccs = np.frombuffer(mfcc_bytes, dtype=np.float32).reshape(rows, cols)
                    batch_features.append(mfccs)
                    batch_labels.append(label)
            
            batch_time = time.time() - start
            
            print(f"   Loaded {len(batch_features)} samples in {batch_time * 1000:.2f}ms")
            print(f"   Throughput: {len(batch_features) / batch_time:.0f} samples/sec\n")
        
        # 6. Feature statistics
        print("6. Feature statistics across dataset...")
        
        with SynaDB(db_path) as db:
            all_means = []
            all_stds = []
            
            for i in range(min(50, len(feature_shapes))):
                mfcc_bytes = db.get_bytes(f"train/mfcc/{i}")
                rows = db.get_int(f"train/mfcc_rows/{i}")
                cols = db.get_int(f"train/mfcc_cols/{i}")
                
                if mfcc_bytes and rows and cols:
                    mfccs = np.frombuffer(mfcc_bytes, dtype=np.float32).reshape(rows, cols)
                    all_means.append(mfccs.mean())
                    all_stds.append(mfccs.std())
            
            print(f"   Global MFCC mean: {np.mean(all_means):.2f}")
            print(f"   Global MFCC std: {np.mean(all_stds):.2f}\n")
        
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

