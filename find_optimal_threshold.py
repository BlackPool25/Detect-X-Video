#!/usr/bin/env python3
"""
Test on larger sample to find optimal threshold
"""
import json
import torch
import gc
import numpy as np
from pathlib import Path
from pipeline_optimized import DeepfakePipeline

def main():
    # Read metadata
    with open('Test-Video/train_sample_videos/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Test on 30 videos (15 FAKE, 15 REAL) for better threshold analysis
    fake_videos = [k for k, v in metadata.items() if v['label'] == 'FAKE'][:15]
    real_videos = [k for k, v in metadata.items() if v['label'] == 'REAL'][:15]
    test_videos = fake_videos + real_videos
    
    print(f"Testing on {len(test_videos)} videos to find optimal threshold...")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = DeepfakePipeline()
    
    fake_probs = []
    real_probs = []
    
    for i, video_name in enumerate(test_videos, 1):
        video_path = f'Test-Video/train_sample_videos/{video_name}'
        ground_truth = metadata[video_name]['label']
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            result = pipeline.detect(video_path, enable_fail_fast=True)
            
            # Extract Layer 2 details
            layer2 = result.layer_results[0]
            avg_prob = layer2.details.get('avg_fake_probability', 0)
            
            if ground_truth == 'FAKE':
                fake_probs.append(avg_prob)
            else:
                real_probs.append(avg_prob)
            
            print(f"[{i:2d}/{len(test_videos)}] {video_name:20s} | {ground_truth:4s} | AvgProb: {avg_prob:.4f}")
            
        except KeyboardInterrupt:
            print("\n\nTest interrupted")
            break
        except Exception as e:
            print(f"[{i:2d}/{len(test_videos)}] {video_name:20s} | ERROR: {str(e)[:50]}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    
    fake_probs = np.array(fake_probs)
    real_probs = np.array(real_probs)
    
    print(f"\nFAKE videos (n={len(fake_probs)}):")
    print(f"  Mean: {fake_probs.mean():.4f}")
    print(f"  Median: {np.median(fake_probs):.4f}")
    print(f"  Min: {fake_probs.min():.4f}")
    print(f"  Max: {fake_probs.max():.4f}")
    print(f"  Std: {fake_probs.std():.4f}")
    
    print(f"\nREAL videos (n={len(real_probs)}):")
    print(f"  Mean: {real_probs.mean():.4f}")
    print(f"  Median: {np.median(real_probs):.4f}")
    print(f"  Min: {real_probs.min():.4f}")
    print(f"  Max: {real_probs.max():.4f}")
    print(f"  Std: {real_probs.std():.4f}")
    
    # Test different thresholds
    print("\n" + "=" * 80)
    print("THRESHOLD PERFORMANCE")
    print("=" * 80)
    
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    
    for thresh in thresholds:
        fake_correct = (fake_probs > thresh).sum()
        real_correct = (real_probs <= thresh).sum()
        total_correct = fake_correct + real_correct
        accuracy = 100 * total_correct / (len(fake_probs) + len(real_probs))
        
        print(f"\nThreshold = {thresh:.2f}:")
        print(f"  Fake Accuracy: {100*fake_correct/len(fake_probs):.1f}% ({fake_correct}/{len(fake_probs)})")
        print(f"  Real Accuracy: {100*real_correct/len(real_probs):.1f}% ({real_correct}/{len(real_probs)})")
        print(f"  Overall: {accuracy:.1f}%")
    
    # Find optimal threshold (maximize accuracy)
    best_thresh = 0.35
    best_acc = 0
    
    for thresh in np.linspace(0.30, 0.60, 31):
        fake_correct = (fake_probs > thresh).sum()
        real_correct = (real_probs <= thresh).sum()
        total_correct = fake_correct + real_correct
        accuracy = total_correct / (len(fake_probs) + len(real_probs))
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_thresh = thresh
    
    print(f"\n{'=' * 80}")
    print(f"OPTIMAL THRESHOLD: {best_thresh:.4f} (Accuracy: {100*best_acc:.1f}%)")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
