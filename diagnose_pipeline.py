#!/usr/bin/env python3
"""
Quick diagnostic to understand why pipeline predicts everything as REAL
"""
import json
import torch
import gc
from pathlib import Path
from pipeline_production import DeepfakePipeline

def main():
    # Read metadata
    with open('Test-Video/train_sample_videos/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Test 2 FAKE and 2 REAL videos
    fake_videos = [k for k, v in metadata.items() if v['label'] == 'FAKE'][:2]
    real_videos = [k for k, v in metadata.items() if v['label'] == 'REAL'][:2]
    test_videos = fake_videos + real_videos
    
    print("=" * 80)
    print("DIAGNOSTIC TEST - Layer-by-Layer Analysis")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = DeepfakePipeline(adaptive_threshold=False)
    
    for video_name in test_videos:
        video_path = f'Test-Video/train_sample_videos/{video_name}'
        ground_truth = metadata[video_name]['label']
        
        print(f"\n{'=' * 80}")
        print(f"Video: {video_name}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'=' * 80}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Run detection with fail-fast DISABLED to see all layers
        result = pipeline.detect(video_path, enable_fail_fast=False)
        
        print(f"\n--- LAYER RESULTS ---")
        for i, layer in enumerate(result.layer_results, 1):
            print(f"\nLayer {i}: {layer.layer_name}")
            print(f"  is_fake: {layer.is_fake}")
            print(f"  confidence: {layer.confidence:.4f}")
            print(f"  time: {layer.processing_time:.2f}s")
            
            # Show key details
            if 'avg_fake_probability' in layer.details:
                print(f"  avg_fake_probability: {layer.details['avg_fake_probability']:.4f}")
            if 'threshold' in layer.details:
                print(f"  threshold: {layer.details['threshold']}")
            if 'error' in layer.details:
                print(f"  ERROR: {layer.details['error'][:100]}")
            if 'skipped' in layer.details:
                print(f"  SKIPPED: {layer.details['skipped']}")
        
        print(f"\n--- FINAL RESULT ---")
        print(f"Prediction: {result.final_verdict}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Stopped at: {result.stopped_at_layer}")
        print(f"Total time: {result.total_time:.2f}s")
        print(f"Correct: {result.final_verdict == ground_truth}")
        print("=" * 80)

if __name__ == "__main__":
    main()
