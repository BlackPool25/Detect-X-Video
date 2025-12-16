#!/usr/bin/env python3
"""
Lightweight DFDC test - skips memory-intensive SyncNet Layer 3
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
    
    # Get all videos
    all_videos = list(Path('Test-Video/train_sample_videos').glob('*.mp4'))
    
    # Test on 20 videos (mix of real and fake)
    test_videos = all_videos[:20]
    
    print(f"Testing on {len(test_videos)} DFDC videos...")
    print("=" * 80)
    
    # Initialize pipeline (no adaptive threshold for DFDC)
    pipeline = DeepfakePipeline(adaptive_threshold=False)
    
    results = []
    correct_count = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    for i, video_path in enumerate(test_videos, 1):
        video_name = video_path.name
        ground_truth = metadata[video_name]['label']
        
        # Clear GPU memory before each video
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Use fail_fast to skip after first fake detection
            result = pipeline.detect(str(video_path), enable_fail_fast=True)
            
            # Extract Layer 2 (SBI) details
            layer2 = result.layer_results[1] if len(result.layer_results) > 1 else None
            avg_prob = layer2.details.get('avg_fake_probability', 0) if layer2 else 0
            
            correct = (result.final_verdict == ground_truth)
            if correct:
                correct_count += 1
            
            if ground_truth == 'REAL':
                real_total += 1
                if correct:
                    real_correct += 1
            else:
                fake_total += 1
                if correct:
                    fake_correct += 1
            
            status = "✓" if correct else "✗"
            print(f"{status} [{i:2d}/{len(test_videos)}] {video_name:20s} | Truth: {ground_truth:4s} | Pred: {result.final_verdict:4s} | L2_AvgProb: {avg_prob:.3f}")
            
            results.append({
                'video': video_name,
                'ground_truth': ground_truth,
                'prediction': result.final_verdict,
                'confidence': result.confidence,
                'layer2_avg_prob': avg_prob,
                'correct': correct
            })
            
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"✗ [{i:2d}/{len(test_videos)}] {video_name:20s} | ERROR: {str(e)[:50]}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("DFDC TEST RESULTS")
    print("=" * 80)
    print(f"Total Tested: {len(results)}")
    print(f"Overall Accuracy: {correct_count}/{len(results)} = {100*correct_count/len(results):.1f}%")
    print()
    print(f"Real Videos: {real_correct}/{real_total} correct ({100*real_correct/real_total if real_total > 0 else 0:.1f}%)")
    print(f"Fake Videos: {fake_correct}/{fake_total} correct ({100*fake_correct/fake_total if fake_total > 0 else 0:.1f}%)")
    print()
    print(f"False Positives (Real → Fake): {real_total - real_correct}")
    print(f"False Negatives (Fake → Real): {fake_total - fake_correct}")
    print("=" * 80)
    
    # Save results
    output_file = 'test_results_dfdc_lightweight.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_tested': len(results),
            'accuracy': correct_count / len(results) if results else 0,
            'real_accuracy': real_correct / real_total if real_total > 0 else 0,
            'fake_accuracy': fake_correct / fake_total if fake_total > 0 else 0,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
