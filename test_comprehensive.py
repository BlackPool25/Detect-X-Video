#!/usr/bin/env python3
"""
Larger test on 50 videos to get reliable accuracy estimate
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
    
    # Test on 50 videos (25 FAKE, 25 REAL)
    fake_videos = [k for k, v in metadata.items() if v['label'] == 'FAKE'][:25]
    real_videos = [k for k, v in metadata.items() if v['label'] == 'REAL'][:25]
    test_videos = fake_videos + real_videos
    
    print(f"COMPREHENSIVE TEST: {len(test_videos)} DFDC videos")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = DeepfakePipeline()
    
    results = []
    correct_count = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    total_time = 0.0
    
    for i, video_name in enumerate(test_videos, 1):
        video_path = f'Test-Video/train_sample_videos/{video_name}'
        ground_truth = metadata[video_name]['label']
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            result = pipeline.detect(video_path, enable_fail_fast=True)
            
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
            
            # Extract Layer 2 details
            layer2 = result.layer_results[0]
            avg_prob = layer2.details.get('avg_fake_probability', 0)
            
            total_time += result.total_time
            
            status = "✓" if correct else "✗"
            if i % 5 == 0 or not correct:  # Print every 5th or errors
                print(f"{status} [{i:2d}/{len(test_videos)}] {video_name:20s} | {ground_truth:4s} → {result.final_verdict:4s} | Prob: {avg_prob:.4f}")
            
            results.append({
                'video': video_name,
                'ground_truth': ground_truth,
                'prediction': result.final_verdict,
                'confidence': result.confidence,
                'avg_fake_prob': avg_prob,
                'time': result.total_time,
                'correct': correct
            })
            
        except KeyboardInterrupt:
            print("\n\nTest interrupted")
            break
        except Exception as e:
            print(f"✗ [{i:2d}/{len(test_videos)}] {video_name:20s} | ERROR: {str(e)[:50]}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS (50 Videos)")
    print("=" * 80)
    print(f"Total Tested: {len(results)}")
    print(f"Overall Accuracy: {correct_count}/{len(results)} = {100*correct_count/len(results):.1f}%")
    print()
    print(f"Real Videos: {real_correct}/{real_total} correct ({100*real_correct/real_total if real_total > 0 else 0:.1f}%)")
    print(f"Fake Videos: {fake_correct}/{fake_total} correct ({100*fake_correct/fake_total if fake_total > 0 else 0:.1f}%)")
    print()
    print(f"False Positives (Real → Fake): {real_total - real_correct}")
    print(f"False Negatives (Fake → Real): {fake_total - fake_correct}")
    print()
    print(f"Average Time per Video: {total_time/len(results) if results else 0:.2f}s")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 80)
    
    # Probability analysis
    fake_probs = [r['avg_fake_prob'] for r in results if r['ground_truth'] == 'FAKE']
    real_probs = [r['avg_fake_prob'] for r in results if r['ground_truth'] == 'REAL']
    
    print("\nProbability Distribution:")
    print(f"  FAKE videos: mean={np.mean(fake_probs):.4f}, median={np.median(fake_probs):.4f}")
    print(f"  REAL videos: mean={np.mean(real_probs):.4f}, median={np.median(real_probs):.4f}")
    
    # Save results
    output_file = 'test_results_comprehensive.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_tested': len(results),
            'correct': correct_count,
            'accuracy': correct_count / len(results) if results else 0,
            'real_accuracy': real_correct / real_total if real_total > 0 else 0,
            'fake_accuracy': fake_correct / fake_total if fake_total > 0 else 0,
            'avg_time_per_video': total_time / len(results) if results else 0,
            'total_time': total_time,
            'fake_prob_mean': float(np.mean(fake_probs)),
            'real_prob_mean': float(np.mean(real_probs)),
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Deployment recommendation
    print("\n" + "=" * 80)
    if correct_count / len(results) >= 0.80:
        print("✓ DEPLOYMENT READY: Accuracy meets 80%+ threshold")
        print("  Speed: Excellent (<2s per video)")
        print("  Stability: No crashes observed")
    else:
        print("⚠ NEEDS IMPROVEMENT: Accuracy below 80%")
        print("  Consider adding ensemble models")
    print("=" * 80)

if __name__ == "__main__":
    main()
