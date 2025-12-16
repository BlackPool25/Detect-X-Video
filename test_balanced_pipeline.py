#!/usr/bin/env python3
"""
Test balanced 3-layer pipeline
"""
import json
import torch
import gc
from pathlib import Path
from pipeline_balanced import DeepfakePipeline

def main():
    with open('Test-Video/train_sample_videos/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Test on 20 videos (10 FAKE, 10 REAL)
    fake_videos = [k for k, v in metadata.items() if v['label'] == 'FAKE'][:10]
    real_videos = [k for k, v in metadata.items() if v['label'] == 'REAL'][:10]
    test_videos = fake_videos + real_videos
    
    print(f"Testing BALANCED 3-Layer Pipeline on {len(test_videos)} videos")
    print("=" * 80)
    
    pipeline = DeepfakePipeline()
    
    results = []
    correct = 0
    fake_correct = 0
    fake_total = 0
    real_correct = 0
    real_total = 0
    total_time = 0.0
    
    for i, video_name in enumerate(test_videos, 1):
        video_path = f'Test-Video/train_sample_videos/{video_name}'
        ground_truth = metadata[video_name]['label']
        
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            result = pipeline.detect(video_path, enable_fail_fast=False)
            
            is_correct = (result.final_verdict == ground_truth)
            if is_correct:
                correct += 1
            
            if ground_truth == 'FAKE':
                fake_total += 1
                if is_correct:
                    fake_correct += 1
            else:
                real_total += 1
                if is_correct:
                    real_correct += 1
            
            total_time += result.total_time
            
            status = "✓" if is_correct else "✗"
            print(f"{status} [{i:2d}/{len(test_videos)}] {video_name:20s} | {ground_truth:4s} → {result.final_verdict:4s} | Conf: {result.confidence:.2%} | Time: {result.total_time:.1f}s")
            
            results.append({
                'video': video_name,
                'truth': ground_truth,
                'prediction': result.final_verdict,
                'confidence': result.confidence,
                'time': result.total_time,
                'correct': is_correct
            })
            
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"✗ [{i}/{len(test_videos)}] {video_name} | ERROR: {str(e)[:60]}")
    
    print("\n" + "=" * 80)
    print("BALANCED PIPELINE RESULTS")
    print("=" * 80)
    print(f"Overall: {correct}/{len(results)} = {100*correct/len(results):.1f}%")
    print(f"Real: {real_correct}/{real_total} = {100*real_correct/real_total if real_total else 0:.1f}%")
    print(f"Fake: {fake_correct}/{fake_total} = {100*fake_correct/fake_total if fake_total else 0:.1f}%")
    print(f"Avg Time: {total_time/len(results) if results else 0:.2f}s per video")
    print("=" * 80)
    
    with open('test_results_balanced.json', 'w') as f:
        json.dump({
            'accuracy': correct / len(results) if results else 0,
            'real_accuracy': real_correct / real_total if real_total else 0,
            'fake_accuracy': fake_correct / fake_total if fake_total else 0,
            'avg_time': total_time / len(results) if results else 0,
            'results': results
        }, f, indent=2)

if __name__ == "__main__":
    main()
