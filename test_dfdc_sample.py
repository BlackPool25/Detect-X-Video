#!/usr/bin/env python3
"""
Test the deepfake pipeline on DFDC (DeepFake Detection Challenge) dataset
DFDC contains face-swap deepfakes - should work well with SBI
"""

import json
import sys
from pathlib import Path
from pipeline_production import DeepfakePipeline
import random

def load_metadata():
    """Load DFDC metadata"""
    metadata_path = Path('Test-Video/train_sample_videos/metadata.json')
    with open(metadata_path, 'r') as f:
        return json.load(f)

def test_dfdc_sample(num_real=5, num_fake=5, adaptive_threshold=True):
    """Test pipeline on DFDC sample videos"""
    
    metadata = load_metadata()
    video_dir = Path('Test-Video/train_sample_videos')
    
    # Separate real and fake videos
    real_videos = [vid for vid, data in metadata.items() if data['label'] == 'REAL']
    fake_videos = [vid for vid, data in metadata.items() if data['label'] == 'FAKE']
    
    # Random sample
    random.seed(42)
    test_real = random.sample(real_videos, min(num_real, len(real_videos)))
    test_fake = random.sample(fake_videos, min(num_fake, len(fake_videos)))
    
    print("=" * 80)
    print("DFDC DEEPFAKE DETECTION TEST")
    print("=" * 80)
    print(f"Testing {len(test_real)} real + {len(test_fake)} fake videos")
    print(f"Adaptive threshold: {adaptive_threshold} (0.6 if True, 0.5 if False)")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = DeepfakePipeline(adaptive_threshold=adaptive_threshold)
    
    results = {
        'real': {'correct': 0, 'total': 0, 'details': []},
        'fake': {'correct': 0, 'total': 0, 'details': []}
    }
    
    # Test real videos
    print("\nTesting REAL videos...")
    for i, video_name in enumerate(test_real, 1):
        video_path = video_dir / video_name
        try:
            result = pipeline.detect(str(video_path), enable_fail_fast=False)
            correct = result.final_verdict == 'REAL'
            
            results['real']['total'] += 1
            if correct:
                results['real']['correct'] += 1
            
            # Get Layer 2 details
            layer2_details = None
            for layer in result.layer_results:
                if 'Visual Artifacts' in layer.layer_name:
                    layer2_details = layer.details
                    break
            
            results['real']['details'].append({
                'video': video_name,
                'prediction': result.final_verdict,
                'confidence': result.confidence,
                'correct': correct,
                'layer2_avg_prob': layer2_details.get('avg_fake_probability', 0) if layer2_details else 0,
                'layer2_threshold': layer2_details.get('threshold', 0) if layer2_details else 0
            })
            
            status = '✓' if correct else '✗'
            print(f"  [{i}/{len(test_real)}] {status} {video_name}: {result.final_verdict} ({result.confidence:.1f}%)")
            
        except Exception as e:
            print(f"  [ERROR] {video_name}: {e}")
            results['real']['total'] += 1
    
    # Test fake videos
    print("\nTesting FAKE videos...")
    for i, video_name in enumerate(test_fake, 1):
        video_path = video_dir / video_name
        try:
            result = pipeline.detect(str(video_path), enable_fail_fast=False)
            correct = result.final_verdict == 'FAKE'
            
            results['fake']['total'] += 1
            if correct:
                results['fake']['correct'] += 1
            
            # Get Layer 2 details
            layer2_details = None
            for layer in result.layer_results:
                if 'Visual Artifacts' in layer.layer_name:
                    layer2_details = layer.details
                    break
            
            results['fake']['details'].append({
                'video': video_name,
                'prediction': result.final_verdict,
                'confidence': result.confidence,
                'correct': correct,
                'layer2_avg_prob': layer2_details.get('avg_fake_probability', 0) if layer2_details else 0,
                'layer2_threshold': layer2_details.get('threshold', 0) if layer2_details else 0
            })
            
            status = '✓' if correct else '✗'
            print(f"  [{i}/{len(test_fake)}] {status} {video_name}: {result.final_verdict} ({result.confidence:.1f}%)")
            
        except Exception as e:
            print(f"  [ERROR] {video_name}: {e}")
            results['fake']['total'] += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    real_acc = 100 * results['real']['correct'] / results['real']['total'] if results['real']['total'] > 0 else 0
    fake_acc = 100 * results['fake']['correct'] / results['fake']['total'] if results['fake']['total'] > 0 else 0
    total_correct = results['real']['correct'] + results['fake']['correct']
    total_videos = results['real']['total'] + results['fake']['total']
    overall_acc = 100 * total_correct / total_videos if total_videos > 0 else 0
    
    print(f"\nReal Videos: {results['real']['correct']}/{results['real']['total']} correct ({real_acc:.1f}%)")
    print(f"Fake Videos: {results['fake']['correct']}/{results['fake']['total']} correct ({fake_acc:.1f}%)")
    print(f"\nOverall Accuracy: {total_correct}/{total_videos} ({overall_acc:.1f}%)")
    
    # False positives/negatives
    false_positives = results['real']['total'] - results['real']['correct']
    false_negatives = results['fake']['total'] - results['fake']['correct']
    print(f"\nFalse Positives (Real marked as Fake): {false_positives}")
    print(f"False Negatives (Fake marked as Real): {false_negatives}")
    
    # Analyze borderline cases
    print("\n" + "=" * 80)
    print("BORDERLINE CASES ANALYSIS")
    print("=" * 80)
    
    # Real videos with high fake probability (near threshold)
    borderline_real = [v for v in results['real']['details'] 
                       if 0.45 <= v['layer2_avg_prob'] <= 0.65]
    if borderline_real:
        print(f"\nReal videos near threshold ({len(borderline_real)}):")
        for v in sorted(borderline_real, key=lambda x: x['layer2_avg_prob'], reverse=True)[:5]:
            print(f"  {v['video']}: prob={v['layer2_avg_prob']:.4f}, "
                  f"threshold={v['layer2_threshold']}, {v['prediction']} {'✓' if v['correct'] else '✗'}")
    
    # Fake videos with low fake probability (near threshold)
    borderline_fake = [v for v in results['fake']['details'] 
                       if 0.45 <= v['layer2_avg_prob'] <= 0.65]
    if borderline_fake:
        print(f"\nFake videos near threshold ({len(borderline_fake)}):")
        for v in sorted(borderline_fake, key=lambda x: x['layer2_avg_prob'])[:5]:
            print(f"  {v['video']}: prob={v['layer2_avg_prob']:.4f}, "
                  f"threshold={v['layer2_threshold']}, {v['prediction']} {'✓' if v['correct'] else '✗'}")
    
    # Save results
    output_file = f"test_results_dfdc_{'adaptive' if adaptive_threshold else 'standard'}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    # Test with both standard and adaptive thresholds
    print("\n" + "=" * 80)
    print("TEST 1: STANDARD THRESHOLD (0.5)")
    print("=" * 80)
    results_standard = test_dfdc_sample(num_real=5, num_fake=5, adaptive_threshold=False)
    
    print("\n\n" + "=" * 80)
    print("TEST 2: ADAPTIVE THRESHOLD (0.6)")
    print("=" * 80)
    results_adaptive = test_dfdc_sample(num_real=5, num_fake=5, adaptive_threshold=True)
    
    # Compare
    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)
    
    std_acc = 100 * (results_standard['real']['correct'] + results_standard['fake']['correct']) / \
              (results_standard['real']['total'] + results_standard['fake']['total'])
    ada_acc = 100 * (results_adaptive['real']['correct'] + results_adaptive['fake']['correct']) / \
              (results_adaptive['real']['total'] + results_adaptive['fake']['total'])
    
    print(f"Standard (0.5): {std_acc:.1f}% overall accuracy")
    print(f"Adaptive (0.6): {ada_acc:.1f}% overall accuracy")
    print(f"Improvement: {ada_acc - std_acc:+.1f}%")
