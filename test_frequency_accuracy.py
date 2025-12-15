"""
Comprehensive Accuracy Test for Frequency-based Deepfake Detection
Tests F3Net and SRM models on real vs fake videos
"""

import torch
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

sys.path.insert(0, '/home/lightdesk/Projects/AI-Video')
from test_frequency_models import FrequencyEnsemble

def test_accuracy(ensemble, real_dir, fake_dir, num_samples=20):
    """Test accuracy on real and fake videos"""
    
    real_videos = list(Path(real_dir).glob('*.mp4'))[:num_samples]
    fake_videos = list(Path(fake_dir).glob('*.mp4'))[:num_samples]
    
    print(f"\n{'='*70}")
    print(f"Testing on {len(real_videos)} REAL and {len(fake_videos)} FAKE videos")
    print(f"{'='*70}\n")
    
    results = {
        'real': {'correct': 0, 'total': 0, 'scores': [], 'predictions': []},
        'fake': {'correct': 0, 'total': 0, 'scores': [], 'predictions': []}
    }
    
    # Test real videos
    print("Testing REAL videos...")
    for video_path in tqdm(real_videos, desc="Real"):
        pred = ensemble.predict_video(str(video_path), max_frames=8)
        
        is_correct = pred['prediction'] == 'REAL'
        results['real']['correct'] += int(is_correct)
        results['real']['total'] += 1
        results['real']['scores'].append(pred['ensemble_score'])
        results['real']['predictions'].append(pred['prediction'])
        
        if not is_correct:
            print(f"  ❌ Misclassified as {pred['prediction']}: {video_path.name} (score: {pred['ensemble_score']:.3f})")
    
    # Test fake videos
    print("\nTesting FAKE videos...")
    for video_path in tqdm(fake_videos, desc="Fake"):
        pred = ensemble.predict_video(str(video_path), max_frames=8)
        
        is_correct = pred['prediction'] == 'FAKE'
        results['fake']['correct'] += int(is_correct)
        results['fake']['total'] += 1
        results['fake']['scores'].append(pred['ensemble_score'])
        results['fake']['predictions'].append(pred['prediction'])
        
        if not is_correct:
            print(f"  ❌ Misclassified as {pred['prediction']}: {video_path.name} (score: {pred['ensemble_score']:.3f})")
    
    return results

def print_results(results):
    """Print accuracy results"""
    
    real_acc = results['real']['correct'] / results['real']['total'] * 100 if results['real']['total'] > 0 else 0
    fake_acc = results['fake']['correct'] / results['fake']['total'] * 100 if results['fake']['total'] > 0 else 0
    
    total_correct = results['real']['correct'] + results['fake']['correct']
    total_samples = results['real']['total'] + results['fake']['total']
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    
    real_scores = results['real']['scores']
    fake_scores = results['fake']['scores']
    
    print(f"\n{'='*70}")
    print("ACCURACY RESULTS")
    print(f"{'='*70}")
    print(f"\n📊 REAL Videos:")
    print(f"   Accuracy: {real_acc:.1f}% ({results['real']['correct']}/{results['real']['total']})")
    print(f"   Avg Score: {np.mean(real_scores):.4f} (±{np.std(real_scores):.4f})")
    print(f"   Score Range: [{min(real_scores):.4f}, {max(real_scores):.4f}]")
    
    print(f"\n📊 FAKE Videos:")
    print(f"   Accuracy: {fake_acc:.1f}% ({results['fake']['correct']}/{results['fake']['total']})")
    print(f"   Avg Score: {np.mean(fake_scores):.4f} (±{np.std(fake_scores):.4f})")
    print(f"   Score Range: [{min(fake_scores):.4f}, {max(fake_scores):.4f}]")
    
    print(f"\n{'='*70}")
    print(f"🎯 OVERALL ACCURACY: {overall_acc:.1f}% ({total_correct}/{total_samples})")
    print(f"{'='*70}\n")
    
    # Calculate separation
    separation = abs(np.mean(fake_scores) - np.mean(real_scores))
    print(f"📈 Score Separation: {separation:.4f}")
    print(f"   (Larger separation = better discrimination)")
    
    # Check if scores are actually different
    if separation < 0.01:
        print(f"\n⚠️  WARNING: Very low score separation!")
        print(f"   Model may not be discriminating between real and fake.")
    
    return overall_acc

if __name__ == "__main__":
    print("="*70)
    print("FREQUENCY-BASED DEEPFAKE DETECTION - ACCURACY TEST")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # Initialize ensemble
    ensemble = FrequencyEnsemble(device=device)
    
    if not ensemble.load_weights():
        print("\n❌ Failed to load models!")
        sys.exit(1)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Celeb-DF Dataset',
            'real_dir': '/home/lightdesk/Projects/AI-Video/Test-Video/Celeb-real',
            'fake_dir': '/home/lightdesk/Projects/AI-Video/Test-Video/Celeb-synthesis',
            'num_samples': 20
        },
        {
            'name': 'YouTube Real vs Celeb Fake',
            'real_dir': '/home/lightdesk/Projects/AI-Video/Test-Video/YouTube-real',
            'fake_dir': '/home/lightdesk/Projects/AI-Video/Test-Video/Celeb-synthesis',
            'num_samples': 15
        }
    ]
    
    overall_results = []
    
    for config in test_configs:
        real_dir = Path(config['real_dir'])
        fake_dir = Path(config['fake_dir'])
        
        if not real_dir.exists() or not fake_dir.exists():
            print(f"\n⚠️  Skipping {config['name']}: directories not found")
            continue
        
        print(f"\n\n{'#'*70}")
        print(f"# TEST: {config['name']}")
        print(f"{'#'*70}")
        
        results = test_accuracy(
            ensemble,
            config['real_dir'],
            config['fake_dir'],
            config['num_samples']
        )
        
        accuracy = print_results(results)
        overall_results.append({
            'name': config['name'],
            'accuracy': accuracy,
            'results': results
        })
    
    # Final summary
    if overall_results:
        print(f"\n\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}\n")
        
        for res in overall_results:
            print(f"  {res['name']}: {res['accuracy']:.1f}%")
        
        avg_accuracy = np.mean([r['accuracy'] for r in overall_results])
        print(f"\n  Average Accuracy: {avg_accuracy:.1f}%")
        print(f"{'='*70}")
        
        if avg_accuracy < 60:
            print("\n⚠️  Accuracy below 60% - Model may need fine-tuning")
        elif avg_accuracy >= 80:
            print("\n✅ Good accuracy! Model is working well.")
        else:
            print("\n📊 Moderate accuracy - Consider threshold adjustment")
