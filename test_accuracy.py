"""
Test script to measure pipeline accuracy on real vs fake videos
"""
import sys
from pathlib import Path
from pipeline_production import DeepfakePipeline

def test_videos(real_videos, fake_videos):
    """Test pipeline on real and fake videos"""
    
    pipeline = DeepfakePipeline()
    
    print("="*80)
    print("TESTING PIPELINE ACCURACY")
    print("="*80)
    
    # Test REAL videos
    print(f"\nTesting {len(real_videos)} REAL videos...")
    real_correct = 0
    real_results = []
    
    for i, video in enumerate(real_videos, 1):
        print(f"[{i}/{len(real_videos)}] {video.name}...", end=" ")
        try:
            result = pipeline.detect(str(video), enable_fail_fast=False)
            is_correct = not result.is_fake
            real_correct += is_correct
            real_results.append({
                'video': video.name,
                'predicted': 'FAKE' if result.is_fake else 'REAL',
                'confidence': result.confidence,
                'correct': is_correct
            })
            print(f"{'✓' if is_correct else '✗'} Predicted: {'FAKE' if result.is_fake else 'REAL'} ({result.confidence:.1f}%)")
        except Exception as e:
            print(f"ERROR: {e}")
            real_results.append({'video': video.name, 'error': str(e)})
    
    # Test FAKE videos
    print(f"\nTesting {len(fake_videos)} FAKE videos...")
    fake_correct = 0
    fake_results = []
    
    for i, video in enumerate(fake_videos, 1):
        print(f"[{i}/{len(fake_videos)}] {video.name}...", end=" ")
        try:
            result = pipeline.detect(str(video), enable_fail_fast=False)
            is_correct = result.is_fake
            fake_correct += is_correct
            fake_results.append({
                'video': video.name,
                'predicted': 'FAKE' if result.is_fake else 'REAL',
                'confidence': result.confidence,
                'correct': is_correct
            })
            print(f"{'✓' if is_correct else '✗'} Predicted: {'FAKE' if result.is_fake else 'REAL'} ({result.confidence:.1f}%)")
        except Exception as e:
            print(f"ERROR: {e}")
            fake_results.append({'video': video.name, 'error': str(e)})
    
    # Calculate accuracy
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nREAL videos: {real_correct}/{len(real_videos)} correct ({100*real_correct/len(real_videos):.1f}%)")
    print(f"FAKE videos: {fake_correct}/{len(fake_videos)} correct ({100*fake_correct/len(fake_videos):.1f}%)")
    
    total_correct = real_correct + fake_correct
    total = len(real_videos) + len(fake_videos)
    print(f"\nOverall Accuracy: {total_correct}/{total} ({100*total_correct/total:.1f}%)")
    
    return real_results, fake_results

if __name__ == "__main__":
    # Test on small subset
    real_videos = list(Path("Test-Video/YouTube-real").glob("*.mp4"))[:3]
    fake_videos = list(Path("Test-Video/Celeb-synthesis").glob("*.mp4"))[:3]
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos\n")
    
    test_videos(real_videos, fake_videos)
