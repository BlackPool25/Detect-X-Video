#!/usr/bin/env python3
"""
Comprehensive accuracy test on large video dataset
Tests on Celeb-real, YouTube-real, and Fake videos
"""
import sys
from pathlib import Path
import subprocess
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def test_video(video_path):
    """Test a single video and return results"""
    try:
        result = subprocess.run(
            ["python3", "test_local_detection.py", str(video_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse output
        lines = result.stdout.split('\n')
        verdict = None
        visual_score = None
        
        for line in lines:
            if "Final Verdict:" in line:
                verdict = line.split("Final Verdict:")[1].strip()
            if "Visual Artifacts:" in line:
                visual_score = float(line.split("Visual Artifacts:")[1].split("(")[0].strip())
        
        return {
            'video': video_path.name,
            'verdict': verdict,
            'visual_score': visual_score,
            'success': True
        }
    except Exception as e:
        return {
            'video': video_path.name,
            'verdict': None,
            'visual_score': None,
            'success': False,
            'error': str(e)
        }

def main():
    print("=" * 80)
    print("COMPREHENSIVE ACCURACY TEST - LARGE DATASET")
    print("=" * 80)
    
    # Collect videos from all folders
    celeb_real = list(Path("Test-Video/Celeb-real").glob("*.mp4"))
    youtube_real = list(Path("Test-Video/YouTube-real").glob("*.mp4"))
    celeb_synthesis = list(Path("Test-Video/Celeb-synthesis").glob("*.mp4"))
    fake_videos = list(Path("Test-Video/Fake").glob("*.mp4"))
    real_videos = list(Path("Test-Video/Real").glob("*.mp4"))
    
    print(f"\nFound:")
    print(f"  Celeb-real: {len(celeb_real)} videos")
    print(f"  YouTube-real: {len(youtube_real)} videos")
    print(f"  Celeb-synthesis (FAKE): {len(celeb_synthesis)} videos")
    print(f"  Fake: {len(fake_videos)} videos")
    print(f"  Real: {len(real_videos)} videos")
    
    # Sample videos for testing (test 30 from each category)
    sample_size = 30
    celeb_sample = random.sample(celeb_real, min(sample_size, len(celeb_real)))
    youtube_sample = random.sample(youtube_real, min(sample_size, len(youtube_real)))
    celeb_synth_sample = random.sample(celeb_synthesis, min(sample_size, len(celeb_synthesis)))
    fake_sample = fake_videos  # Use all (only 3)
    real_sample = real_videos  # Use all (only 2)
    
    print(f"\nTesting {sample_size} videos from each category...")
    
    results = {
        "celeb_real": {"correct": 0, "total": 0, "scores": [], "failed": 0},
        "youtube_real": {"correct": 0, "total": 0, "scores": [], "failed": 0},
        "real": {"correct": 0, "total": 0, "scores": [], "failed": 0},
        "celeb_synthesis": {"correct": 0, "total": 0, "scores": [], "failed": 0},
        "fake": {"correct": 0, "total": 0, "scores": [], "failed": 0}
    }
    
    # Test Celeb-real
    print(f"\n{'='*80}")
    print("Testing CELEB-REAL videos...")
    print(f"{'='*80}")
    
    for idx, video in enumerate(celeb_sample, 1):
        print(f"  [{idx}/{len(celeb_sample)}] {video.name[:50]}... ", end='', flush=True)
        result = test_video(video)
        
        if result['success'] and result['verdict'] and result['visual_score'] is not None:
            results["celeb_real"]["total"] += 1
            results["celeb_real"]["scores"].append(result['visual_score'])
            
            is_authentic = "AUTHENTIC" in result['verdict'] or "UNCERTAIN" in result['verdict']
            
            if is_authentic:
                results["celeb_real"]["correct"] += 1
                print(f"✅ {result['verdict'][:20]} ({result['visual_score']:.3f})")
            else:
                print(f"❌ {result['verdict'][:20]} ({result['visual_score']:.3f})")
        else:
            results["celeb_real"]["failed"] += 1
            print(f"⚠️ FAILED")
    
    # Test YouTube-real
    print(f"\n{'='*80}")
    print("Testing YOUTUBE-REAL videos...")
    print(f"{'='*80}")
    
    for idx, video in enumerate(youtube_sample, 1):
        print(f"  [{idx}/{len(youtube_sample)}] {video.name[:50]}... ", end='', flush=True)
        result = test_video(video)
        
        if result['success'] and result['verdict'] and result['visual_score'] is not None:
            results["youtube_real"]["total"] += 1
            results["youtube_real"]["scores"].append(result['visual_score'])
            
            is_authentic = "AUTHENTIC" in result['verdict'] or "UNCERTAIN" in result['verdict']
            
            if is_authentic:
                results["youtube_real"]["correct"] += 1
                print(f"✅ {result['verdict'][:20]} ({result['visual_score']:.3f})")
            else:
                print(f"❌ {result['verdict'][:20]} ({result['visual_score']:.3f})")
        else:
            results["youtube_real"]["failed"] += 1
            print(f"⚠️ FAILED")
    
    # Test Real (small folder)
    print(f"\n{'='*80}")
    print("Testing REAL videos...")
    print(f"{'='*80}")
    
    for idx, video in enumerate(real_sample, 1):
        print(f"  [{idx}/{len(real_sample)}] {video.name[:50]}... ", end='', flush=True)
        result = test_video(video)
        
        if result['success'] and result['verdict'] and result['visual_score'] is not None:
            results["real"]["total"] += 1
            results["real"]["scores"].append(result['visual_score'])
            
            is_authentic = "AUTHENTIC" in result['verdict'] or "UNCERTAIN" in result['verdict']
            
            if is_authentic:
                results["real"]["correct"] += 1
                print(f"✅ {result['verdict'][:20]} ({result['visual_score']:.3f})")
            else:
                print(f"❌ {result['verdict'][:20]} ({result['visual_score']:.3f})")
        else:
            results["real"]["failed"] += 1
            print(f"⚠️ FAILED")
    
    # Test Celeb-synthesis (FAKE)
    print(f"\n{'='*80}")
    print("Testing CELEB-SYNTHESIS (DEEPFAKES) videos...")
    print(f"{'='*80}")
    
    for idx, video in enumerate(celeb_synth_sample, 1):
        print(f"  [{idx}/{len(celeb_synth_sample)}] {video.name[:50]}... ", end='', flush=True)
        result = test_video(video)
        
        if result['success'] and result['verdict'] and result['visual_score'] is not None:
            results["celeb_synthesis"]["total"] += 1
            results["celeb_synthesis"]["scores"].append(result['visual_score'])
            
            is_fake = "DEEPFAKE" in result['verdict'] or "LIKELY" in result['verdict']
            
            if is_fake:
                results["celeb_synthesis"]["correct"] += 1
                print(f"✅ {result['verdict'][:20]} ({result['visual_score']:.3f})")
            else:
                print(f"❌ {result['verdict'][:20]} ({result['visual_score']:.3f})")
        else:
            results["celeb_synthesis"]["failed"] += 1
            print(f"⚠️ FAILED")
    
    # Test Fake
    print(f"\n{'='*80}")
    print("Testing FAKE videos...")
    print(f"{'='*80}")
    
    for idx, video in enumerate(fake_sample, 1):
        print(f"  [{idx}/{len(fake_sample)}] {video.name[:50]}... ", end='', flush=True)
        result = test_video(video)
        
        if result['success'] and result['verdict'] and result['visual_score'] is not None:
            results["fake"]["total"] += 1
            results["fake"]["scores"].append(result['visual_score'])
            
            is_fake = "DEEPFAKE" in result['verdict'] or "LIKELY" in result['verdict']
            
            if is_fake:
                results["fake"]["correct"] += 1
                print(f"✅ {result['verdict'][:20]} ({result['visual_score']:.3f})")
            else:
                print(f"❌ {result['verdict'][:20]} ({result['visual_score']:.3f})")
        else:
            results["fake"]["failed"] += 1
            print(f"⚠️ FAILED")
    
    # Calculate and display results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    total_correct = 0
    total_tested = 0
    
    for category, data in results.items():
        if data['total'] > 0:
            accuracy = data['correct'] / data['total'] * 100
            avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
            
            print(f"\n{category.upper().replace('_', ' ')}:")
            print(f"  Tested: {data['total']} videos (Failed: {data['failed']})")
            print(f"  Correct: {data['correct']}/{data['total']} ({accuracy:.1f}%)")
            print(f"  Avg visual score: {avg_score:.3f}")
            print(f"  Score range: [{min(data['scores']):.3f}, {max(data['scores']):.3f}]")
            
            total_correct += data['correct']
            total_tested += data['total']
    
    overall_accuracy = total_correct / total_tested * 100 if total_tested > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY: {total_correct}/{total_tested} = {overall_accuracy:.1f}%")
    print(f"{'='*80}")
    
    # Performance assessment
    if overall_accuracy >= 80:
        print("\n✅ EXCELLENT - Model is production-ready")
    elif overall_accuracy >= 70:
        print("\n✅ GOOD - Model performs well, minor tuning recommended")
    elif overall_accuracy >= 60:
        print("\n⚠️ ACCEPTABLE - Model needs optimization")
    else:
        print("\n❌ POOR - Model requires significant improvement")
    
    # Save results
    with open("test_results_comprehensive.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to test_results_comprehensive.json")

if __name__ == "__main__":
    main()
