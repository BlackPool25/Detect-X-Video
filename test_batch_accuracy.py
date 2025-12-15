#!/usr/bin/env python3
"""
Batch test script to evaluate model accuracy on multiple videos
"""
import sys
from pathlib import Path
import subprocess
import json

def test_video(video_path):
    """Test a single video and return results"""
    result = subprocess.run(
        ["python3", "test_local_detection.py", str(video_path)],
        capture_output=True,
        text=True
    )
    
    # Parse output to get verdict and score
    lines = result.stdout.split('\n')
    verdict = None
    score = None
    visual_score = None
    
    for line in lines:
        if "Final Verdict:" in line:
            verdict = line.split("Final Verdict:")[1].strip()
        if "Confidence Score:" in line:
            score = float(line.split("Confidence Score:")[1].strip().replace('%', ''))
        if "Visual Artifacts:" in line:
            visual_score = float(line.split("Visual Artifacts:")[1].split("(")[0].strip())
    
    return verdict, score, visual_score

def main():
    print("=" * 80)
    print("BATCH ACCURACY TEST - MULTIMODAL DEEPFAKE DETECTION")
    print("=" * 80)
    
    # Get test videos
    real_videos = list(Path("Test-Video/Real").glob("*.mp4"))[:5]
    fake_videos = list(Path("Test-Video/Fake").glob("*.mp4"))[:5]
    
    results = {
        "real": {"correct": 0, "total": 0, "scores": []},
        "fake": {"correct": 0, "total": 0, "scores": []}
    }
    
    print(f"\nTesting {len(real_videos)} REAL videos...")
    for video in real_videos:
        print(f"\n  Testing: {video.name}")
        verdict, score, visual_score = test_video(video)
        
        is_authentic = "AUTHENTIC" in verdict or "UNCERTAIN" in verdict
        results["real"]["total"] += 1
        results["real"]["scores"].append(visual_score)
        
        if is_authentic:
            results["real"]["correct"] += 1
            print(f"    ✅ Correctly classified as {verdict} (visual: {visual_score:.3f})")
        else:
            print(f"    ❌ Incorrectly classified as {verdict} (visual: {visual_score:.3f})")
    
    print(f"\nTesting {len(fake_videos)} FAKE videos...")
    for video in fake_videos:
        print(f"\n  Testing: {video.name}")
        verdict, score, visual_score = test_video(video)
        
        is_fake = "DEEPFAKE" in verdict or "LIKELY" in verdict
        results["fake"]["total"] += 1
        results["fake"]["scores"].append(visual_score)
        
        if is_fake:
            results["fake"]["correct"] += 1
            print(f"    ✅ Correctly classified as {verdict} (visual: {visual_score:.3f})")
        else:
            print(f"    ❌ Incorrectly classified as {verdict} (visual: {visual_score:.3f})")
    
    # Calculate accuracy
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    real_accuracy = results["real"]["correct"] / results["real"]["total"] * 100 if results["real"]["total"] > 0 else 0
    fake_accuracy = results["fake"]["correct"] / results["fake"]["total"] * 100 if results["fake"]["total"] > 0 else 0
    overall_accuracy = (results["real"]["correct"] + results["fake"]["correct"]) / (results["real"]["total"] + results["fake"]["total"]) * 100
    
    print(f"\nREAL videos: {results['real']['correct']}/{results['real']['total']} correct ({real_accuracy:.1f}%)")
    print(f"  Average visual score: {sum(results['real']['scores'])/len(results['real']['scores']):.3f}")
    
    print(f"\nFAKE videos: {results['fake']['correct']}/{results['fake']['total']} correct ({fake_accuracy:.1f}%)")
    print(f"  Average visual score: {sum(results['fake']['scores'])/len(results['fake']['scores']):.3f}")
    
    print(f"\nOVERALL ACCURACY: {overall_accuracy:.1f}%")
    
    # Determine if model is performing well
    if overall_accuracy >= 80:
        print("\n✅ Model performance is EXCELLENT")
    elif overall_accuracy >= 60:
        print("\n⚠️  Model performance is ACCEPTABLE but could be improved")
    else:
        print("\n❌ Model performance needs improvement")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
