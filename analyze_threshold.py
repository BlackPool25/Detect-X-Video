#!/usr/bin/env python3
"""
Analyze visual scores distribution and find optimal threshold
"""
import sys
from pathlib import Path
import subprocess

def get_visual_score(video_path):
    """Get visual score from test"""
    result = subprocess.run(
        ["python3", "test_local_detection.py", str(video_path)],
        capture_output=True,
        text=True
    )
    
    for line in result.stdout.split('\n'):
        if "Visual Artifacts:" in line:
            return float(line.split("Visual Artifacts:")[1].split("(")[0].strip())
    return None

print("Analyzing visual score distribution...")
print("=" * 80)

real_videos = list(Path("Test-Video/Real").glob("*.mp4"))[:5]
fake_videos = list(Path("Test-Video/Fake").glob("*.mp4"))[:5]

real_scores = []
fake_scores = []

print("\nREAL videos:")
for video in real_videos:
    score = get_visual_score(video)
    if score:
        real_scores.append(score)
        print(f"  {video.name}: {score:.3f}")

print("\nFAKE videos:")
for video in fake_videos:
    score = get_visual_score(video)
    if score:
        fake_scores.append(score)
        print(f"  {video.name}: {score:.3f}")

# Calculate statistics
if real_scores and fake_scores:
    import numpy as np
    
    real_mean = np.mean(real_scores)
    real_std = np.std(real_scores)
    fake_mean = np.mean(fake_scores)
    fake_std = np.std(fake_scores)
    
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"REAL videos:  mean={real_mean:.3f}, std={real_std:.3f}, range=[{min(real_scores):.3f}, {max(real_scores):.3f}]")
    print(f"FAKE videos:  mean={fake_mean:.3f}, std={fake_std:.3f}, range=[{min(fake_scores):.3f}, {max(fake_scores):.3f}]")
    
    # Find optimal threshold (midpoint between means)
    optimal_threshold = (real_mean + fake_mean) / 2
    print(f"\nOptimal threshold: {optimal_threshold:.3f}")
    
    # Test different thresholds
    print("\nThreshold Analysis:")
    for threshold in [0.4, 0.5, 0.6, 0.65, 0.7]:
        real_correct = sum(1 for score in real_scores if score < threshold)
        fake_correct = sum(1 for score in fake_scores if score >= threshold)
        total_correct = real_correct + fake_correct
        accuracy = total_correct / (len(real_scores) + len(fake_scores)) * 100
        
        print(f"  Threshold {threshold:.2f}: {total_correct}/{len(real_scores)+len(fake_scores)} correct ({accuracy:.1f}%)")
        print(f"    Real: {real_correct}/{len(real_scores)}, Fake: {fake_correct}/{len(fake_scores)}")
