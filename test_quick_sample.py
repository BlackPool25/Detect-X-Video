#!/usr/bin/env python3
"""
Quick accuracy test on small random sample
10 real videos (mixed from all real folders)
10 fake videos (mixed from all fake folders)
"""
import sys
from pathlib import Path
import subprocess
import random

def test_video(video_path):
    """Test a single video and return results"""
    try:
        result = subprocess.run(
            ["python3", "test_local_detection.py", str(video_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        lines = result.stdout.split('\n')
        verdict = None
        visual_score = None
        
        for line in lines:
            if "Final Verdict:" in line:
                verdict = line.split("Final Verdict:")[1].strip()
            if "Visual Artifacts:" in line:
                visual_score = float(line.split("Visual Artifacts:")[1].split("(")[0].strip())
        
        return verdict, visual_score
    except Exception as e:
        return None, None

print("=" * 80)
print("QUICK ACCURACY TEST - 10 REAL + 10 FAKE")
print("=" * 80)

# Collect all real videos
all_real = []
all_real.extend(list(Path("Test-Video/Celeb-real").glob("*.mp4")))
all_real.extend(list(Path("Test-Video/YouTube-real").glob("*.mp4")))
all_real.extend(list(Path("Test-Video/Real").glob("*.mp4")))

# Collect all fake videos
all_fake = []
all_fake.extend(list(Path("Test-Video/Celeb-synthesis").glob("*.mp4")))
all_fake.extend(list(Path("Test-Video/Fake").glob("*.mp4")))

print(f"\nTotal available: {len(all_real)} real, {len(all_fake)} fake")

# Random sample
real_sample = random.sample(all_real, 10)
fake_sample = random.sample(all_fake, 10)

results = {
    "real": {"correct": 0, "total": 0, "scores": []},
    "fake": {"correct": 0, "total": 0, "scores": []}
}

# Test REAL
print(f"\n{'='*80}")
print("Testing 10 REAL videos...")
print(f"{'='*80}")

for idx, video in enumerate(real_sample, 1):
    print(f"  [{idx}/10] {video.name[:60]}... ", end='', flush=True)
    verdict, score = test_video(video)
    
    if verdict and score is not None:
        results["real"]["total"] += 1
        results["real"]["scores"].append(score)
        
        is_authentic = "AUTHENTIC" in verdict or "UNCERTAIN" in verdict
        
        if is_authentic:
            results["real"]["correct"] += 1
            print(f"✅ {verdict[:30]} ({score:.3f})")
        else:
            print(f"❌ {verdict[:30]} ({score:.3f})")
    else:
        print(f"⚠️ FAILED")

# Test FAKE
print(f"\n{'='*80}")
print("Testing 10 FAKE videos...")
print(f"{'='*80}")

for idx, video in enumerate(fake_sample, 1):
    print(f"  [{idx}/10] {video.name[:60]}... ", end='', flush=True)
    verdict, score = test_video(video)
    
    if verdict and score is not None:
        results["fake"]["total"] += 1
        results["fake"]["scores"].append(score)
        
        is_fake = "DEEPFAKE" in verdict or "LIKELY" in verdict
        
        if is_fake:
            results["fake"]["correct"] += 1
            print(f"✅ {verdict[:30]} ({score:.3f})")
        else:
            print(f"❌ {verdict[:30]} ({score:.3f})")
    else:
        print(f"⚠️ FAILED")

# Results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

real_acc = results["real"]["correct"] / results["real"]["total"] * 100 if results["real"]["total"] > 0 else 0
fake_acc = results["fake"]["correct"] / results["fake"]["total"] * 100 if results["fake"]["total"] > 0 else 0
overall = (results["real"]["correct"] + results["fake"]["correct"]) / (results["real"]["total"] + results["fake"]["total"]) * 100

real_avg = sum(results["real"]["scores"]) / len(results["real"]["scores"]) if results["real"]["scores"] else 0
fake_avg = sum(results["fake"]["scores"]) / len(results["fake"]["scores"]) if results["fake"]["scores"] else 0

print(f"\nREAL videos: {results['real']['correct']}/{results['real']['total']} correct ({real_acc:.1f}%)")
print(f"  Avg visual score: {real_avg:.3f}")
print(f"  Range: [{min(results['real']['scores']):.3f}, {max(results['real']['scores']):.3f}]")

print(f"\nFAKE videos: {results['fake']['correct']}/{results['fake']['total']} correct ({fake_acc:.1f}%)")
print(f"  Avg visual score: {fake_avg:.3f}")
print(f"  Range: [{min(results['fake']['scores']):.3f}, {max(results['fake']['scores']):.3f}]")

print(f"\n{'='*80}")
print(f"OVERALL ACCURACY: {overall:.1f}%")
print(f"{'='*80}")

if overall >= 80:
    print("\n✅ EXCELLENT - Model is performing well!")
elif overall >= 60:
    print("\n✅ GOOD - Model is acceptable")
else:
    print("\n⚠️ NEEDS IMPROVEMENT - Consider threshold adjustment")
