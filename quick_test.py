#!/usr/bin/env python3
"""
Quick validation script - Tests the 4-layer pipeline on a few videos
to ensure everything works properly before full testing
"""

import sys
from pathlib import Path
import glob

# Check if we're in the right directory
if not Path("pipeline_production.py").exists():
    print("Error: Please run this script from the AI-Video directory")
    sys.exit(1)

from pipeline_production import DeepfakePipeline

def find_test_video():
    """Find a test video from Test-Video directory"""
    test_dirs = [
        "Test-Video/Real",
        "Test-Video/Fake", 
        "Test-Video/Celeb-real",
        "Test-Video/Celeb-synthesis",
        "Test-Video/YouTube-real"
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            # Find first video
            for ext in ['*.mp4', '*.avi', '*.mov']:
                videos = list(Path(test_dir).glob(ext))
                if videos:
                    return str(videos[0])
                # Try videos subdirectory
                videos = list(Path(test_dir).glob(f'videos/{ext}'))
                if videos:
                    return str(videos[0])
    
    return None

def main():
    print("=" * 80)
    print("QUICK VALIDATION TEST - 4-Layer Deepfake Detection Pipeline")
    print("=" * 80)
    print()
    
    # Check weights
    print("Checking weights...")
    sbi_weights = "weights/SBI/FFc23.tar"
    syncnet_model = "syncnet_python/data/syncnet_v2.model"
    
    if not Path(sbi_weights).exists():
        print(f"✗ SBI weights not found: {sbi_weights}")
        print("  Download from: https://drive.google.com/file/d/1X0-NYT8KPursLZZdxduRQju6E52hauV0")
        return
    else:
        print(f"✓ SBI weights found")
    
    if not Path(syncnet_model).exists():
        print(f"✗ SyncNet model not found: {syncnet_model}")
        print("  Run: cd syncnet_python && bash download_model.sh")
        return
    else:
        print(f"✓ SyncNet model found")
    
    # Find test video
    print("\nFinding test video...")
    test_video = find_test_video()
    
    if not test_video:
        print("✗ No test videos found in Test-Video/")
        print("  Please add videos to Test-Video/Real/ or Test-Video/Fake/")
        return
    
    print(f"✓ Using test video: {test_video}")
    
    # Initialize pipeline
    print("\n" + "=" * 80)
    print("Initializing pipeline...")
    print("=" * 80)
    
    try:
        pipeline = DeepfakePipeline(
            sbi_weights_path=sbi_weights,
            syncnet_model_path=syncnet_model,
            device="cuda"  # Change to "cpu" if no GPU
        )
    except Exception as e:
        print(f"\n✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run detection
    print("\n" + "=" * 80)
    print("Running detection...")
    print("=" * 80)
    
    try:
        result = pipeline.detect(
            video_path=test_video,
            enable_fail_fast=True
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUCCESSFUL!")
        print("=" * 80)
        print(f"\nVideo: {Path(test_video).name}")
        print(f"Final Verdict: {result.final_verdict}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Stopped at: {result.stopped_at_layer}")
        print(f"Total time: {result.total_time:.2f}s")
        
        print("\nLayer-by-layer results:")
        for lr in result.layer_results:
            status = "✓" if not lr.is_fake else "⚠"
            print(f"{status} {lr.layer_name}: {'FAKE' if lr.is_fake else 'REAL'} "
                  f"({lr.confidence:.2%} confidence, {lr.processing_time:.2f}s)")
        
        print("\n" + "=" * 80)
        print("Pipeline is working correctly!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run full test suite: python test_pipeline.py --max-videos 10")
        print("2. Test specific video: python pipeline_production.py --video <path>")
        print()
        
    except Exception as e:
        print(f"\n✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
