#!/usr/bin/env python3
"""
Test script for modal_app_balanced.py deployment
"""

import requests
import json
import time

# Test video URLs (public samples)
TEST_VIDEOS = {
    "real_sample": "https://github.com/ondyari/FaceForensics/raw/master/dataset/original_sequences/youtube/c23/videos/000.mp4",
    "fake_sample": "https://github.com/ondyari/FaceForensics/raw/master/dataset/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4",
}

def test_health_endpoint(base_url: str):
    """Test health check endpoint"""
    print("\n" + "="*80)
    print("Testing Health Endpoint")
    print("="*80)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print(f"✓ Status: {result['status']}")
        print(f"✓ Model: {result['model']}")
        print(f"✓ Layers: {len(result['layers'])}")
        print(f"✓ GPU Available: {result['gpu_available']}")
        print(f"✓ Device: {result['device']}")
        
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_detection_endpoint(base_url: str, video_url: str, label: str):
    """Test detection endpoint"""
    print("\n" + "="*80)
    print(f"Testing Detection: {label}")
    print("="*80)
    print(f"Video URL: {video_url}")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/detect-video",
            json={
                "video_url": video_url,
                "enable_fail_fast": False
            },
            timeout=120  # 2 minutes max
        )
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        
        # Check for errors
        if "error" in result:
            print(f"✗ Error: {result['error']}")
            return False
        
        print(f"\n✓ Detection completed in {elapsed:.2f}s")
        print(f"  Final Verdict: {result['final_verdict']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Stopped at: {result['stopped_at_layer']}")
        print(f"  Total Processing Time: {result['total_time']:.2f}s")
        
        print(f"\n  Layer Results:")
        for lr in result['layer_results']:
            print(f"    - {lr['layer_name']}: {'FAKE' if lr['is_fake'] else 'REAL'} ({lr['confidence']:.2%}) [{lr['processing_time']:.2f}s]")
        
        return True
        
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Modal endpoint URL (e.g., https://your-app.modal.run)")
    parser.add_argument("--test-video", help="Optional: Test with custom video URL")
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')
    
    print("="*80)
    print("Modal Balanced 3-Layer Deepfake Detector - Test Suite")
    print("="*80)
    print(f"Base URL: {base_url}")
    
    # Test 1: Health check
    if not test_health_endpoint(base_url):
        print("\n❌ Health check failed - aborting tests")
        return
    
    # Test 2: Detection with custom video or test samples
    if args.test_video:
        test_detection_endpoint(base_url, args.test_video, "Custom Video")
    else:
        print("\n⚠ No test video provided - skipping detection test")
        print("  Use --test-video <url> to test detection")
    
    print("\n" + "="*80)
    print("Test Suite Complete")
    print("="*80)


if __name__ == "__main__":
    main()
