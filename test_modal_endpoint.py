#!/usr/bin/env python3
"""
Test Modal deepfake detection endpoint
"""
import requests
import json
import time

MODAL_URL = "https://blackpool25--deepfake-detector-optimized-detect-video.modal.run"
HEALTH_URL = "https://blackpool25--deepfake-detector-optimized-health.modal.run"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(HEALTH_URL, timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"✓ Health check passed")
        print(f"  Model: {result.get('model')}")
        print(f"  Device: {result.get('device')}")
        print(f"  GPU: {result.get('gpu_available')}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_detection(video_url: str, threshold: float = 0.33):
    """Test video detection endpoint"""
    print(f"\nTesting detection with video: {video_url[:50]}...")
    print(f"Threshold: {threshold}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            MODAL_URL,
            params={"video_url": video_url, "threshold": threshold},
            timeout=180  # 3 minutes max
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
        
        result = response.json()
        print(f"\n✓ Detection completed in {elapsed:.2f}s")
        print(f"\nRaw response:")
        print(json.dumps(result, indent=2))
        
        # Check if error in response
        if 'error' in result:
            print(f"\n✗ API returned error: {result.get('error')}")
            return False
        
        print(f"\nResults:")
        print(f"  Is Fake: {result.get('is_fake')}")
        if result.get('confidence') is not None:
            print(f"  Confidence: {result.get('confidence'):.2%}")
        else:
            print(f"  Confidence: None (no faces detected?)")
        print(f"  Label: {result.get('label')}")
        if result.get('probability_fake') is not None:
            print(f"  Probability Fake: {result.get('probability_fake'):.4f}")
        else:
            print(f"  Probability Fake: None")
        if result.get('processing_time') is not None:
            print(f"  Processing Time: {result.get('processing_time'):.2f}s")
        print(f"  Frames Analyzed: {result.get('frames_analyzed')}")
        print(f"  Model: {result.get('model_version')}")
        
        return True
        
    except requests.Timeout:
        print(f"✗ Request timed out after {time.time() - start_time:.2f}s")
        return False
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Modal Deepfake Detection Endpoint Test")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n⚠ Health check failed, but continuing with detection test...")
    
    # Test with a small public video
    print("\n" + "=" * 60)
    print("Test 1: Small sample video")
    print("=" * 60)
    test_detection(
        "https://www.w3schools.com/html/mov_bbb.mp4",
        threshold=0.33
    )
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
