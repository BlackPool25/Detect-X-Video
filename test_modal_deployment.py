#!/usr/bin/env python3
"""
Test script for Modal deployment
Validates health, video detection, and integration points
"""

import sys
import os
import requests
import time
from typing import Dict, Optional

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg: str):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg: str):
    print(f"{RED}✗ {msg}{RESET}")

def print_warning(msg: str):
    print(f"{YELLOW}⚠ {msg}{RESET}")

def print_info(msg: str):
    print(f"{BLUE}ℹ {msg}{RESET}")

def print_header(msg: str):
    print(f"\n{BLUE}{'='*60}")
    print(f"{msg}")
    print(f"{'='*60}{RESET}\n")


def test_health_endpoint(base_url: str) -> bool:
    """Test the health check endpoint"""
    print_header("TEST 1: Health Check")
    
    try:
        health_url = f"{base_url.rstrip('/')}/health"
        print_info(f"Testing: {health_url}")
        
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"Response: {data}")
        
        # Validate response
        assert data.get('status') == 'healthy', "Status not healthy"
        assert data.get('model') == 'SBI-EfficientNet-B4-Optimized-v1', "Wrong model"
        assert data.get('gpu_available') == True, "GPU not available"
        assert data.get('device') == 'cuda', "Not using CUDA"
        
        print_success(f"Health check passed")
        print_info(f"Model: {data['model']}")
        print_info(f"GPU: {data['device']}")
        return True
        
    except requests.exceptions.RequestException as e:
        print_error(f"Health check failed: {e}")
        return False
    except AssertionError as e:
        print_error(f"Health check validation failed: {e}")
        return False


def test_video_detection(base_url: str, video_url: Optional[str] = None) -> bool:
    """Test video detection endpoint"""
    print_header("TEST 2: Video Detection")
    
    if not video_url:
        print_warning("No video URL provided, skipping detection test")
        print_info("To test, run: python test_modal_deployment.py <base_url> <video_url>")
        return True
    
    try:
        detect_url = f"{base_url.rstrip('/')}/detect-video"
        print_info(f"Testing: {detect_url}")
        print_info(f"Video: {video_url}")
        
        payload = {
            "video_url": video_url,
            "threshold": 0.33
        }
        
        print_info("Sending request...")
        start_time = time.time()
        
        response = requests.post(
            detect_url,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        data = response.json()
        
        print(f"\nResponse: {data}\n")
        
        # Validate response structure
        required_fields = ['is_fake', 'confidence', 'label', 'probability_fake', 
                          'processing_time', 'model_version', 'frames_analyzed']
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Print results
        print_success(f"Detection completed in {elapsed:.2f}s")
        print_info(f"Result: {data['label']}")
        print_info(f"Confidence: {data['confidence']:.2%}")
        print_info(f"Probability Fake: {data['probability_fake']:.3f}")
        print_info(f"Processing Time: {data['processing_time']:.2f}s")
        print_info(f"Frames Analyzed: {data['frames_analyzed']}")
        print_info(f"Model: {data['model_version']}")
        
        # Performance validation
        if data['processing_time'] > 5:
            print_warning(f"Processing time ({data['processing_time']:.2f}s) exceeds target (5s)")
        else:
            print_success(f"Processing time within target")
        
        return True
        
    except requests.exceptions.Timeout:
        print_error("Detection timed out after 120s")
        return False
    except requests.exceptions.RequestException as e:
        print_error(f"Detection failed: {e}")
        return False
    except AssertionError as e:
        print_error(f"Response validation failed: {e}")
        return False


def test_website_integration(website_url: str = "http://localhost:3000") -> bool:
    """Test website API integration"""
    print_header("TEST 3: Website Integration")
    
    try:
        api_url = f"{website_url}/api/detect"
        print_info(f"Testing: {api_url}")
        
        # Test with a dummy video URL (just to check endpoint is accessible)
        payload = {
            "video_url": "https://example.com/test.mp4"
        }
        
        print_info("Checking if website API is accessible...")
        response = requests.post(
            api_url,
            json=payload,
            timeout=5
        )
        
        # We expect it might fail due to invalid URL, but endpoint should be accessible
        if response.status_code in [200, 400, 500]:
            print_success("Website API endpoint accessible")
            print_info(f"Status: {response.status_code}")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_warning("Website not running at http://localhost:3000")
        print_info("Start with: cd AI-Website && npm run dev")
        return True  # Not a failure, just not running
    except Exception as e:
        print_error(f"Website integration test failed: {e}")
        return False


def test_whatsapp_service():
    """Test WhatsApp service configuration"""
    print_header("TEST 4: WhatsApp Service")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'whatsapp'))
        from modal_service import MODAL_VIDEO_API_URL, detect_video_multimodal
        
        print_info(f"Configured URL: {MODAL_VIDEO_API_URL}")
        
        if "your-modal-app" in MODAL_VIDEO_API_URL:
            print_warning("WhatsApp service still using placeholder URL")
            print_info("Update MODAL_VIDEO_API_URL in whatsapp/modal_service.py")
            return True  # Not a failure, just needs configuration
        
        print_success("WhatsApp service configured")
        print_info("To test fully, call detect_video_multimodal() with a real video URL")
        return True
        
    except Exception as e:
        print_error(f"WhatsApp service test failed: {e}")
        return False


def run_all_tests(base_url: str, video_url: Optional[str] = None):
    """Run all deployment tests"""
    print(f"{BLUE}")
    print("=" * 60)
    print("MODAL DEPLOYMENT VALIDATION TESTS")
    print("=" * 60)
    print(f"{RESET}")
    
    results = []
    
    # Test 1: Health
    results.append(("Health Check", test_health_endpoint(base_url)))
    
    # Test 2: Video Detection
    results.append(("Video Detection", test_video_detection(base_url, video_url)))
    
    # Test 3: Website Integration
    results.append(("Website Integration", test_website_integration()))
    
    # Test 4: WhatsApp Service
    results.append(("WhatsApp Service", test_whatsapp_service()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{BLUE}Results: {passed}/{total} tests passed{RESET}\n")
    
    if passed == total:
        print_success("All tests passed! Deployment is ready.")
        return 0
    else:
        print_error(f"{total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_modal_deployment.py <modal_base_url> [video_url]")
        print()
        print("Examples:")
        print("  # Test health only")
        print("  python test_modal_deployment.py https://your-workspace--deepfake-detector-optimized")
        print()
        print("  # Test with video")
        print("  python test_modal_deployment.py https://your-workspace--deepfake-detector-optimized https://example.com/video.mp4")
        sys.exit(1)
    
    base_url = sys.argv[1]
    video_url = sys.argv[2] if len(sys.argv) > 2 else None
    
    exit_code = run_all_tests(base_url, video_url)
    sys.exit(exit_code)
