#!/usr/bin/env python3
"""
Quick test of Modal balanced 3-layer deployment with a sample video URL
"""
import requests
import json

# Modal endpoint
MODAL_URL = "https://blackpool25--deepfake-detector-balanced-3layer-detect-video.modal.run"

# Test with a small sample video (you can replace with any public video URL)
test_video_url = "https://www.w3schools.com/html/mov_bbb.mp4"  # Big Buck Bunny sample

print(f"Testing Modal Balanced 3-Layer Detection")
print(f"=" * 80)
print(f"Endpoint: {MODAL_URL}")
print(f"Video: {test_video_url}")
print(f"=" * 80)

try:
    print("\nSending request...")
    # FastAPI endpoint expects form data or query params, not JSON body for simple params
    response = requests.post(
        MODAL_URL,
        params={
            "video_url": test_video_url,
            "enable_fail_fast": False
        },
        timeout=120
    )
    
    response.raise_for_status()
    result = response.json()
    
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
    else:
        print(f"\n✅ Detection Complete!")
        print(f"\n{'=' * 80}")
        print(f"RESULT")
        print(f"{'=' * 80}")
        print(f"Final Verdict: {result['final_verdict']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Stopped At: {result['stopped_at_layer']}")
        print(f"Total Time: {result['total_time']:.2f}s")
        
        print(f"\n{'=' * 80}")
        print(f"LAYER BREAKDOWN")
        print(f"{'=' * 80}")
        for layer in result['layer_results']:
            print(f"\n{layer['layer_name']}:")
            print(f"  Verdict: {'FAKE' if layer['is_fake'] else 'REAL'}")
            print(f"  Confidence: {layer['confidence']:.2%}")
            print(f"  Time: {layer['processing_time']:.2f}s")
        
        print(f"\n{'=' * 80}")
        print(json.dumps(result, indent=2))

except requests.exceptions.Timeout:
    print("\n❌ Request timed out (>120s)")
except requests.exceptions.RequestException as e:
    print(f"\n❌ Request failed: {e}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
