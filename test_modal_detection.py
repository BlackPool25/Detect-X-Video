#!/usr/bin/env python3
"""
Test script for multimodal deepfake detection system
Tests both videos in Test-Video/ folder and evaluates accuracy
"""
import requests
import time
import json
from pathlib import Path
import sys

MODAL_API_URL = "https://blackpool25--deepfake-detection-complete-fastapi-app.modal.run"

def test_health():
    """Test if Modal API is healthy"""
    print("🔍 Testing Modal API health...")
    response = requests.get(f"{MODAL_API_URL}/health")
    if response.status_code == 200:
        print(f"✅ Health check passed: {response.json()}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def upload_to_supabase(video_path):
    """
    Upload video to Supabase storage
    NOTE: You'll need to implement this with your Supabase client
    For now, returning a mock URL
    """
    # TODO: Implement actual Supabase upload
    # from supabase import create_client
    # client = create_client(url, key)
    # with open(video_path, 'rb') as f:
    #     result = client.storage.from_('video-uploads').upload(
    #         Path(video_path).name, f
    #     )
    # return public_url
    
    print(f"⚠️  Skipping Supabase upload - using local file path for testing")
    return f"file://{video_path}"

def submit_detection(video_url, task_id):
    """Submit video for detection"""
    print(f"\n📤 Submitting detection for task: {task_id}")
    payload = {
        "video_url": video_url,
        "task_id": task_id
    }
    
    response = requests.post(
        f"{MODAL_API_URL}/detect_video",
        json=payload,
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Detection submitted: {result}")
        return result
    else:
        print(f"❌ Detection submission failed: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def check_status(task_id, max_wait=300):
    """Poll for detection results"""
    print(f"\n⏳ Waiting for results (max {max_wait}s)...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{MODAL_API_URL}/status/{task_id}")
        
        if response.status_code == 200:
            status_data = response.json()
            current_status = status_data.get("status")
            
            if current_status == "completed":
                print(f"✅ Detection completed!")
                return status_data.get("result")
            elif current_status == "failed":
                print(f"❌ Detection failed: {status_data.get('error')}")
                return None
            else:
                elapsed = int(time.time() - start_time)
                print(f"⏳ Status: {current_status} ({elapsed}s elapsed)")
                time.sleep(5)
        else:
            print(f"⚠️  Status check failed: {response.status_code}")
            time.sleep(5)
    
    print(f"⏰ Timeout after {max_wait}s")
    return None

def analyze_results(results, video_name):
    """Analyze and display detection results"""
    print(f"\n" + "="*80)
    print(f"📊 RESULTS FOR: {video_name}")
    print("="*80)
    
    if not results:
        print("❌ No results available")
        return
    
    print(f"\n🎯 Verdict: {results.get('verdict', 'UNKNOWN')}")
    print(f"📈 Final Score: {results.get('final_score', 0):.4f}")
    print(f"💯 Confidence: {results.get('confidence_percent', 0):.2f}%")
    print(f"🎵 Has Audio: {results.get('has_audio', False)}")
    
    breakdown = results.get('breakdown', {})
    if breakdown:
        print(f"\n📊 Detector Breakdown:")
        print(f"  🔍 Visual Artifacts: {breakdown.get('visual_artifacts', 0):.4f}")
        print(f"  🎬 Temporal Consistency: {breakdown.get('temporal_consistency', 0):.4f}")
        
        if results.get('has_audio'):
            print(f"  🎵 Audio Synthesis: {breakdown.get('audio_synthesis', 0):.4f}")
        
        print(f"  👤 Face Quality: {breakdown.get('face_quality', 0):.4f}")
    
    metadata = results.get('model_metadata', {})
    if metadata:
        print(f"\n⚙️  Processing Metadata:")
        print(f"  Models: {', '.join(metadata.get('models_used', []))}")
        print(f"  Processing Time: {metadata.get('processing_time_seconds', 0):.2f}s")
        print(f"  Frames Analyzed: {metadata.get('frames_analyzed', 0)}")
        print(f"  Video Duration: {metadata.get('video_duration_seconds', 0):.1f}s")
    
    print("="*80)

def main():
    """Run comprehensive test on both videos"""
    print("\n" + "🎬 MULTIMODAL DEEPFAKE DETECTION - COMPREHENSIVE TEST")
    print("="*80)
    
    # Health check
    if not test_health():
        print("\n❌ Modal API is not healthy. Exiting.")
        sys.exit(1)
    
    # Get test videos
    test_dir = Path(__file__).parent / "Test-Video"
    videos = list(test_dir.glob("*.mp4"))
    
    if not videos:
        print(f"\n❌ No test videos found in {test_dir}")
        sys.exit(1)
    
    print(f"\n✅ Found {len(videos)} test video(s)")
    
    results_summary = []
    
    # Test each video
    for idx, video_path in enumerate(videos, 1):
        video_name = video_path.name
        task_id = f"test-{idx}-{int(time.time())}"
        
        print(f"\n{'='*80}")
        print(f"📹 TEST {idx}/{len(videos)}: {video_name}")
        print(f"{'='*80}")
        
        # For now, we'll test with file:// URLs since Supabase upload is not implemented
        # In production, you'd upload to Supabase first
        video_url = f"file://{video_path.absolute()}"
        
        # Note: Modal won't be able to access local files
        # This is just to show the flow. You need to upload to Supabase first.
        print(f"⚠️  WARNING: Testing with local path - Modal cannot access this")
        print(f"   You need to upload videos to Supabase storage first")
        print(f"   Skipping actual detection for local file")
        
        results_summary.append({
            "video": video_name,
            "task_id": task_id,
            "status": "skipped - local file",
            "note": "Upload to Supabase storage first"
        })
        
        continue
        
        # Uncomment below when videos are uploaded to Supabase
        """
        # Submit detection
        submission = submit_detection(video_url, task_id)
        if not submission:
            results_summary.append({
                "video": video_name,
                "status": "submission_failed"
            })
            continue
        
        # Wait for results
        results = check_status(task_id, max_wait=300)
        
        # Analyze results
        analyze_results(results, video_name)
        
        results_summary.append({
            "video": video_name,
            "task_id": task_id,
            "status": "completed" if results else "failed",
            "verdict": results.get('verdict') if results else None,
            "score": results.get('final_score') if results else None
        })
        """
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("📝 FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    for idx, summary in enumerate(results_summary, 1):
        print(f"{idx}. {summary['video']}")
        print(f"   Status: {summary['status']}")
        if 'verdict' in summary and summary['verdict']:
            print(f"   Verdict: {summary['verdict']}")
            print(f"   Score: {summary['score']:.4f}")
        print()
    
    print("\n💡 NEXT STEPS:")
    print("1. Upload test videos to Supabase storage bucket 'video-uploads'")
    print("2. Get the public URLs for the uploaded videos")
    print("3. Update this script to use those URLs")
    print("4. Re-run the test to see actual detection results")
    print("\nExample Supabase upload code:")
    print("""
from supabase import create_client
import os

client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

with open('Test-Video/video.mp4', 'rb') as f:
    result = client.storage.from_('video-uploads').upload(
        'test-video-1.mp4', f
    )

public_url = client.storage.from_('video-uploads').get_public_url('test-video-1.mp4')
print(f"Video URL: {public_url}")
""")

if __name__ == "__main__":
    main()
