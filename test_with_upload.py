#!/usr/bin/env python3
"""
Upload test videos to Supabase and run detection
"""
import os
import sys
from pathlib import Path
from supabase import create_client
import requests
import time

# Load environment
from dotenv import load_dotenv
load_dotenv('whatsapp/.env')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
MODAL_API_URL = os.getenv('MODAL_VIDEO_API_URL')

def upload_video_to_supabase(video_path):
    """Upload video to Supabase storage"""
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    file_name = f"test_{int(time.time())}_{Path(video_path).name}"
    
    print(f"📤 Uploading {Path(video_path).name} to Supabase...")
    
    with open(video_path, 'rb') as f:
        result = client.storage.from_('video-uploads').upload(
            file_name, f
        )
    
    # Get public URL
    public_url = client.storage.from_('video-uploads').get_public_url(file_name)
    
    print(f"✅ Uploaded to: {public_url}")
    return public_url, file_name

def run_detection(video_url, task_id):
    """Submit video for detection and wait for results"""
    print(f"\n🔍 Starting detection for task: {task_id}")
    
    # Submit
    response = requests.post(
        f"{MODAL_API_URL}/detect_video",
        json={"video_url": video_url, "task_id": task_id}
    )
    
    if response.status_code != 200:
        print(f"❌ Submission failed: {response.text}")
        return None
    
    print(f"✅ Detection submitted: {response.json()}")
    
    # Poll for results
    max_wait = 600  # 10 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status_response = requests.get(f"{MODAL_API_URL}/status/{task_id}")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            current_status = status_data.get("status")
            
            if current_status == "completed":
                return status_data.get("result")
            elif current_status == "failed":
                print(f"❌ Detection failed: {status_data.get('error')}")
                return None
            else:
                elapsed = int(time.time() - start_time)
                print(f"⏳ Status: {current_status} ({elapsed}s elapsed)")
                time.sleep(10)
        else:
            time.sleep(10)
    
    print(f"⏰ Timeout after {max_wait}s")
    return None

def display_results(results, video_name):
    """Display detection results"""
    print(f"\n{'='*80}")
    print(f"📊 RESULTS: {video_name}")
    print('='*80)
    
    if not results:
        print("❌ No results")
        return
    
    print(f"\n🎯 Verdict: {results.get('verdict')}")
    print(f"📈 Score: {results.get('final_score', 0):.4f}")
    print(f"💯 Confidence: {results.get('confidence_percent', 0):.1f}%")
    print(f"🎵 Audio: {'Yes' if results.get('has_audio') else 'No'}")
    
    breakdown = results.get('breakdown', {})
    print(f"\n📊 Breakdown:")
    print(f"  Visual: {breakdown.get('visual_artifacts', 0):.3f}")
    print(f"  Temporal: {breakdown.get('temporal_consistency', 0):.3f}")
    if results.get('has_audio'):
        print(f"  Audio: {breakdown.get('audio_synthesis', 0):.3f}")
    print(f"  Face Quality: {breakdown.get('face_quality', 0):.3f}")
    
    metadata = results.get('model_metadata', {})
    print(f"\n⚙️  Metadata:")
    print(f"  Time: {metadata.get('processing_time_seconds', 0):.1f}s")
    print(f"  Frames: {metadata.get('frames_analyzed', 0)}")
    print(f"  Duration: {metadata.get('video_duration_seconds', 0):.1f}s")

def main():
    print("🎬 MULTIMODAL DEEPFAKE DETECTION TEST")
    print("="*80)
    
    # Check test videos
    test_dir = Path("Test-Video")
    videos = list(test_dir.glob("*.mp4"))
    
    if not videos:
        print("❌ No videos in Test-Video/")
        sys.exit(1)
    
    print(f"✅ Found {len(videos)} video(s)")
    
    all_results = []
    
    for idx, video_path in enumerate(videos[:2], 1):  # Test first 2 videos
        print(f"\n\n{'#'*80}")
        print(f"VIDEO {idx}: {video_path.name}")
        print('#'*80)
        
        try:
            # Upload
            public_url, file_name = upload_video_to_supabase(video_path)
            
            # Detect
            task_id = f"test_{idx}_{int(time.time())}"
            results = run_detection(public_url, task_id)
            
            # Display
            display_results(results, video_path.name)
            
            all_results.append({
                'name': video_path.name,
                'verdict': results.get('verdict') if results else 'FAILED',
                'score': results.get('final_score') if results else 0,
                'results': results
            })
            
        except Exception as e:
            print(f"❌ Error testing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📝 SUMMARY")
    print('='*80)
    
    for idx, res in enumerate(all_results, 1):
        print(f"\n{idx}. {res['name']}")
        print(f"   Verdict: {res['verdict']}")
        print(f"   Score: {res['score']:.4f}")
    
    print(f"\n✅ Testing complete!")

if __name__ == "__main__":
    main()
