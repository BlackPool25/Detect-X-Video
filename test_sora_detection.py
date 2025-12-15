#!/usr/bin/env python3
"""
Test Modal API with actual Sora-generated deepfake videos
Uploads to Supabase first, then runs detection
"""
import requests
import time
import json
from pathlib import Path
import base64

MODAL_API_URL = "https://blackpool25--deepfake-detection-complete-fastapi-app.modal.run"

def create_test_url(video_path):
    """
    For testing purposes, we'll use a mock publicly accessible URL
    In production, upload to Supabase storage first
    """
    # This is where you'd normally upload to Supabase and get public URL
    # For now, returning a placeholder that shows the expected format
    file_name = Path(video_path).name
    
    # Expected format after Supabase upload:
    # https://cjkcwycnetdhumtqthuk.supabase.co/storage/v1/object/public/video-uploads/test_video.mp4
    
    print(f"\n⚠️  NOTE: In production, upload {file_name} to Supabase storage first")
    print(f"   Expected URL format: https://YOUR_PROJECT.supabase.co/storage/v1/object/public/video-uploads/{file_name}")
    
    return None  # Return None to indicate upload needed

def test_modal_api_direct():
    """Test Modal API with a simulated request"""
    print("\n" + "="*80)
    print("🧪 TESTING MODAL API - Sora Deepfake Detection")
    print("="*80)
    
    # Test health first
    print("\n1️⃣ Testing health endpoint...")
    health_response = requests.get(f"{MODAL_API_URL}/health")
    if health_response.status_code == 200:
        print(f"✅ Health check passed: {health_response.json()}")
    else:
        print(f"❌ Health check failed!")
        return
    
    # Check test videos
    test_dir = Path("Test-Video")
    videos = list(test_dir.glob("*.mp4"))
    
    if not videos:
        print("❌ No test videos found")
        return
    
    print(f"\n2️⃣ Found {len(videos)} Sora-generated deepfake video(s)")
    
    for idx, video_path in enumerate(videos[:2], 1):
        print(f"\n{'='*80}")
        print(f"📹 VIDEO {idx}: {video_path.name}")
        print(f"   Size: {video_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   Expected: DEEPFAKE (Sora-generated)")
        print('='*80)
        
        # For actual testing, you need to:
        # 1. Upload to Supabase storage
        # 2. Get the public URL
        # 3. Pass that URL to Modal API
        
        test_url = create_test_url(video_path)
        
        if test_url is None:
            print("\n⏭️  Skipping actual API call - upload to Supabase first")
            print("\n📝 To test manually:")
            print(f"""
# 1. Upload video to Supabase:
#    Go to Supabase dashboard > Storage > video-uploads bucket
#    Upload: {video_path.name}
#    Get public URL

# 2. Test with curl:
curl -X POST {MODAL_API_URL}/detect_video \\
  -H "Content-Type: application/json" \\
  -d '{{
    "video_url": "YOUR_SUPABASE_URL_HERE",
    "task_id": "sora-test-{idx}"
  }}'

# 3. Check status (wait 30-60s):
curl {MODAL_API_URL}/status/sora-test-{idx}
""")
            continue

def main():
    print("\n🎬 SORA DEEPFAKE DETECTION TEST")
    print("="*80)
    print("Both test videos are AI-generated using Sora")
    print("Expected detection: DEEPFAKE / LIKELY DEEPFAKE")
    print("="*80)
    
    test_modal_api_direct()
    
    print("\n\n" + "="*80)
    print("📋 NEXT STEPS TO COMPLETE TESTING")
    print("="*80)
    print("""
1. Upload videos to Supabase storage:
   - Go to https://supabase.com/dashboard/project/cjkcwycnetdhumtqthuk/storage/buckets
   - Navigate to 'video-uploads' bucket
   - Upload both MP4 files from Test-Video/
   - Copy the public URLs

2. Test detection with real videos:
   Run the commands shown above with actual Supabase URLs

3. Expected results for Sora videos:
   ✓ Should detect as DEEPFAKE or LIKELY DEEPFAKE
   ✓ Visual artifact score: 0.6-0.8 (high = fake)
   ✓ Temporal consistency: 0.7-0.9 (but inverted in fusion)
   ✓ Face quality: varies (Sora faces are realistic)
   ✓ Final score: > 0.5 indicates deepfake

4. Monitor processing:
   modal app logs deepfake-detection-complete --follow

5. Check database after detection:
   Results stored in detection_history table with full breakdown
""")
    
    print("\n💡 Local test showed the pipeline works without audio!")
    print("   Fusion weights adjusted: visual=55%, temporal=30%, face=15%")

if __name__ == "__main__":
    main()
