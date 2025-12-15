# Multimodal Deepfake Detection System - Deployment Guide

## ✅ Implementation Complete

All components have been implemented successfully:

### 1. ✅ Database Schema Updated
- Added `detector_scores` JSONB column for multimodal scores
- Added `model_metadata` JSONB column for processing information
- Applied migration to Supabase

### 2. ✅ Model Weights Organized
Location: `/home/lightdesk/Projects/AI-Video/modal_services/weights/`
- **EfficientNet-B7**: `efficientnet_b7_deepfake.pt` (256 MB)
- **Wav2Vec2**: `model.safetensors` (361 MB) + config files
- **RetinaFace**: `retinaface_resnet50.pth` (105 MB)

### 3. ✅ Modal Services Created
- **Preprocessing** (`preprocessing.py`): Face extraction with MTCNN, audio extraction
- **Visual Detection** (`visual_detector.py`): EfficientNet-B7 on T4 GPU
- **Audio Detection** (`audio_detector.py`): Wav2Vec2 on T4 GPU
- **Fusion Layer** (`fusion_layer.py`): Weighted multimodal score combination
- **Main API** (`deepfake_detector.py`): All-in-one FastAPI orchestration

### 4. ✅ WhatsApp Bot Integration
- Updated `message_handler.py` to trigger video detection
- Added `/api/detection_callback` endpoint in `app.py`
- Created `modal_service.py` for Modal API calls

### 5. ✅ Website UI Updated
- Created `DetectorBreakdown.tsx` component for multimodal score visualization
- Updated `dashboard/page.tsx` to show detailed analysis for videos
- Added expandable sections for multimodal breakdowns
- Updated TypeScript types with new fields

---

## 🚀 Deployment Instructions

### Step 1: Authenticate with Modal

```bash
# Navigate to project directory
cd /home/lightdesk/Projects/AI-Video

# Authenticate Modal (one-time setup)
modal token new
```

**→ Browser will open**: https://modal.com/token-flow/tf-1Ijv9UXTg8nQFMhHjEkGnh

Follow the instructions to authorize your account.

### Step 2: Deploy to Modal

```bash
# Deploy the deepfake detection app
modal deploy modal_services/deepfake_detector.py
```

**Expected output:**
```
✓ Created objects.
├── 🔨 Created mount /home/lightdesk/Projects/AI-Video/modal_services/weights
├── 🔨 Created function DeepfakeDetector.process_video_pipeline.
└── 🔨 Created web function FastAPI app => https://your-username--deepfake-detection-complete-fastapi-app.modal.run
```

**Copy the URL** from the output (e.g., `https://your-username--deepfake-detection-complete-fastapi-app.modal.run`)

### Step 3: Update Environment Variables

Update your WhatsApp bot `.env` file:

```bash
# /home/lightdesk/Projects/AI-Video/whatsapp/.env
MODAL_VIDEO_API_URL=https://your-username--deepfake-detection-complete-fastapi-app.modal.run
FLASK_BASE_URL=https://your-ngrok-url.ngrok.io  # Your WhatsApp bot webhook URL
```

### Step 4: Test the Modal API

```bash
# Test health endpoint
curl https://your-username--deepfake-detection-complete-fastapi-app.modal.run/health

# Test with a video (using test video from Test-Video folder)
# First, upload a test video to Supabase storage and get its public URL
# Then:
curl -X POST https://your-username--deepfake-detection-complete-fastapi-app.modal.run/detect_video \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://your-supabase-url.co/storage/v1/object/public/video-uploads/test.mp4",
    "task_id": "test-001"
  }'

# Check status
curl https://your-username--deepfake-detection-complete-fastapi-app.modal.run/status/test-001
```

### Step 5: Test End-to-End with Test Video

**Test videos available in**: `/home/lightdesk/Projects/AI-Video/Test-Video/`
- `20251214_2115_New Video_simple_compose_01kcerkqdkexmtad54h2s9fw9p.mp4`
- `20251214_2115_New Video_simple_compose_01kcerkqegez9snkd7tmx969rs.mp4`

**Testing workflow:**

1. **Upload test video to Supabase**:
```bash
# Use Supabase dashboard or CLI to upload to video-uploads bucket
# Get the public URL
```

2. **Send video via WhatsApp**:
   - Send video to your WhatsApp bot
   - Should receive: "🎥 Video uploaded successfully! 🔍 Analyzing for deepfakes..."

3. **Monitor processing**:
```bash
# Check WhatsApp bot logs
tail -f /home/lightdesk/Projects/AI-Video/whatsapp/logs/*.log

# Check Modal logs
modal app logs deepfake-detection-complete
```

4. **Verify database update**:
   - Check Supabase `detection_history` table
   - Should see `detection_result` change from "processing" → verdict
   - `detector_scores` should be populated with JSON
   - `model_metadata` should contain processing info

5. **Check website dashboard**:
   - Navigate to `/dashboard`
   - Find the video entry
   - Click "View Detailed Analysis"
   - Should see multimodal breakdown with score bars

---

## 📊 Expected Results

### Multimodal Score Breakdown

For a typical video, you should see:

```json
{
  "final_score": 0.6543,
  "is_deepfake": true,
  "confidence_percent": 30.86,
  "verdict": "LIKELY DEEPFAKE",
  "has_audio": true,
  "breakdown": {
    "visual_artifacts": 0.7234,
    "temporal_consistency": 0.8456,
    "audio_synthesis": 0.6123,
    "face_quality": 0.9501
  },
  "model_metadata": {
    "models_used": ["MTCNN", "EfficientNet-B7", "Wav2Vec2"],
    "processing_time_seconds": 45.23,
    "frames_analyzed": 60,
    "video_duration_seconds": 30.0
  }
}
```

### Dashboard UI

Videos will show:
- ✅ Standard detection card with file info
- ✅ "View Detailed Analysis" button (only for videos)
- ✅ Expandable section with:
  - Visual Authenticity bar (blue)
  - Temporal Consistency bar (green)
  - Audio Authenticity bar (purple) - if audio present
  - Face Detection Quality bar (orange)

---

## 🐛 Troubleshooting

### Issue: Modal deployment fails

**Solution 1**: Check authentication
```bash
modal token new
```

**Solution 2**: Verify weights directory exists
```bash
ls -lh modal_services/weights/
# Should show 3 files: efficientnet_b7_deepfake.pt, model.safetensors, retinaface_resnet50.pth
```

### Issue: Video detection stays "processing"

**Check 1**: Modal logs
```bash
modal app logs deepfake-detection-complete --follow
```

**Check 2**: Callback URL is reachable
```bash
# Test if WhatsApp bot callback endpoint is accessible
curl -X POST https://your-ngrok-url.ngrok.io/api/detection_callback \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test", "result": {"verdict": "TEST"}}'
```

**Check 3**: Video URL is publicly accessible
```bash
# Test if Modal can download the video
curl -I https://your-supabase-url.co/storage/v1/object/public/video-uploads/test.mp4
# Should return 200 OK
```

### Issue: No faces detected

**Possible causes**:
- Video quality too low
- Face too small in frame
- Poor lighting
- Face not visible for enough frames

**Solution**: Try with a video that has:
- Clear, front-facing face
- Good lighting
- Face takes up >20% of frame
- At least 5 seconds duration

### Issue: Audio analysis skipped

**This is normal if**:
- Video has no audio track (some screen recordings)
- Audio extraction failed (corrupted file)

**Verification**:
```bash
# Check if video has audio using ffmpeg
ffprobe -v error -show_entries stream=codec_type /path/to/video.mp4
```

---

## 💰 Cost Estimates (Modal GPU Pricing)

### T4 GPU Pricing
- **Rate**: $0.00060/second
- **Per minute**: ~$0.036
- **Average 30s video**: ~$0.05 per detection

### Breakdown
- Preprocessing (face extraction): ~10s
- Visual detection (EfficientNet-B7): ~25s
- Audio detection (Wav2Vec2): ~10s
- **Total**: ~45 seconds = **$0.027 per video**

### Monthly estimates
- 100 videos/month: **$2.70**
- 1,000 videos/month: **$27**
- 10,000 videos/month: **$270**

---

## 🔄 Next Steps (Optional Enhancements)

### 1. Add VideoMAE Temporal Analysis
Currently using simplified temporal analysis (std of visual scores). To add full VideoMAE:

```python
# In modal_services/deepfake_detector.py, add:
from transformers import VideoMAEForVideoClassification

# Add temporal detection function
async def detect_temporal_artifacts(frames):
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base").cuda()
    # Process frame sequences...
    return temporal_score
```

### 2. Add Result Caching
Prevent re-processing same videos:

```python
# Check file hash before processing
import hashlib
file_hash = hashlib.sha256(file_content).hexdigest()
# Query database for existing results
```

### 3. Add WhatsApp Notifications
Send results back to user via WhatsApp:

```python
# In detection_callback endpoint
from whatsapp_service import send_whatsapp_message

# Send notification
send_whatsapp_message(
    user_phone,
    f"✅ Detection complete: {result['verdict']}\n"
    f"Confidence: {result['confidence_percent']}%"
)
```

### 4. Batch Processing
Process multiple videos concurrently:

```python
@app.function()
async def batch_process_videos(video_urls: List[str]):
    results = await asyncio.gather(*[
        process_video_pipeline.spawn(url, None, f"batch-{i}")
        for i, url in enumerate(video_urls)
    ])
    return results
```

---

## 📝 Architecture Summary

```
User uploads video
     ↓
WhatsApp Bot / Website
     ↓
Upload to Supabase Storage
     ↓
Create detection_history record (status: "pending")
     ↓
Call Modal API (/detect_video)
     ↓
Modal spawns async task (status: "processing")
     ↓
┌─────────────────────────────────┐
│  Modal T4 GPU Processing        │
│  1. Extract faces (MTCNN)       │
│  2. Extract audio (FFmpeg)      │
│  3. Visual detection (EffNet)   │
│  4. Audio detection (Wav2Vec2)  │
│  5. Fuse scores (weighted)      │
└─────────────────────────────────┘
     ↓
Send callback to WhatsApp bot
     ↓
Update detection_history with results
     ↓
User sees results in dashboard
```

---

## ✅ Verification Checklist

Before marking as complete, verify:

- [ ] Modal app deployed successfully
- [ ] Health endpoint returns `{"status": "healthy"}`
- [ ] Test video detection returns task_id
- [ ] Status endpoint shows "processing" then "completed"
- [ ] Database record updated with detector_scores
- [ ] Dashboard shows multimodal breakdown
- [ ] WhatsApp bot triggers detection for videos
- [ ] Callback endpoint receives and stores results
- [ ] All 3 model weights load without errors in Modal logs

---

**Status**: ✅ All code implemented, ready for deployment and testing

**Last Updated**: December 14, 2025
