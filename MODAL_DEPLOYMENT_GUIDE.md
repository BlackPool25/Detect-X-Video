# Modal Deployment Guide - Optimized Deepfake Detection

## Overview
This guide walks through deploying the optimized single-layer deepfake detection pipeline to Modal with T4 GPU.

**Key Features:**
- 80% accuracy on DFDC dataset
- ~1.78s processing time per video
- T4 GPU optimized with FP16 inference
- Fast cold start with model pre-loading
- Keep-warm=1 for faster subsequent requests

## Prerequisites

1. **Modal Account Setup**
```bash
pip install modal
modal token new
```

2. **Required Files**
- `modal_app_optimized.py` - Deployment script
- `weights/SBI/FFc23.tar` - SBI model weights

## Deployment Steps

### 1. Test Locally (Optional)
```bash
# Test with a sample video
modal run modal_app_optimized.py --video-path Test-Video/Real/sample.mp4
```

### 2. Deploy to Modal
```bash
# Deploy the app
modal deploy modal_app_optimized.py

# Expected output:
# ✓ Created objects.
# ├── 🔨 Created mount /root/weights/SBI
# ├── 🔨 Created DeepfakeDetector => https://your-workspace--deepfake-detector-optimized-deepfakedetector.modal.run
# ├── 🔨 Created detect_video => https://your-workspace--deepfake-detector-optimized-detect-video.modal.run
# └── 🔨 Created health => https://your-workspace--deepfake-detector-optimized-health.modal.run
# 
# View Deployment: https://modal.com/apps/your-workspace/deepfake-detector-optimized
```

### 3. Get Your Deployment URLs

After deployment, Modal will provide endpoints like:
```
Health Check: https://your-workspace--deepfake-detector-optimized-health.modal.run
Video Detection: https://your-workspace--deepfake-detector-optimized-detect-video.modal.run
```

**Save these URLs** - you'll need them for integration.

## Integration Setup

### Website Integration

1. **Update Environment Variables**

Edit `AI-Website/.env.local`:
```bash
MODAL_VIDEO_API_URL=https://your-workspace--deepfake-detector-optimized-detect-video.modal.run
# Optional: Add API key if you configure one on Modal
# MODAL_API_KEY=your_api_key_here
```

2. **Test Website API**
```bash
cd AI-Website
npm run dev

# The /api/detect route should now call your Modal endpoint
```

### WhatsApp Integration

1. **Update Modal URL**

Edit `whatsapp/modal_service.py` or set environment variable:
```bash
export MODAL_VIDEO_API_URL="https://your-workspace--deepfake-detector-optimized-detect-video.modal.run"
```

2. **Test WhatsApp Handler**
```python
from whatsapp.modal_service import detect_video_multimodal

result = detect_video_multimodal(
    video_url="https://example.com/video.mp4",
    threshold=0.33
)
print(result)
```

## API Usage

### Video Detection Endpoint

**POST** `/detect-video`

**Request:**
```json
{
  "video_url": "https://example.com/video.mp4",
  "threshold": 0.33
}
```

**Response:**
```json
{
  "is_fake": false,
  "confidence": 0.92,
  "label": "REAL",
  "probability_fake": 0.08,
  "processing_time": 1.78,
  "model_version": "SBI-EfficientNet-B4-Optimized-v1",
  "frames_analyzed": 8
}
```

### Health Check Endpoint

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "model": "SBI-EfficientNet-B4-Optimized-v1",
  "gpu_available": true,
  "device": "cuda"
}
```

## Testing

### 1. Health Check
```bash
curl https://your-workspace--deepfake-detector-optimized-health.modal.run
```

Expected: `{"status":"healthy","model":"SBI-EfficientNet-B4-Optimized-v1","gpu_available":true,"device":"cuda"}`

### 2. Video Detection Test
```bash
curl -X POST https://your-workspace--deepfake-detector-optimized-detect-video.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/test-video.mp4",
    "threshold": 0.33
  }'
```

### 3. Performance Benchmarks

Expected metrics:
- **Cold Start:** ~10-15 seconds (first request after idle)
- **Warm Start:** ~1.5-2.5 seconds (subsequent requests)
- **GPU Utilization:** ~60-80% on T4
- **Memory:** ~4-6GB GPU memory

### 4. End-to-End Website Test

1. Go to your website's upload page
2. Upload a video
3. Check that:
   - Video uploads to Supabase storage
   - Public URL is generated
   - API calls Modal endpoint
   - Results display correctly

### 5. End-to-End WhatsApp Test

1. Send video to WhatsApp bot
2. Bot should:
   - Upload video to temporary storage
   - Call Modal API
   - Return result with confidence

## Monitoring

### Modal Dashboard
- View logs: `modal app logs deepfake-detector-optimized`
- Monitor usage: https://modal.com/apps/your-workspace/deepfake-detector-optimized

### Key Metrics to Watch
- Invocation count
- Average processing time
- Error rate
- GPU utilization

## Troubleshooting

### Issue: "No faces detected in video"
**Solution:** Video quality too low or no clear faces. Recommend:
- Minimum 480p resolution
- Clear frontal face views
- Good lighting

### Issue: Slow cold starts (>20s)
**Causes:**
- Weights not properly mounted
- Model extraction taking time
- T4 GPU not available

**Solution:**
```python
# Check keep_warm setting in modal_app_optimized.py
@app.cls(
    ...
    keep_warm=1  # Keep 1 container warm
)
```

### Issue: "GPU out of memory"
**Causes:**
- Video too long (>1min)
- Too many concurrent requests

**Solution:**
- Limit video duration to 30-60 seconds
- Increase GPU memory if needed

### Issue: Website API timeout
**Causes:**
- Modal cold start
- Very long video

**Solution:**
- Increase timeout in `AI-Website/app/api/detect/route.ts`
- Add loading indicator to UI

## Cost Optimization

### Modal Pricing (Approximate)
- T4 GPU: ~$0.60/hour
- Storage: ~$0.15/GB/month
- Egress: ~$0.10/GB

### Optimization Tips
1. **Keep-Warm Setting**
   - `keep_warm=1`: 1 container always ready (faster, slightly higher cost)
   - `keep_warm=0`: Spin up on demand (slower cold start, lower cost)

2. **Video Length Limits**
   - Recommend max 60 seconds
   - Longer videos = higher processing cost

3. **Batch Processing**
   - For multiple videos, consider batch endpoint
   - Amortize cold start cost

## Security

### API Key Setup (Optional)
1. Generate API key in Modal dashboard
2. Add to environment variables:
   ```bash
   export MODAL_API_KEY="your_secure_key"
   ```
3. Modal will validate requests with `Authorization: Bearer <key>` header

### Rate Limiting
Consider implementing rate limiting in your website API to prevent abuse.

## Production Checklist

- [ ] Modal app deployed successfully
- [ ] Health endpoint returns healthy
- [ ] Test video detection works
- [ ] Website environment variables updated
- [ ] WhatsApp service configured
- [ ] End-to-end website test passed
- [ ] End-to-end WhatsApp test passed
- [ ] Monitoring dashboard set up
- [ ] Error alerting configured
- [ ] API rate limiting implemented
- [ ] Cost alerts configured in Modal

## Support

- Modal Docs: https://modal.com/docs
- Modal Community: https://modal.com/slack
- Project Issues: Check your project repository

## Performance Comparison

| Metric | Old 4-Layer | New Optimized |
|--------|-------------|---------------|
| Accuracy | 0% (broken) | 80% |
| Speed | 3min | 1.78s |
| GPU | Multiple | T4 only |
| Architecture | 4 layers | 1 layer (SBI) |
| Threshold | 0.5 | 0.33 |
| Model | Ensemble | EfficientNet-B4 |

## Next Steps

1. **Monitor Performance**
   - Track accuracy on real-world data
   - Monitor processing times
   - Check cost metrics

2. **Iterate if Needed**
   - Adjust threshold if false positive/negative rates change
   - Consider adding ensemble if accuracy needs boost
   - Optimize frame count for speed/accuracy tradeoff

3. **Scale**
   - Increase `keep_warm` if traffic increases
   - Consider autoscaling settings
   - Implement caching for repeated videos
