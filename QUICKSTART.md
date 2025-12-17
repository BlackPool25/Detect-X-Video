# Quick Start: Modal Deployment

## 1. Deploy to Modal

```bash
./deploy_modal.sh
```

This will:
- Check Modal CLI is installed
- Verify authentication
- Check SBI weights exist
- Deploy to Modal
- Show your deployment URLs

## 2. Update Configuration

Copy your deployment URL from the output and update:

**Website (.env.local):**
```bash
cd AI-Website
echo "MODAL_VIDEO_API_URL=https://your-workspace--deepfake-detector-optimized-detect-video.modal.run" >> .env.local
```

**WhatsApp (environment variable):**
```bash
export MODAL_VIDEO_API_URL="https://your-workspace--deepfake-detector-optimized-detect-video.modal.run"
```

## 3. Test Deployment

```bash
# Test health
python test_modal_deployment.py https://your-workspace--deepfake-detector-optimized

# Test with video
python test_modal_deployment.py https://your-workspace--deepfake-detector-optimized https://example.com/video.mp4
```

## 4. Test Website

```bash
cd AI-Website
npm run dev
# Visit http://localhost:3000 and upload a video
```

## 5. Monitor

View logs:
```bash
modal app logs deepfake-detector-optimized
```

Dashboard: https://modal.com/apps

## API Endpoints

**Health Check:**
```bash
curl https://your-workspace--deepfake-detector-optimized-health.modal.run
```

**Video Detection:**
```bash
curl -X POST https://your-workspace--deepfake-detector-optimized-detect-video.modal.run \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "threshold": 0.33}'
```

## Expected Performance

- **Accuracy:** 80% on DFDC dataset
- **Speed:** 1.5-2.5s per video (warm start)
- **Cold Start:** 10-15s (first request)
- **GPU:** T4 with FP16 inference
- **Frames:** 8 frames analyzed per video

## Troubleshooting

**Issue:** Deployment fails
- Check weights exist: `ls weights/SBI/FFc23.tar`
- Check Modal token: `modal token show`

**Issue:** Health check fails
- Wait 30s for deployment to complete
- Check Modal dashboard for errors

**Issue:** Slow detection (>5s)
- Check GPU is being used (health endpoint)
- Monitor Modal dashboard for cold starts
- Consider increasing `keep_warm` setting

## Files Changed

✓ `modal_app_optimized.py` - New optimized deployment
✓ `AI-Website/app/api/detect/route.ts` - Updated to call Modal
✓ `whatsapp/modal_service.py` - Updated to use new endpoint
✓ `deploy_modal.sh` - Deployment automation script
✓ `test_modal_deployment.py` - Validation test suite
✓ `MODAL_DEPLOYMENT_GUIDE.md` - Complete documentation
✓ `QUICKSTART.md` - This file

## Support

Detailed docs: [MODAL_DEPLOYMENT_GUIDE.md](./MODAL_DEPLOYMENT_GUIDE.md)
