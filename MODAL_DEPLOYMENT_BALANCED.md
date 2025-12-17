# Modal Deployment Guide - Balanced 3-Layer Pipeline

## Overview

This deployment implements the full **pipeline_balanced.py** architecture on Modal with T4 GPU:

- **Layer 1**: Visual Artifacts (SBI EfficientNet-B4) - 80% weight
- **Layer 2**: Temporal Consistency - 15% weight  
- **Layer 3**: Audio-Visual Sync - 5% weight

**Target**: 85-90% accuracy in <5 seconds per video

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal setup
   ```

3. **Weights**: Ensure SBI weights are in `./weights/SBI/FFc23.tar`

## Deployment Steps

### 1. Deploy to Modal

```bash
# Deploy the app
modal deploy modal_app_balanced.py
```

This will:
- Build the Docker image with all dependencies
- Upload SBI weights to Modal
- Deploy 3 endpoints: `/detect-video`, `/health`, and a class method

### 2. Get Your Endpoint URL

After deployment, Modal will output:
```
✓ Created web function detect_video => https://your-username--deepfake-detector-balanced-3layer-detect-video.modal.run
✓ Created web function health => https://your-username--deepfake-detector-balanced-3layer-health.modal.run
```

### 3. Test Health Endpoint

```bash
curl https://your-username--deepfake-detector-balanced-3layer-health.modal.run
```

Expected response:
```json
{
  "status": "healthy",
  "model": "Balanced-3Layer-Pipeline-v1",
  "layers": [
    "Layer 1: Visual Artifacts (SBI EfficientNet-B4)",
    "Layer 2: Temporal Consistency",
    "Layer 3: Audio-Visual Sync"
  ],
  "gpu_available": true,
  "device": "cuda",
  "weights": [0.80, 0.15, 0.05]
}
```

### 4. Test Detection Endpoint

Using curl:
```bash
curl -X POST https://your-username--deepfake-detector-balanced-3layer-detect-video.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/test-video.mp4",
    "enable_fail_fast": false
  }'
```

Using Python:
```python
import requests

response = requests.post(
    "https://your-username--deepfake-detector-balanced-3layer-detect-video.modal.run",
    json={
        "video_url": "https://example.com/test-video.mp4",
        "enable_fail_fast": False
    }
)

result = response.json()
print(f"Verdict: {result['final_verdict']}")
print(f"Confidence: {result['confidence']:.2%}")
```

Using the test script:
```bash
python test_modal_balanced.py \
  --url https://your-username--deepfake-detector-balanced-3layer-detect-video.modal.run \
  --test-video https://example.com/test-video.mp4
```

## Response Format

Successful detection:
```json
{
  "video_path": "https://example.com/video.mp4",
  "final_verdict": "FAKE",
  "confidence": 0.85,
  "stopped_at_layer": "Ensemble",
  "layer_results": [
    {
      "layer_name": "Layer 1: Visual Artifacts",
      "is_fake": true,
      "confidence": 0.87,
      "processing_time": 1.82,
      "details": {
        "avg_fake_probability": 0.87,
        "max_fake_probability": 0.92,
        "threshold": 0.33,
        "frames_analyzed": 24
      }
    },
    {
      "layer_name": "Layer 2: Temporal Consistency",
      "is_fake": false,
      "confidence": 0.45,
      "processing_time": 0.52,
      "details": {
        "temporal_score": 0.12,
        "std_diff": 0.034,
        "mean_diff": 0.28,
        "threshold": 0.15
      }
    },
    {
      "layer_name": "Layer 3: Audio-Visual Sync",
      "is_fake": false,
      "confidence": 0.50,
      "processing_time": 1.45,
      "details": {
        "coefficient_variation": 0.68,
        "energy_std": 0.042,
        "energy_mean": 0.062,
        "note": "Basic heuristic, low confidence"
      }
    }
  ],
  "total_time": 3.85
}
```

Error response:
```json
{
  "error": "Failed to download video: 404 Not Found",
  "error_type": "DownloadError"
}
```

## Key Features

### 1. Proper Error Handling
- Uses `requests` library instead of `urllib` to avoid serialization errors
- All exceptions are caught and returned as serializable dicts
- No unpicklable objects (like HTTPError with BufferedReader)

### 2. Full Pipeline Implementation
- All 3 layers from `pipeline_balanced.py`
- Weighted ensemble voting (80/15/5)
- Same thresholds and parameters

### 3. GPU Optimization
- T4 GPU for cost-effective inference
- FP16 (half precision) for faster processing
- `keep_warm=1` to reduce cold starts

### 4. Fail-Fast Mode
- Set `enable_fail_fast=true` to stop after Layer 1 if confidence > 0.8
- Saves ~2 seconds for obvious fakes

## Cost Optimization

- **keep_warm=1**: Keeps 1 container warm (minimal cost)
- **T4 GPU**: Most cost-effective GPU for this workload
- **FP16**: Faster inference, lower memory
- **Fail-fast**: Optional early exit for high-confidence cases

## Monitoring

Check logs in Modal dashboard:
```bash
modal logs deepfake-detector-balanced-3layer
```

View running containers:
```bash
modal container list
```

## Troubleshooting

### Issue: "No faces detected"
**Solution**: Video may not contain faces or faces are too small
- Check video quality
- Ensure faces are visible and well-lit

### Issue: "Failed to download video"
**Solution**: Video URL is invalid or inaccessible
- Verify URL is public and valid
- Check if video format is supported (MP4 recommended)

### Issue: Slow processing (>10s)
**Solution**: 
- Enable fail-fast mode for obvious fakes
- Check if container is cold (first request takes longer)
- Verify T4 GPU is being used

### Issue: Serialization errors
**Solution**: This version uses `requests` instead of `urllib` to avoid unpicklable objects
- All errors are now returned as simple dicts
- No HTTPError exceptions with BufferedReader

## Next Steps

1. **Deploy**: `modal deploy modal_app_balanced.py`
2. **Get URL**: Copy the endpoint URL from deployment output
3. **Test**: Use `test_modal_balanced.py` or curl to verify
4. **Integrate**: Use the endpoint in your app (see AI-Website integration)

## Support

- Modal Docs: https://modal.com/docs
- Modal Discord: https://discord.gg/modal
- Issue Tracker: GitHub repo issues
