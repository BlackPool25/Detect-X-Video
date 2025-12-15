## Deepfake Detection Architecture Analysis

### Current Model Behavior (Post-Inversion)
- **Real videos**: Average score = 0.195 (range: 0.06 - 0.46)
- **Fake videos (Celeb-synthesis)**: Average score = 0.105 (range: 0.02 - 0.21)

### The "Too Good to Be True" Paradox
**Why fakes score LOWER than real:**
1. **Real videos** contain natural imperfections:
   - Sensor noise, compression artifacts
   - Irregular lighting, skin micro-textures
   - Natural video inconsistencies

2. **High-quality deepfakes** (GANs/Diffusion models):
   - Generate mathematically "perfect" pixels
   - Smooth out natural noise
   - Result in hyper-realistic but unnaturally clean images

3. **Model interpretation**:
   - Model learned: "some imperfection = real"
   - Sees ultra-smooth deepfakes as "too perfect"
   - Assigns them LOW fake scores (paradoxically correct!)

### Solution: Double-Sided Anomaly Detection

Instead of single threshold (score > 0.5 = fake), use **two thresholds**:

```python
# FAKE if EITHER:
# 1. Too obvious (traditional deepfakes): score > 0.5
# 2. Too perfect (modern deepfakes): score < 0.12

if score > 0.5:
    verdict = "FAKE (Traditional - High Artifacts)"
elif score < 0.12:
    verdict = "FAKE (Hyper-Real - Too Perfect)"
elif score > 0.20:
    verdict = "UNCERTAIN"
else:
    verdict = "AUTHENTIC"
```

### Optimal Thresholds (Empirically Derived)
Based on test data:
- **Lower bound**: 0.12 (catches "too perfect" deepfakes)
- **Safe zone**: 0.12 - 0.20 (likely authentic)
- **Upper bound**: 0.50 (traditional deepfakes with visible artifacts)

### Long-Term Improvements
Visual-only detection is insufficient for 2025 deepfakes. Future enhancements:

1. **Frequency Domain Analysis** (F3-Net, DCT filters)
   - Detects GAN upsampling artifacts invisible to human eye
   
2. **Biological Signal Detection** (rPPG)
   - Real humans: subtle pulse/heartbeat in skin color
   - Deepfakes: no pulse signal (dead face)
   
3. **Temporal Inconsistencies** (VideoMAE)
   - Analyze motion patterns across frames
   
4. **Audio-Visual Sync** (Wav2Vec2 + lip sync)
   - Check if voice matches lip movements

### Current Architecture
```
Input Video
    ↓
Face Extraction (MTCNN)
    ↓
Visual Analysis (EfficientNet-B7) → Inverted Score
    ↓
Fusion Layer (85% visual weight)
    ↓
Double-Sided Threshold Detection
    ↓
Final Verdict
```
