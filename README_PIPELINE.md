# 4-Layer Cascade Deepfake Detection Pipeline

A production-ready, GPU-optimized deepfake detection system that uses a fail-fast cascade architecture to efficiently detect AI-generated videos.

## 🎯 Architecture Overview

The pipeline implements **4 detection layers** that run sequentially with **early termination** on fake detection:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT VIDEO                               │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Audio Analysis (Wav2Vec2)          ~200ms         │
│  ► Detects synthetic/robotic audio artifacts                │
│  ► Model: facebook/wav2vec2-base                            │
│  ► Fail-Fast: Stop if FAKE detected                         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼ (if REAL)
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Visual Artifacts (SBI)             ~500ms         │
│  ► Detects pixel-level face manipulations                   │
│  ► Model: EfficientNet-B4 (Self-Blended Images)             │
│  ► Weights: weights/SBI/FFc23.tar                           │
│  ► Fail-Fast: Stop if FAKE detected                         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼ (if REAL)
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Lip-Sync Detection (SyncNet)       ~1s            │
│  ► Detects audio-visual synchronization mismatch            │
│  ► Model: SyncNet (syncnet_python)                          │
│  ► Logic: offset > 5 frames OR confidence < 2.0 = FAKE      │
│  ► Fail-Fast: Stop if FAKE detected                         │
└─────────────────────┬───────────────────────────────────────┘
                      ▼ (if REAL)
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Semantic Analysis (CLIP)           ~2s            │
│  ► Detects AI-generated videos (Sora/Midjourney style)      │
│  ► Model: UniversalFakeDetect CLIP ViT-L/14                 │
│  ► Catches "perfect" deepfakes with semantic understanding  │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              FINAL VERDICT: REAL or FAKE                     │
│              + Confidence Score                              │
│              + Layer-wise Results                            │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

1. **Fail-Fast Optimization**: Most fakes are caught in Layer 1-2, saving GPU time
2. **Comprehensive Coverage**: 
   - Layer 1: Catches audio deepfakes (voice cloning)
   - Layer 2: Catches face-swap artifacts
   - Layer 3: Catches lip-sync issues (HeyGen, D-ID)
   - Layer 4: Catches AI-generated videos (Sora, RunwayML)
3. **Production-Ready**: ~200ms-4s latency depending on fail-fast exit
4. **Extensible**: Easy to add new layers or swap models

## 📦 Repository Structure

```
AI-Video/
├── deepfake_pipeline.py       # Main 4-layer pipeline implementation
├── supabase_logger.py          # Database integration for results
├── test_pipeline.py            # Testing script with metrics
├── requirements_pipeline.txt   # Python dependencies
│
├── weights/                    # Model weights
│   ├── SBI/
│   │   ├── FFc23.tar          # EfficientNet-B4 (FF++ c23)
│   │   └── FFraw.tar          # EfficientNet-B4 (FF++ raw)
│   └── Lips/
│       └── lipforensics_ff.pth # (Optional, not used - using SyncNet)
│
├── syncnet_python/             # Layer 3: SyncNet repo
│   ├── SyncNetInstance.py
│   ├── SyncNetModel.py
│   └── data/syncnet_v2.model
│
├── SelfBlendedImages/          # Layer 2: SBI repo
│   └── src/
│       ├── model.py
│       └── inference/
│
├── UniversalFakeDetect/        # Layer 4: CLIP repo
│   ├── models/
│   │   ├── clip_models.py
│   │   └── __init__.py
│   └── validate.py
│
└── Test-Video/                 # Test datasets
    ├── Real/
    ├── Fake/
    ├── Celeb-real/
    ├── Celeb-synthesis/
    └── YouTube-real/
```

## 🚀 Installation

### 1. Clone Repository
```bash
cd /home/lightdesk/Projects/AI-Video
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows

# Install requirements
pip install -r requirements_pipeline.txt
```

### 3. Download Model Weights

#### SBI Weights (Layer 2)
```bash
# Already available in weights/SBI/
# FFc23.tar - trained on FaceForensics++ c23
# FFraw.tar - trained on FaceForensics++ raw
```

#### SyncNet Weights (Layer 3)
```bash
cd syncnet_python
bash download_model.sh
cd ..
```

#### UniversalFakeDetect Weights (Layer 4)
The model uses CLIP from HuggingFace - downloads automatically on first run.

### 4. Setup Environment Variables (for Supabase)
```bash
# Create .env file
cat > .env << EOF
SUPABASE_URL=https://cjkcwycnetdhumtqthuk.supabase.co
SUPABASE_KEY=sb_publishable_kYQsl9DIOWNzkcZNUojI1w_yIyL70XH
EOF
```

## 🎮 Usage

### Basic Detection (Single Video)

```python
from deepfake_pipeline import DeepfakePipeline

# Initialize pipeline
pipeline = DeepfakePipeline(
    sbi_weights_path="weights/SBI/FFc23.tar",
    device="cuda"  # or "cpu"
)

# Run detection with fail-fast
result = pipeline.detect(
    video_path="path/to/video.mp4",
    enable_fail_fast=True
)

# Access results
print(f"Verdict: {result.final_verdict}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Stopped at: {result.stopped_at_layer}")
print(f"Total time: {result.total_time:.2f}s")

# Layer-wise results
for layer_result in result.layer_results:
    print(f"{layer_result.layer_name}: {layer_result.is_fake}")
```

### Command-Line Usage

```bash
# Single video detection
python deepfake_pipeline.py \
    --video Test-Video/Real/example.mp4 \
    --sbi-weights weights/SBI/FFc23.tar \
    --device cuda

# Run all layers (disable fail-fast)
python deepfake_pipeline.py \
    --video Test-Video/Fake/deepfake.mp4 \
    --no-fail-fast
```

### Testing on Dataset

```bash
# Test on up to 10 videos per category (Real/Fake)
python test_pipeline.py \
    --test-dir Test-Video \
    --max-videos 10 \
    --device cuda

# Test with all layers (no fail-fast)
python test_pipeline.py \
    --test-dir Test-Video \
    --max-videos 20 \
    --no-fail-fast

# Log results to Supabase
python test_pipeline.py \
    --test-dir Test-Video \
    --log-supabase
```

### Integration with Supabase

```python
from deepfake_pipeline import DeepfakePipeline
from supabase_logger import get_logger

# Initialize
pipeline = DeepfakePipeline(sbi_weights_path="weights/SBI/FFc23.tar")
logger = get_logger()

# Run detection
result = pipeline.detect("video.mp4")

# Log to database
logger.log_detection(
    result,
    user_id="user-uuid-here",
    session_id="session-123",
    file_url="https://storage.example.com/video.mp4"
)

# Get statistics
stats = logger.get_statistics(days=30)
print(f"Fake detection rate: {stats['fake_percentage']:.1f}%")
```

## 📊 Database Schema

The pipeline logs results to Supabase `detection_history` table:

```sql
{
  "id": "uuid",
  "user_id": "uuid",
  "session_id": "text",
  "filename": "video.mp4",
  "file_type": "video",
  "file_size": 12345678,
  "file_extension": "mp4",
  "file_url": "https://...",
  "model_used": "4-Layer-Cascade-v1",
  "detection_result": "FAKE",
  "confidence_score": 0.87,
  "metadata": {
    "stopped_at_layer": "Layer 2: Visual Artifacts",
    "total_processing_time": 0.65,
    "layer_results": [
      {
        "layer_name": "Layer 1: Audio Analysis",
        "is_fake": false,
        "confidence": 0.78,
        "processing_time": 0.21,
        "details": {...}
      },
      {
        "layer_name": "Layer 2: Visual Artifacts",
        "is_fake": true,
        "confidence": 0.92,
        "processing_time": 0.44,
        "details": {...}
      }
    ]
  },
  "created_at": "2024-12-16T10:30:00Z"
}
```

## ⚡ Performance Metrics

### Expected Latency (GPU: T4/A10G)

| Layer | Model | Avg Time | Cumulative |
|-------|-------|----------|------------|
| Layer 1 | Wav2Vec2 | ~200ms | 0.2s |
| Layer 2 | EfficientNet-B4 | ~500ms | 0.7s |
| Layer 3 | SyncNet | ~1000ms | 1.7s |
| Layer 4 | CLIP ViT-L/14 | ~2000ms | 3.7s |

**With Fail-Fast**: Most videos exit at Layer 1-2 → **~0.2-0.7s average**

### Expected Accuracy (on FaceForensics++)

- **Layer 2 (SBI)**: ~98% AUC on FF++ c23
- **Layer 3 (SyncNet)**: ~85% accuracy on lip-sync fakes
- **Layer 4 (UniversalFakeDetect)**: ~95%+ on generative models

**Combined Pipeline**: Expected ~95%+ accuracy on modern deepfakes

## 🔧 Configuration & Tuning

### Adjusting Thresholds

```python
# In deepfake_pipeline.py

# Layer 1: Audio
self.threshold = 0.5  # Higher = more strict (more false positives)

# Layer 2: Visual
self.threshold = 0.5  # Adjust based on your dataset

# Layer 3: Lip-Sync
self.offset_threshold = 5      # frames (higher = more lenient)
self.confidence_threshold = 2.0  # (lower = more strict)

# Layer 4: Semantic
self.threshold = 0.5  # Adjust for Sora-style videos
```

### Using Different Weights

```python
# SBI: Use FFraw for uncompressed videos
pipeline = DeepfakePipeline(
    sbi_weights_path="weights/SBI/FFraw.tar"
)
```

### GPU Memory Optimization

```python
# For low VRAM (< 8GB)
# 1. Use CPU for some layers
layer4 = Layer4SemanticDetector(device='cpu')

# 2. Process fewer frames
face_crops = self.extract_face_crops(video_path, n_frames=5)  # vs 10

# 3. Use torch.cuda.empty_cache()
import torch
torch.cuda.empty_cache()
```

## 🐛 Troubleshooting

### Common Issues

1. **"No faces detected"**
   - Check video quality
   - Ensure face is visible and well-lit
   - Try adjusting RetinaFace threshold

2. **SyncNet errors**
   - Ensure `syncnet_v2.model` is downloaded
   - Check audio track exists in video
   - Run `bash syncnet_python/download_model.sh`

3. **CUDA Out of Memory**
   - Reduce `n_frames` in frame extraction
   - Process on CPU: `--device cpu`
   - Use smaller batch sizes

4. **Slow inference**
   - Enable fail-fast (should exit early)
   - Use GPU acceleration
   - Check if using correct CUDA version

## 📈 Testing Results Format

The test script outputs JSON with comprehensive metrics:

```json
{
  "timestamp": "2024-12-16 10:30:00",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.97,
    "f1_score": 0.95,
    "true_positive": 48,
    "true_negative": 47,
    "false_positive": 3,
    "false_negative": 2
  },
  "layer_stats": {
    "Layer 1: Audio Analysis": 12,
    "Layer 2: Visual Artifacts": 35,
    "Layer 3: Lip-Sync Analysis": 8,
    "Layer 4: Semantic Analysis": 5
  },
  "avg_processing_times": {
    "Layer 1: Audio Analysis": 0.21,
    "Layer 2: Visual Artifacts": 0.48,
    "Layer 3: Lip-Sync Analysis": 1.12,
    "Layer 4: Semantic Analysis": 1.95
  }
}
```

## 🔬 Research References

1. **SBI (Self-Blended Images)**
   - Paper: [Detecting Deepfakes with Self-Blended Images](https://arxiv.org/abs/2204.08376)
   - Repo: [mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages)

2. **SyncNet**
   - Paper: [Out of time: automated lip sync in the wild](http://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/)
   - Repo: [joonson/syncnet_python](https://github.com/joonson/syncnet_python)

3. **UniversalFakeDetect**
   - Paper: [Towards Universal Fake Image Detectors](https://arxiv.org/abs/2302.10174)
   - Repo: [WisconsinAIVision/UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect)

4. **Wav2Vec2**
   - Paper: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
   - Repo: [HuggingFace Transformers](https://huggingface.co/facebook/wav2vec2-base)

## 🤝 Contributing

To add a new detection layer:

1. Create a new class inheriting from base pattern:
```python
class Layer5NewDetector:
    def __init__(self, device='cuda'):
        # Initialize model
        pass
    
    def detect(self, video_path: str) -> DetectionResult:
        # Run detection
        pass
```

2. Add to pipeline in `DeepfakePipeline.__init__`
3. Add to cascade in `DeepfakePipeline.detect`

## 📝 License

This project uses multiple open-source components with different licenses:
- **SBI**: Research use only (see LICENSE in SelfBlendedImages/)
- **SyncNet**: MIT License
- **UniversalFakeDetect**: MIT License
- **Pipeline code**: MIT License

For commercial use, ensure compliance with all component licenses.

## 🙏 Acknowledgments

- SBI team at University of Tokyo
- SyncNet authors at University of Oxford
- UniversalFakeDetect team at University of Wisconsin
- HuggingFace for Transformers library

---

**Note**: This is a research/educational implementation. For production deployment, consider:
- Model fine-tuning on your specific dataset
- Adversarial robustness testing
- Rate limiting and abuse prevention
- Regular model updates to counter new deepfake techniques
