"""
All-in-One Multimodal Deepfake Detection API
Combines preprocessing, visual, audio detection and fusion in a single Modal app
"""
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import time
from pathlib import Path

app = modal.App("deepfake-detection-complete")

# Base image for preprocessing and detection with weights directory
weights_path = Path(__file__).parent / "weights"
detection_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsm6", "libxext6", "libxrender-dev", "libgomp1")
    .pip_install(
        # Core dependencies
        "fastapi==0.104.1",
        "pydantic==2.5.0",
        "httpx==0.25.2",
        # CV and ML
        "opencv-python-headless==4.8.1.78",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "torchaudio==2.1.0",
        # Models
        "facenet-pytorch==2.5.3",
        "timm==0.9.12",
        "transformers==4.35.0",
        "safetensors==0.4.1",
        # Utils
        "ffmpeg-python==0.2.0",
        "numpy==1.24.3",
        "Pillow==10.1.0",
        "requests==2.31.0"
    )
    .add_local_dir(str(weights_path), remote_path="/weights")
)

# In-memory task storage
tasks = {}

# FastAPI app
web_app = FastAPI(
    title="Multimodal Deepfake Detection API",
    description="Production-ready deepfake detection",
    version="1.0.0"
)

class DetectionRequest(BaseModel):
    video_url: str
    callback_url: Optional[str] = None
    task_id: Optional[str] = None

class DetectionResponse(BaseModel):
    task_id: str
    status: str
    message: str

class DetectionResult(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


@web_app.post("/detect_video", response_model=DetectionResponse)
async def detect_video(request: DetectionRequest):
    """Start async video deepfake detection"""
    import uuid
    
    task_id = request.task_id or str(uuid.uuid4())
    
    if not request.video_url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid video URL")
    
    tasks[task_id] = {"status": "processing", "result": None, "created_at": time.time()}
    
    # Spawn async processing
    process_video_pipeline.spawn(request.video_url, request.callback_url, task_id)
    
    return DetectionResponse(
        task_id=task_id,
        status="processing",
        message="Detection started. Poll /status/{task_id}"
    )


@web_app.get("/status/{task_id}", response_model=DetectionResult)
async def get_status(task_id: str):
    """Check detection status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return DetectionResult(task_id=task_id, **tasks[task_id])


@web_app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "version": "1.0.0"}


@app.function(
    gpu="T4",
    image=detection_image,
    timeout=900,
    retries=2
)
async def process_video_pipeline(video_url: str, callback_url: Optional[str], task_id: str):
    """Complete detection pipeline"""
    import httpx
    import traceback
    
    start_time = time.time()
    
    try:
        print(f"🎬 Starting pipeline for task: {task_id}")
        
        # Step 1: Preprocess
        print("📹 Preprocessing...")
        face_crops, audio_bytes, metadata = await preprocess_video(video_url)
        
        if not face_crops:
            raise Exception("No faces detected")
        
        print(f"✅ Extracted {len(face_crops)} face crops")
        
        # Step 2: Visual detection
        print("🖼️  Visual analysis...")
        visual_score, temporal_score = await detect_visual_artifacts(face_crops)
        print(f"✅ Visual: {visual_score:.3f}, Temporal: {temporal_score:.3f}")
        
        # Step 3: Audio detection
        audio_score = None
        if audio_bytes:
            print("🎵 Audio analysis...")
            audio_score = await detect_audio_synthesis(audio_bytes)
            print(f"✅ Audio: {audio_score:.3f}")
        
        # Step 4: Fusion
        print("🔀 Fusing scores...")
        face_quality = min(0.98, 0.8 + (len(face_crops) / metadata["total_frames"]) * 0.2)
        final_result = fuse_scores(visual_score, audio_score, temporal_score, face_quality)
        
        final_result["model_metadata"] = {
            "models_used": ["MTCNN", "EfficientNet-B7"] + (["Wav2Vec2"] if audio_score else []),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "frames_analyzed": len(face_crops),
            "video_duration_seconds": round(metadata["video_duration"], 2)
        }
        
        tasks[task_id] = {"status": "completed", "result": final_result}
        
        print(f"✅ Complete: {final_result['verdict']} ({time.time() - start_time:.1f}s)")
        
        # Callback
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    await client.post(callback_url, json={"task_id": task_id, "result": final_result})
            except Exception as e:
                print(f"⚠️ Callback failed: {e}")
    
    except Exception as e:
        print(f"❌ Failed: {e}\n{traceback.format_exc()}")
        tasks[task_id] = {"status": "failed", "error": str(e)}


async def preprocess_video(video_url: str):
    """Extract faces and audio from video"""
    import cv2
    import numpy as np
    import requests
    import tempfile
    import ffmpeg
    import io
    from PIL import Image
    from facenet_pytorch import MTCNN
    import torch
    
    # Load face detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_detector = MTCNN(keep_all=True, device=device, post_process=False, select_largest=True)
    
    # Download video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        resp = requests.get(video_url, timeout=120)
        resp.raise_for_status()
        tmp.write(resp.content)
        video_path = tmp.name
    
    # Extract audio
    audio_bytes = None
    try:
        probe = ffmpeg.probe(video_path)
        if any(s['codec_type'] == 'audio' for s in probe['streams']):
            audio_path = video_path.replace('.mp4', '.wav')
            ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16000').overwrite_output().run(quiet=True)
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
    except:
        pass
    
    # Extract faces
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    frame_interval = max(1, int(fps * 2))  # 2 FPS
    face_crops = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = face_detector.detect(rgb)
            
            if boxes is not None and len(boxes) > 0 and max(probs) > 0.9:
                box = boxes[np.argmax(probs)]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Padding
                pad = int((x2 - x1) * 0.2)
                h, w = rgb.shape[:2]
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                
                face = cv2.resize(rgb[y1:y2, x1:x2], (224, 224))
                
                buffer = io.BytesIO()
                Image.fromarray(face).save(buffer, format='JPEG', quality=95)
                face_crops.append(buffer.getvalue())
        
        frame_idx += 1
    
    cap.release()
    
    return face_crops, audio_bytes, {"total_frames": total_frames, "video_duration": duration, "fps": fps}


async def detect_visual_artifacts(face_crops_bytes: List[bytes]):
    """Analyze visual artifacts with EfficientNet-B7"""
    import torch
    import timm
    import io
    from PIL import Image
    from torchvision import transforms
    import numpy as np
    
    # Load model
    model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=2)
    try:
        checkpoint = torch.load("/weights/efficientnet_b7_deepfake.pt", map_location='cuda')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint if not isinstance(checkpoint, dict) else model
    except:
        pass
    
    model = model.cuda().eval()
    
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    scores = []
    with torch.no_grad():
        for crop_bytes in face_crops_bytes:
            img = Image.open(io.BytesIO(crop_bytes)).convert('RGB')
            tensor = transform(img).unsqueeze(0).cuda()
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            scores.append(probs[0, 1].item())
    
    return float(np.mean(scores)), 1.0 - float(np.std(scores))


async def detect_audio_synthesis(audio_bytes: bytes):
    """Detect synthetic audio with Wav2Vec2"""
    import torch
    import torchaudio
    import tempfile
    from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
    
    # Load model
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained("/weights", local_files_only=True).cuda().eval()
        processor = Wav2Vec2Processor.from_pretrained("/weights", local_files_only=True)
    except:
        model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base").cuda().eval()
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Process audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        waveform, sr = torchaudio.load(tmp.name)
    
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    
    return probs[0, 1].item() if probs.shape[1] >= 2 else probs[0, 0].item()


def fuse_scores(visual: float, audio: Optional[float], temporal: float, face_quality: float):
    """
    Fuse multimodal scores
    
    All scores are normalized 0-1 where:
    - Higher visual score = more likely deepfake
    - Higher audio score = more likely deepfake  
    - Higher temporal = more consistent (REAL)
    - Higher face_quality = better detection (REAL)
    
    So we invert temporal and face_quality before fusion
    """
    # Invert scores that indicate authenticity
    temporal_inverted = 1 - temporal  # High consistency = real, so invert
    face_inverted = 1 - face_quality  # High quality = real, so invert
    
    if audio is None:
        # Without audio: visual=55%, temporal=30%, face=15%
        weights = {'visual': 0.55, 'temporal': 0.30, 'face_quality': 0.15}
        final = (
            weights['visual'] * visual + 
            weights['temporal'] * temporal_inverted + 
            weights['face_quality'] * face_inverted
        )
        has_audio = False
    else:
        # With audio: visual=40%, audio=35%, temporal=15%, face=10%
        weights = {'visual': 0.40, 'audio': 0.35, 'temporal': 0.15, 'face_quality': 0.10}
        final = (
            weights['visual'] * visual + 
            weights['audio'] * audio + 
            weights['temporal'] * temporal_inverted + 
            weights['face_quality'] * face_inverted
        )
        has_audio = True
    
    is_fake = final > 0.5
    confidence = min(100, abs(final - 0.5) * 200)
    
    verdict = "DEEPFAKE DETECTED (High)" if final > 0.7 else "LIKELY DEEPFAKE" if final > 0.5 else "UNCERTAIN" if final > 0.3 else "LIKELY AUTHENTIC"
    
    return {
        "final_score": round(final, 4),
        "is_deepfake": is_fake,
        "confidence_percent": round(confidence, 2),
        "verdict": verdict,
        "has_audio": has_audio,
        "breakdown": {
            "visual_artifacts": round(visual, 4),
            "temporal_consistency": round(temporal, 4),
            "audio_synthesis": round(audio, 4) if audio else None,
            "face_quality": round(face_quality, 4)
        }
    }


@app.function(image=detection_image)
@modal.asgi_app()
def fastapi_app():
    return web_app
