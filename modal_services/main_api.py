"""
Main Modal Orchestration API for Deepfake Detection
Coordinates preprocessing, visual, audio detection and fusion
"""
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

app = modal.App("deepfake-detection-api")

# Base image for the API
api_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.104.1",
    "pydantic==2.5.0",
    "httpx==0.25.2"
)

# We'll use stub references to other apps instead of direct imports
# This allows Modal to properly resolve dependencies across apps

# Create FastAPI web app
web_app = FastAPI(
    title="Multimodal Deepfake Detection API",
    description="Production-ready deepfake detection with visual, audio, and temporal analysis",
    version="1.0.0"
)

# Get references to other Modal apps
preprocessing_app = modal.App.lookup("deepfake-preprocessing", create_if_missing=False)
visual_app = modal.App.lookup("deepfake-visual-detector", create_if_missing=False)
audio_app = modal.App.lookup("deepfake-audio-detector", create_if_missing=False)
fusion_app = modal.App.lookup("deepfake-fusion", create_if_missing=False)

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
    processing_time: Optional[float] = None


# In-memory task storage (use Redis in production)
tasks = {}

@web_app.post("/detect_video", response_model=DetectionResponse)
async def detect_video(request: DetectionRequest):
    """
    Start async video deepfake detection
    
    Args:
        video_url: Public URL to video file (Supabase storage)
        callback_url: Optional webhook to POST results back
        task_id: Optional external task ID (database record ID)
    
    Returns:
        task_id for polling status
    """
    import uuid
    
    if not request.task_id:
        task_id = str(uuid.uuid4())
    else:
        task_id = request.task_id
    
    # Validate video URL
    if not request.video_url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid video URL")
    
    # Mark task as processing
    tasks[task_id] = {
        "status": "processing",
        "result": None,
        "created_at": time.time()
    }
    
    # Trigger async processing
    process_video.spawn(request.video_url, request.callback_url, task_id)
    
    return DetectionResponse(
        task_id=task_id,
        status="processing",
        message="Video detection started. Use /status/{task_id} to check progress."
    )


@app.function(timeout=600, retries=2)
async def process_video(video_url: str, callback_url: Optional[str], task_id: str):
    """
    Background worker that runs the full detection pipeline
    """
    import httpx
    import traceback
    
    start_time = time.time()
    
    try:
        print(f"🎬 Starting detection pipeline for task: {task_id}")
        
        # Step 1: Preprocessing - Extract faces and audio
        print("📹 Step 1/4: Preprocessing video...")
        preprocessor = VideoPreprocessor()
        prep_result = preprocessor.extract_faces_and_audio.remote(video_url, sample_rate=2)
        
        face_crops = prep_result["face_crops"]
        audio_bytes = prep_result["audio_bytes"]
        metadata = prep_result["metadata"]
        
        if len(face_crops) == 0:
            raise Exception("No faces detected in video - cannot analyze")
        
        print(f"✅ Preprocessing complete: {len(face_crops)} face crops extracted")
        
        # Step 2: Visual Detection
        print("🖼️  Step 2/4: Analyzing visual artifacts...")
        visual_detector = VisualArtifactDetector()
        visual_result = visual_detector.detect_artifacts.remote(face_crops)
        
        if "error" in visual_result:
            raise Exception(f"Visual detection failed: {visual_result['error']}")
        
        visual_score = visual_result["mean_score"]
        temporal_score = 1.0 - visual_result["std_score"]  # Low variance = high consistency
        
        print(f"✅ Visual analysis complete: score={visual_score:.3f}")
        
        # Step 3: Audio Detection (if audio available)
        audio_score = None
        if audio_bytes and metadata["has_audio"]:
            print("🎵 Step 3/4: Analyzing audio synthesis...")
            audio_detector = AudioSynthesisDetector()
            audio_result = audio_detector.detect_synthetic_audio.remote(audio_bytes)
            
            if audio_result.get("has_audio", False):
                audio_score = audio_result["synthesis_score"]
                print(f"✅ Audio analysis complete: score={audio_score:.3f}")
            else:
                print("⚠️ Audio analysis skipped - no valid audio")
        else:
            print("⚠️ Step 3/4: No audio track - skipping audio analysis")
        
        # Step 4: Fusion Layer
        print("🔀 Step 4/4: Fusing multimodal scores...")
        
        # Estimate face quality from preprocessing (simple heuristic)
        face_quality = min(0.98, 0.8 + (len(face_crops) / metadata["total_frames"]) * 0.2)
        
        final_result = fuse_multimodal_scores.remote(
            visual_score=visual_score,
            audio_score=audio_score,
            face_quality=face_quality,
            temporal_score=temporal_score
        )
        
        processing_time = time.time() - start_time
        
        # Add metadata
        final_result["model_metadata"] = {
            "models_used": ["MTCNN", "EfficientNet-B7"] + (["Wav2Vec2"] if audio_score else []),
            "processing_time_seconds": round(processing_time, 2),
            "frames_analyzed": len(face_crops),
            "video_duration_seconds": round(metadata["video_duration"], 2),
            "total_frames": metadata["total_frames"],
            "fps": round(metadata["fps"], 2)
        }
        
        # Update task status
        tasks[task_id] = {
            "status": "completed",
            "result": final_result,
            "processing_time": processing_time
        }
        
        print(f"✅ Detection complete: {final_result['verdict']}")
        print(f"⏱️  Total processing time: {processing_time:.2f}s")
        
        # Send callback if provided
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    await client.post(callback_url, json={
                        "task_id": task_id,
                        "result": final_result
                    })
                print(f"✅ Callback sent to {callback_url}")
            except Exception as e:
                print(f"⚠️ Callback failed: {e}")
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Detection failed: {error_msg}")
        print(traceback.format_exc())
        
        tasks[task_id] = {
            "status": "failed",
            "error": error_msg,
            "processing_time": time.time() - start_time
        }


@web_app.get("/status/{task_id}", response_model=DetectionResult)
async def get_status(task_id: str):
    """Poll endpoint to check detection status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = tasks[task_id]
    
    return DetectionResult(
        task_id=task_id,
        status=task_data["status"],
        result=task_data.get("result"),
        error=task_data.get("error"),
        processing_time=task_data.get("processing_time")
    )


@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Multimodal Deepfake Detection API",
        "version": "1.0.0"
    }


@app.function()
@modal.asgi_app()
def fastapi_app():
    """Expose FastAPI app via Modal"""
    return web_app


@app.local_entrypoint()
def test_api():
    """Test the full API locally"""
    # Test with a public video URL
    test_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
    
    import requests
    
    # Get the Modal app URL (you'll need to deploy first)
    print("⚠️  Deploy the app first with: modal deploy modal_services/main_api.py")
    print("Then use the provided URL to test the API")
