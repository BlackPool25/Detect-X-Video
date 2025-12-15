"""
Modal Preprocessing Service for Deepfake Detection
Handles face extraction and frame sampling from videos using MTCNN
"""
import modal
import io
import os
from pathlib import Path

app = modal.App("deepfake-preprocessing")

# Container image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsm6", "libxext6", "libxrender-dev", "libgomp1")
    .pip_install(
        "opencv-python-headless==4.8.1.78",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "facenet-pytorch==2.5.3",
        "ffmpeg-python==0.2.0",
        "numpy==1.24.3",
        "Pillow==10.1.0",
        "requests==2.31.0"
    )
)

@app.cls(
    gpu="T4",
    image=image,
    container_idle_timeout=300,
    timeout=600
)
class VideoPreprocessor:
    @modal.enter()
    def load_models(self):
        """Load MTCNN face detector on GPU"""
        from facenet_pytorch import MTCNN
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detector = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
            select_largest=True
        )
        print(f"✅ MTCNN Face detector loaded on {device}")
    
    @modal.method()
    def extract_faces_and_audio(self, video_url: str, sample_rate: int = 2):
        """
        Download video, extract faces from sampled frames and audio track
        
        Args:
            video_url: Public URL to video file (Supabase storage)
            sample_rate: Process 1 frame every N seconds (default 2 FPS effective)
        
        Returns:
            dict: {
                "face_crops": List[bytes],  # JPEG encoded face images
                "audio_bytes": bytes,  # WAV audio file
                "metadata": {
                    "total_frames": int,
                    "faces_detected": int,
                    "video_duration": float,
                    "fps": float,
                    "has_audio": bool
                }
            }
        """
        import cv2
        import numpy as np
        import requests
        import tempfile
        import ffmpeg
        from PIL import Image
        
        print(f"🎥 Processing video from: {video_url[:50]}...")
        
        # Download video to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            response = requests.get(video_url, timeout=120)
            response.raise_for_status()
            tmp_video.write(response.content)
            video_path = tmp_video.name
        
        try:
            # Extract audio using ffmpeg
            audio_path = video_path.replace('.mp4', '.wav')
            has_audio = False
            audio_bytes = None
            
            try:
                # Check if video has audio stream
                probe = ffmpeg.probe(video_path)
                audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
                
                if audio_streams:
                    ffmpeg.input(video_path).output(
                        audio_path, 
                        acodec='pcm_s16le', 
                        ac=1, 
                        ar='16000'
                    ).overwrite_output().run(quiet=True, capture_stderr=True)
                    
                    with open(audio_path, 'rb') as f:
                        audio_bytes = f.read()
                    has_audio = True
                    print(f"✅ Audio extracted: {len(audio_bytes)} bytes")
                else:
                    print("⚠️ No audio stream found in video")
            except Exception as e:
                print(f"⚠️ Audio extraction failed: {e}")
            
            # Extract and process frames
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            frame_interval = max(1, int(fps * sample_rate))
            face_crops_bytes = []
            frame_idx = 0
            frames_with_faces = 0
            
            print(f"📊 Video stats: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process sampled frames
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB for face detector
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    boxes, probs = self.face_detector.detect(rgb_frame)
                    
                    if boxes is not None and len(boxes) > 0:
                        # Take the face with highest confidence
                        best_idx = np.argmax(probs)
                        box = boxes[best_idx]
                        prob = probs[best_idx]
                        
                        if prob > 0.9:  # High confidence threshold
                            # Crop face with some padding
                            x1, y1, x2, y2 = [int(b) for b in box]
                            h, w = rgb_frame.shape[:2]
                            
                            # Add 20% padding
                            padding = int((x2 - x1) * 0.2)
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(w, x2 + padding)
                            y2 = min(h, y2 + padding)
                            
                            face = rgb_frame[y1:y2, x1:x2]
                            
                            # Resize to 224x224 for EfficientNet
                            face_resized = cv2.resize(face, (224, 224))
                            
                            # Convert to PIL and encode as JPEG
                            pil_image = Image.fromarray(face_resized)
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format='JPEG', quality=95)
                            face_crops_bytes.append(buffer.getvalue())
                            frames_with_faces += 1
                
                frame_idx += 1
            
            cap.release()
            
            print(f"✅ Extracted {len(face_crops_bytes)} face crops from {frames_with_faces} frames")
            
            return {
                "face_crops": face_crops_bytes,
                "audio_bytes": audio_bytes,
                "metadata": {
                    "total_frames": total_frames,
                    "faces_detected": len(face_crops_bytes),
                    "video_duration": duration,
                    "fps": fps,
                    "sample_rate": sample_rate,
                    "has_audio": has_audio
                }
            }
        
        finally:
            # Cleanup temp files
            if os.path.exists(video_path):
                os.unlink(video_path)
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.unlink(audio_path)


@app.local_entrypoint()
def test_preprocessing():
    """Test the preprocessing service locally"""
    # Test with a sample video URL (replace with actual test video)
    test_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
    
    preprocessor = VideoPreprocessor()
    result = preprocessor.extract_faces_and_audio.remote(test_url)
    
    print(f"\n📊 Preprocessing Results:")
    print(f"  - Face crops extracted: {len(result['face_crops'])}")
    print(f"  - Audio available: {result['metadata']['has_audio']}")
    print(f"  - Video duration: {result['metadata']['video_duration']:.2f}s")
    print(f"  - Faces detected: {result['metadata']['faces_detected']}")
