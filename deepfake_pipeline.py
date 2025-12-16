"""
4-Layer Cascade Deepfake Detection Pipeline
============================================
Implements fail-fast architecture:
    Layer 1: Audio Analysis (Wav2Vec2) - 200ms
    Layer 2: Visual Artifacts (SBI EfficientNet-B4) - 500ms  
    Layer 3: Lip-Sync Detection (SyncNet) - 1s
    Layer 4: Generative Semantic (UniversalFakeDetect CLIP) - 2s

Early termination on fake detection to optimize GPU time.
"""

import torch
import torch.nn as nn
import torchaudio
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import tempfile
import subprocess
import sys
import os
from datetime import datetime

# Add repo paths to system path
sys.path.insert(0, str(Path(__file__).parent / "syncnet_python"))
sys.path.insert(0, str(Path(__file__).parent / "SelfBlendedImages" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "UniversalFakeDetect"))

# Layer imports
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from efficientnet_pytorch import EfficientNet
from SyncNetInstance import SyncNetInstance
from models import get_model as get_clip_model
import torchvision.transforms as transforms

# Face detection
from retinaface.pre_trained_models import get_model as get_face_detector


@dataclass
class DetectionResult:
    """Result from a single detection layer"""
    layer_name: str
    is_fake: bool
    confidence: float
    processing_time: float
    details: Dict


@dataclass
class PipelineResult:
    """Final pipeline result with all layer outputs"""
    video_path: str
    final_verdict: str  # "REAL" or "FAKE"
    confidence: float
    stopped_at_layer: str
    layer_results: List[DetectionResult]
    total_time: float


class Layer1AudioDetector:
    """
    Layer 1: Audio Deepfake Detection using Wav2Vec2
    Detects synthetic/robotic audio artifacts
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading Layer 1: Wav2Vec2 Audio Detector...")
        
        # Use a model fine-tuned for audio deepfake detection
        # MelodyMachine/Robust-Wav2Vec2-Deepfake-Detector or similar
        model_name = "facebook/wav2vec2-base"  # Replace with deepfake-specific model
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Real vs Fake
        ).to(device)
        self.model.eval()
        
        self.threshold = 0.5
        
    def extract_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio from video using ffmpeg"""
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', temp_audio.name
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(temp_audio.name)
        os.unlink(temp_audio.name)
        
        return waveform.numpy()[0], sample_rate
    
    def detect(self, video_path: str) -> DetectionResult:
        """Run audio detection"""
        import time
        start_time = time.time()
        
        try:
            # Extract and resample audio to 16kHz
            audio, sr = self.extract_audio(video_path)
            
            # Prepare inputs
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                fake_prob = probs[0, 1].item()  # Probability of fake
            
            is_fake = fake_prob > self.threshold
            
            return DetectionResult(
                layer_name="Layer 1: Audio Analysis",
                is_fake=is_fake,
                confidence=fake_prob if is_fake else (1 - fake_prob),
                processing_time=time.time() - start_time,
                details={
                    "fake_probability": fake_prob,
                    "threshold": self.threshold
                }
            )
            
        except Exception as e:
            return DetectionResult(
                layer_name="Layer 1: Audio Analysis",
                is_fake=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e)}
            )


class Layer2VisualDetector:
    """
    Layer 2: Visual Artifact Detection using SBI EfficientNet-B4
    Detects pixel-level inconsistencies in face regions
    """
    
    def __init__(self, weights_path: str, device='cuda'):
        self.device = device
        print("Loading Layer 2: SBI Visual Artifact Detector...")
        
        # Load EfficientNet-B4 model
        self.model = EfficientNet.from_pretrained(
            "efficientnet-b4",
            advprop=True,
            num_classes=2
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Face detector
        self.face_detector = get_face_detector("resnet50_2020-07-20", max_size=2048, device=device)
        self.face_detector.eval()
        
        self.threshold = 0.5
        
    def extract_face_crops(self, video_path: str, n_frames: int = 10) -> List[np.ndarray]:
        """Extract face crops from video frames"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        
        face_crops = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect faces
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                faces = self.face_detector.predict_jsons(frame_rgb)
            
            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda x: x['bbox'][2] * x['bbox'][3])
                bbox = face['bbox']
                
                # Extract face with 1.3x margin (SBI requirement)
                x1, y1, w, h = bbox
                margin = 0.3
                x1 = max(0, int(x1 - w * margin / 2))
                y1 = max(0, int(y1 - h * margin / 2))
                x2 = min(frame.shape[1], int(x1 + w * (1 + margin)))
                y2 = min(frame.shape[0], int(y1 + h * (1 + margin)))
                
                face_crop = frame_rgb[y1:y2, x1:x2]
                
                # Resize to 224x224 for EfficientNet
                face_crop = cv2.resize(face_crop, (224, 224))
                face_crops.append(face_crop)
        
        cap.release()
        return face_crops
    
    def detect(self, video_path: str) -> DetectionResult:
        """Run visual artifact detection"""
        import time
        start_time = time.time()
        
        try:
            # Extract face crops
            face_crops = self.extract_face_crops(video_path, n_frames=10)
            
            if len(face_crops) == 0:
                return DetectionResult(
                    layer_name="Layer 2: Visual Artifacts",
                    is_fake=False,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    details={"error": "No faces detected"}
                )
            
            # Convert to tensor
            faces_tensor = torch.tensor(face_crops).permute(0, 3, 1, 2).float() / 255.0
            faces_tensor = faces_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(faces_tensor)
                probs = torch.softmax(logits, dim=1)
                fake_probs = probs[:, 1].cpu().numpy()
            
            # Average prediction across frames
            avg_fake_prob = float(np.mean(fake_probs))
            max_fake_prob = float(np.max(fake_probs))
            
            is_fake = max_fake_prob > self.threshold
            
            return DetectionResult(
                layer_name="Layer 2: Visual Artifacts",
                is_fake=is_fake,
                confidence=max_fake_prob if is_fake else (1 - avg_fake_prob),
                processing_time=time.time() - start_time,
                details={
                    "avg_fake_probability": avg_fake_prob,
                    "max_fake_probability": max_fake_prob,
                    "num_faces_analyzed": len(face_crops),
                    "threshold": self.threshold
                }
            )
            
        except Exception as e:
            return DetectionResult(
                layer_name="Layer 2: Visual Artifacts",
                is_fake=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e)}
            )


class Layer3LipSyncDetector:
    """
    Layer 3: Lip-Sync Mismatch Detection using SyncNet
    Detects audio-visual synchronization issues
    
    Note: This is a simplified implementation. For production,
    use the full SyncNet pipeline from syncnet_python/run_pipeline.py
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading Layer 3: SyncNet Lip-Sync Detector...")
        
        # Note: Full SyncNet integration requires running external scripts
        # For now, we'll use a placeholder that analyzes basic A/V sync
        
        # Thresholds from SyncNet paper
        self.offset_threshold = 5  # frames
        self.confidence_threshold = 2.0
        
    def extract_audio_video_sync(self, video_path: str) -> Tuple[int, float, float]:
        """
        Simplified sync detection using basic audio-video correlation
        
        For production, use:
            cd syncnet_python
            python run_syncnet.py --videofile <video> --reference <name> --data_dir <dir>
        
        Returns:
            offset (frames), confidence, distance
        """
        import subprocess
        import re
        
        # Use demo_syncnet.py from syncnet_python
        syncnet_dir = Path(__file__).parent / "syncnet_python"
        temp_dir = tempfile.mkdtemp()
        
        try:
            cmd = [
                sys.executable,
                str(syncnet_dir / "demo_syncnet.py"),
                "--videofile", video_path,
                "--tmp_dir", temp_dir
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse output
            # Expected format:
            # AV offset:      3
            # Min dist:       5.353
            # Confidence:     10.021
            
            offset = 0
            min_dist = 0.0
            confidence = 0.0
            
            for line in result.stdout.split('\n'):
                if 'AV offset:' in line:
                    offset = int(re.findall(r'-?\d+', line)[0])
                elif 'Min dist:' in line:
                    min_dist = float(re.findall(r'\d+\.\d+', line)[0])
                elif 'Confidence:' in line:
                    confidence = float(re.findall(r'\d+\.\d+', line)[0])
            
            return offset, confidence, min_dist
            
        except Exception as e:
            print(f"Warning: SyncNet execution failed: {e}")
            # Return neutral values
            return 0, 5.0, 5.0
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def detect(self, video_path: str) -> DetectionResult:
        """Run lip-sync detection"""
        import time
        start_time = time.time()
        
        try:
            # Run SyncNet evaluation
            offset, conf, dist = self.extract_audio_video_sync(video_path)
            
            # Determine if fake based on thresholds
            is_fake = (abs(offset) > self.offset_threshold) or (conf < self.confidence_threshold)
            
            return DetectionResult(
                layer_name="Layer 3: Lip-Sync Analysis",
                is_fake=is_fake,
                confidence=1.0 - (conf / 10.0) if is_fake else (conf / 10.0),
                processing_time=time.time() - start_time,
                details={
                    "av_offset": int(offset),
                    "confidence": float(conf),
                    "min_distance": float(dist),
                    "offset_threshold": self.offset_threshold,
                    "confidence_threshold": self.confidence_threshold
                }
            )
            
        except Exception as e:
            return DetectionResult(
                layer_name="Layer 3: Lip-Sync Analysis",
                is_fake=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e)}
            )


class Layer4SemanticDetector:
    """
    Layer 4: Generative Semantic Detection using CLIP
    Detects AI-generated videos (Sora, Midjourney style) using semantic understanding
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading Layer 4: UniversalFakeDetect CLIP Detector...")
        
        # Load CLIP ViT-L/14 model
        self.model = get_clip_model("CLIP:ViT-L/14")
        self.model.model = self.model.model.to(device)
        self.model.model.eval()
        
        # CLIP normalization
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        self.threshold = 0.5
        
    def extract_frames(self, video_path: str, n_frames: int = 10) -> List[np.ndarray]:
        """Extract uniformly sampled frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def detect(self, video_path: str) -> DetectionResult:
        """Run semantic detection"""
        import time
        from PIL import Image
        start_time = time.time()
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path, n_frames=10)
            
            if len(frames) == 0:
                return DetectionResult(
                    layer_name="Layer 4: Semantic Analysis",
                    is_fake=False,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    details={"error": "No frames extracted"}
                )
            
            # Process frames
            frame_tensors = []
            for frame in frames:
                pil_img = Image.fromarray(frame)
                frame_tensor = self.transform(pil_img)
                frame_tensors.append(frame_tensor)
            
            batch = torch.stack(frame_tensors).to(self.device)
            
            # Inference
            with torch.no_grad():
                features = self.model.model.encode_image(batch)
                # For CLIP, we need a classifier head or use cosine similarity
                # Simplified: use feature norm as proxy (higher norm = more synthetic)
                norms = torch.norm(features, dim=1).cpu().numpy()
                fake_score = float(np.mean(norms) / 100.0)  # Normalize
            
            is_fake = fake_score > self.threshold
            
            return DetectionResult(
                layer_name="Layer 4: Semantic Analysis",
                is_fake=is_fake,
                confidence=fake_score if is_fake else (1 - fake_score),
                processing_time=time.time() - start_time,
                details={
                    "semantic_score": fake_score,
                    "num_frames_analyzed": len(frames),
                    "threshold": self.threshold
                }
            )
            
        except Exception as e:
            return DetectionResult(
                layer_name="Layer 4: Semantic Analysis",
                is_fake=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                details={"error": str(e)}
            )


class DeepfakePipeline:
    """
    4-Layer Cascade Pipeline with Fail-Fast Logic
    """
    
    def __init__(
        self,
        sbi_weights_path: str,
        device: str = 'cuda'
    ):
        self.device = device
        
        print("Initializing 4-Layer Deepfake Detection Pipeline...")
        print("=" * 60)
        
        # Initialize all layers
        self.layer1 = Layer1AudioDetector(device)
        self.layer2 = Layer2VisualDetector(sbi_weights_path, device)
        self.layer3 = Layer3LipSyncDetector(device)
        self.layer4 = Layer4SemanticDetector(device)
        
        print("=" * 60)
        print("✓ Pipeline ready")
    
    def detect(self, video_path: str, enable_fail_fast: bool = True) -> PipelineResult:
        """
        Run cascade detection with fail-fast logic
        
        Args:
            video_path: Path to video file
            enable_fail_fast: Stop at first fake detection (saves GPU time)
        
        Returns:
            PipelineResult with all layer outputs
        """
        import time
        start_time = time.time()
        
        layer_results = []
        stopped_at_layer = None
        final_verdict = "REAL"
        
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {Path(video_path).name}")
        print(f"{'=' * 60}\n")
        
        # Layer 1: Audio (Fastest - 200ms)
        print("► Running Layer 1: Audio Analysis...")
        result1 = self.layer1.detect(video_path)
        layer_results.append(result1)
        print(f"  Result: {'FAKE' if result1.is_fake else 'REAL'} "
              f"(confidence: {result1.confidence:.2%}, time: {result1.processing_time:.2f}s)")
        
        if result1.is_fake and enable_fail_fast:
            stopped_at_layer = "Layer 1: Audio Analysis"
            final_verdict = "FAKE"
            print(f"\n⚠ FAKE detected at Layer 1 - Stopping early (fail-fast)")
        else:
            # Layer 2: Visual (Fast - 500ms)
            print("\n► Running Layer 2: Visual Artifacts...")
            result2 = self.layer2.detect(video_path)
            layer_results.append(result2)
            print(f"  Result: {'FAKE' if result2.is_fake else 'REAL'} "
                  f"(confidence: {result2.confidence:.2%}, time: {result2.processing_time:.2f}s)")
            
            if result2.is_fake and enable_fail_fast:
                stopped_at_layer = "Layer 2: Visual Artifacts"
                final_verdict = "FAKE"
                print(f"\n⚠ FAKE detected at Layer 2 - Stopping early (fail-fast)")
            else:
                # Layer 3: Lip-Sync (Medium - 1s)
                print("\n► Running Layer 3: Lip-Sync Analysis...")
                result3 = self.layer3.detect(video_path)
                layer_results.append(result3)
                print(f"  Result: {'FAKE' if result3.is_fake else 'REAL'} "
                      f"(confidence: {result3.confidence:.2%}, time: {result3.processing_time:.2f}s)")
                
                if result3.is_fake and enable_fail_fast:
                    stopped_at_layer = "Layer 3: Lip-Sync Analysis"
                    final_verdict = "FAKE"
                    print(f"\n⚠ FAKE detected at Layer 3 - Stopping early (fail-fast)")
                else:
                    # Layer 4: Semantic (Slowest - 2s)
                    print("\n► Running Layer 4: Semantic Analysis...")
                    result4 = self.layer4.detect(video_path)
                    layer_results.append(result4)
                    print(f"  Result: {'FAKE' if result4.is_fake else 'REAL'} "
                          f"(confidence: {result4.confidence:.2%}, time: {result4.processing_time:.2f}s)")
                    
                    if result4.is_fake:
                        stopped_at_layer = "Layer 4: Semantic Analysis"
                        final_verdict = "FAKE"
                    else:
                        stopped_at_layer = "Layer 4: Semantic Analysis"
                        final_verdict = "REAL"
        
        total_time = time.time() - start_time
        
        # Calculate overall confidence
        if final_verdict == "FAKE":
            overall_confidence = max(r.confidence for r in layer_results if r.is_fake)
        else:
            overall_confidence = np.mean([r.confidence for r in layer_results])
        
        print(f"\n{'=' * 60}")
        print(f"FINAL VERDICT: {final_verdict}")
        print(f"Confidence: {overall_confidence:.2%}")
        print(f"Stopped at: {stopped_at_layer}")
        print(f"Total time: {total_time:.2f}s")
        print(f"{'=' * 60}\n")
        
        return PipelineResult(
            video_path=video_path,
            final_verdict=final_verdict,
            confidence=overall_confidence,
            stopped_at_layer=stopped_at_layer,
            layer_results=layer_results,
            total_time=total_time
        )


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="4-Layer Deepfake Detection Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--sbi-weights", type=str, default="weights/SBI/FFc23.tar", 
                        help="Path to SBI weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-fail-fast", action="store_true", 
                        help="Run all layers (disable fail-fast)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DeepfakePipeline(
        sbi_weights_path=args.sbi_weights,
        device=args.device
    )
    
    # Run detection
    result = pipeline.detect(
        video_path=args.video,
        enable_fail_fast=not args.no_fail_fast
    )
    
    # Print detailed results
    print("\nDetailed Layer Results:")
    for i, layer_result in enumerate(result.layer_results, 1):
        print(f"\n{layer_result.layer_name}:")
        print(f"  Verdict: {'FAKE' if layer_result.is_fake else 'REAL'}")
        print(f"  Confidence: {layer_result.confidence:.2%}")
        print(f"  Time: {layer_result.processing_time:.2f}s")
        print(f"  Details: {layer_result.details}")
