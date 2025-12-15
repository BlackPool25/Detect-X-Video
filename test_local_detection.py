#!/usr/bin/env python3
"""
Local end-to-end test for deepfake detection pipeline
Tests individual components before deploying to Modal
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import time

print("=" * 80)
print("LOCAL DEEPFAKE DETECTION PIPELINE TEST")
print("=" * 80)

# Check GPU availability
import torch
print(f"\nGPU Status:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Current device: {torch.cuda.current_device()}")
else:
    print(f"  ⚠️ WARNING: Running on CPU - will be slower")

def test_preprocessing(video_path):
    """Test face extraction and audio extraction"""
    print("\n[STEP 1] Testing Preprocessing - Face Extraction...")
    
    try:
        from facenet_pytorch import MTCNN
        import ffmpeg
        import tempfile
        
        # Load face detector
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")
        
        face_detector = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
            select_largest=True
        )
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"  Video: {video_path.name}")
        print(f"  FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Sample frames (2 FPS)
        sample_rate = 2
        frame_interval = max(1, int(fps * sample_rate))
        face_crops = []
        frame_idx = 0
        
        while cap.isOpened() and len(face_crops) < 30:  # Max 30 frames for test
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                boxes, probs = face_detector.detect(rgb_frame)
                
                if boxes is not None and len(boxes) > 0:
                    best_idx = np.argmax(probs)
                    box = boxes[best_idx]
                    
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = rgb_frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face_resized = cv2.resize(face, (224, 224))
                        face_crops.append(face_resized)
            
            frame_idx += 1
        
        cap.release()
        
        print(f"  ✅ Extracted {len(face_crops)} face crops")
        
        # Test audio extraction
        print("\n  Testing audio extraction...")
        try:
            audio_path = tempfile.mktemp(suffix='.wav')
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
            
            audio_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            has_audio = audio_size > 1000  # At least 1KB
            
            if has_audio:
                print(f"  ✅ Audio extracted: {audio_size / 1024:.1f} KB")
            else:
                print(f"  ⚠️ No audio track or audio too short")
            
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        except Exception as e:
            print(f"  ⚠️ Audio extraction failed: {e}")
            has_audio = False
        
        return face_crops, has_audio, {"total_frames": total_frames, "fps": fps, "duration": duration}
    
    except Exception as e:
        print(f"  ❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return [], False, {}

def test_visual_detector(face_crops):
    """Test EfficientNet-B7 visual artifact detection"""
    print("\n[STEP 2] Testing Visual Artifact Detector...")
    
    try:
        from torchvision import transforms
        from PIL import Image
        
        # Load model - it's a TorchScript model
        print("  Loading EfficientNet-B7...")
        try:
            model = torch.jit.load("modal_services/weights/efficientnet_b7_deepfake.pt", map_location='cpu')
            model_type = "torchscript"
            print("  ✅ Loaded TorchScript model")
        except:
            # Fallback to regular model
            import timm
            model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=2)
            checkpoint = torch.load("modal_services/weights/efficientnet_b7_deepfake.pt", map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            model_type = "regular"
            print("  ✅ Loaded regular PyTorch model")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        scores = []
        
        print(f"  Analyzing {len(face_crops)} face crops...")
        with torch.no_grad():
            for idx, crop in enumerate(face_crops[:10]):  # Test first 10
                pil_image = Image.fromarray(crop.astype(np.uint8))
                tensor = transform(pil_image).unsqueeze(0).to(device)
                
                logits = model(tensor)
                
                # Handle different output formats
                # IMPORTANT: Model has inverted labels - we invert the prediction
                if model_type == "torchscript" or logits.shape[1] == 1:
                    # Sigmoid output - invert it
                    fake_prob = 1.0 - torch.sigmoid(logits[0, 0]).item()
                else:
                    # Softmax output - use class 0 instead of class 1
                    probs = torch.softmax(logits, dim=1)
                    fake_prob = probs[0, 0].item()
                
                scores.append(fake_prob)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"  ✅ Visual artifact score: {mean_score:.3f} (std: {std_score:.3f})")
        print(f"     Range: [{min(scores):.3f}, {max(scores):.3f}]")
        
        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "artifact_scores": scores
        }
    
    except Exception as e:
        print(f"  ❌ Visual detection failed: {e}")
        import traceback
        traceback.print_exc()
        return {"mean_score": 0.5, "std_score": 0.0}

def test_audio_detector(video_path):
    """Test Wav2Vec2 audio synthesis detection"""
    print("\n[STEP 3] Testing Audio Synthesis Detector...")
    
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
        import torchaudio
        import ffmpeg
        import tempfile
        
        # Extract audio
        audio_path = tempfile.mktemp(suffix='.wav')
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16000')
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            print("  ⚠️ No audio track - skipping audio detection")
            return {"synthesis_score": 0.0, "has_audio": False}
        
        # Load model
        print("  Loading Wav2Vec2...")
        try:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "modal_services/weights",
                local_files_only=True
            )
            processor = Wav2Vec2Processor.from_pretrained(
                "modal_services/weights",
                local_files_only=True
            )
        except:
            print("  ⚠️ Using base Wav2Vec2 model")
            model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base")
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            max_length=160000,
            truncation=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        if probs.shape[1] >= 2:
            fake_prob = probs[0, 1].item()
        else:
            fake_prob = probs[0, 0].item()
        
        print(f"  ✅ Audio synthesis score: {fake_prob:.3f}")
        
        os.unlink(audio_path)
        
        return {"synthesis_score": fake_prob, "has_audio": True}
    
    except Exception as e:
        print(f"  ⚠️ Audio detection failed: {e}")
        return {"synthesis_score": 0.0, "has_audio": False}

def test_fusion(visual_score, audio_score, has_audio, std_score):
    """Test fusion layer with double-sided anomaly detection"""
    print("\n[STEP 4] Testing Fusion Layer...")
    
    temporal_score = 1.0 - std_score  # High consistency = low std
    face_quality = 0.95
    
    if not has_audio:
        weights = {'visual': 0.85, 'temporal': 0.10, 'face_quality': 0.05}
        final_score = (
            weights['visual'] * visual_score +
            weights['temporal'] * (1.0 - temporal_score) +
            weights['face_quality'] * (1.0 - face_quality)
        )
    else:
        weights = {'visual': 0.50, 'audio': 0.35, 'temporal': 0.10, 'face_quality': 0.05}
        final_score = (
            weights['visual'] * visual_score +
            weights['audio'] * audio_score +
            weights['temporal'] * (1.0 - temporal_score) +
            weights['face_quality'] * (1.0 - face_quality)
        )
    
    # Double-sided anomaly detection
    threshold_lower = 0.08  # Extremely perfect = fake
    threshold_upper = 0.50  # Obvious artifacts = fake
    safe_zone_upper = 0.25  # Uncertain zone
    
    if final_score > threshold_upper:
        verdict = "DEEPFAKE DETECTED (High Confidence - Visible Artifacts)"
    elif final_score < threshold_lower:
        verdict = "DEEPFAKE DETECTED (High Confidence - Hyper-Real)"
    elif final_score > safe_zone_upper:
        verdict = "UNCERTAIN - Review Recommended"
    else:
        verdict = "AUTHENTIC"
    
    print(f"  Weights used: {weights}")
    print(f"  Final score: {final_score:.3f}")
    print(f"  Verdict: {verdict}")
    
    return {
        "final_score": final_score,
        "verdict": verdict,
        "weights": weights,
        "breakdown": {
            "visual_artifacts": visual_score,
            "audio_synthesis": audio_score if has_audio else None,
            "temporal_consistency": temporal_score,
            "face_quality": face_quality
        }
    }

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python3 test_local_detection.py <video_path>")
        print("\nExample:")
        print("  python3 test_local_detection.py Test-Video/Real/00003.mp4")
        print("  python3 test_local_detection.py Test-Video/Fake/01_03__hugging_happy__ISF9SP4G.mp4")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    start_time = time.time()
    
    # Run pipeline
    face_crops, has_audio, metadata = test_preprocessing(video_path)
    
    if len(face_crops) == 0:
        print("\n❌ No faces detected - cannot analyze video")
        sys.exit(1)
    
    visual_result = test_visual_detector(face_crops)
    audio_result = test_audio_detector(video_path)
    
    final_result = test_fusion(
        visual_result["mean_score"],
        audio_result["synthesis_score"],
        audio_result["has_audio"],
        visual_result["std_score"]
    )
    
    processing_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    print(f"Category: {'FAKE' if 'Fake' in str(video_path) else 'REAL'} (ground truth)")
    print(f"\nFinal Verdict: {final_result['verdict']}")
    print(f"Confidence Score: {final_result['final_score']:.1%}")
    print(f"\nDetector Breakdown:")
    print(f"  Visual Artifacts:      {visual_result['mean_score']:.3f} (higher = more fake)")
    print(f"  Temporal Consistency:  {final_result['breakdown']['temporal_consistency']:.3f} (higher = more consistent)")
    if audio_result["has_audio"]:
        print(f"  Audio Synthesis:       {audio_result['synthesis_score']:.3f} (higher = more synthetic)")
    else:
        print(f"  Audio Synthesis:       N/A (no audio)")
    print(f"  Face Quality:          {final_result['breakdown']['face_quality']:.3f}")
    print(f"\nProcessing Time: {processing_time:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
