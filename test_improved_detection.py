#!/usr/bin/env python3
"""
Improved Deepfake Detection Test Using Production-Ready Models
Uses actual deepfake-trained models from HuggingFace
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import sys
from PIL import Image
import torchaudio
import tempfile
import subprocess

class ImprovedDetectionTest:
    """Test deepfake detection with proper models"""
    
    def __init__(self):
        self.test_video_dir = Path("Test-Video")
        self.results = {}
        
    def test_visual_detection_with_huggingface(self):
        """Use HuggingFace deepfake detector (actually trained on deepfakes)"""
        print("\n" + "="*80)
        print("TEST: Visual Detection with Pre-trained Deepfake Detector")
        print("="*80)
        
        try:
            from transformers import pipeline
            from facenet_pytorch import MTCNN
            
            print("   Loading deepfake detector from HuggingFace...")
            
            # Use a simpler model that's actually available
            # dima806/deepfake_vs_real_image_detection is a ViT trained on deepfakes
            try:
                detector = pipeline("image-classification", 
                                  model="dima806/deepfake_vs_real_image_detection",
                                  device=-1)  # CPU
                print("   ✅ Loaded ViT deepfake detector")
            except Exception as e:
                print(f"   ⚠️ Could not load HF model: {e}")
                print("   Using fallback: Basic CNN with texture analysis")
                return self.test_with_frequency_analysis()
            
            face_detector = MTCNN(keep_all=True, device='cpu')
            
            # Test on real and fake videos
            test_cases = [
                ("REAL", self.test_video_dir / "Real" / "01__kitchen_pan.mp4"),
                ("FAKE", self.test_video_dir / "Fake" / "01_03__hugging_happy__ISF9SP4G.mp4")
            ]
            
            for label, video_path in test_cases:
                if not video_path.exists():
                    print(f"   ⚠️ Video not found: {video_path}")
                    continue
                
                print(f"\n   Testing {label} video: {video_path.name}")
                cap = cv2.VideoCapture(str(video_path))
                
                fake_scores = []
                frames_processed = 0
                
                # Sample 15 frames
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_indices = np.linspace(0, total_frames-1, min(15, total_frames), dtype=int)
                
                current_frame = 0
                for target_idx in frame_indices:
                    # Seek to frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, probs = face_detector.detect(rgb_frame)
                    
                    if boxes is not None and len(boxes) > 0:
                        # Take first/largest face
                        box = boxes[0]
                        x1, y1, x2, y2 = [int(b) for b in box]
                        x1, y1 = max(0, x1), max(0, y1)
                        face = rgb_frame[y1:y2, x1:x2]
                        
                        if face.size > 0:
                            # Run deepfake detector
                            face_pil = Image.fromarray(face)
                            result = detector(face_pil, top_k=2)
                            
                            # Parse results - look for 'fake' or 'real' labels
                            fake_score = 0.5  # default
                            for pred in result:
                                label_lower = pred['label'].lower()
                                if 'fake' in label_lower or 'deepfake' in label_lower:
                                    fake_score = pred['score']
                                    break
                                elif 'real' in label_lower or 'authentic' in label_lower:
                                    fake_score = 1.0 - pred['score']
                                    break
                            
                            fake_scores.append(fake_score)
                            frames_processed += 1
                            
                            if frames_processed <= 3:  # Print first 3 for debugging
                                print(f"      Frame {target_idx}: {result[0]['label']} ({result[0]['score']:.3f})")
                
                cap.release()
                
                if fake_scores:
                    mean_score = np.mean(fake_scores)
                    std_score = np.std(fake_scores)
                    
                    self.results[f"{label}_visual"] = {
                        "mean_fake_score": mean_score,
                        "std": std_score,
                        "frames": frames_processed,
                        "verdict": "FAKE" if mean_score > 0.5 else "REAL"
                    }
                    
                    print(f"   Frames processed: {frames_processed}")
                    print(f"   Mean fake probability: {mean_score:.4f} ± {std_score:.4f}")
                    print(f"   Verdict: {self.results[f'{label}_visual']['verdict']}")
                else:
                    print(f"   ❌ No faces detected!")
            
            # Check discrimination
            if "REAL_visual" in self.results and "FAKE_visual" in self.results:
                real_score = self.results["REAL_visual"]["mean_fake_score"]
                fake_score = self.results["FAKE_visual"]["mean_fake_score"]
                
                print(f"\n   DISCRIMINATION ANALYSIS:")
                print(f"   Real video fake-score: {real_score:.4f}")
                print(f"   Fake video fake-score: {fake_score:.4f}")
                print(f"   Separation: {abs(fake_score - real_score):.4f}")
                
                # Check if verdicts are correct
                real_correct = self.results["REAL_visual"]["verdict"] == "REAL"
                fake_correct = self.results["FAKE_visual"]["verdict"] == "FAKE"
                
                if real_correct and fake_correct:
                    print(f"   ✅ Both videos classified correctly!")
                    return True
                elif real_correct or fake_correct:
                    print(f"   ⚠️ One video misclassified")
                    return False
                else:
                    print(f"   ❌ Both videos misclassified!")
                    return False
            
            return False
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_with_frequency_analysis(self):
        """Fallback: Use frequency domain analysis (catches GAN artifacts)"""
        print("\n   Using Frequency Domain Analysis (FFT-based detection)...")
        
        try:
            from facenet_pytorch import MTCNN
            
            face_detector = MTCNN(keep_all=True, device='cpu')
            
            test_cases = [
                ("REAL", self.test_video_dir / "Real" / "01__kitchen_pan.mp4"),
                ("FAKE", self.test_video_dir / "Fake" / "01_03__hugging_happy__ISF9SP4G.mp4")
            ]
            
            for label, video_path in test_cases:
                if not video_path.exists():
                    continue
                
                print(f"\n   Testing {label} video: {video_path.name}")
                cap = cv2.VideoCapture(str(video_path))
                
                fft_scores = []
                frames_processed = 0
                
                for i in range(10):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Compute FFT
                    f_transform = np.fft.fft2(gray)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude = np.abs(f_shift)
                    
                    # AI-generated images often have periodic patterns in high frequencies
                    # Check for grid-like artifacts
                    h, w = magnitude.shape
                    center_h, center_w = h//2, w//2
                    
                    # Sample high-frequency region (outer 30%)
                    mask = np.zeros_like(magnitude)
                    cv2.circle(mask, (center_w, center_h), int(min(h, w) * 0.45), 0, -1)
                    cv2.circle(mask, (center_w, center_h), int(min(h, w) * 0.3), 1, -1)
                    
                    high_freq_energy = np.sum(magnitude * (mask > 0))
                    total_energy = np.sum(magnitude)
                    
                    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
                    
                    # Real images have natural high-freq noise
                    # AI images often have either too clean or periodic high-freq
                    # This is a simplified heuristic
                    fft_scores.append(high_freq_ratio)
                    frames_processed += 1
                
                cap.release()
                
                if fft_scores:
                    mean_score = np.mean(fft_scores)
                    std_score = np.std(fft_scores)
                    
                    print(f"   High-frequency ratio: {mean_score:.6f} ± {std_score:.6f}")
                    
                    self.results[f"{label}_fft"] = {
                        "hf_ratio": mean_score,
                        "std": std_score
                    }
            
            return True
            
        except Exception as e:
            print(f"   ❌ FFT analysis failed: {e}")
            return False
    
    def test_audio_detection(self):
        """Test audio synthesis detection"""
        print("\n" + "="*80)
        print("TEST: Audio Synthesis Detection")
        print("="*80)
        
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
            
            # Load model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "modal_services/weights",
                local_files_only=True
            ).eval()
            
            try:
                processor = Wav2Vec2Processor.from_pretrained(
                    "modal_services/weights",
                    local_files_only=True
                )
            except:
                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            
            print("   ✅ Audio model loaded")
            
            # Test on videos with audio
            test_videos = [
                ("FAKE", self.test_video_dir / "Fake" / "01_03__hugging_happy__ISF9SP4G.mp4")
            ]
            
            for label, video_path in test_videos:
                if not video_path.exists():
                    continue
                
                print(f"\n   Testing {label} video: {video_path.name}")
                
                # Extract audio with ffmpeg
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    audio_path = tmp.name
                
                try:
                    subprocess.run([
                        'ffmpeg', '-i', str(video_path),
                        '-vn', '-acodec', 'pcm_s16le',
                        '-ar', '16000', '-ac', '1',
                        audio_path, '-y'
                    ], check=True, capture_output=True)
                    
                    # Load and process audio
                    waveform, sr = torchaudio.load(audio_path)
                    
                    # Take 5-second chunks
                    chunk_duration = 5 * 16000
                    scores = []
                    
                    for start in range(0, min(waveform.shape[1], 20*16000), chunk_duration):
                        chunk = waveform[:, start:start+chunk_duration]
                        
                        if chunk.shape[1] < 16000:
                            break
                        
                        inputs = processor(
                            chunk.squeeze().numpy(),
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=True
                        )
                        
                        with torch.no_grad():
                            logits = model(**inputs).logits
                            probs = torch.softmax(logits, dim=-1)
                        
                        if probs.shape[1] >= 2:
                            fake_prob = probs[0, 1].item()
                            scores.append(fake_prob)
                    
                    if scores:
                        mean_score = np.mean(scores)
                        print(f"   Audio chunks processed: {len(scores)}")
                        print(f"   Mean synthetic score: {mean_score:.4f}")
                        print(f"   Verdict: {'SYNTHETIC' if mean_score > 0.5 else 'REAL'}")
                        
                        self.results[f"{label}_audio"] = {
                            "mean_score": mean_score,
                            "chunks": len(scores)
                        }
                    
                    # Cleanup
                    Path(audio_path).unlink()
                    
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️ No audio track found (expected for some test videos)")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Audio detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("DETECTION TEST SUMMARY")
        print("="*80)
        
        if not self.results:
            print("❌ No results to display")
            return
        
        print("\nVisual Detection Results:")
        for key in ["REAL_visual", "FAKE_visual"]:
            if key in self.results:
                data = self.results[key]
                print(f"  {key.split('_')[0]}:")
                print(f"    - Mean fake score: {data['mean_fake_score']:.4f}")
                print(f"    - Verdict: {data['verdict']}")
                print(f"    - Frames analyzed: {data['frames']}")
        
        print("\nFrequency Analysis Results:")
        for key in ["REAL_fft", "FAKE_fft"]:
            if key in self.results:
                data = self.results[key]
                print(f"  {key.split('_')[0]}: HF ratio = {data['hf_ratio']:.6f}")
        
        print("\nAudio Detection Results:")
        for key in ["FAKE_audio"]:
            if key in self.results:
                data = self.results[key]
                print(f"  Synthetic score: {data['mean_score']:.4f}")
                print(f"  Chunks analyzed: {data['chunks']}")
        
        # Overall accuracy
        correct = 0
        total = 0
        
        if "REAL_visual" in self.results:
            total += 1
            if self.results["REAL_visual"]["verdict"] == "REAL":
                correct += 1
        
        if "FAKE_visual" in self.results:
            total += 1
            if self.results["FAKE_visual"]["verdict"] == "FAKE":
                correct += 1
        
        if total > 0:
            accuracy = correct / total * 100
            print(f"\n{'='*80}")
            print(f"OVERALL ACCURACY: {correct}/{total} = {accuracy:.1f}%")
            print(f"{'='*80}")
            
            if accuracy == 100:
                print("🎉 PERFECT! All test videos classified correctly!")
            elif accuracy >= 50:
                print("✅ Decent performance, but room for improvement")
            else:
                print("❌ Poor performance - models may need better training data")

def main():
    """Run improved tests"""
    print("="*80)
    print("IMPROVED DEEPFAKE DETECTION TEST SUITE")
    print("Using Production-Ready Models")
    print("="*80)
    
    suite = ImprovedDetectionTest()
    
    # Run tests
    suite.test_visual_detection_with_huggingface()
    suite.test_audio_detection()
    
    # Print summary
    suite.print_summary()

if __name__ == "__main__":
    main()
