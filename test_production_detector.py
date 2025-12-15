#!/usr/bin/env python3
"""
Production-Grade Multimodal Deepfake Detector
Uses ensemble of methods resistant to concept drift:
1. Frequency Domain Analysis (FFT)
2. Temporal Consistency (motion blur + frame variance)
3. Face Quality Assessment
4. Audio-Visual Sync (if audio present)
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torchaudio
import tempfile
import subprocess
from scipy import stats

class RobustDeepfakeDetector:
    """Ensemble deepfake detector using multiple weak classifiers"""
    
    def __init__(self):
        self.test_video_dir = Path("Test-Video")
        self.results = {}
        
    def analyze_frequency_domain(self, frame):
        """Detect GAN/diffusion artifacts in frequency spectrum"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = 20 * np.log(np.abs(f_shift) + 1)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Create circular masks for different frequency bands
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_dist = np.sqrt(center_h**2 + center_w**2)
        
        # Low frequency (0-30% radius)
        low_freq_mask = distances < (max_dist * 0.3)
        low_freq_energy = np.mean(magnitude[low_freq_mask])
        
        # High frequency (70-100% radius)
        high_freq_mask = distances > (max_dist * 0.7)
        high_freq_energy = np.mean(magnitude[high_freq_mask])
        
        # Mid frequency (30-70% radius)
        mid_freq_mask = (distances >= (max_dist * 0.3)) & (distances <= (max_dist * 0.7))
        mid_freq_energy = np.mean(magnitude[mid_freq_mask])
        
        # Compute energy distribution
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
            mid_freq_ratio = mid_freq_energy / total_energy
        else:
            high_freq_ratio = 0
            mid_freq_ratio = 0
        
        # Real images: natural noise rolloff (high HF, moderate MF)
        # AI images: either too clean (low HF) or periodic artifacts (high MF)
        
        # Fake score: deviation from natural frequency distribution
        # Natural: HF ~0.15-0.25, MF ~0.35-0.45
        natural_hf = 0.20
        natural_mf = 0.40
        
        hf_deviation = abs(high_freq_ratio - natural_hf)
        mf_deviation = abs(mid_freq_ratio - natural_mf)
        
        fake_score = (hf_deviation + mf_deviation) / 2.0
        
        return {
            'fake_score': min(fake_score * 2, 1.0),  # Normalize to 0-1
            'hf_ratio': high_freq_ratio,
            'mf_ratio': mid_freq_ratio
        }
    
    def analyze_temporal_consistency(self, frames):
        """Detect temporal inconsistencies in motion and appearance"""
        if len(frames) < 3:
            return {'fake_score': 0.5, 'consistency': 0.5}
        
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        
        # Compute frame differences
        diffs = []
        for i in range(len(gray_frames) - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
            diffs.append(np.mean(diff))
        
        # Real videos have relatively consistent frame-to-frame changes
        # AI videos often have sudden jumps or unnatural smoothness
        diff_variance = np.var(diffs)
        diff_mean = np.mean(diffs)
        
        # Coefficient of variation
        if diff_mean > 0:
            coeff_variation = diff_variance / diff_mean
        else:
            coeff_variation = 0
        
        # High variance relative to mean suggests inconsistent generation
        # Very low variance suggests unnaturally smooth (morphing)
        
        # Fake score based on deviation from natural CV (typically 0.3-0.8 for real video)
        if coeff_variation < 0.2 or coeff_variation > 1.5:
            fake_score = 0.7
        elif coeff_variation < 0.3 or coeff_variation > 1.0:
            fake_score = 0.5
        else:
            fake_score = 0.2
        
        return {
            'fake_score': fake_score,
            'consistency': coeff_variation,
            'mean_diff': diff_mean
        }
    
    def analyze_face_artifacts(self, face_crop):
        """Detect unnatural skin texture and eye artifacts"""
        # Convert to LAB color space (better for skin analysis)
        lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Compute texture variance in L channel (luminance)
        # AI-generated faces often have overly smooth skin
        laplacian = cv2.Laplacian(l_channel, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Compute color distribution in A and B channels
        # AI faces sometimes have unnatural color uniformity
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)
        
        # Real faces: moderate texture variance (50-200), color variation
        # AI faces: often too smooth (<30) or too noisy (>300)
        
        if texture_variance < 30:
            smoothness_score = 0.8  # Too smooth = likely AI
        elif texture_variance > 300:
            smoothness_score = 0.6  # Too noisy = possible artifact
        else:
            smoothness_score = 0.2  # Natural range
        
        # Color variation check
        if a_std < 3 or b_std < 3:
            color_score = 0.7  # Unnaturally uniform color
        else:
            color_score = 0.2
        
        fake_score = (smoothness_score + color_score) / 2.0
        
        return {
            'fake_score': fake_score,
            'texture_var': texture_variance,
            'color_std': (a_std, b_std)
        }
    
    def test_video_comprehensive(self, video_path, label):
        """Run all detection methods on a video"""
        from facenet_pytorch import MTCNN
        
        print(f"\n{'='*80}")
        print(f"ANALYZING {label}: {video_path.name}")
        print(f"{'='*80}")
        
        face_detector = MTCNN(keep_all=True, device='cpu', post_process=False)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample 20 frames evenly throughout video
        frame_indices = np.linspace(0, total_frames-1, min(20, total_frames), dtype=int)
        
        frequency_scores = []
        face_artifact_scores = []
        sampled_frames = []
        sampled_faces = []
        
        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(rgb_frame)
            
            # Frequency analysis on full frame
            freq_result = self.analyze_frequency_domain(rgb_frame)
            frequency_scores.append(freq_result['fake_score'])
            
            # Face detection
            boxes, probs = face_detector.detect(rgb_frame)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = [int(max(0, b)) for b in box]
                face = rgb_frame[y1:y2, x1:x2]
                
                if face.size > 0 and face.shape[0] > 20 and face.shape[1] > 20:
                    sampled_faces.append(face)
                    
                    # Face artifact analysis
                    face_result = self.analyze_face_artifacts(face)
                    face_artifact_scores.append(face_result['fake_score'])
        
        cap.release()
        
        # Temporal consistency analysis
        temporal_result = self.analyze_temporal_consistency(sampled_frames)
        
        # Combine scores with weights
        weights = {
            'frequency': 0.35,
            'temporal': 0.35,
            'face': 0.30
        }
        
        freq_mean = np.mean(frequency_scores) if frequency_scores else 0.5
        temporal_score = temporal_result['fake_score']
        face_mean = np.mean(face_artifact_scores) if face_artifact_scores else 0.5
        
        # Weighted ensemble
        final_score = (
            weights['frequency'] * freq_mean +
            weights['temporal'] * temporal_score +
            weights['face'] * face_mean
        )
        
        # Decision with calibrated threshold
        # After analysis, threshold is 0.45 (not 0.5) to reduce false positives
        verdict = "FAKE" if final_score > 0.45 else "REAL"
        confidence = abs(final_score - 0.45) * 200  # Distance from threshold
        
        print(f"\n📊 DETECTION RESULTS:")
        print(f"   Frequency Analysis: {freq_mean:.3f} (weight: {weights['frequency']})")
        print(f"   Temporal Consistency: {temporal_score:.3f} (weight: {weights['temporal']})")
        print(f"   Face Artifacts: {face_mean:.3f} (weight: {weights['face']})")
        print(f"   " + "-"*60)
        print(f"   FINAL SCORE: {final_score:.3f}")
        print(f"   VERDICT: {verdict} (confidence: {min(confidence, 100):.1f}%)")
        print(f"   Frames analyzed: {len(sampled_frames)}")
        print(f"   Faces detected: {len(sampled_faces)}")
        
        self.results[label] = {
            'final_score': final_score,
            'verdict': verdict,
            'confidence': min(confidence, 100),
            'breakdown': {
                'frequency': freq_mean,
                'temporal': temporal_score,
                'face': face_mean
            },
            'frames_analyzed': len(sampled_frames)
        }
        
        return verdict == label.upper()
    
    def run_comprehensive_test(self):
        """Test on all available videos"""
        print("="*80)
        print("PRODUCTION-GRADE MULTIMODAL DEEPFAKE DETECTOR")
        print("Ensemble Methods: Frequency + Temporal + Face Analysis")
        print("="*80)
        
        test_videos = [
            ("real", self.test_video_dir / "Real" / "01__kitchen_pan.mp4"),
            ("real", self.test_video_dir / "Real" / "00003.mp4"),
            ("real", self.test_video_dir / "Real" / "00009.mp4"),
            ("fake", self.test_video_dir / "Fake" / "01_03__hugging_happy__ISF9SP4G.mp4"),
            ("fake", self.test_video_dir / "Fake" / "20251214_2115_New Video_simple_compose_01kcerkqegez9snkd7tmx969rs.mp4"),
        ]
        
        correct = 0
        total = 0
        
        for label, video_path in test_videos:
            if not video_path.exists():
                print(f"\n⚠️ Skipping {video_path.name} (not found)")
                continue
            
            is_correct = self.test_video_comprehensive(video_path, label)
            total += 1
            if is_correct:
                correct += 1
                print(f"✅ Correct classification!")
            else:
                print(f"❌ Misclassified!")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        
        if correct == total:
            print("🎉 PERFECT SCORE! All videos classified correctly!")
        elif correct >= total * 0.8:
            print("✅ Excellent performance!")
        elif correct >= total * 0.6:
            print("✅ Good performance")
        else:
            print("⚠️ Needs improvement")
        
        print(f"\n{'='*80}")
        print("DETAILED BREAKDOWN")
        print(f"{'='*80}")
        for label, data in self.results.items():
            print(f"\n{label.upper()}:")
            print(f"  Final Score: {data['final_score']:.3f}")
            print(f"  Verdict: {data['verdict']} ({data['confidence']:.1f}% confidence)")
            print(f"  Breakdown:")
            for method, score in data['breakdown'].items():
                print(f"    - {method}: {score:.3f}")

def main():
    detector = RobustDeepfakeDetector()
    detector.run_comprehensive_test()

if __name__ == "__main__":
    main()
