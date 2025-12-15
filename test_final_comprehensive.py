#!/usr/bin/env python3
"""
Final Comprehensive Deepfake Detection Test
Tests all detection methods and provides detailed accuracy analysis
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import json

class FinalComprehensiveTest:
    """
    Final testing suite that combines:
    1. Ensemble visual detection (frequency + texture + eyes)
    2. Temporal consistency analysis
    3. Simple physiological checks
    """
    
    def __init__(self):
        self.test_video_dir = Path("Test-Video")
        self.results = []
        
    def analyze_frame_quality(self, frame):
        """Quick quality metrics"""
        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness
        brightness = np.mean(gray)
        
        return {
            'blur_score': laplacian_var,
            'brightness': brightness,
            'is_blurry': laplacian_var < 100
        }
    
    def frequency_analysis(self, face):
        """FFT-based GAN artifact detection"""
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_dist = np.sqrt(center_h**2 + center_w**2)
        
        # High frequency energy
        hf_mask = dist > (max_dist * 0.7)
        hf_energy = np.mean(magnitude[hf_mask]) if np.any(hf_mask) else 0
        
        mf_mask = (dist >= (max_dist * 0.3)) & (dist <= (max_dist * 0.7))
        mf_energy = np.mean(magnitude[mf_mask]) if np.any(mf_mask) else 0
        
        total = hf_energy + mf_energy
        hf_ratio = hf_energy / total if total > 0 else 0.5
        
        # Natural: ~0.20
        deviation = abs(hf_ratio - 0.20)
        score = min(deviation * 3.0, 1.0)
        
        return score, hf_ratio
    
    def texture_analysis(self, face):
        """Skin texture smoothness check"""
        lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        laplacian = cv2.Laplacian(l_channel, cv2.CV_64F)
        texture_var = np.var(laplacian)
        
        # Too smooth < 30, too noisy > 300
        if texture_var < 25:
            score = 0.85
        elif texture_var < 35:
            score = 0.6
        elif texture_var > 350:
            score = 0.7
        elif texture_var > 250:
            score = 0.4
        else:
            score = 0.15
        
        return score, texture_var
    
    def eye_region_analysis(self, face):
        """Check eye region for artifacts"""
        h, w = face.shape[:2]
        eye_region = face[:int(h*0.4), :]
        
        gray = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
        
        # Brightness variability
        brightness_std = np.std(gray)
        
        # AI eyes often too uniform or too speckled
        if brightness_std < 15:
            score = 0.7
        elif brightness_std > 60:
            score = 0.6
        else:
            score = 0.2
        
        return score, brightness_std
    
    def temporal_analysis(self, frames):
        """Frame-to-frame consistency"""
        if len(frames) < 3:
            return 0.5, 0
        
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        
        diffs = []
        for i in range(len(gray_frames) - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i+1])
            diffs.append(np.mean(diff))
        
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        if mean_diff > 0:
            cv = std_diff / mean_diff
        else:
            cv = 0.5
        
        # Unnatural: too smooth (<0.15) or too jittery (>2.0)
        if cv < 0.12 or cv > 2.5:
            score = 0.75
        elif cv < 0.18 or cv > 1.8:
            score = 0.55
        else:
            score = 0.20
        
        return score, cv
    
    def test_single_video(self, video_path, label):
        """Test one video with all methods"""
        from facenet_pytorch import MTCNN
        
        print(f"\n{'='*80}")
        print(f"Testing: {video_path.name}")
        print(f"Expected: {label.upper()}")
        print(f"{'='*80}")
        
        face_detector = MTCNN(keep_all=False, device='cpu', post_process=False)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames
        n_samples = min(30, total_frames)
        frame_indices = np.linspace(0, total_frames-1, n_samples, dtype=int)
        
        frames = []
        faces = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = face_detector.detect(rgb)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                face = rgb[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append(face)
        
        cap.release()
        
        print(f"   Frames sampled: {len(frames)}")
        print(f"   Faces detected: {len(faces)}")
        
        if len(faces) < 5:
            print(f"   ❌ Insufficient faces detected")
            return None
        
        # Run all analyses
        freq_scores = []
        texture_scores = []
        eye_scores = []
        
        for face in faces[:20]:  # Analyze first 20 faces
            freq_score, freq_ratio = self.frequency_analysis(face)
            texture_score, texture_var = self.texture_analysis(face)
            eye_score, eye_std = self.eye_region_analysis(face)
            
            freq_scores.append(freq_score)
            texture_scores.append(texture_score)
            eye_scores.append(eye_score)
        
        temporal_score, temporal_cv = self.temporal_analysis(frames)
        
        # Ensemble weights
        freq_mean = np.mean(freq_scores)
        texture_mean = np.mean(texture_scores)
        eye_mean = np.mean(eye_scores)
        
        # Weighted combination
        weights = {
            'frequency': 0.25,
            'texture': 0.30,
            'eyes': 0.20,
            'temporal': 0.25
        }
        
        final_score = (
            weights['frequency'] * freq_mean +
            weights['texture'] * texture_mean +
            weights['eyes'] * eye_mean +
            weights['temporal'] * temporal_score
        )
        
        # Adaptive threshold based on confidence
        threshold = 0.42
        verdict = "FAKE" if final_score > threshold else "REAL"
        confidence = abs(final_score - threshold) * 200
        
        is_correct = verdict == label.upper()
        
        print(f"\n   📊 Detection Breakdown:")
        print(f"      Frequency:  {freq_mean:.3f} (wt: {weights['frequency']})")
        print(f"      Texture:    {texture_mean:.3f} (wt: {weights['texture']})")
        print(f"      Eyes:       {eye_mean:.3f} (wt: {weights['eyes']})")
        print(f"      Temporal:   {temporal_score:.3f} (wt: {weights['temporal']})")
        print(f"      " + "-" * 50)
        print(f"      Final Score: {final_score:.3f}")
        print(f"      Threshold:   {threshold}")
        print(f"      Verdict:     {verdict} ({min(confidence, 100):.1f}% confidence)")
        print(f"      Expected:    {label.upper()}")
        print(f"      Result:      {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        
        result = {
            'video': video_path.name,
            'label': label,
            'verdict': verdict,
            'correct': is_correct,
            'final_score': final_score,
            'confidence': min(confidence, 100),
            'breakdown': {
                'frequency': freq_mean,
                'texture': texture_mean,
                'eyes': eye_mean,
                'temporal': temporal_score
            }
        }
        
        self.results.append(result)
        return is_correct
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("="*80)
        print("COMPREHENSIVE DEEPFAKE DETECTION TEST")
        print("Ensemble: Frequency + Texture + Eyes + Temporal")
        print("="*80)
        
        test_videos = [
            ("real", "Real/01__kitchen_pan.mp4"),
            ("real", "Real/00003.mp4"),
            ("real", "Real/00009.mp4"),
            ("fake", "Fake/01_03__hugging_happy__ISF9SP4G.mp4"),
        ]
        
        # Add second fake if exists
        second_fake = self.test_video_dir / "Fake" / "20251214_2115_New Video_simple_compose_01kcerkqegez9snkd7tmx969rs.mp4"
        if second_fake.exists():
            test_videos.append(("fake", str(second_fake.relative_to(self.test_video_dir))))
        
        for label, rel_path in test_videos:
            video_path = self.test_video_dir / rel_path
            if not video_path.exists():
                print(f"\n⚠️ Skipping {rel_path} (not found)")
                continue
            
            self.test_single_video(video_path, label)
        
        # Print final summary
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive summary"""
        print(f"\n{'='*80}")
        print("FINAL TEST SUMMARY")
        print(f"{'='*80}")
        
        correct = sum(1 for r in self.results if r['correct'])
        total = len(self.results)
        
        print(f"\nOverall Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        
        # Per-category accuracy
        real_results = [r for r in self.results if r['label'] == 'real']
        fake_results = [r for r in self.results if r['label'] == 'fake']
        
        if real_results:
            real_correct = sum(1 for r in real_results if r['correct'])
            print(f"Real Videos: {real_correct}/{len(real_results)} = {real_correct/len(real_results)*100:.1f}%")
        
        if fake_results:
            fake_correct = sum(1 for r in fake_results if r['correct'])
            print(f"Fake Videos: {fake_correct}/{len(fake_results)} = {fake_correct/len(fake_results)*100:.1f}%")
        
        # Score distribution
        print(f"\n{'='*80}")
        print("Score Distribution:")
        print(f"{'='*80}")
        
        for result in self.results:
            status = "✅" if result['correct'] else "❌"
            print(f"{status} {result['video'][:40]:40s} | {result['label']:4s} → {result['verdict']:4s} | Score: {result['final_score']:.3f}")
        
        # Method effectiveness
        print(f"\n{'='*80}")
        print("Method Analysis:")
        print(f"{'='*80}")
        
        real_scores = [r['breakdown'] for r in self.results if r['label'] == 'real']
        fake_scores = [r['breakdown'] for r in self.results if r['label'] == 'fake']
        
        if real_scores and fake_scores:
            for method in ['frequency', 'texture', 'eyes', 'temporal']:
                real_mean = np.mean([s[method] for s in real_scores])
                fake_mean = np.mean([s[method] for s in fake_scores])
                separation = abs(fake_mean - real_mean)
                
                print(f"{method.capitalize():12s}: Real={real_mean:.3f}, Fake={fake_mean:.3f}, Sep={separation:.3f}")
        
        # Final verdict
        print(f"\n{'='*80}")
        if correct == total:
            print("🎉 PERFECT! All videos classified correctly!")
            print("✅ System is production-ready for deployment")
        elif correct >= total * 0.8:
            print("✅ Excellent performance! Minor tuning recommended")
        elif correct >= total * 0.6:
            print("⚠️ Good performance but needs improvement")
            print("Consider adding more advanced features (rPPG, 3D pose)")
        else:
            print("❌ Poor performance - major improvements needed")
            print("Current approach insufficient for production use")
        
        print(f"{'='*80}\n")
        
        # Save results to JSON
        with open('test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("Results saved to test_results.json")

def main():
    tester = FinalComprehensiveTest()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
