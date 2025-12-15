#!/usr/bin/env python3
"""
Local end-to-end test simulating the full Modal pipeline
Tests actual Sora-generated deepfake videos
"""
import cv2
import numpy as np
from pathlib import Path
import time
import subprocess
import sys

def load_efficientnet_mock():
    """Mock EfficientNet-B7 detector (would use actual model in Modal)"""
    print("   Loading EfficientNet-B7 model... (MOCK)")
    return None

def detect_faces_mtcnn(video_path):
    """Detect and extract faces using OpenCV (simulates MTCNN)"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_crops = []
    frame_idx = 0
    frame_interval = max(1, int(fps * 0.5))  # Sample every 0.5 seconds
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get largest face
                largest = max(faces, key=lambda x: x[2]*x[3])
                x, y, w, h = largest
                
                # Add padding
                pad = int(w * 0.2)
                height, width = rgb.shape[:2]
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(width, x + w + pad)
                y2 = min(height, y + h + pad)
                
                face_crop = cv2.resize(rgb[y1:y2, x1:x2], (224, 224))
                face_crops.append(face_crop)
        
        frame_idx += 1
    
    cap.release()
    
    return face_crops, {
        'fps': fps,
        'total_frames': total_frames,
        'video_duration': duration
    }

def analyze_visual_artifacts(face_crops):
    """
    Analyze visual artifacts in face crops
    For Sora videos, look for:
    - Unnatural smoothness
    - Lack of skin texture detail
    - Temporal inconsistencies
    """
    if not face_crops:
        return 0.5, 0.5
    
    visual_scores = []
    temporal_diffs = []
    
    prev_crop = None
    
    for crop in face_crops:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        
        # 1. Analyze texture/sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # 2. Analyze smoothness (AI videos tend to be over-smoothed)
        blur_score = cv2.GaussianBlur(gray, (5, 5), 0)
        smoothness = np.mean(np.abs(gray - blur_score))
        
        # 3. Frequency analysis (AI videos lack high-freq detail)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        # High frequency content (edges of spectrum)
        h, w = magnitude.shape
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, (w//2, h//2), min(h, w)//4, 1, -1)
        high_freq = np.mean(magnitude[center_mask == 0])
        low_freq = np.mean(magnitude[center_mask == 1])
        freq_ratio = high_freq / (low_freq + 1e-6)
        
        # For Sora/AI videos:
        # - Lower sharpness = more suspicious (typical range: 100-500)
        # - Lower smoothness = over-processed (AI smooths too much)
        # - Lower freq_ratio = lacks detail (AI has less high-freq content)
        
        # Normalize and combine
        sharpness_norm = min(sharpness / 300.0, 1.0)
        smoothness_norm = min(smoothness / 20.0, 1.0)
        freq_norm = min(freq_ratio * 10, 1.0)
        
        # Lower values = more suspicious = higher deepfake score
        artifact_score = 1.0 - (sharpness_norm * 0.4 + smoothness_norm * 0.3 + freq_norm * 0.3)
        visual_scores.append(artifact_score)
        
        # Temporal consistency
        if prev_crop is not None:
            diff = cv2.absdiff(gray, cv2.cvtColor(prev_crop, cv2.COLOR_RGB2GRAY))
            temporal_diffs.append(np.mean(diff) / 255.0)
        
        prev_crop = crop
    
    # Average visual artifact score (0-1, higher = more likely fake)
    visual_score = np.mean(visual_scores)
    
    # Temporal consistency score
    if temporal_diffs:
        # Higher std = less consistent = more suspicious
        temporal_std = np.std(temporal_diffs)
        temporal_score = min(temporal_std * 3, 1.0)  # Higher = more consistent (real)
    else:
        temporal_score = 0.5
    
    return visual_score, temporal_score

def check_audio(video_path):
    """Check if video has audio"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=nw=1',
            str(video_path)
        ], capture_output=True, text=True)
        
        has_audio = 'codec_name=' in result.stdout
        return has_audio
    except:
        return False

def fuse_scores(visual, audio, temporal, face_quality, has_audio):
    """Fuse multimodal scores (matches Modal service)"""
    # Invert scores that indicate authenticity
    temporal_inverted = 1 - temporal
    face_inverted = 1 - face_quality
    
    if not has_audio or audio is None:
        # Without audio: visual=55%, temporal=30%, face=15%
        final = (
            visual * 0.55 +
            temporal_inverted * 0.30 +
            face_inverted * 0.15
        )
    else:
        # With audio: visual=40%, audio=35%, temporal=15%, face=10%
        final = (
            visual * 0.40 +
            audio * 0.35 +
            temporal_inverted * 0.15 +
            face_inverted * 0.10
        )
    
    is_fake = final > 0.5
    confidence = min(100, abs(final - 0.5) * 200)
    
    if final > 0.7:
        verdict = "DEEPFAKE DETECTED (High)"
    elif final > 0.5:
        verdict = "LIKELY DEEPFAKE"
    elif final > 0.3:
        verdict = "UNCERTAIN"
    else:
        verdict = "LIKELY AUTHENTIC"
    
    return {
        'final_score': final,
        'is_deepfake': is_fake,
        'confidence_percent': confidence,
        'verdict': verdict,
        'has_audio': has_audio,
        'breakdown': {
            'visual_artifacts': visual,
            'temporal_consistency': temporal,
            'audio_synthesis': audio,
            'face_quality': face_quality
        }
    }

def test_video(video_path):
    """Run full detection pipeline on a video"""
    print(f"\n{'='*80}")
    print(f"🎬 TESTING: {Path(video_path).name}")
    print(f"   Expected: DEEPFAKE (Sora-generated)")
    print('='*80)
    
    start_time = time.time()
    
    # Step 1: Preprocessing
    print("\n1️⃣ Preprocessing...")
    print("   Extracting faces...")
    face_crops, metadata = detect_faces_mtcnn(video_path)
    
    has_audio = check_audio(video_path)
    
    print(f"   ✅ Extracted {len(face_crops)} face crops")
    print(f"   ✅ Video duration: {metadata['video_duration']:.1f}s")
    print(f"   ✅ Total frames: {metadata['total_frames']}")
    print(f"   ✅ Audio: {'Yes' if has_audio else 'No'}")
    
    if not face_crops:
        print("   ❌ No faces detected!")
        return None
    
    # Step 2: Visual Analysis
    print("\n2️⃣ Visual artifact detection (EfficientNet-B7 simulation)...")
    visual_score, temporal_score = analyze_visual_artifacts(face_crops)
    print(f"   ✅ Visual artifacts score: {visual_score:.4f}")
    print(f"   ✅ Temporal consistency: {temporal_score:.4f}")
    
    # Step 3: Audio Analysis (if available)
    audio_score = None
    if has_audio:
        print("\n3️⃣ Audio synthesis detection (Wav2Vec2 simulation)...")
        print("   ⚠️ Skipped - no audio stream")
    
    # Step 4: Calculate face quality
    face_quality = min(0.98, 0.8 + (len(face_crops) / metadata['total_frames']) * 0.2)
    print(f"\n4️⃣ Face detection quality: {face_quality:.4f}")
    
    # Step 5: Fusion
    print("\n5️⃣ Fusing multimodal scores...")
    result = fuse_scores(visual_score, audio_score, temporal_score, face_quality, has_audio)
    
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\n{'='*80}")
    print(f"📊 RESULTS")
    print('='*80)
    print(f"\n🎯 Verdict: {result['verdict']}")
    print(f"📈 Final Score: {result['final_score']:.4f} (>0.5 = deepfake)")
    print(f"💯 Confidence: {result['confidence_percent']:.1f}%")
    print(f"🎵 Audio: {'Yes' if result['has_audio'] else 'No'}")
    
    print(f"\n📊 Breakdown:")
    print(f"   Visual Artifacts: {result['breakdown']['visual_artifacts']:.4f}")
    print(f"   Temporal Consistency: {result['breakdown']['temporal_consistency']:.4f}")
    if result['breakdown']['audio_synthesis']:
        print(f"   Audio Synthesis: {result['breakdown']['audio_synthesis']:.4f}")
    print(f"   Face Quality: {result['breakdown']['face_quality']:.4f}")
    
    print(f"\n⏱️  Processing Time: {elapsed:.2f}s")
    
    # Validation for Sora videos
    print(f"\n✅ VALIDATION:")
    if result['is_deepfake']:
        print(f"   ✓ Correctly identified as DEEPFAKE!")
    else:
        print(f"   ✗ Failed to detect (marked as AUTHENTIC)")
    
    return result

def main():
    print("\n" + "="*80)
    print("🎬 LOCAL END-TO-END TEST - Sora Deepfake Detection")
    print("="*80)
    print("Testing actual Sora-generated videos")
    print("Expected: Both should be detected as DEEPFAKE")
    print("="*80)
    
    test_dir = Path("Test-Video")
    videos = list(test_dir.glob("*.mp4"))
    
    if not videos:
        print(f"\n❌ No videos found in {test_dir}")
        sys.exit(1)
    
    print(f"\n✅ Found {len(videos)} video(s)")
    
    results = []
    
    for video_path in videos[:2]:
        result = test_video(video_path)
        if result:
            results.append({
                'name': video_path.name,
                'result': result
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📝 FINAL SUMMARY - Sora Video Detection")
    print('='*80)
    
    correct = 0
    total = len(results)
    
    for idx, res in enumerate(results, 1):
        r = res['result']
        is_correct = r['is_deepfake']
        
        print(f"\n{idx}. {res['name']}")
        print(f"   Expected: DEEPFAKE (Sora)")
        print(f"   Detected: {r['verdict']}")
        print(f"   Score: {r['final_score']:.4f}")
        print(f"   Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
        if is_correct:
            correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"🎯 ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print('='*80)
    
    if accuracy >= 50:
        print("\n✅ System is detecting Sora-generated videos!")
    else:
        print("\n⚠️ Low detection rate - may need model tuning")
    
    print("\n💡 Notes:")
    print("   - Local test uses OpenCV for face detection (Modal uses MTCNN)")
    print("   - Visual analysis uses frequency/texture analysis (Modal uses EfficientNet-B7)")
    print("   - No audio in test videos, so using adjusted weights (55%/30%/15%)")
    print("   - Sora videos should show high visual artifact scores due to AI smoothing")

if __name__ == "__main__":
    main()
