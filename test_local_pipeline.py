#!/usr/bin/env python3
"""
Local test of the detection pipeline without Modal
Tests face detection, video processing, and score fusion logic
"""
import cv2
import numpy as np
from pathlib import Path
import time

def test_face_detection(video_path):
    """Test face detection with OpenCV (MTCNN alternative for local testing)"""
    print(f"\n🔍 Testing face detection on: {Path(video_path).name}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Total frames: {total_frames}")
    
    # Load OpenCV face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces_detected = []
    frames_with_faces = 0
    frame_count = 0
    
    # Sample every 5th frame for speed
    sample_rate = 5
    
    print(f"\n🎬 Processing frames (sampling every {sample_rate} frames)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % sample_rate != 0:
            continue
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            frames_with_faces += 1
            for (x, y, w, h) in faces:
                face_size = w * h
                faces_detected.append({
                    'frame': frame_count,
                    'bbox': (x, y, w, h),
                    'size': face_size,
                    'confidence': 0.9  # Mock confidence
                })
    
    cap.release()
    
    sampled_frames = frame_count // sample_rate
    face_detection_rate = frames_with_faces / sampled_frames if sampled_frames > 0 else 0
    
    print(f"\n✅ Face Detection Results:")
    print(f"   Sampled frames: {sampled_frames}")
    print(f"   Frames with faces: {frames_with_faces}")
    print(f"   Detection rate: {face_detection_rate*100:.1f}%")
    print(f"   Total faces detected: {len(faces_detected)}")
    
    if len(faces_detected) > 0:
        avg_face_size = np.mean([f['size'] for f in faces_detected])
        print(f"   Average face size: {avg_face_size:.0f} pixels²")
        
        # Calculate face quality score (based on size and detection rate)
        # Normalize face size (assume 480x854 video, good face size is ~40k pixels²)
        size_score = min(avg_face_size / 40000.0, 1.0)
        quality_score = (size_score * 0.6) + (face_detection_rate * 0.4)
        print(f"   Face quality score: {quality_score:.4f}")
        
        return {
            'success': True,
            'faces_detected': len(faces_detected),
            'detection_rate': face_detection_rate,
            'quality_score': quality_score,
            'avg_face_size': avg_face_size
        }
    else:
        print(f"   ❌ No faces detected!")
        return {
            'success': False,
            'quality_score': 0.0
        }

def test_temporal_consistency(video_path):
    """Test temporal consistency (frame-to-frame similarity)"""
    print(f"\n🎬 Testing temporal consistency...")
    
    cap = cv2.VideoCapture(str(video_path))
    
    prev_frame = None
    frame_diffs = []
    frame_count = 0
    sample_rate = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
        
        # Convert to grayscale and resize for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 160))
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(small, prev_frame)
            diff_mean = np.mean(diff) / 255.0  # Normalize to 0-1
            frame_diffs.append(diff_mean)
        
        prev_frame = small
    
    cap.release()
    
    if len(frame_diffs) > 0:
        avg_diff = np.mean(frame_diffs)
        std_diff = np.std(frame_diffs)
        
        # Lower std = more consistent = less likely to be manipulated
        # Normalize: high consistency (low std) = high score
        consistency_score = max(0, 1.0 - (std_diff * 5))  # Scale factor
        
        print(f"✅ Temporal Consistency Results:")
        print(f"   Avg frame difference: {avg_diff:.4f}")
        print(f"   Std of differences: {std_diff:.4f}")
        print(f"   Consistency score: {consistency_score:.4f}")
        
        return {
            'success': True,
            'avg_diff': avg_diff,
            'std_diff': std_diff,
            'consistency_score': consistency_score
        }
    else:
        return {'success': False, 'consistency_score': 0.5}

def simulate_visual_detection(video_path):
    """Simulate visual artifact detection (mock EfficientNet-B7)"""
    print(f"\n🖼️  Simulating visual artifact detection...")
    
    # In real system, EfficientNet-B7 analyzes face crops
    # For local test, we'll use simple image statistics
    
    cap = cv2.VideoCapture(str(video_path))
    
    sharpness_scores = []
    noise_scores = []
    frame_count = 0
    sample_rate = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_scores.append(sharpness)
        
        # Calculate noise level (high frequency content)
        noise = np.std(gray)
        noise_scores.append(noise)
    
    cap.release()
    
    if len(sharpness_scores) > 0:
        avg_sharpness = np.mean(sharpness_scores)
        avg_noise = np.mean(noise_scores)
        
        # Mock visual artifact score
        # Real videos tend to have consistent sharpness and moderate noise
        # Deepfakes might have irregular patterns
        
        # Normalize sharpness (typical range 0-500)
        sharpness_norm = min(avg_sharpness / 200.0, 1.0)
        
        # This is a mock - in reality EfficientNet-B7 learns patterns
        # Higher score = more likely real
        visual_score = 0.5 + (np.random.random() * 0.3)  # Random for mock
        
        print(f"✅ Visual Analysis Results:")
        print(f"   Avg sharpness: {avg_sharpness:.2f}")
        print(f"   Avg noise: {avg_noise:.2f}")
        print(f"   Visual authenticity score: {visual_score:.4f} (MOCK)")
        
        return {
            'success': True,
            'visual_score': visual_score
        }
    else:
        return {'success': False, 'visual_score': 0.5}

def test_audio_extraction(video_path):
    """Test audio extraction"""
    print(f"\n🎵 Testing audio extraction...")
    
    import subprocess
    
    # Try to extract audio info using ffprobe
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_name,duration',
            '-of', 'default=nw=1',
            str(video_path)
        ], capture_output=True, text=True)
        
        has_audio = 'codec_name=' in result.stdout
        
        if has_audio:
            print(f"✅ Audio stream detected")
            return {'has_audio': True, 'audio_score': None}
        else:
            print(f"⚠️  No audio stream detected")
            return {'has_audio': False, 'audio_score': None}
    except Exception as e:
        print(f"❌ Error checking audio: {e}")
        return {'has_audio': False, 'audio_score': None}

def fuse_scores(face_result, temporal_result, visual_result, audio_result):
    """Simulate multimodal fusion (matches Modal service logic)"""
    print(f"\n🔀 Fusing multimodal scores...")
    
    has_audio = audio_result.get('has_audio', False)
    
    # Get individual scores
    visual_score = visual_result.get('visual_score', 0.5)
    temporal_score = temporal_result.get('consistency_score', 0.5)
    face_quality = face_result.get('quality_score', 0.5)
    audio_score = audio_result.get('audio_score') if has_audio else None
    
    print(f"\n📊 Individual Scores:")
    print(f"   Visual artifacts: {visual_score:.4f}")
    print(f"   Temporal consistency: {temporal_score:.4f}")
    print(f"   Face quality: {face_quality:.4f}")
    if has_audio and audio_score is not None:
        print(f"   Audio synthesis: {audio_score:.4f}")
    else:
        print(f"   Audio synthesis: N/A (no audio)")
    
    # Weighted fusion (matches modal_services/deepfake_detector.py)
    if has_audio and audio_score is not None:
        # With audio: visual=40%, audio=35%, temporal=15%, face=10%
        final_score = (
            visual_score * 0.40 +
            audio_score * 0.35 +
            temporal_score * 0.15 +
            face_quality * 0.10
        )
        weights_used = "visual=40%, audio=35%, temporal=15%, face=10%"
    else:
        # Without audio: visual=55%, temporal=30%, face=15%
        final_score = (
            visual_score * 0.55 +
            temporal_score * 0.30 +
            face_quality * 0.15
        )
        weights_used = "visual=55%, temporal=30%, face=15% (no audio)"
    
    # Determine verdict
    if final_score > 0.7:
        verdict = "LIKELY REAL"
        confidence = (final_score - 0.7) / 0.3 * 100
    elif final_score < 0.3:
        verdict = "LIKELY DEEPFAKE"
        confidence = (0.3 - final_score) / 0.3 * 100
    else:
        verdict = "UNCERTAIN"
        confidence = 100 - abs(final_score - 0.5) / 0.2 * 100
    
    print(f"\n🎯 Final Results:")
    print(f"   Fusion weights: {weights_used}")
    print(f"   Final score: {final_score:.4f}")
    print(f"   Verdict: {verdict}")
    print(f"   Confidence: {confidence:.1f}%")
    
    return {
        'final_score': final_score,
        'verdict': verdict,
        'confidence_percent': confidence,
        'has_audio': has_audio,
        'breakdown': {
            'visual_artifacts': visual_score,
            'temporal_consistency': temporal_score,
            'audio_synthesis': audio_score,
            'face_quality': face_quality
        }
    }

def test_video(video_path):
    """Run complete local test on a video"""
    print(f"\n{'='*80}")
    print(f"🎬 TESTING: {Path(video_path).name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Test each component
    face_result = test_face_detection(video_path)
    temporal_result = test_temporal_consistency(video_path)
    visual_result = simulate_visual_detection(video_path)
    audio_result = test_audio_extraction(video_path)
    
    # Fuse scores
    final_result = fuse_scores(face_result, temporal_result, visual_result, audio_result)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total processing time: {elapsed:.2f}s")
    
    return final_result

def main():
    """Test both videos"""
    print("\n" + "="*80)
    print("🧪 LOCAL PIPELINE TEST - Multimodal Deepfake Detection")
    print("="*80)
    
    test_dir = Path("Test-Video")
    videos = list(test_dir.glob("*.mp4"))
    
    if not videos:
        print(f"❌ No videos found in {test_dir}")
        return
    
    print(f"\n✅ Found {len(videos)} video(s)")
    
    results = []
    
    for video_path in videos[:2]:  # Test first 2
        result = test_video(video_path)
        results.append({
            'name': video_path.name,
            'result': result
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📝 SUMMARY")
    print('='*80)
    
    for idx, res in enumerate(results, 1):
        r = res['result']
        print(f"\n{idx}. {res['name']}")
        print(f"   Verdict: {r['verdict']}")
        print(f"   Score: {r['final_score']:.4f}")
        print(f"   Confidence: {r['confidence_percent']:.1f}%")
        print(f"   Has Audio: {'Yes' if r['has_audio'] else 'No'}")
    
    print(f"\n✅ Local testing complete!")
    print(f"\n💡 Note: Visual artifact scores are MOCKED in local test.")
    print(f"   Real Modal deployment uses EfficientNet-B7 for actual detection.")

if __name__ == "__main__":
    main()
