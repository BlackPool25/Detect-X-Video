"""
Test the 4-layer deepfake detection pipeline on actual video files
Tests a sample of REAL and FAKE videos to validate end-to-end performance
"""

import torch
import numpy as np
import sys
import cv2
import subprocess
from pathlib import Path
import time
from typing import Dict, List

# Add repo paths
sys.path.insert(0, str(Path(__file__).parent / "UniversalFakeDetect"))

print("="*80)
print("REAL-WORLD VIDEO TESTING - 4-Layer Deepfake Detection")
print("="*80)

# ============================================================================
# LOAD ALL MODELS
# ============================================================================

def load_models():
    """Load all 4 detection models"""
    models = {}
    
    # Layer 1: Audio (Wav2Vec2)
    print("\n[LOADING] Layer 1: Audio Detection...")
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    audio_path = Path(__file__).parent / "Deepfake-audio-detection-V2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models['audio_extractor'] = Wav2Vec2FeatureExtractor.from_pretrained(str(audio_path))
    models['audio_model'] = Wav2Vec2ForSequenceClassification.from_pretrained(str(audio_path)).to(device)
    models['audio_model'].eval()
    print("✓ Audio model loaded")
    
    # Layer 2: Visual (EfficientNet-B4)
    print("\n[LOADING] Layer 2: Visual Artifacts Detection...")
    from efficientnet_pytorch import EfficientNet
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    weights_path = Path(__file__).parent / "weights" / "SBI" / "FFc23.tar"
    models['visual_model'] = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    models['visual_model'].load_state_dict(checkpoint['model'], strict=False)
    models['visual_model'].eval()
    
    models['visual_transform'] = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    print("✓ Visual model loaded")
    
    # Layer 4: Semantic (CLIP + FC)
    print("\n[LOADING] Layer 4: Generative Semantic Detection...")
    import clip
    import torch.nn as nn
    
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    
    fc_layer = nn.Linear(768, 1).to(device)
    fc_weights_path = Path(__file__).parent / "UniversalFakeDetect" / "pretrained_weights" / "fc_weights.pth"
    fc_checkpoint = torch.load(fc_weights_path, map_location=device, weights_only=False)
    fc_layer.load_state_dict(fc_checkpoint, strict=False)
    fc_layer.eval()
    
    models['clip_model'] = clip_model
    models['clip_preprocess'] = preprocess
    models['fc_layer'] = fc_layer
    print("✓ Semantic model loaded")
    
    models['device'] = device
    return models


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_audio_fake(audio_path: str, models: Dict) -> Dict:
    """Layer 1: Audio deepfake detection"""
    import librosa
    
    start = time.time()
    device = models['device']
    
    # Load audio (16kHz)
    audio, sr = librosa.load(audio_path, sr=16000, mono=True, duration=10)  # First 10 seconds
    
    # Process
    inputs = models['audio_extractor'](
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Inference
    with torch.no_grad():
        logits = models['audio_model'](**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        fake_prob = probs[0][1].item()
    
    return {
        'confidence': fake_prob if fake_prob > 0.5 else (1 - fake_prob),
        'is_fake': fake_prob > 0.5,
        'fake_probability': fake_prob,
        'time': time.time() - start
    }


def detect_visual_artifacts(video_path: str, models: Dict, num_frames: int = 16) -> Dict:
    """Layer 2: Visual artifact detection"""
    start = time.time()
    device = models['device']
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, min(total_frames - 1, 300), num_frames, dtype=int)  # Max 300 frames
    
    predictions = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed = models['visual_transform'](image=frame_rgb)
            img_tensor = transformed['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = models['visual_model'](img_tensor)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)
    
    cap.release()
    
    avg_score = np.mean(predictions)
    
    return {
        'confidence': avg_score if avg_score > 0.5 else (1 - avg_score),
        'is_fake': avg_score > 0.5,
        'fake_probability': avg_score,
        'time': time.time() - start,
        'frames_analyzed': len(predictions)
    }


def detect_semantic_fake(video_path: str, models: Dict, num_frames: int = 8) -> Dict:
    """Layer 4: Generative semantic detection"""
    from PIL import Image
    
    start = time.time()
    device = models['device']
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, min(total_frames - 1, 300), num_frames, dtype=int)
    
    predictions = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # CLIP preprocessing
            image_input = models['clip_preprocess'](frame_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = models['clip_model'].encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
                logit = models['fc_layer'](features.float())
                prob = torch.sigmoid(logit).item()
                predictions.append(prob)
    
    cap.release()
    
    avg_prob = np.mean(predictions)
    
    return {
        'confidence': avg_prob if avg_prob > 0.5 else (1 - avg_prob),
        'is_fake': avg_prob > 0.5,
        'fake_probability': avg_prob,
        'time': time.time() - start,
        'frames_analyzed': len(predictions)
    }


# ============================================================================
# PIPELINE
# ============================================================================

def analyze_video(video_path: str, models: Dict) -> Dict:
    """Run full detection pipeline"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {Path(video_path).name}")
    print(f"{'='*80}")
    
    # Extract audio
    audio_path = "/tmp/test_audio.wav"
    result = subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path, "-y"
    ], capture_output=True, text=True)
    
    if not Path(audio_path).exists():
        print(f"⚠️  Warning: Could not extract audio, skipping audio analysis")
        print(f"FFmpeg error: {result.stderr[:200]}")
        audio_result = {
            'confidence': 0.5,
            'is_fake': False,
            'fake_probability': 0.5,
            'time': 0.0
        }
    else:
        # Layer 1: Audio
        print("\n[Layer 1] Audio Analysis...")
        audio_result = detect_audio_fake(audio_path, models)
        print(f"  Fake Prob: {audio_result['fake_probability']:.2%} | Time: {audio_result['time']:.2f}s")
    
    results = {}
    results['audio'] = audio_result
    total_start = time.time()
    
    # FAIL-FAST: If audio is >90% confident fake, stop here
    if audio_result['is_fake'] and audio_result['confidence'] > 0.90:
        print(f"\n⚡ FAIL-FAST TRIGGERED: Audio confidence {audio_result['confidence']:.2%}")
        results['final_verdict'] = True
        results['final_confidence'] = audio_result['confidence']
        results['layers_executed'] = ['audio']
        results['total_time'] = time.time() - total_start
        return results
    
    # Layer 2: Visual
    print("\n[Layer 2] Visual Artifacts Analysis...")
    visual_result = detect_visual_artifacts(video_path, models, num_frames=16)
    results['visual'] = visual_result
    print(f"  Fake Prob: {visual_result['fake_probability']:.2%} | Frames: {visual_result['frames_analyzed']} | Time: {visual_result['time']:.2f}s")
    
    # Layer 4: Semantic (Skip Layer 3 LipSync for now - needs full implementation)
    print("\n[Layer 4] Semantic Analysis...")
    semantic_result = detect_semantic_fake(video_path, models, num_frames=8)
    results['semantic'] = semantic_result
    print(f"  Fake Prob: {semantic_result['fake_probability']:.2%} | Frames: {semantic_result['frames_analyzed']} | Time: {semantic_result['time']:.2f}s")
    
    # Aggregate (weights: Audio 40%, Visual 30%, Semantic 30%)
    weighted_score = (
        audio_result['fake_probability'] * 0.40 +
        visual_result['fake_probability'] * 0.30 +
        semantic_result['fake_probability'] * 0.30
    )
    
    results['final_verdict'] = weighted_score > 0.5
    results['final_confidence'] = weighted_score if weighted_score > 0.5 else (1 - weighted_score)
    results['final_fake_probability'] = weighted_score
    results['layers_executed'] = ['audio', 'visual', 'semantic']
    results['total_time'] = time.time() - total_start
    
    return results


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    # Load models once
    print("\nLoading all models (this may take a minute)...")
    models = load_models()
    
    print("\n" + "="*80)
    print("MODELS LOADED - Starting video tests")
    print("="*80)
    
    # Test videos
    test_videos = {
        'REAL': [
            "/home/lightdesk/Projects/AI-Video/Test-Video/Real/00003.mp4",
            "/home/lightdesk/Projects/AI-Video/Test-Video/Real/00009.mp4",
        ],
        'FAKE': [
            "/home/lightdesk/Projects/AI-Video/Test-Video/Fake/01_03__hugging_happy__ISF9SP4G.mp4",
            "/home/lightdesk/Projects/AI-Video/Test-Video/Celeb-synthesis/id53_id56_0008.mp4",
            "/home/lightdesk/Projects/AI-Video/Test-Video/Celeb-synthesis/id28_id19_0008.mp4",
        ]
    }
    
    results_summary = {
        'REAL': {'correct': 0, 'total': 0, 'results': []},
        'FAKE': {'correct': 0, 'total': 0, 'results': []}
    }
    
    # Test real videos
    for video_path in test_videos['REAL']:
        if not Path(video_path).exists():
            continue
            
        result = analyze_video(video_path, models)
        
        ground_truth = 'REAL'
        prediction = 'FAKE' if result['final_verdict'] else 'REAL'
        correct = ground_truth == prediction
        
        results_summary['REAL']['total'] += 1
        if correct:
            results_summary['REAL']['correct'] += 1
        
        results_summary['REAL']['results'].append({
            'file': Path(video_path).name,
            'prediction': prediction,
            'confidence': result['final_confidence'],
            'correct': correct,
            'time': result['total_time']
        })
        
        status = "✅ CORRECT" if correct else "❌ WRONG"
        print(f"\n{status} | Ground Truth: {ground_truth} | Predicted: {prediction} ({result['final_confidence']:.1%})")
        print(f"Total Time: {result['total_time']:.2f}s")
    
    # Test fake videos
    for video_path in test_videos['FAKE']:
        if not Path(video_path).exists():
            continue
            
        result = analyze_video(video_path, models)
        
        ground_truth = 'FAKE'
        prediction = 'FAKE' if result['final_verdict'] else 'REAL'
        correct = ground_truth == prediction
        
        results_summary['FAKE']['total'] += 1
        if correct:
            results_summary['FAKE']['correct'] += 1
        
        results_summary['FAKE']['results'].append({
            'file': Path(video_path).name,
            'prediction': prediction,
            'confidence': result['final_confidence'],
            'correct': correct,
            'time': result['total_time']
        })
        
        status = "✅ CORRECT" if correct else "❌ WRONG"
        print(f"\n{status} | Ground Truth: {ground_truth} | Predicted: {prediction} ({result['final_confidence']:.1%})")
        print(f"Total Time: {result['total_time']:.2f}s")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    total_correct = results_summary['REAL']['correct'] + results_summary['FAKE']['correct']
    total_videos = results_summary['REAL']['total'] + results_summary['FAKE']['total']
    
    print(f"\nREAL Videos: {results_summary['REAL']['correct']}/{results_summary['REAL']['total']} correct")
    for r in results_summary['REAL']['results']:
        emoji = "✅" if r['correct'] else "❌"
        print(f"  {emoji} {r['file'][:40]:40} -> {r['prediction']:4} ({r['confidence']:.1%}) [{r['time']:.1f}s]")
    
    print(f"\nFAKE Videos: {results_summary['FAKE']['correct']}/{results_summary['FAKE']['total']} correct")
    for r in results_summary['FAKE']['results']:
        emoji = "✅" if r['correct'] else "❌"
        print(f"  {emoji} {r['file'][:40]:40} -> {r['prediction']:4} ({r['confidence']:.1%}) [{r['time']:.1f}s]")
    
    accuracy = (total_correct / total_videos * 100) if total_videos > 0 else 0
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY: {total_correct}/{total_videos} ({accuracy:.1f}%)")
    print(f"{'='*80}")
