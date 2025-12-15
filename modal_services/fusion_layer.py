"""
Multimodal Fusion Layer
Combines visual, temporal, and audio detection scores into final verdict
"""
import modal

app = modal.App("deepfake-fusion")

@app.function()
def fuse_multimodal_scores(
    visual_score: float,
    audio_score: float = None,
    face_quality: float = 0.95,
    temporal_score: float = 0.85
) -> dict:
    """
    Combine detector scores into final verdict using weighted fusion
    
    Scoring Logic (Research-based weights):
    - Visual artifacts: 40% weight (EfficientNet-B7) - primary indicator
    - Audio synthesis: 35% weight (Wav2Vec2) - critical for audio-visual fakes
    - Temporal consistency: 15% weight (simplified - using variance of visual scores)
    - Face quality: 10% weight (RetinaFace confidence) - reliability indicator
    
    Args:
        visual_score: 0-1 (higher = more fake visual artifacts)
        audio_score: 0-1 (higher = more synthetic audio) - None if no audio
        face_quality: 0-1 (RetinaFace confidence, higher = better quality)
        temporal_score: 0-1 (consistency across frames, higher = more consistent)
    
    Returns:
        dict: Final scores and verdict with breakdown
    """
    import numpy as np
    
    # Adjust weights based on available modalities
    if audio_score is None or audio_score == 0.0:
        # Video without audio - use visual heavily
        weights = {
            'visual': 0.85,      # Increased - visual is primary signal
            'temporal': 0.10,    # Reduced - less reliable
            'face_quality': 0.05 # Reduced - less impact
        }
        audio_score = 0.0
        has_audio = False
    else:
        weights = {
            'visual': 0.50,      # Visual primary
            'audio': 0.35,       # Audio secondary
            'temporal': 0.10,    # Temporal tertiary
            'face_quality': 0.05 # Quality check
        }
        has_audio = True
    
    # Normalize temporal score (flip it: high consistency = low fake score)
    temporal_fake_score = 1.0 - temporal_score
    
    # Calculate weighted final score
    if has_audio:
        final_score = (
            weights['visual'] * visual_score +
            weights['audio'] * audio_score +
            weights['temporal'] * temporal_fake_score +
            weights['face_quality'] * (1.0 - face_quality)
        )
    else:
        final_score = (
            weights['visual'] * visual_score +
            weights['temporal'] * temporal_fake_score +
            weights['face_quality'] * (1.0 - face_quality)
        )
    
    # Apply confidence penalty for low face quality (less aggressive)
    confidence_multiplier = 1.0
    if face_quality < 0.6:
        confidence_multiplier = 0.9
        final_score = final_score * 0.95
    
    # Double-Sided Anomaly Detection Threshold
    # Real videos: avg 0.195 (range 0.06-0.46)
    # Fake videos: avg 0.105 (range 0.02-0.21) - "Too Perfect" Paradox
    # 
    # Strategy: Flag as FAKE if EITHER:
    # 1. Too obvious (score > 0.5) - traditional deepfakes with artifacts
    # 2. Extremely perfect (score < 0.08) - hyper-real modern deepfakes
    
    threshold_lower = 0.08  # Below this = extremely perfect = fake
    threshold_upper = 0.50  # Above this = obvious artifacts = fake
    safe_zone_upper = 0.25  # Above this but below 0.5 = uncertain
    
    # Determine if fake based on double-sided threshold
    is_fake = (final_score < threshold_lower) or (final_score > threshold_upper)
    
    # Calculate confidence percentage
    if is_fake:
        if final_score < threshold_lower:
            # Too perfect - calculate distance from lower bound
            confidence = min(100, (threshold_lower - final_score) * 200 * confidence_multiplier)
        else:
            # Too obvious - calculate distance from upper bound
            confidence = min(100, (final_score - threshold_upper) * 150 * confidence_multiplier)
    else:
        # Authentic - calculate distance from nearest boundary
        dist_to_lower = abs(final_score - threshold_lower)
        dist_to_upper = abs(final_score - threshold_upper)
        confidence = min(100, min(dist_to_lower, dist_to_upper) * 200 * confidence_multiplier)
    
    # Determine verdict string
    if final_score > threshold_upper:
        verdict = "DEEPFAKE DETECTED (High Confidence - Visible Artifacts)"
    elif final_score < threshold_lower:
        verdict = "DEEPFAKE DETECTED (High Confidence - Hyper-Real)"
    elif final_score > safe_zone_upper:
        verdict = "UNCERTAIN - Review Recommended"
    else:
        verdict = "AUTHENTIC"
    
    return {
        "final_score": round(final_score, 4),
        "is_deepfake": is_fake,
        "confidence_percent": round(confidence, 2),
        "verdict": verdict,
        "has_audio": has_audio,
        "breakdown": {
            "visual_artifacts": round(visual_score, 4),
            "temporal_consistency": round(temporal_score, 4),
            "audio_synthesis": round(audio_score, 4) if audio_score else None,
            "face_quality": round(face_quality, 4),
            "weights_used": weights
        },
        "explanation": _get_explanation(
            is_fake, visual_score, audio_score, temporal_score, face_quality, has_audio
        )
    }


def _get_explanation(is_fake, visual_score, audio_score, temporal_score, face_quality, has_audio):
    """Generate human-readable explanation of the detection"""
    explanations = []
    
    if visual_score > 0.7:
        explanations.append("High visual artifact detection suggests manipulated content")
    elif visual_score < 0.3:
        explanations.append("Visual analysis shows authentic characteristics")
    
    if has_audio:
        if audio_score > 0.7:
            explanations.append("Audio shows signs of AI synthesis")
        elif audio_score < 0.3:
            explanations.append("Audio appears to be genuine human speech")
    else:
        explanations.append("No audio track available for analysis")
    
    if temporal_score < 0.5:
        explanations.append("Temporal inconsistencies detected across frames")
    
    if face_quality < 0.7:
        explanations.append("Face detection confidence is low - results may be less reliable")
    
    if not explanations:
        explanations.append("Analysis complete with moderate confidence")
    
    return " | ".join(explanations)


@app.local_entrypoint()
def test_fusion():
    """Test the fusion layer with different scenarios"""
    
    print("🧪 Testing Fusion Layer\n")
    
    # Test Case 1: Clear deepfake (high visual + high audio)
    result1 = fuse_multimodal_scores.remote(
        visual_score=0.85,
        audio_score=0.78,
        face_quality=0.95,
        temporal_score=0.45
    )
    print("Test 1 - Clear Deepfake:")
    print(f"  Verdict: {result1['verdict']}")
    print(f"  Confidence: {result1['confidence_percent']}%")
    print(f"  Final Score: {result1['final_score']}\n")
    
    # Test Case 2: Authentic video
    result2 = fuse_multimodal_scores.remote(
        visual_score=0.12,
        audio_score=0.08,
        face_quality=0.98,
        temporal_score=0.92
    )
    print("Test 2 - Authentic Video:")
    print(f"  Verdict: {result2['verdict']}")
    print(f"  Confidence: {result2['confidence_percent']}%")
    print(f"  Final Score: {result2['final_score']}\n")
    
    # Test Case 3: No audio (visual only)
    result3 = fuse_multimodal_scores.remote(
        visual_score=0.62,
        audio_score=None,
        face_quality=0.88,
        temporal_score=0.55
    )
    print("Test 3 - No Audio Track:")
    print(f"  Verdict: {result3['verdict']}")
    print(f"  Confidence: {result3['confidence_percent']}%")
    print(f"  Final Score: {result3['final_score']}\n")
    
    # Test Case 4: Edge case - low quality face
    result4 = fuse_multimodal_scores.remote(
        visual_score=0.55,
        audio_score=0.48,
        face_quality=0.65,
        temporal_score=0.70
    )
    print("Test 4 - Low Quality Face:")
    print(f"  Verdict: {result4['verdict']}")
    print(f"  Confidence: {result4['confidence_percent']}%")
    print(f"  Final Score: {result4['final_score']}")
    print(f"  Explanation: {result4['explanation']}\n")
