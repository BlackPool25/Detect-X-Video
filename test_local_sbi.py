#!/usr/bin/env python3
"""
Test SBI locally to verify it works before Modal deployment
"""
import sys
import torch
import numpy as np
import time

# Add SBI paths
sys.path.insert(0, './SelfBlendedImages/src')
sys.path.insert(0, './SelfBlendedImages/src/inference')

from model import Detector as SBIDetector
from retinaface.pre_trained_models import get_model as get_face_detector
from preprocess import extract_frames

def test_local_sbi():
    print("=" * 60)
    print("Local SBI Test")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load model
    print("\nLoading SBI model...")
    model = SBIDetector().to(device)
    weights_path = "./weights/SBI/FFc23.tar"
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval().half()
    print("✓ Model loaded in FP16 mode")
    
    # Load face detector
    print("\nLoading RetinaFace...")
    face_detector = get_face_detector("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()
    print("✓ Face detector loaded")
    
    # Test on a real video
    video_path = "./Test-Video/Real/00003.mp4"
    print(f"\nTesting on: {video_path}")
    
    start_time = time.time()
    
    # Extract frames
    print("Extracting frames...")
    n_frames = 8
    face_list, idx_list = extract_frames(
        video_path, n_frames, face_detector, image_size=(380, 380)
    )
    
    print(f"✓ Extracted {len(face_list)} face crops from {len(set(idx_list))} frames")
    print(f"  idx_list: {idx_list}")
    
    # Convert to tensor
    faces_tensor = torch.tensor(face_list).float() / 255.0
    faces_tensor = faces_tensor.half().to(device)
    print(f"✓ Tensor shape: {faces_tensor.shape}")
    
    # Inference
    print("Running inference...")
    with torch.no_grad(), torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
        pred = model(faces_tensor).softmax(1)[:, 1]
    
    print(f"✓ Predictions shape: {pred.shape}")
    print(f"  Raw predictions: {pred.cpu().numpy()}")
    
    # Aggregate
    print("\nAggregating predictions...")
    pred_list = []
    idx_img = -1
    for i in range(len(pred)):
        if idx_list[i] != idx_img:
            pred_list.append([])
            idx_img = idx_list[i]
        pred_list[-1].append(pred[i].item())
    
    print(f"✓ Grouped predictions by frame: {len(pred_list)} frames")
    for i, p in enumerate(pred_list):
        print(f"  Frame {i}: {p} -> max={max(p):.4f}")
    
    # Final result
    pred_res = np.array([max(p) for p in pred_list])
    avg_prob = float(pred_res.mean())
    
    processing_time = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"RESULT")
    print(f"=" * 60)
    print(f"Average Fake Probability: {avg_prob:.4f}")
    print(f"Is Fake (threshold=0.33): {avg_prob >= 0.33}")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"=" * 60)

if __name__ == "__main__":
    try:
        test_local_sbi()
    except Exception as e:
        import traceback
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
