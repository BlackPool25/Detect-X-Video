"""Quick test of frequency models on single frames"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/home/lightdesk/Projects/AI-Video/modal_services')
from frequency_detector import FAD_Head, SRMConv2d_simple

print("Testing Frequency Detectors on Single Frames")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load models
print("Loading F3Net FAD Head...")
fad_head = FAD_Head(256).to(device)

checkpoint = torch.load('/home/lightdesk/Projects/AI-Video/Weights/organized/f3net_best.pth', 
                       map_location='cpu', weights_only=False)

fad_weights = {k.replace('FAD_head.', ''): v for k, v in checkpoint.items() if 'FAD_head' in k}
fad_head.load_state_dict(fad_weights, strict=False)
fad_head.eval()
print("✅ F3Net FAD loaded\n")

print("Loading SRM...")
srm_conv = SRMConv2d_simple(inc=3, learnable=False).to(device)
srm_conv.eval()
print("✅ SRM loaded\n")

# Test on single frame
test_video = '/home/lightdesk/Projects/AI-Video/Test-Video/Celeb-real/id0_0000.mp4'
if Path(test_video).exists():
    cap = cv2.VideoCapture(test_video)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"Testing on frame from: {Path(test_video).name}")
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (256, 256))
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # F3Net frequency analysis
            print("\nF3Net Frequency Analysis:")
            fad_output = fad_head(frame_tensor)
            print(f"  Output shape: {fad_output.shape}")
            print(f"  Frequency bands (4 x 3 channels): {fad_output.shape[1]}")
            
            # Analyze frequency energy
            low_freq = fad_output[:, 0:3].abs().mean().item()
            mid_freq = fad_output[:, 3:6].abs().mean().item()
            high_freq = fad_output[:, 6:9].abs().mean().item()
            all_freq = fad_output[:, 9:12].abs().mean().item()
            
            print(f"  Low frequency energy: {low_freq:.4f}")
            print(f"  Mid frequency energy: {mid_freq:.4f}")
            print(f"  High frequency energy: {high_freq:.4f}")
            print(f"  All frequency energy: {all_freq:.4f}")
            
            # Simple heuristic: high/low ratio
            ratio = high_freq / (low_freq + 1e-6)
            f3net_score = 1.0 / (1.0 + np.exp(-(ratio - 1.0) * 2.0))  # Sigmoid
            print(f"  High/Low Ratio: {ratio:.4f}")
            print(f"  Fake Score (heuristic): {f3net_score:.4f}")
            
            # SRM noise analysis
            print("\nSRM Noise Residual Analysis:")
            x_norm = frame_tensor * 2.0 - 1.0
            srm_output = srm_conv(x_norm)
            print(f"  Output shape: {srm_output.shape}")
            print(f"  Noise residual channels: {srm_output.shape[1]}")
            
            # Analyze noise patterns
            noise_std = srm_output.std(dim=[2, 3]).mean().item()
            noise_mean_abs = srm_output.abs().mean().item()
            
            print(f"  Noise std deviation: {noise_std:.4f}")
            print(f"  Noise mean absolute: {noise_mean_abs:.4f}")
            
            # Simple heuristic: higher noise variance in fakes
            srm_score = 1.0 / (1.0 + np.exp(-( noise_std - 0.5) * 5.0))
            print(f"  Fake Score (heuristic): {srm_score:.4f}")
            
            # Ensemble
            ensemble_score = (f3net_score + srm_score) / 2.0
            prediction = "FAKE" if ensemble_score > 0.5 else "REAL"
            
            print(f"\n{'='*60}")
            print(f"ENSEMBLE PREDICTION: {prediction}")
            print(f"Confidence: {abs(ensemble_score - 0.5) * 200:.1f}%")
            print(f"Fake Score: {ensemble_score:.4f}")
            print(f"{'='*60}")

print("\n✅ Test complete!")
