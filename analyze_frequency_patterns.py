"""
Analyze frequency patterns to calibrate detection threshold
"""

import torch
import sys
import numpy as np
from pathlib import Path
import cv2

sys.path.insert(0, '/home/lightdesk/Projects/AI-Video/modal_services')
from frequency_detector import FAD_Head, SRMConv2d_simple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load F3Net FAD
fad_head = FAD_Head(256).to(device)
checkpoint = torch.load('/home/lightdesk/Projects/AI-Video/Weights/organized/f3net_best.pth', 
                       map_location='cpu', weights_only=False)
fad_weights = {k.replace('FAD_head.', ''): v for k, v in checkpoint.items() if 'FAD_head' in k}
fad_head.load_state_dict(fad_weights, strict=False)
fad_head.eval()

def analyze_video(video_path, label):
    """Analyze frequency patterns in video"""
    cap = cv2.VideoCapture(video_path)
    
    ratios = []
    high_vals = []
    low_vals = []
    
    for _ in range(8):  # Sample 8 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (256, 256))
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            fea_FAD = fad_head(frame_tensor)
            
            low_freq = fea_FAD[:, 0:3].abs().mean(dim=[2, 3]).mean(dim=1).item()
            high_freq = fea_FAD[:, 6:9].abs().mean(dim=[2, 3]).mean(dim=1).item()
            
            ratio = high_freq / (low_freq + 1e-6)
            
            ratios.append(ratio)
            high_vals.append(high_freq)
            low_vals.append(low_freq)
    
    cap.release()
    
    return {
        'label': label,
        'video': Path(video_path).name,
        'ratio_mean': np.mean(ratios),
        'ratio_std': np.std(ratios),
        'high_mean': np.mean(high_vals),
        'low_mean': np.mean(low_vals)
    }

print("Analyzing Frequency Patterns...")
print("="*70)

# Sample real videos
real_dir = Path('/home/lightdesk/Projects/AI-Video/Test-Video/Celeb-real')
fake_dir = Path('/home/lightdesk/Projects/AI-Video/Test-Video/Celeb-synthesis')

real_videos = list(real_dir.glob('*.mp4'))[:5]
fake_videos = list(fake_dir.glob('*.mp4'))[:5]

real_results = []
fake_results = []

print("\n📹 REAL Videos:")
for video in real_videos:
    result = analyze_video(str(video), 'REAL')
    real_results.append(result)
    print(f"  {result['video'][:20]:20s} | Ratio: {result['ratio_mean']:.4f} | High: {result['high_mean']:.4f} | Low: {result['low_mean']:.4f}")

print("\n📹 FAKE Videos:")
for video in fake_videos:
    result = analyze_video(str(video), 'FAKE')
    fake_results.append(result)
    print(f"  {result['video'][:20]:20s} | Ratio: {result['ratio_mean']:.4f} | High: {result['high_mean']:.4f} | Low: {result['low_mean']:.4f}")

# Calculate statistics
real_ratios = [r['ratio_mean'] for r in real_results]
fake_ratios = [r['ratio_mean'] for r in fake_results]

print(f"\n{'='*70}")
print("STATISTICS:")
print(f"{'='*70}")
print(f"\nREAL Videos:")
print(f"  Mean Ratio: {np.mean(real_ratios):.4f} (±{np.std(real_ratios):.4f})")
print(f"  Range: [{min(real_ratios):.4f}, {max(real_ratios):.4f}]")

print(f"\nFAKE Videos:")
print(f"  Mean Ratio: {np.mean(fake_ratios):.4f} (±{np.std(fake_ratios):.4f})")
print(f"  Range: [{min(fake_ratios):.4f}, {max(fake_ratios):.4f}]")

separation = abs(np.mean(real_ratios) - np.mean(fake_ratios))
print(f"\nSeparation: {separation:.4f}")

# Find optimal threshold
all_ratios = real_ratios + fake_ratios
all_labels = [0]*len(real_ratios) + [1]*len(fake_ratios)  # 0=real, 1=fake

best_threshold = None
best_accuracy = 0

for threshold in np.linspace(min(all_ratios), max(all_ratios), 100):
    correct = 0
    for ratio, label in zip(all_ratios, all_labels):
        # If ratio < threshold → classify as FAKE
        predicted_fake = ratio < threshold
        if predicted_fake == label:
            correct += 1
    
    accuracy = correct / len(all_labels)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"\n📊 Optimal Threshold: {best_threshold:.4f}")
print(f"   Expected Accuracy: {best_accuracy*100:.1f}%")
print(f"\nRecommendation: Use threshold = {best_threshold:.4f} in sigmoid function")
print(f"   fake_score = sigmoid(({best_threshold:.4f} - ratio) * scale)")
