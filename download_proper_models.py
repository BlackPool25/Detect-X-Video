#!/usr/bin/env python3
"""
Download proper deepfake detection models from HuggingFace
"""

from pathlib import Path
import torch
from transformers import AutoModel, AutoFeatureExtractor, pipeline
import timm

weights_dir = Path("modal_services/weights")
weights_dir.mkdir(exist_ok=True, parents=True)

print("="*80)
print("Downloading Proper Deepfake Detection Models")
print("="*80)

# Option 1: Use a pretrained deepfake detector from HuggingFace
print("\n1. Checking for HuggingFace deepfake detectors...")

try:
    # Try the SigLIP-based detector mentioned in research
    print("   Attempting to load prithivMLmods/deepfake-detector-model-v1...")
    detector = pipeline("image-classification", model="prithivMLmods/deepfake-detector-model-v1")
    print("   ✅ Successfully loaded SigLIP detector (will use via transformers)")
except Exception as e:
    print(f"   ⚠️ Could not load: {e}")

# Option 2: Use standard EfficientNet-B7 and fine-tune it ourselves
# For now, we'll use a properly configured EfficientNet-B7 with 2 output classes
print("\n2. Creating proper EfficientNet-B7 architecture for binary classification...")

try:
    # Create EfficientNet-B7 with 2 output classes
    model = timm.create_model('tf_efficientnet_b7', pretrained=True, num_classes=2)
    
    # Initialize the classification layer with proper weights
    # This will give us a baseline - not trained on deepfakes but proper architecture
    print(f"   Created EfficientNet-B7 with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"   Output shape check: {model(torch.randn(1, 3, 600, 600)).shape}")
    
    # Save it
    save_path = weights_dir / "efficientnet_b7_binary.pth"
    torch.save({
        'state_dict': model.state_dict(),
        'num_classes': 2,
        'architecture': 'tf_efficientnet_b7'
    }, save_path)
    
    print(f"   ✅ Saved proper binary classifier to {save_path}")
    print(f"   ⚠️ Note: This is pretrained on ImageNet, not deepfakes")
    print(f"   For production, you would fine-tune on FaceForensics++ dataset")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("""
For accurate deepfake detection, you have 3 options:

1. USE HUGGINGFACE PIPELINE (Easiest):
   - Use 'prithivMLmods/deepfake-detector-model-v1' via transformers pipeline
   - This is already trained on deepfakes
   - Simply call: pipeline("image-classification", model="...")

2. DOWNLOAD DEEFAKEBENCH WEIGHTS (Best Accuracy):
   - Clone: https://github.com/SCLBD/DeepfakeBench
   - Download their pretrained EfficientNet-B7 from Google Drive
   - This is trained on FaceForensics++ dataset

3. USE PRETRAINED IMAGENET + SIMPLE HEURISTICS (Current):
   - Use the efficientnet_b7_binary.pth we just created
   - Will detect some patterns but not specifically trained
   - Better than nothing for MVP testing
""")
