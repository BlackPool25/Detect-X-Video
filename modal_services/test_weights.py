"""
Test script to verify model weights load correctly locally
"""
import torch
import timm
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification

weights_dir = Path(__file__).parent / "weights"

print("🔍 Testing Model Weights Loading...")
print("=" * 50)

# Test 1: EfficientNet-B7
print("\n1️⃣ Testing EfficientNet-B7...")
try:
    model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=2)
    checkpoint_path = weights_dir / "efficientnet_b7_deepfake.pt"
    
    if checkpoint_path.exists():
        print(f"   ✅ Weight file found: {checkpoint_path.stat().st_size / 1e6:.1f} MB")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"   ✅ Checkpoint loaded. Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("   ✅ Loaded from checkpoint['state_dict']")
            else:
                print(f"   📋 Checkpoint keys: {list(checkpoint.keys())}")
        else:
            print("   ✅ Checkpoint is the model itself")
        
        print(f"   ✅ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    else:
        print(f"   ❌ Weight file not found: {checkpoint_path}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Wav2Vec2
print("\n2️⃣ Testing Wav2Vec2...")
try:
    config_path = weights_dir / "config.json"
    model_path = weights_dir / "model.safetensors"
    
    if config_path.exists() and model_path.exists():
        print(f"   ✅ Config found: {config_path.stat().st_size} bytes")
        print(f"   ✅ Model found: {model_path.stat().st_size / 1e6:.1f} MB")
        
        # Try loading
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            str(weights_dir),
            local_files_only=True
        )
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    else:
        print(f"   ❌ Missing files in {weights_dir}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: RetinaFace
print("\n3️⃣ Testing RetinaFace...")
try:
    retinaface_path = weights_dir / "retinaface_resnet50.pth"
    
    if retinaface_path.exists():
        print(f"   ✅ Weight file found: {retinaface_path.stat().st_size / 1e6:.1f} MB")
        checkpoint = torch.load(retinaface_path, map_location='cpu')
        print(f"   ✅ Loaded. Type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"   📋 Keys: {list(checkpoint.keys())[:5]}...")
    else:
        print(f"   ❌ Weight file not found: {retinaface_path}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 50)
print("✅ Weight verification complete!")
