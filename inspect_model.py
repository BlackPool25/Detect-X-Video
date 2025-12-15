#!/usr/bin/env python3
"""
Inspect the EfficientNet model to understand its structure
"""
import torch
import sys

model_path = "modal_services/weights/efficientnet_b7_deepfake.pt"

print("Inspecting EfficientNet model...")
print("=" * 80)

try:
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"Is TorchScript: {isinstance(checkpoint, torch.jit.ScriptModule)}")
    
    if isinstance(checkpoint, torch.jit.ScriptModule):
        print("\nThis is a TorchScript model!")
        print("We need to use it directly, not load into timm")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 600, 600)
        with torch.no_grad():
            output = checkpoint(dummy_input)
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Output: {output}")
        
        # Check if it's sigmoid output (single value) or softmax (two values)
        if output.shape[1] == 1:
            print("\n✅ Model outputs single value (sigmoid) - binary classification")
            print("   Interpretation: value > 0.5 = fake, value < 0.5 = real")
        else:
            print(f"\n✅ Model outputs {output.shape[1]} values")
    
    elif isinstance(checkpoint, dict):
        print(f"\nKeys in checkpoint: {list(checkpoint.keys())}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Number of parameters: {len(state_dict)}")
            print(f"Sample keys: {list(state_dict.keys())[:5]}")
        
    else:
        print(f"\nUnexpected checkpoint type: {type(checkpoint)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
