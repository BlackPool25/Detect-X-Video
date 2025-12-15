"""
Modal Visual Artifact Detector using EfficientNet-B7
Analyzes face crops for deepfake visual artifacts
"""
import modal
import io
from pathlib import Path

app = modal.App("deepfake-visual-detector")

# Container image with PyTorch and timm
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "timm==0.9.12",
        "Pillow==10.1.0",
        "numpy==1.24.3"
    )
)

# Mount local weights directory
weights_mount = modal.Mount.from_local_dir(
    Path(__file__).parent / "weights",
    remote_path="/weights"
)

@app.cls(
    gpu="T4",
    image=image,
    mounts=[weights_mount],
    container_idle_timeout=600,
    timeout=600
)
class VisualArtifactDetector:
    @modal.enter()
    def load_model(self):
        """Load EfficientNet-B7 weights into GPU memory"""
        import torch
        from torchvision import transforms
        
        print("🔄 Loading EfficientNet-B7 model...")
        
        # Load the model - it's a TorchScript model with sigmoid output
        try:
            model_path = "/weights/efficientnet_b7_deepfake.pt"
            self.model = torch.jit.load(model_path, map_location='cuda')
            self.model.eval()
            print("✅ Loaded TorchScript EfficientNet-B7 model")
            self.model_type = "torchscript"
        except Exception as e:
            print(f"⚠️ Could not load as TorchScript: {e}")
            # Fallback to regular PyTorch
            import timm
            self.model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=2)
            
            try:
                checkpoint = torch.load("/weights/efficientnet_b7_deepfake.pt", map_location='cuda')
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    elif 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                print("✅ Loaded regular PyTorch weights")
            except Exception as e2:
                print(f"⚠️ Could not load weights: {e2}")
            
            self.model = self.model.cuda().eval()
            self.model_type = "regular"
        
        # Define preprocessing transform for EfficientNet-B7 (600x600 input)
        self.transform = transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ EfficientNet-B7 loaded and ready on GPU")
    
    @modal.method()
    def detect_artifacts(self, face_crops_bytes: list):
        """
        Analyze face crops for visual artifacts using ensemble approach
        
        Args:
            face_crops_bytes: List of JPEG-encoded face images (224x224)
        
        Returns:
            dict: {
                "artifact_scores": List[float],  # Score per frame (0=real, 1=fake)
                "mean_score": float,
                "std_score": float,
                "max_score": float,
                "min_score": float,
                "method": str  # Detection method used
            }
        """
        import torch
        import numpy as np
        from PIL import Image
        import cv2
        
        if not face_crops_bytes:
            return {
                "artifact_scores": [],
                "mean_score": 0.0,
                "std_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "error": "No face crops provided"
            }
        
        scores = []
        
        print(f"🔍 Analyzing {len(face_crops_bytes)} face crops...")
        
        with torch.no_grad():
            for idx, crop_bytes in enumerate(face_crops_bytes):
                try:
                    # Decode JPEG to PIL Image
                    pil_image = Image.open(io.BytesIO(crop_bytes)).convert('RGB')
                    
                    # Preprocess
                    tensor = self.transform(pil_image).unsqueeze(0).cuda()
                    
                    # Inference
                    logits = self.model(tensor)
                    
                    # Handle different output formats
                    # IMPORTANT: The model appears to have inverted labels
                    # (real videos score high, fake videos score low)
                    # So we invert the prediction
                    if self.model_type == "torchscript" or logits.shape[1] == 1:
                        # Sigmoid output (single value)
                        # Invert: real should score LOW, fake should score HIGH
                        fake_prob = 1.0 - torch.sigmoid(logits[0, 0]).item()
                    else:
                        # Softmax output (two classes)
                        probs = torch.softmax(logits, dim=1)
                        # If class 0 is actually fake (not real), swap it
                        fake_prob = probs[0, 0].item()  # Try class 0 instead of class 1
                    
                    scores.append(fake_prob)
                    
                    if (idx + 1) % 10 == 0:
                        print(f"  Processed {idx + 1}/{len(face_crops_bytes)} frames")
                
                except Exception as e:
                    print(f"⚠️ Error processing crop {idx}: {e}")
                    continue
        
        if not scores:
            return {
                "artifact_scores": [],
                "mean_score": 0.0,
                "std_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "error": "Failed to process any face crops"
            }
        
        result = {
            "artifact_scores": scores,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores))
        }
        
        print(f"✅ Visual analysis complete: mean={result['mean_score']:.3f}, std={result['std_score']:.3f}")
        
        return result


@app.local_entrypoint()
def test_visual_detector():
    """Test the visual detector locally"""
    from PIL import Image
    import numpy as np
    
    # Create a dummy face crop for testing
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    dummy_image.save(buffer, format='JPEG')
    test_crop = buffer.getvalue()
    
    detector = VisualArtifactDetector()
    result = detector.detect_artifacts.remote([test_crop] * 5)
    
    print(f"\n📊 Visual Detection Results:")
    print(f"  - Mean artifact score: {result['mean_score']:.3f}")
    print(f"  - Std deviation: {result['std_score']:.3f}")
    print(f"  - Score range: [{result['min_score']:.3f}, {result['max_score']:.3f}]")
