"""
Robust Visual Artifact Detector using Ensemble Methods
Combines frequency analysis, texture analysis, and optional model-based detection
"""
import modal
import io
from pathlib import Path
import numpy as np

app = modal.App("deepfake-visual-detector-robust")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "timm==0.9.12",
        "Pillow==10.1.0",
        "numpy==1.24.3",
        "opencv-python-headless==4.8.1.78",
        "scipy==1.11.4"
    )
)

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
class RobustVisualDetector:
    @modal.enter()
    def load_model(self):
        """Load model and initialize ensemble components"""
        import torch
        import timm
        from torchvision import transforms
        
        print("🔄 Initializing Robust Visual Detector...")
        
        # Try to load fine-tuned model, but don't fail if it doesn't work
        self.model = None
        self.use_model = False
        
        try:
            self.model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=2)
            checkpoint = torch.load("/weights/efficientnet_b7_binary.pth", map_location='cuda')
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint
            
            self.model = self.model.cuda().eval()
            self.use_model = True
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            print("   Continuing with heuristic-only detection")
        
        self.transform = transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Robust Visual Detector ready")
    
    def _analyze_frequency_domain(self, img_array):
        """Detect GAN/diffusion artifacts in frequency spectrum"""
        import cv2
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = 20 * np.log(np.abs(f_shift) + 1)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_dist = np.sqrt(center_h**2 + center_w**2)
        
        # Analyze different frequency bands
        high_freq_mask = distances > (max_dist * 0.7)
        high_freq_energy = np.mean(magnitude[high_freq_mask]) if np.any(high_freq_mask) else 0
        
        mid_freq_mask = (distances >= (max_dist * 0.3)) & (distances <= (max_dist * 0.7))
        mid_freq_energy = np.mean(magnitude[mid_freq_mask]) if np.any(mid_freq_mask) else 0
        
        low_freq_mask = distances < (max_dist * 0.3)
        low_freq_energy = np.mean(magnitude[low_freq_mask]) if np.any(low_freq_mask) else 0
        
        total = high_freq_energy + mid_freq_energy + low_freq_energy
        if total > 0:
            hf_ratio = high_freq_energy / total
            mf_ratio = mid_freq_energy / total
        else:
            hf_ratio = 0.2
            mf_ratio = 0.4
        
        # Natural images: HF ~0.15-0.25, MF ~0.35-0.45
        hf_deviation = abs(hf_ratio - 0.20)
        mf_deviation = abs(mf_ratio - 0.40)
        
        fake_score = min((hf_deviation + mf_deviation) * 2.0, 1.0)
        
        return {
            'score': fake_score,
            'hf_ratio': hf_ratio,
            'mf_ratio': mf_ratio
        }
    
    def _analyze_texture(self, img_array):
        """Detect unnaturally smooth or noisy skin texture"""
        import cv2
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Texture variance in luminance
        laplacian = cv2.Laplacian(l_channel, cv2.CV_64F)
        texture_var = np.var(laplacian)
        
        # Color variation
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)
        
        # Scoring
        if texture_var < 30:
            smoothness_score = 0.8  # Too smooth = likely AI
        elif texture_var > 300:
            smoothness_score = 0.6  # Too noisy
        else:
            smoothness_score = 0.2
        
        if a_std < 3 or b_std < 3:
            color_score = 0.7  # Unnaturally uniform
        else:
            color_score = 0.2
        
        fake_score = (smoothness_score + color_score) / 2.0
        
        return {
            'score': fake_score,
            'texture_var': texture_var,
            'color_std': (a_std, b_std)
        }
    
    def _analyze_eyes(self, img_array):
        """Detect eye artifacts (glossiness, specular highlights)"""
        import cv2
        
        # Simple eye region detection (top 40% of face)
        h = img_array.shape[0]
        eye_region = img_array[:int(h*0.4), :]
        
        # Convert to grayscale
        gray_eyes = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
        
        # Check for unnatural brightness patterns
        max_brightness = np.max(gray_eyes)
        bright_pixels = np.sum(gray_eyes > 250)
        total_pixels = gray_eyes.size
        
        # AI eyes often lack proper specular highlights or have too many
        bright_ratio = bright_pixels / total_pixels
        
        if bright_ratio < 0.001:  # No highlights
            score = 0.6
        elif bright_ratio > 0.05:  # Too many bright spots
            score = 0.7
        else:
            score = 0.2
        
        return {
            'score': score,
            'bright_ratio': bright_ratio
        }
    
    @modal.method()
    def detect_artifacts(self, face_crops_bytes: list):
        """
        Analyze face crops using robust ensemble methods
        
        Args:
            face_crops_bytes: List of JPEG-encoded face images
        
        Returns:
            dict with scores and detailed breakdown
        """
        import torch
        from PIL import Image
        
        if not face_crops_bytes:
            return {
                "artifact_scores": [],
                "mean_score": 0.5,
                "std_score": 0.0,
                "max_score": 0.5,
                "min_score": 0.5,
                "method": "none",
                "breakdown": {}
            }
        
        scores = []
        freq_scores = []
        texture_scores = []
        eye_scores = []
        model_scores = []
        
        for idx, crop_bytes in enumerate(face_crops_bytes):
            try:
                # Decode image
                pil_img = Image.open(io.BytesIO(crop_bytes)).convert('RGB')
                img_array = np.array(pil_img)
                
                # Run ensemble methods
                freq_result = self._analyze_frequency_domain(img_array)
                texture_result = self._analyze_texture(img_array)
                eye_result = self._analyze_eyes(img_array)
                
                freq_scores.append(freq_result['score'])
                texture_scores.append(texture_result['score'])
                eye_scores.append(eye_result['score'])
                
                # Model-based (if available)
                model_score = 0.5
                if self.use_model and self.model is not None:
                    try:
                        with torch.no_grad():
                            tensor = self.transform(pil_img).unsqueeze(0).cuda()
                            logits = self.model(tensor)
                            
                            if logits.shape[1] >= 2:
                                probs = torch.softmax(logits, dim=1)
                                model_score = probs[0, 1].item()
                    except:
                        pass
                
                model_scores.append(model_score)
                
                # Weighted ensemble: Frequency 25%, Texture 30%, Eyes 20%, Model 25%
                ensemble_score = (
                    0.25 * freq_result['score'] +
                    0.30 * texture_result['score'] +
                    0.20 * eye_result['score'] +
                    0.25 * model_score
                )
                
                scores.append(ensemble_score)
                
            except Exception as e:
                print(f"⚠️ Error processing crop {idx}: {e}")
                scores.append(0.5)
        
        return {
            "artifact_scores": scores,
            "mean_score": float(np.mean(scores)) if scores else 0.5,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.5,
            "min_score": float(np.min(scores)) if scores else 0.5,
            "method": "robust_ensemble_v1",
            "breakdown": {
                "frequency_mean": float(np.mean(freq_scores)) if freq_scores else 0.5,
                "texture_mean": float(np.mean(texture_scores)) if texture_scores else 0.5,
                "eye_mean": float(np.mean(eye_scores)) if eye_scores else 0.5,
                "model_mean": float(np.mean(model_scores)) if model_scores else 0.5
            }
        }
