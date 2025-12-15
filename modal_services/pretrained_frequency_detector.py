"""
Pretrained Frequency Detector Inference
Loads F3Net and SRM pretrained weights from DeepfakeBench
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import sys

# Add DeepfakeBench to path
sys.path.insert(0, '/home/lightdesk/Projects/AI-Video/DeepfakeBench/training')

from detectors.f3net_detector import F3netDetector
from detectors.srm_detector import SRMDetector


class PretrainedFrequencyDetector:
    """
    Wrapper for pretrained F3Net and SRM models
    """
    def __init__(self, device='cuda'):
        self.device = device
        
        # Configuration for F3Net
        self.f3net_config = {
            'backbone_name': 'xception',
            'backbone_config': {'mode': 'original', 'num_classes': 2, 'inc': 3, 'dropout': 0.5},
            'loss_func': 'cross_entropy',
            'pretrained': '/home/lightdesk/Projects/AI-Video/DeepfakeBench/training/pretrained/xception-b5690688.pth',
            'resolution': 256
        }
        
        # Configuration for SRM
        self.srm_config = {
            'backbone_name': 'xception',
            'backbone_config': {'mode': 'original', 'num_classes': 2, 'inc': 3, 'dropout': 0.5},
            'loss_func': 'cross_entropy',
            'pretrained': '/home/lightdesk/Projects/AI-Video/DeepfakeBench/training/pretrained/xception-b5690688.pth',
        }
        
        self.f3net = None
        self.srm = None
        self.models_loaded = False
    
    def load_models(self):
        """Load pretrained weights"""
        print("🔄 Loading pretrained frequency detection models...")
        
        # Check if Xception pretrained weights exist
        xception_path = Path(self.f3net_config['pretrained'])
        if not xception_path.exists():
            print(f"⚠️  Xception pretrained weights not found at {xception_path}")
            print("📥 Downloading Xception pretrained weights...")
            self._download_xception_weights()
        
        try:
            # Load F3Net
            f3net_weight_path = '/home/lightdesk/Projects/AI-Video/Weights/organized/f3net_best.pth'
            if Path(f3net_weight_path).exists():
                print(f"✅ Loading F3Net from {f3net_weight_path}")
                self.f3net = F3netDetector(self.f3net_config)
                
                checkpoint = torch.load(f3net_weight_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from DataParallel)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.f3net.load_state_dict(state_dict, strict=False)
                self.f3net = self.f3net.to(self.device)
                self.f3net.eval()
                print("✅ F3Net loaded successfully")
            else:
                print(f"⚠️  F3Net weights not found at {f3net_weight_path}")
        
        except Exception as e:
            print(f"⚠️  Failed to load F3Net: {e}")
            self.f3net = None
        
        try:
            # Load SRM
            srm_weight_path = '/home/lightdesk/Projects/AI-Video/Weights/organized/srm_best.pth'
            if Path(srm_weight_path).exists():
                print(f"✅ Loading SRM from {srm_weight_path}")
                self.srm = SRMDetector(self.srm_config)
                
                checkpoint = torch.load(srm_weight_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.srm.load_state_dict(state_dict, strict=False)
                self.srm = self.srm.to(self.device)
                self.srm.eval()
                print("✅ SRM loaded successfully")
            else:
                print(f"⚠️  SRM weights not found at {srm_weight_path}")
        
        except Exception as e:
            print(f"⚠️  Failed to load SRM: {e}")
            self.srm = None
        
        self.models_loaded = (self.f3net is not None or self.srm is not None)
        return self.models_loaded
    
    def _download_xception_weights(self):
        """Download Xception ImageNet weights if missing"""
        import urllib.request
        
        xception_url = "http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth"
        save_path = '/home/lightdesk/Projects/AI-Video/DeepfakeBench/training/pretrained/xception-b5690688.pth'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"Downloading from {xception_url}...")
            urllib.request.urlretrieve(xception_url, save_path)
            print(f"✅ Downloaded to {save_path}")
        except Exception as e:
            print(f"❌ Failed to download Xception weights: {e}")
    
    def predict_frame(self, frame):
        """
        Predict on a single frame
        
        Args:
            frame: numpy array (H, W, 3) in BGR or RGB format
        
        Returns:
            dict with predictions from each model
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        frame_resized = cv2.resize(frame_rgb, (256, 256))
        
        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        results = {}
        
        with torch.no_grad():
            # F3Net prediction
            if self.f3net is not None:
                try:
                    data_dict = {'image': frame_tensor, 'label': torch.tensor([0])}
                    pred_dict = self.f3net(data_dict, inference=True)
                    f3net_prob = pred_dict['prob'].item()
                    results['f3net_score'] = f3net_prob
                except Exception as e:
                    print(f"F3Net prediction error: {e}")
                    results['f3net_score'] = 0.5
            
            # SRM prediction
            if self.srm is not None:
                try:
                    data_dict = {'image': frame_tensor, 'label': torch.tensor([0])}
                    pred_dict = self.srm(data_dict, inference=True)
                    srm_prob = pred_dict['prob'].item()
                    results['srm_score'] = srm_prob
                except Exception as e:
                    print(f"SRM prediction error: {e}")
                    results['srm_score'] = 0.5
        
        # Ensemble prediction (average of available models)
        available_scores = [v for k, v in results.items() if k.endswith('_score')]
        results['ensemble_score'] = np.mean(available_scores) if available_scores else 0.5
        
        return results
    
    def predict_video(self, video_path, max_frames=32):
        """
        Predict on video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to analyze
        
        Returns:
            dict with aggregated predictions
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        f3net_scores = []
        srm_scores = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            results = self.predict_frame(frame)
            
            if 'f3net_score' in results:
                f3net_scores.append(results['f3net_score'])
            if 'srm_score' in results:
                srm_scores.append(results['srm_score'])
        
        cap.release()
        
        # Aggregate results
        final_results = {
            'f3net_score': np.mean(f3net_scores) if f3net_scores else None,
            'f3net_std': np.std(f3net_scores) if f3net_scores else None,
            'srm_score': np.mean(srm_scores) if srm_scores else None,
            'srm_std': np.std(srm_scores) if srm_scores else None,
            'frames_analyzed': len(frame_indices)
        }
        
        # Ensemble prediction
        available_scores = [v for k, v in final_results.items() if k.endswith('_score') and v is not None]
        final_results['ensemble_score'] = np.mean(available_scores) if available_scores else 0.5
        
        # Classification (threshold at 0.5)
        final_results['prediction'] = 'FAKE' if final_results['ensemble_score'] > 0.5 else 'REAL'
        final_results['confidence'] = abs(final_results['ensemble_score'] - 0.5) * 2
        
        return final_results


if __name__ == "__main__":
    # Test the detector
    print("=" * 60)
    print("Testing Pretrained Frequency Detector")
    print("=" * 60)
    
    detector = PretrainedFrequencyDetector(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    if detector.load_models():
        print("\n✅ Models loaded successfully!")
        
        # Test on a sample video
        test_video = '/home/lightdesk/Projects/AI-Video/Test-Video/Real/id0_id1_0000.mp4'
        
        if Path(test_video).exists():
            print(f"\n🎬 Testing on video: {test_video}")
            results = detector.predict_video(test_video, max_frames=16)
            
            print("\n📊 Results:")
            print(f"  Prediction: {results['prediction']}")
            print(f"  Confidence: {results['confidence']:.2%}")
            print(f"  Ensemble Score: {results['ensemble_score']:.4f}")
            
            if results['f3net_score'] is not None:
                print(f"  F3Net Score: {results['f3net_score']:.4f} (±{results['f3net_std']:.4f})")
            
            if results['srm_score'] is not None:
                print(f"  SRM Score: {results['srm_score']:.4f} (±{results['srm_std']:.4f})")
            
            print(f"  Frames Analyzed: {results['frames_analyzed']}")
        else:
            print(f"\n⚠️  Test video not found: {test_video}")
    else:
        print("\n❌ Failed to load models")
