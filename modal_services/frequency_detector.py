"""
Frequency Domain Deepfake Detector
Combines SRM filters and F3-Net DCT analysis for high-quality deepfake detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import cv2


class SRMConv2d_simple(nn.Module):
    """
    Spatial Rich Model (SRM) Filter Layer
    Extracts noise residuals to reveal GAN/Diffusion artifacts
    """
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        """
        x: imgs (Batch, C, H, W) - RGB images normalized to [-1, 1]
        Returns: (Batch, 3, H, W) - Noise residual maps
        """
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)
        return out

    def _build_kernel(self, inc):
        # Filter 1: KB (Basic high-pass)
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        
        # Filter 2: KV (Edge detection)
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        
        # Filter 3: Horizontal 2nd derivative
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        
        filters = [[filter1], [filter2], [filter3]]
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)
        return filters


class Filter(nn.Module):
    """Frequency band filter for F3-Net"""
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, size)), 
            requires_grad=False
        )
        
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), 
                requires_grad=False
            )

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class FAD_Head(nn.Module):
    """
    Frequency-Aware Decomposition (FAD) Head from F3-Net
    Uses DCT to analyze frequency domain artifacts
    """
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # Init DCT matrix
        self._DCT_all = nn.Parameter(
            torch.tensor(DCT_mat(size)).float(), 
            requires_grad=False
        )
        self._DCT_all_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), 
            requires_grad=False
        )

        # Define frequency band filters
        # Low: 0 - 1/2.82, Middle: 1/2.82 - 1/2, High: 1/2 - 2, All: 0 - 2
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        """
        x: (Batch, 3, H, W) - RGB image
        Returns: (Batch, 12, H, W) - 4 frequency bands concatenated
        """
        # Apply DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T

        # Extract 4 frequency bands
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)
            y = self._DCT_all_T @ x_pass @ self._DCT_all
            y_list.append(y)
        
        out = torch.cat(y_list, dim=1)  # Concatenate along channel dimension
        return out


class HybridFrequencyDetector(nn.Module):
    """
    Hybrid detector combining SRM noise analysis and F3-Net frequency analysis
    """
    def __init__(self, input_size=256):
        super(HybridFrequencyDetector, self).__init__()
        
        # SRM for noise residual extraction
        self.srm_conv = SRMConv2d_simple(inc=3, learnable=False)
        
        # F3-Net FAD head for frequency decomposition
        self.fad_head = FAD_Head(input_size)
        
        # Feature extraction from SRM output (3 channels)
        self.srm_features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Feature extraction from FAD output (12 channels)
        self.fad_features = nn.Sequential(
            nn.Conv2d(12, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (Batch, 3, H, W) - RGB images in range [0, 1]
        Returns: (Batch,) - Fake probability [0, 1]
        """
        # Normalize to [-1, 1] for SRM
        x_norm = x * 2.0 - 1.0
        
        # Extract SRM noise residuals
        srm_out = self.srm_conv(x_norm)
        srm_feat = self.srm_features(srm_out).flatten(1)
        
        # Extract FAD frequency features
        fad_out = self.fad_head(x)
        fad_feat = self.fad_features(fad_out).flatten(1)
        
        # Concatenate features
        combined = torch.cat([srm_feat, fad_feat], dim=1)
        
        # Classify
        prob = self.classifier(combined).squeeze(1)
        
        return prob, {
            'srm_residual': srm_out,
            'fad_frequency': fad_out,
            'srm_features': srm_feat,
            'fad_features': fad_feat
        }


# Utility functions
def DCT_mat(size):
    """Create DCT transformation matrix"""
    m = [
        [
            (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * 
            np.cos((j + 0.5) * np.pi * i / size) 
            for j in range(size)
        ] 
        for i in range(size)
    ]
    return m


def generate_filter(start, end, size):
    """Generate frequency band filter"""
    return [
        [0. if i + j > end or i + j < start else 1. for j in range(size)] 
        for i in range(size)
    ]


def norm_sigma(x):
    """Normalize sigma for learnable filters"""
    return 2. * torch.sigmoid(x) - 1.


def extract_frequency_score(video_path: str, model: HybridFrequencyDetector, device='cuda') -> Dict:
    """
    Extract frequency-based deepfake score from video
    
    Args:
        video_path: Path to video file
        model: HybridFrequencyDetector model
        device: Device to run on
    
    Returns:
        Dict with frequency scores and diagnostics
    """
    cap = cv2.VideoCapture(video_path)
    
    frame_scores = []
    srm_scores = []
    fad_scores = []
    
    model.eval()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)
            
            # Get predictions
            prob, debug_info = model(frame_tensor)
            
            frame_scores.append(prob.item())
            srm_scores.append(debug_info['srm_features'].abs().mean().item())
            fad_scores.append(debug_info['fad_features'].abs().mean().item())
    
    cap.release()
    
    if len(frame_scores) == 0:
        return {
            'frequency_score': 0.5,
            'confidence': 0.0,
            'srm_signal': 0.0,
            'fad_signal': 0.0,
            'analysis': 'No frames extracted'
        }
    
    # Aggregate scores
    avg_score = np.mean(frame_scores)
    std_score = np.std(frame_scores)
    avg_srm = np.mean(srm_scores)
    avg_fad = np.mean(fad_scores)
    
    return {
        'frequency_score': float(avg_score),
        'confidence': float(1.0 - std_score),  # Lower variance = higher confidence
        'srm_signal': float(avg_srm),
        'fad_signal': float(avg_fad),
        'frame_count': len(frame_scores),
        'score_std': float(std_score),
        'analysis': 'Frequency domain analysis complete'
    }


if __name__ == "__main__":
    # Test the model
    model = HybridFrequencyDetector(input_size=256)
    
    # Test with random input
    test_input = torch.rand(2, 3, 256, 256)
    prob, debug = model(test_input)
    
    print(f"Model output shape: {prob.shape}")
    print(f"Sample probabilities: {prob}")
    print(f"SRM features shape: {debug['srm_features'].shape}")
    print(f"FAD features shape: {debug['fad_features'].shape}")
    print("\n✅ Frequency detector initialized successfully!")
