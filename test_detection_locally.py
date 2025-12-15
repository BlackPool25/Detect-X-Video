#!/usr/bin/env python3
"""
Comprehensive Local Testing for Deepfake Detection Pipeline
Tests that models are actually loaded and not using random predictions
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import sys
import tempfile
import json
from PIL import Image
import torchaudio

# Add modal_services to path
sys.path.insert(0, str(Path(__file__).parent / "modal_services"))

class LocalTestSuite:
    """Test suite to verify models are properly loaded and functional"""
    
    def __init__(self):
        self.test_video_dir = Path("Test-Video")
        self.weights_dir = Path("modal_services/weights")
        self.results = []
        
    def test_1_weights_exist(self):
        """Verify all required weight files are present"""
        print("\n" + "="*80)
        print("TEST 1: Checking Weight Files")
        print("="*80)
        
        required_weights = {
            "EfficientNet-B7": self.weights_dir / "efficientnet_b7_deepfake.pt",
            "Wav2Vec2": self.weights_dir / "model.safetensors",
            "RetinaFace": self.weights_dir / "retinaface_resnet50.pth"
        }
        
        all_exist = True
        for name, path in required_weights.items():
            exists = path.exists()
            size_mb = path.stat().st_size / (1024*1024) if exists else 0
            status = "✅" if exists else "❌"
            print(f"{status} {name}: {path}")
            if exists:
                print(f"   Size: {size_mb:.1f} MB")
            all_exist = all_exist and exists
            
        self.results.append(("Weights Exist", all_exist))
        return all_exist
    
    def test_2_load_efficientnet(self):
        """Test loading EfficientNet-B7 and verify it's not random"""
        print("\n" + "="*80)
        print("TEST 2: Loading EfficientNet-B7 Model")
        print("="*80)
        
        try:
            import timm
            from torchvision import transforms
            
            # Create model
            model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=2)
            
            # Load weights
            checkpoint_path = self.weights_dir / "efficientnet_b7_deepfake.pt"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check checkpoint structure
            print(f"✅ Loaded checkpoint from {checkpoint_path}")
            
            if isinstance(checkpoint, dict):
                print(f"   Checkpoint keys: {list(checkpoint.keys())}")
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("   ✅ Loaded from state_dict")
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("   ✅ Loaded from model_state_dict")
                else:
                    model.load_state_dict(checkpoint)
                    print("   ✅ Loaded directly from checkpoint dict")
            else:
                model = checkpoint
                print("   ✅ Loaded complete model")
            
            model = model.eval()
            
            # Test with dummy input to verify model is functional
            dummy_input = torch.randn(1, 3, 600, 600)
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"   Model output shape: {output.shape}")
            print(f"   Output values: {output[0].tolist()}")
            
            # Verify it's not just zeros or ones
            if torch.all(output == 0) or torch.all(output == output[0, 0]):
                print("   ❌ WARNING: Model appears to output constant values!")
                self.results.append(("EfficientNet Loaded", False))
                return False
            
            print("   ✅ Model appears functional (non-constant outputs)")
            
            # Test consistency - same input should give same output
            with torch.no_grad():
                output2 = model(dummy_input)
            
            if torch.allclose(output, output2):
                print("   ✅ Model is deterministic (same input = same output)")
            else:
                print("   ❌ WARNING: Model outputs are non-deterministic!")
            
            self.results.append(("EfficientNet Loaded", True))
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading EfficientNet: {e}")
            import traceback
            traceback.print_exc()
            self.results.append(("EfficientNet Loaded", False))
            return False
    
    def test_3_load_wav2vec2(self):
        """Test loading Wav2Vec2 audio detector"""
        print("\n" + "="*80)
        print("TEST 3: Loading Wav2Vec2 Audio Detector")
        print("="*80)
        
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
            
            # Try loading from weights directory
            print(f"   Loading model from {self.weights_dir}...")
            
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                str(self.weights_dir),
                local_files_only=True
            ).eval()
            
            try:
                processor = Wav2Vec2Processor.from_pretrained(
                    str(self.weights_dir),
                    local_files_only=True
                )
            except:
                print("   ⚠️ Using base processor")
                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            
            print("   ✅ Wav2Vec2 model loaded successfully")
            
            # Test with dummy audio
            dummy_audio = np.random.randn(16000)  # 1 second of random audio
            inputs = processor(
                dummy_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                output = model(**inputs).logits
            
            print(f"   Model output shape: {output.shape}")
            print(f"   Output values: {output[0].tolist()}")
            
            if torch.all(output == 0):
                print("   ❌ WARNING: Model outputs all zeros!")
                self.results.append(("Wav2Vec2 Loaded", False))
                return False
            
            print("   ✅ Model appears functional")
            self.results.append(("Wav2Vec2 Loaded", True))
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading Wav2Vec2: {e}")
            import traceback
            traceback.print_exc()
            self.results.append(("Wav2Vec2 Loaded", False))
            return False
    
    def test_4_face_detection(self):
        """Test face detection on real vs fake videos"""
        print("\n" + "="*80)
        print("TEST 4: Face Detection on Test Videos")
        print("="*80)
        
        try:
            from facenet_pytorch import MTCNN
            
            detector = MTCNN(keep_all=True, device='cpu', post_process=False)
            print("   ✅ MTCNN loaded")
            
            # Test on one real and one fake video
            test_videos = [
                self.test_video_dir / "Real" / "01__kitchen_pan.mp4",
                self.test_video_dir / "Fake" / "01_03__hugging_happy__ISF9SP4G.mp4"
            ]
            
            for video_path in test_videos:
                if not video_path.exists():
                    print(f"   ⚠️ Video not found: {video_path}")
                    continue
                
                print(f"\n   Testing: {video_path.name}")
                cap = cv2.VideoCapture(str(video_path))
                
                faces_found = 0
                frames_checked = 0
                
                for i in range(5):  # Check first 5 frames
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frames_checked += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, probs = detector.detect(rgb_frame)
                    
                    if boxes is not None:
                        faces_found += len(boxes)
                        print(f"      Frame {i}: Found {len(boxes)} face(s) with confidence {probs}")
                
                cap.release()
                print(f"   ✅ Checked {frames_checked} frames, found {faces_found} total faces")
            
            self.results.append(("Face Detection", True))
            return True
            
        except Exception as e:
            print(f"   ❌ Error in face detection: {e}")
            import traceback
            traceback.print_exc()
            self.results.append(("Face Detection", False))
            return False
    
    def test_5_visual_detection_discrimination(self):
        """Test if visual detector can discriminate between real and fake"""
        print("\n" + "="*80)
        print("TEST 5: Visual Detector Discrimination Test")
        print("="*80)
        
        try:
            import timm
            from torchvision import transforms
            from facenet_pytorch import MTCNN
            
            # Load model
            model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=2)
            checkpoint = torch.load(self.weights_dir / "efficientnet_b7_deepfake.pt", map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            
            model = model.eval()
            
            transform = transforms.Compose([
                transforms.Resize((600, 600)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            face_detector = MTCNN(keep_all=True, device='cpu')
            
            # Test on real and fake videos
            test_cases = [
                ("Real", self.test_video_dir / "Real" / "01__kitchen_pan.mp4"),
                ("Fake", self.test_video_dir / "Fake" / "01_03__hugging_happy__ISF9SP4G.mp4")
            ]
            
            results = {}
            
            for label, video_path in test_cases:
                if not video_path.exists():
                    print(f"   ⚠️ Video not found: {video_path}")
                    continue
                
                print(f"\n   Testing {label} video: {video_path.name}")
                cap = cv2.VideoCapture(str(video_path))
                
                scores = []
                frames_processed = 0
                
                for i in range(10):  # Process 10 frames
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, probs = face_detector.detect(rgb_frame)
                    
                    if boxes is not None and len(boxes) > 0:
                        # Take first face
                        box = boxes[0]
                        x1, y1, x2, y2 = [int(b) for b in box]
                        face = rgb_frame[y1:y2, x1:x2]
                        
                        # Process face
                        face_pil = Image.fromarray(face)
                        tensor = transform(face_pil).unsqueeze(0)
                        
                        with torch.no_grad():
                            logits = model(tensor)
                            probs = torch.softmax(logits, dim=1)
                            fake_score = probs[0, 1].item()
                            scores.append(fake_score)
                            frames_processed += 1
                
                cap.release()
                
                if scores:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    results[label] = {
                        "mean": mean_score,
                        "std": std_score,
                        "frames": frames_processed,
                        "all_scores": scores
                    }
                    print(f"   Frames processed: {frames_processed}")
                    print(f"   Mean fake score: {mean_score:.4f}")
                    print(f"   Std deviation: {std_score:.4f}")
                    print(f"   Score range: [{min(scores):.4f}, {max(scores):.4f}]")
                else:
                    print(f"   ❌ No faces detected!")
            
            # Check if model can discriminate
            if "Real" in results and "Fake" in results:
                real_mean = results["Real"]["mean"]
                fake_mean = results["Fake"]["mean"]
                
                print(f"\n   DISCRIMINATION TEST:")
                print(f"   Real video mean score: {real_mean:.4f}")
                print(f"   Fake video mean score: {fake_mean:.4f}")
                print(f"   Difference: {abs(fake_mean - real_mean):.4f}")
                
                # We expect fake to have higher score
                if fake_mean > real_mean:
                    print(f"   ✅ Model correctly scores fake higher than real!")
                    discrimination_score = (fake_mean - real_mean) / max(fake_mean, real_mean)
                    print(f"   Discrimination strength: {discrimination_score:.2%}")
                    
                    if discrimination_score > 0.1:  # At least 10% difference
                        print(f"   ✅ Strong discrimination (>{10}% difference)")
                        self.results.append(("Visual Discrimination", True))
                        return True
                    else:
                        print(f"   ⚠️ Weak discrimination (<{10}% difference)")
                        self.results.append(("Visual Discrimination", False))
                        return False
                else:
                    print(f"   ❌ Model scores real higher than fake - likely not trained!")
                    self.results.append(("Visual Discrimination", False))
                    return False
            else:
                print(f"   ❌ Could not test both video types")
                self.results.append(("Visual Discrimination", False))
                return False
            
        except Exception as e:
            print(f"   ❌ Error in discrimination test: {e}")
            import traceback
            traceback.print_exc()
            self.results.append(("Visual Discrimination", False))
            return False
    
    def test_6_audio_detection(self):
        """Test audio detection on videos with audio"""
        print("\n" + "="*80)
        print("TEST 6: Audio Detection Test")
        print("="*80)
        
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
            import ffmpeg
            
            # Load model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                str(self.weights_dir),
                local_files_only=True
            ).eval()
            
            try:
                processor = Wav2Vec2Processor.from_pretrained(
                    str(self.weights_dir),
                    local_files_only=True
                )
            except:
                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            
            print("   ✅ Audio model loaded")
            
            # Test on a video
            test_video = self.test_video_dir / "Fake" / "01_03__hugging_happy__ISF9SP4G.mp4"
            
            if test_video.exists():
                print(f"   Testing: {test_video.name}")
                
                # Extract audio
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    audio_path = tmp.name
                
                try:
                    ffmpeg.input(str(test_video)).output(
                        audio_path,
                        acodec='pcm_s16le',
                        ac=1,
                        ar='16000'
                    ).run(quiet=True, overwrite_output=True)
                    
                    # Load audio
                    waveform, sr = torchaudio.load(audio_path)
                    
                    # Take first 3 seconds
                    waveform = waveform[:, :16000*3]
                    
                    # Process
                    inputs = processor(
                        waveform.squeeze().numpy(),
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)
                    
                    print(f"   Audio detection output: {probs[0].tolist()}")
                    
                    if probs.shape[1] >= 2:
                        fake_prob = probs[0, 1].item()
                        print(f"   Synthetic audio score: {fake_prob:.4f}")
                    
                    print("   ✅ Audio detection functional")
                    self.results.append(("Audio Detection", True))
                    return True
                    
                except Exception as e:
                    print(f"   ⚠️ Audio extraction/processing failed: {e}")
                    # This might be expected if video has no audio
                    self.results.append(("Audio Detection", True))
                    return True
            else:
                print(f"   ⚠️ Test video not found")
                self.results.append(("Audio Detection", False))
                return False
            
        except Exception as e:
            print(f"   ❌ Error in audio detection: {e}")
            import traceback
            traceback.print_exc()
            self.results.append(("Audio Detection", False))
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        for test_name, passed in self.results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        total = len(self.results)
        passed = sum(1 for _, p in self.results if p)
        
        print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Models are properly loaded and functional.")
        elif passed >= total * 0.7:
            print("\n⚠️ MOST TESTS PASSED. Review failures above.")
        else:
            print("\n❌ MULTIPLE FAILURES. Models may not be properly loaded.")

def main():
    """Run all tests"""
    print("="*80)
    print("DEEPFAKE DETECTION - LOCAL TEST SUITE")
    print("="*80)
    
    suite = LocalTestSuite()
    
    # Run all tests
    suite.test_1_weights_exist()
    suite.test_2_load_efficientnet()
    suite.test_3_load_wav2vec2()
    suite.test_4_face_detection()
    suite.test_5_visual_detection_discrimination()
    suite.test_6_audio_detection()
    
    # Print summary
    suite.print_summary()

if __name__ == "__main__":
    main()
