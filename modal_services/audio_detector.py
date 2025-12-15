"""
Modal Audio Synthesis Detector using Wav2Vec2
Detects AI-generated/synthetic voice patterns in audio
"""
import modal
import io
import tempfile
from pathlib import Path

app = modal.App("deepfake-audio-detector")

# Container image with transformers and audio processing
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "transformers==4.35.0",
        "safetensors==0.4.1",
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
class AudioSynthesisDetector:
    @modal.enter()
    def load_model(self):
        """Load Wav2Vec2 audio deepfake detector"""
        from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
        import torch
        
        print("🔄 Loading Wav2Vec2 audio detector...")
        
        try:
            # Try loading the fine-tuned model from local weights
            # The Deepfake-audio-detection-V2 model is a fine-tuned Wav2Vec2
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "/weights",
                local_files_only=True
            ).cuda().eval()
            
            # Load processor (or use base if not available)
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(
                    "/weights",
                    local_files_only=True
                )
            except:
                print("⚠️ Using base Wav2Vec2 processor")
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            
            print("✅ Loaded fine-tuned Wav2Vec2 model")
        
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned model: {e}")
            print("Loading base Wav2Vec2 model instead")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "facebook/wav2vec2-base"
            ).cuda().eval()
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        print("✅ Wav2Vec2 audio detector loaded and ready")
    
    @modal.method()
    def detect_synthetic_audio(self, audio_bytes: bytes):
        """
        Analyze audio for AI-generated speech artifacts
        
        Args:
            audio_bytes: WAV file bytes (16kHz mono)
        
        Returns:
            dict: {
                "synthesis_score": float,  # 0=real, 1=synthetic
                "confidence": float,
                "has_audio": bool
            }
        """
        import torch
        import torchaudio
        import tempfile
        import os
        
        if not audio_bytes:
            return {
                "synthesis_score": 0.0,
                "confidence": 0.0,
                "has_audio": False,
                "error": "No audio provided"
            }
        
        print(f"🎵 Analyzing audio ({len(audio_bytes)} bytes)...")
        
        try:
            # Save audio bytes to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                tmp_audio.write(audio_bytes)
                audio_path = tmp_audio.name
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Cleanup temp file
            os.unlink(audio_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Process with Wav2Vec2 processor
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=160000,  # Max 10 seconds
                truncation=True
            )
            
            # Move to GPU
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Assuming class 0 = Real, class 1 = Fake (adjust based on model)
            if probs.shape[1] >= 2:
                fake_prob = probs[0, 1].item()
                confidence = max(probs[0].tolist())
            else:
                # Binary classification, single output
                fake_prob = probs[0, 0].item()
                confidence = fake_prob
            
            result = {
                "synthesis_score": fake_prob,
                "confidence": confidence,
                "has_audio": True
            }
            
            print(f"✅ Audio analysis complete: synthesis_score={fake_prob:.3f}")
            
            return result
        
        except Exception as e:
            print(f"⚠️ Audio processing failed: {e}")
            return {
                "synthesis_score": 0.0,
                "confidence": 0.0,
                "has_audio": False,
                "error": str(e)
            }


@app.local_entrypoint()
def test_audio_detector():
    """Test the audio detector locally"""
    import numpy as np
    import wave
    
    # Create a dummy WAV file for testing
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # A4 note
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    # Write to WAV bytes
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    test_audio = buffer.getvalue()
    
    detector = AudioSynthesisDetector()
    result = detector.detect_synthetic_audio.remote(test_audio)
    
    print(f"\n📊 Audio Detection Results:")
    print(f"  - Synthesis score: {result['synthesis_score']:.3f}")
    print(f"  - Confidence: {result['confidence']:.3f}")
    print(f"  - Has audio: {result['has_audio']}")
