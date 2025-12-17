"""
Test WhatsApp Modal Integration - All Three Detection Types
Tests video, image, and text detection through WhatsApp modal_service
"""

import sys
import os
import base64
sys.path.append(os.path.join(os.path.dirname(__file__), 'whatsapp'))

from whatsapp.modal_service import detect_video_multimodal, detect_image_ai, detect_text_ai, ModalDetectionError


def test_video_detection():
    """Test video detection with balanced 3-layer pipeline"""
    print("\n" + "="*60)
    print("Testing Video Detection (Balanced 3-Layer Pipeline)")
    print("="*60)
    
    # Use public Big Buck Bunny video
    test_url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
    
    try:
        result = detect_video_multimodal(
            video_url=test_url,
            enable_fail_fast=False
        )
        
        print(f"✅ Detection successful!")
        print(f"\n📊 Results:")
        print(f"   Final Verdict: {result.get('final_verdict')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Total Time: {result.get('total_time', 0):.2f}s")
        
        print(f"\n📋 Layer Breakdown:")
        for layer in result.get('layer_results', []):
            layer_name = layer.get('layer_name', 'Unknown')
            is_fake = layer.get('is_fake')
            verdict = 'FAKE' if is_fake else 'REAL' if is_fake is not None else 'N/A'
            confidence = layer.get('confidence', 0)
            time = layer.get('processing_time', 0)
            
            print(f"   {layer_name}:")
            print(f"      Verdict: {verdict}")
            print(f"      Confidence: {confidence:.1%}")
            print(f"      Time: {time:.2f}s")
        
        return True
        
    except ModalDetectionError as e:
        print(f"❌ Detection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_detection():
    """Test image detection with EfficientFormer"""
    print("\n" + "="*60)
    print("Testing Image Detection (EfficientFormer)")
    print("="*60)
    
    # Create a simple test image (1x1 red pixel PNG)
    # This is a minimal valid PNG file
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    test_image_bytes = base64.b64decode(test_image_base64)
    
    try:
        result = detect_image_ai(
            file_content=test_image_bytes,
            mime_type="image/png"
        )
        
        print(f"✅ Detection successful!")
        print(f"\n📊 Results:")
        print(f"   Top Prediction: {result.get('top_prediction')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Predictions: {result.get('predictions', [])}")
        
        return True
        
    except ModalDetectionError as e:
        print(f"❌ Detection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_detection():
    """Test text detection with ensemble AI detector"""
    print("\n" + "="*60)
    print("Testing Text Detection (Ensemble AI Detector)")
    print("="*60)
    
    test_text = "Hello world this is a test message to check if the AI text detector is working properly with the ensemble model."
    
    try:
        result = detect_text_ai(text=test_text)
        
        if result.get('success'):
            print(f"✅ Detection successful!")
            print(f"\n📊 Results:")
            res = result['result']
            print(f"   Prediction: {res.get('prediction')}")
            print(f"   Confidence: {res.get('confidence_percent')}")
            print(f"   Is AI: {res.get('is_ai')}")
            print(f"   Agreement: {res.get('agreement')}")
            print(f"   Domain: {res.get('domain')}")
            
            if result.get('breakdown'):
                print(f"\n📋 Detector Breakdown:")
                for detector, data in result['breakdown'].items():
                    print(f"   {detector.upper()}:")
                    print(f"      Prediction: {data.get('prediction')}")
                    print(f"      Confidence: {data.get('confidence', 0):.1%}")
            
            return True
        else:
            print(f"❌ Detection failed: {result}")
            return False
        
    except ModalDetectionError as e:
        print(f"❌ Detection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n🧪 WhatsApp Modal Integration Tests")
    print("Testing all three detection types...")
    
    results = []
    
    # Test all three detection types
    results.append(("Video Detection", test_video_detection()))
    results.append(("Image Detection", test_image_detection()))
    results.append(("Text Detection", test_text_detection()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
