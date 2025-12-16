#!/usr/bin/env python3
"""
Comprehensive Test Suite for 4-Layer Deepfake Detection Pipeline
Tests systematically on real and fake videos to measure accuracy
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
import traceback

if not Path("pipeline_production.py").exists():
    print("Error: Please run this script from the AI-Video directory")
    sys.exit(1)

from pipeline_production import DeepfakePipeline

class TestSuite:
    def __init__(self, sbi_weights: str, syncnet_model: str, device: str = 'cuda'):
        self.pipeline = DeepfakePipeline(
            sbi_weights_path=sbi_weights,
            syncnet_model_path=syncnet_model,
            device=device
        )
        self.results = {
            'test_date': datetime.now().isoformat(),
            'videos_tested': 0,
            'real_videos': {'correct': 0, 'incorrect': 0, 'errors': 0, 'details': []},
            'fake_videos': {'correct': 0, 'incorrect': 0, 'errors': 0, 'details': []},
            'layer_stats': {
                'Layer 1: Audio Analysis': {'real_correct': 0, 'real_fake': 0, 'fake_correct': 0, 'fake_real': 0},
                'Layer 2: Visual Artifacts': {'real_correct': 0, 'real_fake': 0, 'fake_correct': 0, 'fake_real': 0},
                'Layer 3: Lip-Sync Analysis': {'real_correct': 0, 'real_fake': 0, 'fake_correct': 0, 'fake_real': 0},
                'Layer 4: Semantic Analysis': {'real_correct': 0, 'real_fake': 0, 'fake_correct': 0, 'fake_real': 0}
            }
        }
    
    def test_video(self, video_path: str, is_fake: bool) -> Dict:
        """Test a single video and return results"""
        try:
            print(f"\n{'='*80}")
            print(f"Testing: {Path(video_path).name}")
            print(f"Ground Truth: {'FAKE' if is_fake else 'REAL'}")
            print(f"{'='*80}")
            
            result = self.pipeline.detect(video_path, enable_fail_fast=False)
            
            # Check if prediction is correct
            prediction_is_fake = (result.final_verdict == "FAKE")
            is_correct = (prediction_is_fake == is_fake)
            
            # Update overall stats
            category = 'fake_videos' if is_fake else 'real_videos'
            if is_correct:
                self.results[category]['correct'] += 1
            else:
                self.results[category]['incorrect'] += 1
            
            # Update layer-by-layer stats
            for layer_result in result.layer_results:
                layer_name = layer_result.layer_name
                if is_fake:
                    # For fake videos: layer should detect fake
                    if layer_result.is_fake:
                        self.results['layer_stats'][layer_name]['fake_correct'] += 1
                    else:
                        self.results['layer_stats'][layer_name]['fake_real'] += 1
                else:
                    # For real videos: layer should NOT detect fake
                    if not layer_result.is_fake:
                        self.results['layer_stats'][layer_name]['real_correct'] += 1
                    else:
                        self.results['layer_stats'][layer_name]['real_fake'] += 1
            
            detail = {
                'video': str(video_path),
                'ground_truth': 'FAKE' if is_fake else 'REAL',
                'prediction': result.final_verdict,
                'correct': is_correct,
                'confidence': result.confidence,
                'total_time': result.total_time,
                'stopped_at': result.stopped_at_layer,
                'layers': []
            }
            
            for lr in result.layer_results:
                detail['layers'].append({
                    'name': lr.layer_name,
                    'prediction': 'FAKE' if lr.is_fake else 'REAL',
                    'confidence': lr.confidence,
                    'time': lr.processing_time,
                    'details': lr.details
                })
            
            self.results[category]['details'].append(detail)
            self.results['videos_tested'] += 1
            
            # Print summary
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"\n{status}")
            print(f"Prediction: {result.final_verdict} (Confidence: {result.confidence:.2%})")
            print(f"Time: {result.total_time:.2f}s")
            
            return detail
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            traceback.print_exc()
            
            self.results[category]['errors'] += 1
            self.results['videos_tested'] += 1
            
            error_detail = {
                'video': str(video_path),
                'ground_truth': 'FAKE' if is_fake else 'REAL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.results[category]['details'].append(error_detail)
            
            return error_detail
    
    def test_directory(self, directory: str, is_fake: bool, max_videos: int = None):
        """Test all videos in a directory"""
        video_dir = Path(directory)
        
        if not video_dir.exists():
            print(f"Warning: Directory {directory} does not exist")
            return
        
        # Find all videos
        videos = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            videos.extend(list(video_dir.glob(ext)))
        
        if len(videos) == 0:
            print(f"Warning: No videos found in {directory}")
            return
        
        if max_videos:
            videos = videos[:max_videos]
        
        print(f"\n{'='*80}")
        print(f"Testing {len(videos)} videos from {directory}")
        print(f"Expected: {'FAKE' if is_fake else 'REAL'}")
        print(f"{'='*80}")
        
        for video in videos:
            self.test_video(str(video), is_fake)
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*80)
        
        total = self.results['videos_tested']
        real_total = self.results['real_videos']['correct'] + self.results['real_videos']['incorrect'] + self.results['real_videos']['errors']
        fake_total = self.results['fake_videos']['correct'] + self.results['fake_videos']['incorrect'] + self.results['fake_videos']['errors']
        
        print(f"\nTotal Videos Tested: {total}")
        print(f"  Real Videos: {real_total}")
        print(f"  Fake Videos: {fake_total}")
        
        print("\n" + "-"*80)
        print("OVERALL ACCURACY")
        print("-"*80)
        
        if real_total > 0:
            real_accuracy = self.results['real_videos']['correct'] / real_total * 100
            print(f"Real Videos: {self.results['real_videos']['correct']}/{real_total} correct ({real_accuracy:.1f}%)")
            if self.results['real_videos']['incorrect'] > 0:
                print(f"  ⚠ False Positives (Real marked as Fake): {self.results['real_videos']['incorrect']}")
        
        if fake_total > 0:
            fake_accuracy = self.results['fake_videos']['correct'] / fake_total * 100
            print(f"Fake Videos: {self.results['fake_videos']['correct']}/{fake_total} correct ({fake_accuracy:.1f}%)")
            if self.results['fake_videos']['incorrect'] > 0:
                print(f"  ⚠ False Negatives (Fake marked as Real): {self.results['fake_videos']['incorrect']}")
        
        if total > 0:
            overall_accuracy = (self.results['real_videos']['correct'] + self.results['fake_videos']['correct']) / total * 100
            print(f"\nOverall Accuracy: {overall_accuracy:.1f}%")
        
        print("\n" + "-"*80)
        print("LAYER-BY-LAYER PERFORMANCE")
        print("-"*80)
        
        for layer_name, stats in self.results['layer_stats'].items():
            print(f"\n{layer_name}:")
            
            # Real video performance
            real_tested = stats['real_correct'] + stats['real_fake']
            if real_tested > 0:
                real_layer_acc = stats['real_correct'] / real_tested * 100
                print(f"  Real Videos: {stats['real_correct']}/{real_tested} correct ({real_layer_acc:.1f}%)")
                if stats['real_fake'] > 0:
                    print(f"    ⚠ False Positives: {stats['real_fake']}")
            
            # Fake video performance
            fake_tested = stats['fake_correct'] + stats['fake_real']
            if fake_tested > 0:
                fake_layer_acc = stats['fake_correct'] / fake_tested * 100
                print(f"  Fake Videos: {stats['fake_correct']}/{fake_tested} correct ({fake_layer_acc:.1f}%)")
                if stats['fake_real'] > 0:
                    print(f"    ⚠ False Negatives: {stats['fake_real']}")
        
        print("\n" + "="*80)
    
    def save_results(self, output_file: str = "test_results.json"):
        """Save detailed results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Detailed results saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Pipeline Testing")
    parser.add_argument("--sbi-weights", default="weights/SBI/FFc23.tar")
    parser.add_argument("--syncnet-model", default="syncnet_python/data/syncnet_v2.model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-per-category", type=int, default=5, 
                        help="Max videos to test per category")
    parser.add_argument("--output", default="test_results.json")
    
    args = parser.parse_args()
    
    # Check weights
    if not Path(args.sbi_weights).exists():
        print(f"✗ SBI weights not found: {args.sbi_weights}")
        return
    
    if not Path(args.syncnet_model).exists():
        print(f"✗ SyncNet model not found: {args.syncnet_model}")
        return
    
    # Initialize test suite
    print("\n" + "="*80)
    print("INITIALIZING TEST SUITE")
    print("="*80)
    
    suite = TestSuite(args.sbi_weights, args.syncnet_model, args.device)
    
    # Test real videos
    suite.test_directory("Test-Video/Real", is_fake=False, max_videos=args.max_per_category)
    suite.test_directory("Test-Video/YouTube-real", is_fake=False, max_videos=args.max_per_category)
    suite.test_directory("Test-Video/Celeb-real", is_fake=False, max_videos=args.max_per_category)
    
    # Test fake videos
    suite.test_directory("Test-Video/Fake", is_fake=True, max_videos=args.max_per_category)
    suite.test_directory("Test-Video/Celeb-synthesis", is_fake=True, max_videos=args.max_per_category)
    
    # Print summary and save results
    suite.print_summary()
    suite.save_results(args.output)


if __name__ == "__main__":
    main()
