"""
Testing Script for 4-Layer Deepfake Detection Pipeline
Tests on Real/Fake video datasets and measures performance
"""

import sys
from pathlib import Path
import argparse
import json
import time
from typing import List, Dict
import pandas as pd

from pipeline_production import DeepfakePipeline, PipelineResult
from supabase_logger import get_logger


def get_test_videos(test_dir: str) -> Dict[str, List[Path]]:
    """
    Collect test videos from Test-Video directory
    
    Expected structure:
        Test-Video/
            Real/
            Fake/
            Celeb-real/
            Celeb-synthesis/
            YouTube-real/
    
    Returns:
        Dictionary with 'real' and 'fake' video lists
    """
    test_path = Path(test_dir)
    
    videos = {
        'real': [],
        'fake': []
    }
    
    # Real videos
    real_dirs = ['Real', 'Celeb-real', 'YouTube-real']
    for dir_name in real_dirs:
        dir_path = test_path / dir_name
        if dir_path.exists():
            # Common video extensions
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                videos['real'].extend(dir_path.glob(ext))
                # Recursively check videos subfolder
                videos['real'].extend(dir_path.glob(f'videos/{ext}'))
    
    # Fake videos
    fake_dirs = ['Fake', 'Celeb-synthesis']
    for dir_name in fake_dirs:
        dir_path = test_path / dir_name
        if dir_path.exists():
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                videos['fake'].extend(dir_path.glob(ext))
                videos['fake'].extend(dir_path.glob(f'videos/{ext}'))
    
    return videos


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate accuracy, precision, recall, F1 score
    """
    true_positive = sum(1 for r in results if r['ground_truth'] == 'FAKE' and r['prediction'] == 'FAKE')
    true_negative = sum(1 for r in results if r['ground_truth'] == 'REAL' and r['prediction'] == 'REAL')
    false_positive = sum(1 for r in results if r['ground_truth'] == 'REAL' and r['prediction'] == 'FAKE')
    false_negative = sum(1 for r in results if r['ground_truth'] == 'FAKE' and r['prediction'] == 'REAL')
    
    total = len(results)
    
    accuracy = (true_positive + true_negative) / total if total > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positive': true_positive,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'total': total
    }


def run_tests(
    pipeline: DeepfakePipeline,
    test_videos: Dict[str, List[Path]],
    max_per_category: int = 10,
    enable_fail_fast: bool = True,
    save_results: bool = True,
    log_to_supabase: bool = False
) -> Dict:
    """
    Run tests on video datasets
    
    Args:
        pipeline: Initialized DeepfakePipeline
        test_videos: Dictionary with 'real' and 'fake' video lists
        max_per_category: Maximum videos to test per category
        enable_fail_fast: Use fail-fast cascade logic
        save_results: Save results to JSON file
        log_to_supabase: Log results to Supabase database
    
    Returns:
        Test results and metrics
    """
    results = []
    
    # Optional Supabase logger
    supabase_logger = None
    if log_to_supabase:
        try:
            supabase_logger = get_logger()
            print("✓ Supabase logging enabled")
        except Exception as e:
            print(f"⚠ Could not initialize Supabase logger: {e}")
    
    print("\n" + "=" * 80)
    print("TESTING 4-LAYER DEEPFAKE DETECTION PIPELINE")
    print("=" * 80)
    
    # Test real videos
    print(f"\n{'=' * 80}")
    print(f"Testing REAL videos (max {max_per_category})")
    print(f"{'=' * 80}")
    
    for i, video_path in enumerate(test_videos['real'][:max_per_category], 1):
        print(f"\n[{i}/{min(len(test_videos['real']), max_per_category)}] Testing: {video_path.name}")
        
        try:
            result = pipeline.detect(str(video_path), enable_fail_fast=enable_fail_fast)
            
            test_result = {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'ground_truth': 'REAL',
                'prediction': result.final_verdict,
                'confidence': result.confidence,
                'stopped_at_layer': result.stopped_at_layer,
                'total_time': result.total_time,
                'correct': result.final_verdict == 'REAL',
                'layer_results': [
                    {
                        'layer': lr.layer_name,
                        'is_fake': lr.is_fake,
                        'confidence': lr.confidence,
                        'time': lr.processing_time
                    }
                    for lr in result.layer_results
                ]
            }
            
            results.append(test_result)
            
            # Log to Supabase
            if supabase_logger:
                try:
                    supabase_logger.log_detection(
                        result,
                        session_id=f"test_session_{int(time.time())}"
                    )
                except Exception as e:
                    print(f"  ⚠ Supabase logging error: {e}")
            
            status = "✓" if test_result['correct'] else "✗"
            print(f"{status} Prediction: {result.final_verdict} (Confidence: {result.confidence:.2%})")
            
        except Exception as e:
            print(f"✗ Error processing video: {e}")
            results.append({
                'video_path': str(video_path),
                'video_name': video_path.name,
                'ground_truth': 'REAL',
                'prediction': 'ERROR',
                'error': str(e),
                'correct': False
            })
    
    # Test fake videos
    print(f"\n{'=' * 80}")
    print(f"Testing FAKE videos (max {max_per_category})")
    print(f"{'=' * 80}")
    
    for i, video_path in enumerate(test_videos['fake'][:max_per_category], 1):
        print(f"\n[{i}/{min(len(test_videos['fake']), max_per_category)}] Testing: {video_path.name}")
        
        try:
            result = pipeline.detect(str(video_path), enable_fail_fast=enable_fail_fast)
            
            test_result = {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'ground_truth': 'FAKE',
                'prediction': result.final_verdict,
                'confidence': result.confidence,
                'stopped_at_layer': result.stopped_at_layer,
                'total_time': result.total_time,
                'correct': result.final_verdict == 'FAKE',
                'layer_results': [
                    {
                        'layer': lr.layer_name,
                        'is_fake': lr.is_fake,
                        'confidence': lr.confidence,
                        'time': lr.processing_time
                    }
                    for lr in result.layer_results
                ]
            }
            
            results.append(test_result)
            
            # Log to Supabase
            if supabase_logger:
                try:
                    supabase_logger.log_detection(
                        result,
                        session_id=f"test_session_{int(time.time())}"
                    )
                except Exception as e:
                    print(f"  ⚠ Supabase logging error: {e}")
            
            status = "✓" if test_result['correct'] else "✗"
            print(f"{status} Prediction: {result.final_verdict} (Confidence: {result.confidence:.2%})")
            
        except Exception as e:
            print(f"✗ Error processing video: {e}")
            results.append({
                'video_path': str(video_path),
                'video_name': video_path.name,
                'ground_truth': 'FAKE',
                'prediction': 'ERROR',
                'error': str(e),
                'correct': False
            })
    
    # Calculate metrics
    valid_results = [r for r in results if r['prediction'] != 'ERROR']
    metrics = calculate_metrics(valid_results)
    
    # Calculate layer-wise statistics
    layer_stats = {}
    for result in valid_results:
        stopped_layer = result.get('stopped_at_layer', 'Unknown')
        if stopped_layer not in layer_stats:
            layer_stats[stopped_layer] = 0
        layer_stats[stopped_layer] += 1
    
    # Average processing time per layer
    avg_times = {}
    for result in valid_results:
        for layer_result in result.get('layer_results', []):
            layer_name = layer_result['layer']
            if layer_name not in avg_times:
                avg_times[layer_name] = []
            avg_times[layer_name].append(layer_result['time'])
    
    avg_processing_times = {
        layer: sum(times) / len(times)
        for layer, times in avg_times.items()
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    print(f"\nTotal videos tested: {len(results)}")
    print(f"Valid results: {len(valid_results)}")
    print(f"Errors: {len(results) - len(valid_results)}")
    
    print(f"\n{'METRIC':<20} {'VALUE':<15}")
    print("-" * 35)
    print(f"{'Accuracy':<20} {metrics['accuracy']:.2%}")
    print(f"{'Precision':<20} {metrics['precision']:.2%}")
    print(f"{'Recall':<20} {metrics['recall']:.2%}")
    print(f"{'F1 Score':<20} {metrics['f1_score']:.2%}")
    
    print(f"\n{'CONFUSION MATRIX':<20} {'COUNT':<15}")
    print("-" * 35)
    print(f"{'True Positive':<20} {metrics['true_positive']}")
    print(f"{'True Negative':<20} {metrics['true_negative']}")
    print(f"{'False Positive':<20} {metrics['false_positive']}")
    print(f"{'False Negative':<20} {metrics['false_negative']}")
    
    print(f"\n{'LAYER STOP DISTRIBUTION':<40} {'COUNT':<10} {'%':<10}")
    print("-" * 60)
    for layer, count in layer_stats.items():
        percentage = (count / len(valid_results)) * 100
        print(f"{layer:<40} {count:<10} {percentage:.1f}%")
    
    print(f"\n{'AVG PROCESSING TIME PER LAYER':<40} {'TIME (s)':<10}")
    print("-" * 60)
    for layer, avg_time in avg_processing_times.items():
        print(f"{layer:<40} {avg_time:.3f}s")
    
    # Save results
    if save_results:
        output = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'max_per_category': max_per_category,
                'enable_fail_fast': enable_fail_fast
            },
            'metrics': metrics,
            'layer_stats': layer_stats,
            'avg_processing_times': avg_processing_times,
            'detailed_results': results
        }
        
        output_file = f"test_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    return {
        'metrics': metrics,
        'layer_stats': layer_stats,
        'avg_processing_times': avg_processing_times,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test 4-Layer Deepfake Detection Pipeline"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="Test-Video",
        help="Path to test video directory"
    )
    parser.add_argument(
        "--sbi-weights",
        type=str,
        default="weights/SBI/FFc23.tar",
        help="Path to SBI model weights"
    )
    parser.add_argument(
        "--syncnet-model",
        type=str,
        default="syncnet_python/data/syncnet_v2.model",
        help="Path to SyncNet model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=10,
        help="Maximum videos to test per category (real/fake)"
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Disable fail-fast (run all layers)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON file"
    )
    parser.add_argument(
        "--log-supabase",
        action="store_true",
        help="Log results to Supabase database"
    )
    
    args = parser.parse_args()
    
    # Collect test videos
    print("Collecting test videos...")
    test_videos = get_test_videos(args.test_dir)
    
    print(f"Found {len(test_videos['real'])} REAL videos")
    print(f"Found {len(test_videos['fake'])} FAKE videos")
    
    if len(test_videos['real']) == 0 and len(test_videos['fake']) == 0:
        print("\n✗ No test videos found!")
        print(f"  Please check directory: {args.test_dir}")
        return
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = DeepfakePipeline(
        sbi_weights_path=args.sbi_weights,
        syncnet_model_path=args.syncnet_model,
        device=args.device
    )
    
    # Run tests
    results = run_tests(
        pipeline=pipeline,
        test_videos=test_videos,
        max_per_category=args.max_videos,
        enable_fail_fast=not args.no_fail_fast,
        save_results=not args.no_save,
        log_to_supabase=args.log_supabase
    )
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
