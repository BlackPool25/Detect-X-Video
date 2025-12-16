"""Show detailed confidence scores from each layer"""
import sys
from pipeline_production import DeepfakePipeline

video = sys.argv[1] if len(sys.argv) > 1 else "Test-Video/Celeb-synthesis/id0_id1_0000.mp4"

print(f"\nAnalyzing: {video}")
print("="*80)

pipeline = DeepfakePipeline()
result = pipeline.detect(video, enable_fail_fast=False)

print("\nLAYER-BY-LAYER BREAKDOWN:")
print("="*80)
for layer_result in result.layer_results:
    verdict = "FAKE" if layer_result.is_fake else "REAL"
    print(f"\n{layer_result.layer_name}:")
    print(f"  Verdict: {verdict}")
    print(f"  Confidence: {layer_result.confidence:.2%}")
    print(f"  Time: {layer_result.processing_time:.2f}s")
    if 'avg_fake_probability' in layer_result.details:
        print(f"  Avg Fake Prob: {layer_result.details['avg_fake_probability']:.4f}")
    if 'max_fake_probability' in layer_result.details:
        print(f"  Max Fake Prob: {layer_result.details['max_fake_probability']:.4f}")
    if 'error' in layer_result.details:
        print(f"  Error: {layer_result.details['error']}")

print("\n" + "="*80)
print(f"FINAL VERDICT: {result.final_verdict}")
print(f"Overall Confidence: {result.confidence:.2%}")
print(f"Stopped at: {result.stopped_at_layer}")
print(f"Total Time: {result.total_time:.2f}s")
print("="*80)
