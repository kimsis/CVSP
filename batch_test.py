"""
Batch Testing Script for Multiple Models
Runs multiple models through the video pipeline and collects performance metrics.

Usage:
    # Test all models on a video folder (headless mode, faster)
    python batch_test.py --video_folder test_videos/ --headless
    
    # Test specific models with output videos
    python batch_test.py --video_folder test_videos/ --models hierlight-n-visdrone,yolo11n-visdrone --save_output
    
    # Use custom model config file
    python batch_test.py --video_folder test_videos/ --config my_models.yaml --headless
"""

import sys
sys.path.insert(0, "./models")

from hierlight_modules import IRDCB, LDown

import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

modules.IRDCB = IRDCB
modules.LDown = LDown
tasks.IRDCB = IRDCB
tasks.LDown = LDown

# Now import other modules
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import yaml
from ultralytics import YOLO
from video_utils import get_video_sources
from main import run_pipeline


def load_model_configs(config_file: str = "model_configs.yaml") -> list:
    """
    Load model configurations from YAML file.
    Falls back to hardcoded configs if file doesn't exist.
    """
    config_path = Path(config_file)
    
    if config_path.exists():
        print(f"Loading model configs from: {config_path}")
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            models = [m for m in config_data.get("models", []) if m.get("enabled", True)]
            return models
    else:
        return Exception('Config file not found')


def test_model_on_videos(model_config: dict, video_sources: list, args: argparse.Namespace) -> dict:
    """
    Test a single model on all video sources and collect metrics.
    
    Returns:
        dict: Performance metrics including FPS, detection counts, timing, etc.
    """
    model_name = model_config["name"]
    model_path = Path(model_config["path"])
    
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}")
    
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        return {
            "model_name": model_name,
            "status": "SKIPPED",
            "reason": "Model file not found",
            "model_path": str(model_path)
        }
    
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {
            "model_name": model_name,
            "status": "ERROR",
            "reason": f"Failed to load model: {str(e)}",
            "model_path": str(model_path)
        }
    
    # Set output directory if saving
    output_dir = None
    if args.save_output:
        output_dir = Path(args.output_dir) / model_name
    
    # Run the pipeline using main.py's run_pipeline function
    metrics = run_pipeline(
        model=model,
        video_sources=video_sources,
        save_output=args.save_output,
        output_dir=str(output_dir) if output_dir else None,
        headless=args.headless,
        model_name=model_name,
        collect_metrics=True,
        target_fps=args.target_fps
    )
    
    # Add model path to metrics
    metrics["model_path"] = str(model_path)
    
    print(f"\n✅ Model {model_name} completed:")
    print(f"   Total Frames: {metrics['total_frames']}")
    print(f"   Total Detections: {metrics['total_detections']}")
    print(f"   Total Inference Time: {metrics['total_inference_time']:.2f} seconds")
    
    return metrics


def main(args: argparse.Namespace):
    """Main batch testing function."""
    print("="*80)
    print("BATCH MODEL TESTING")
    print("="*80)
    
    # Get video sources
    video_sources = get_video_sources(args)
    if not video_sources:
        print("❌ No video sources found!")
        return
    
    print(f"\nFound {len(video_sources)} video(s) to process")
    for i, src in enumerate(video_sources, 1):
        print(f"  {i}. {src}")
    
    # Load model configurations
    all_model_configs = load_model_configs(args.config)

    # Test all models
    all_results = []
    start_time = time.time()
    
    for model_config in all_model_configs:
        result = test_model_on_videos(model_config, video_sources, args)
        all_results.append(result)
    
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_models_tested": len(all_results),
            "total_time_seconds": total_time,
            "video_sources": [str(v) for v in video_sources],
            "results": all_results
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("BATCH TESTING COMPLETED")
    print("="*80)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Status':<12} {'Frames':<10} {'Detections':<12} {'Avg FPS':<10}")
    print("-"*80)
    for result in all_results:
        status = result.get("status", "UNKNOWN")
        frames = result.get("total_frames", 0)
        detections = result.get("total_detections", 0)
        fps = result.get("avg_fps", 0.0)
        print(f"{result['model_name']:<30} {status:<12} {frames:<10} {detections:<12} {fps:<10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch test multiple models on video dataset")
    parser.add_argument("--video_path", type=str, default=None, help="Path to single video file")
    parser.add_argument("--video_folder", type=str, default=None, help="Folder containing video files")
    parser.add_argument("--config", type=str, default="model_configs.yaml", help="Path to model configuration YAML")
    parser.add_argument("--output_dir", type=str, default="batch_test_results", help="Directory to save results")
    parser.add_argument("--save_output", action="store_true", help="Save output videos with detections")
    parser.add_argument("--target_fps", type=int, default=60, help="Target FPS for video processing (default: 60)")
    parser.add_argument("--headless", action="store_true", help="Run without displaying windows (faster)")
    
    args = parser.parse_args()
    main(args)
