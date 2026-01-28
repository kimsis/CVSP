# Batch Model Testing Guide

## Overview

The `batch_test.py` script allows you to test multiple trained models on the same video dataset in one run, collecting performance metrics for comparison.

## Quick Start

### 1. Configure Your Models

Edit `model_configs.yaml` to add your trained models:

```yaml
models:
  - name: hierlight-n-visdrone
    path: runs/train/hierlight-yolo8-n/weights/best.pt
    enabled: true
  
  - name: yolo11n-visdrone
    path: runs/train/yolo11n-visdrone/weights/best.pt
    enabled: true
```

Set `enabled: false` to temporarily skip a model without deleting it.

### 2. Run Batch Tests

**Test all models (headless mode for speed):**
```bash
python batch_test.py --video_folder test_videos/ --headless
```

**Save output videos with detections:**
```bash
python batch_test.py --video_folder test_videos/ --save_output
```

**Use a custom config file:**
```bash
python batch_test.py --video_folder test_videos/ --config my_models.yaml --headless
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--video_path` | Path to single video file | None |
| `--video_folder` | Folder containing video files (MP4) | None |
| `--config` | Path to model config YAML | `model_configs.yaml` |
| `--output_dir` | Directory to save results | `batch_test_results` |
| `--save_output` | Save output videos with detections | False |
| `--headless` | Run without display windows (faster) | False |

## Output

### Results JSON

Results are saved to `batch_test_results/batch_test_results_TIMESTAMP.json` with:

- **Per-model metrics**: frames processed, detections, FPS, inference time
- **Per-video metrics**: breakdown for each video
- **Overall summary**: total time, model count, video list

Example structure:
```json
{
  "timestamp": "2026-01-20T10:30:00",
  "total_models_tested": 6,
  "total_time_seconds": 1234.5,
  "results": [
    {
      "model_name": "hierlight-n-visdrone",
      "status": "SUCCESS",
      "total_frames": 5000,
      "total_detections": 12500,
      "avg_fps": 45.2,
      "videos": [...]
    }
  ]
}
```

### Console Summary

After all tests, a summary table is printed:

```
SUMMARY
================================================================================
Model                          Status       Frames     Detections   Avg FPS   
--------------------------------------------------------------------------------
hierlight-n-visdrone           SUCCESS      5000       12500        45.20     
hierlight-s-visdrone           SUCCESS      5000       13200        38.50     
yolo11n-visdrone               SUCCESS      5000       11800        48.30     
```

### Output Videos (Optional)

With `--save_output`, annotated videos are saved to:
```
batch_test_results/
  hierlight-n-visdrone/
    video1.mp4
    video2.mp4
  yolo11n-visdrone/
    video1.mp4
    video2.mp4
```

## Typical Workflow

### 1. Train All Models

```bash
# HierLight-YOLO variants on COCO
python train.py --scale n  # Edit data to COCO first
python train.py --scale s
python train.py --scale m

# YOLOv11 baselines (use standard yolo11 training)
yolo train model=yolo11n.pt data=coco.yaml
yolo train model=yolo11s.pt data=coco.yaml
yolo train model=yolo11n.pt data=visdrone.yaml
# etc...
```

### 2. Update Model Configs

Edit `model_configs.yaml` with actual paths to your `best.pt` files.

### 3. Run Batch Tests

```bash
# Quick headless test
python batch_test.py --video_folder test_videos/ --headless

# Or test specific comparison
python batch_test.py --video_folder test_videos/ \
  --models hierlight-n-visdrone,yolo11n-visdrone \
  --headless
```

### 4. Analyze Results

Open the JSON file in `batch_test_results/` or parse it with a script to generate comparison charts.
