# CVSP
Aerial Person Detection and Tracking in Drone Footage

## Project Structure

```
CVSP/
├── batch_test.py              # Batch testing script for multiple models
├── main.py                    # Single model inference pipeline
├── train.py                   # Model training script
├── validate_all_models.py     # Batch validation script
├── configs/
│   ├── model_configs.yaml     # Model configuration for batch operations
│   └── train_config.yaml      # Training configuration
├── scripts/
│   ├── analyze_results.py     # Comprehensive results analysis
│   ├── plot_batch_test_results.py  # Visualization tool
│   ├── video_utils.py         # Video processing utilities
│   └── static_object_filter.py     # Object filtering utilities
├── models/
│   └── hierlight_modules.py   # Custom HierLight-YOLO modules
├── datasets/                  # Dataset directory (VisDrone, etc.)
├── runs/                      # Training and validation outputs
└── batch_test_results/        # Batch testing results

```

## Setup

### Prerequisites

1. Check CUDA version: `nvidia-smi`
2. Activate virtual environment: `source venv/bin/activate`
3. Install PyTorch with CUDA support:
   ```bash
   # For CUDA 13.0
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
4. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

Place your dataset in the following structure:
```
datasets/
└── VisDrone-VID/
    ├── data.yaml
    ├── train/
    ├── val/
    └── test/
        └── videos/
```

# Training

## Configuration

Edit `configs/train_config.yaml` to set parameters for the training session and model configuration.

## Running Training

**Start fresh training:**
```bash
python train.py
```

**Resume interrupted training:**
```bash
python train.py --resume runs/train/model-folder/weights/last.pt
```

---

# Batch Testing

Test multiple models on video datasets and collect performance metrics.

## 1. Configure Your Models

Edit `configs/model_configs.yaml

### YOLO baseline models
python train.py yolo11n.pt
python train.py yolo11s.pt
python train.py yolo11m.pt

Training outputs are saved to `runs/train/`.

## 2. Run Batch Tests

**Test all models (headless mode for speed):**
```bash
python batch_test.py --video_folder test_videos/ --headless
```

**Test on specific video:**
```bash
python batch_test.py --video_path test_videos/video.mp4 --headless
```

**Save output videos with detections:**
```bash
python batch_test.py --video_folder test_videos/ --save_output
```
---

# Batch Validation

Validate all trained models and generate performance metrics with visualizations.

## Configuration

Edit `configs/model_configs.yaml` to specify which models to validate:

```yaml
models:
  - name: hierlight-yolo8-m
    path: runs/train/hierlight-yolo8-m/weights/best.pt
    enabled: true
  
  - name: yolo11m-visdrone
    path: runs/train/yolo11m-visdrone/weights/best.pt
    enabled: true
```

## Running Validation

```bash
# Validate all enabled models
python validate_all_models.py

# Use specific dataset
python validate_all_models.py --data visdrone-vid.yaml

# Use custom config
python validate_all_models.py --config configs/my_models.yaml
```

## Output

Results are saved to `validation_results/`:
- `validation_results_TIMESTAMP.json` - Detailed metrics for all models
- `validation_table.png` - Formatted table with mAP, Precision, Recall, and Validation Time

# Batch Testing

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
python scripts/batch_test.py --video_folder test_videos/ --headless
```

**Save output videos with detections:**
```bash
python scripts/batch_test.py --video_folder test_videos/ --save_output
```

## Output

Results are saved to `batch_test_results/batch_test_results_TIMESTAMP.json` with:

- **Per-model metrics**: frames processed, detections, FPS, inference time
- **Per-video metrics**: breakdown for each video
- **Overall summary**: total time, model count, video list

---
Complete Workflow

## Typical Pipeline

### 1. Train Models

```bash
# HierLight-YOLO variants
python train.py --scale n
python train.py --scale s
python train.py --scale m

# YOLO baselines
python train.py yolo11n.pt
python train.py yolo11s.pt
python train.py yolo11m.pt
```

### 2. Configure Models

Edit `configs/model_configs.yaml` with paths to your trained `best.pt` files:

```yaml
models:
  - name: hierlight-yolo8-m
    path: runs/train/hierlight-yolo8-m/weights/best.pt
    enabled: true
  - name: yolo11m-baseline
    path: runs/train/yolo11m-baseline/weights/best.pt
    enabled: true
```

### 3. Validate Models

```bash
python validate_all_models.py --data visdrone-vid.yaml
```

Check `validation_results/` for metrics and performance comparison table.

### 4. Run Batch Tests

```bash
# Test with output saving
python batch_test.py --video_folder test_videos/ --save_output
```

### 5. Visualize Results

```bash
# Comprehensive analysis
python scripts/analyze_results.py batch_test_results/batch_test_results_TIMESTAMP.json
```

### 6. Test Individual Models

```bash
# Run inference with a specific model
python main.py --trained_model runs/train/best-model/weights/best.pt --video_path test.mp4
```

# Scripts Directory

Utility scripts located in `scripts/`:

| Script | Purpose |
|--------|---------|
| `video_utils.py` | Video processing utilities (camera listing, frame resizing, source handling) |
| `static_object_filter.py` | Filter static/non-moving objects from detections |
| `analyze_results.py` | Comprehensive analysis with multiple chart types |
| `plot_batch_test_results.py` | Generate 4-panel comparison charts |
| `convert_vis_to_yolo.py` | Convert VisDrone annotations to YOLO format |
| `display_image_with_annotation.py` | Visualization tool for annotated images |
| `filter_coco_person.py` | Filter COCO dataset to person class only |

---

# Citation

---
## Output

### Results JSON

Results are saved to `batch_test_results/batch_test_results_TIMESTAMP.json` with:

- **Per-model metrics**: frames processed, detections, FPS, inference time
- **Per-video metrics**: breakdown for each video
- **Overall summary**: total time, model count, video list

# Using the pipeline
## Typical Workflow

### 1. Train All Models

```bash
# HierLight-YOLO variants
python scripts/train.py --scale n
python scripts/train.py --scale s
python scripts/train.py --scale m

# YOLOv11 baselines
python scripts/train.py yolo11n.pt
python scripts/train.py yolo11s.pt
python scripts/train.py yolo11m.pt
# etc...
```

### 2. Update Model Configs

Edit `model_configs.yaml` with actual paths to your `best.pt` files.

### 3. Run Batch Tests

```bash
# Quick headless test
python batch_test.py --video_folder test_videos/ --headless

# Or test on specific video
python batch_test.py --video_path test_videos/video.mp4
```

### 4. Analyze Results

Open the JSON file in `batch_test_results/` or the parsed data in the folder with the same name.







@article{zhu2021detection,
  title={Detection and tracking meet drones challenge},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={7380--7399},
  year={2021},
  publisher={IEEE}
}