"""
Validation Script for All Trained Models
Runs validation on all models in runs/train/ and collects metrics.

Usage:
    # Validate all models on a specific dataset
    python validate_all_models.py --data visdrone-vid.yaml
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

import sys
sys.path.insert(0, "./models")

from models.hierlight_modules import IRDCB, LDown

import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

modules.IRDCB = IRDCB
modules.LDown = LDown
tasks.IRDCB = IRDCB
tasks.LDown = LDown


def load_models_from_config(config_path: str = "model_configs.yaml") -> list:
    """
    Load model configurations from YAML file.
    Returns list of enabled models with their paths.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_path}")
        return []
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    models = []
    for model in config.get('models', []):
        if not model.get('enabled', True):
            continue
        
        name = model['name']
        path = model['path']
        
        # Determine stage from name
        if '-base' in name or name.endswith('n.pt') or name.endswith('s.pt') or name.endswith('m.pt'):
            stage = 'base'
        elif '-pretrain' in name:
            stage = 'pretrain'
        elif '-finetune' in name:
            stage = 'finetune'
        else:
            stage = 'unknown'
        
        models.append({
            'name': name,
            'path': path,
            'stage': stage,
            'dir': str(Path(path).parent)
        })
    
    return models


def validate_model(model_info: dict, data: str) -> dict:
    """
    Validate a single model and return metrics.
    """
    model_name = model_info['name']
    model_path = model_info['path']
    stage = model_info['stage']
    
    print(f"\n{'='*80}")
    print(f"Validating: {model_name}")
    print(f"Path: {model_path}")
    print(f"Stage: {stage}")
    print(f"Dataset: {data}")
    print(f"{'='*80}")
    
    result = {
        'model_name': model_name,
        'model_path': model_path,
        'stage': stage,
        'dataset': data,
        'status': 'UNKNOWN'
    }
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"⚠️  Model not found: {model_path}")
        result['status'] = 'SKIPPED'
        result['reason'] = 'Model file not found'
        return result
    
    # Load model
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        result['status'] = 'ERROR'
        result['reason'] = f"Failed to load model: {str(e)}"
        return result
    
    # Run validation
    try:
        start_time = time.time()
        metrics = model.val(data=data)
        val_time = time.time() - start_time
        
        # Extract metrics from results object
        result['status'] = 'SUCCESS'
        result['validation_time_seconds'] = val_time
        result['mAP50'] = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None
        result['mAP50_95'] = float(metrics.box.map) if hasattr(metrics.box, 'map') else None
        result['precision'] = float(metrics.box.mp) if hasattr(metrics.box, 'mp') else None
        result['recall'] = float(metrics.box.mr) if hasattr(metrics.box, 'mr') else None
        
        print(f"✓ Validation completed in {val_time:.2f}s")
        print(f"  mAP50: {result['mAP50']:.4f}" if result['mAP50'] else "  mAP50: N/A")
        print(f"  mAP50-95: {result['mAP50_95']:.4f}" if result['mAP50_95'] else "  mAP50-95: N/A")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        result['status'] = 'ERROR'
        result['reason'] = f"Validation failed: {str(e)}"
    
    return result


def main(args: argparse.Namespace):
    """Main validation function."""
    print("="*80)
    print("BATCH MODEL VALIDATION")
    print("="*80)
    
    # Load models from config file
    print(f"Loading models from config: {args.config}")
    models = load_models_from_config(args.config)
    
    if not models:
        print(f"❌ No models found")
        return
    
    print(f"\nFound {len(models)} model(s):")
    for model in models:
        print(f"  - {model['name']} ({model['stage']})")
    
    # Validate all models
    all_results = []
    start_time = time.time()
    
    for model_info in models:
        result = validate_model(model_info, args.data)
        all_results.append(result)
    
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_models_validated": len(all_results),
            "total_time_seconds": total_time,
            "dataset": args.data,
            "results": all_results
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETED")
    print("="*80)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Model':<35} {'Stage':<10} {'Status':<10} {'mAP50':<10} {'mAP50-95':<10}")
    print("-"*80)
    for result in all_results:
        status = result.get("status", "UNKNOWN")
        stage = result.get("stage", "unknown")
        mAP50 = result.get("mAP50", None)
        mAP50_95 = result.get("mAP50_95", None)
        
        mAP50_str = f"{mAP50:.4f}" if mAP50 is not None else "N/A"
        mAP50_95_str = f"{mAP50_95:.4f}" if mAP50_95 is not None else "N/A"
        
        print(f"{result['model_name']:<35} {stage:<10} {status:<10} {mAP50_str:<10} {mAP50_95_str:<10}")


    # --- Table/Image Reporting Logic (in-memory) ---
    import pandas as pd
    from tabulate import tabulate
    import matplotlib.pyplot as plt

    df = pd.DataFrame(all_results)

    # Select and rename columns to match the paper's style
    df_table = df[[col for col in ["model_name", "mAP50", "mAP50_95", "precision", "recall", "validation_time_seconds"] if col in df.columns]].copy()
    df_table.rename(columns={
        "model_name": "Model",
        "mAP50": "AP$_{0.5}$",
        "mAP50_95": "AP$_{0.5:0.95}$",
        "precision": "Precision",
        "recall": "Recall",
        "validation_time_seconds": "Val Time (s)"
    }, inplace=True)

    # Format floats for pretty printing
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else x

    df_table = df_table.applymap(fmt)

    print("\n--- Table Output ---")
    print(tabulate(df_table, headers="keys", tablefmt="github", showindex=False))

    # Save as image using matplotlib table
    fig, ax = plt.subplots(figsize=(12, 0.5 + 0.5 * len(df_table)))
    ax.axis('off')
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df_table.columns))))
    plt.tight_layout()
    table_img_path = output_dir / "validation_table.png"
    plt.savefig(table_img_path, bbox_inches='tight', dpi=200)
    print(f"Saved table as {table_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate all trained models on a specific dataset")
    parser.add_argument("--config", type=str, default="model_configs.yaml", help="YAML config file with model list")
    parser.add_argument("--data", type=str, default="visdrone-vid.yaml", help="Dataset YAML to validate on")
    parser.add_argument("--output_dir", type=str, default="validation_results", help="Directory to save results")
    
    args = parser.parse_args()
    main(args)
