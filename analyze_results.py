"""
Results Analysis Script
Generates comparison charts and tables from batch test results JSON.

Usage:
    python analyze_results.py batch_test_results/batch_test_results_20260120_103000.json
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_results(json_path: str) -> dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def generate_comparison_table(results: dict) -> pd.DataFrame:
    """Generate pandas DataFrame for easy comparison."""
    data = []
    for result in results["results"]:
        if result["status"] == "SUCCESS":
            data.append({
                "Model": result["model_name"],
                "Total Frames": result["total_frames"],
                "Total Detections": result["total_detections"],
                "Avg FPS": round(result["avg_fps"], 2),
                "Detections/Frame": round(result["avg_detections_per_frame"], 2),
                "Total Time (s)": round(result["total_inference_time"], 2)
            })
    
    return pd.DataFrame(data)


def plot_fps_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot FPS comparison bar chart."""
    plt.figure(figsize=(12, 6))
    plt.bar(df["Model"], df["Avg FPS"], color='skyblue', edgecolor='black')
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Average FPS", fontsize=12)
    plt.title("Inference Speed Comparison (Higher is Better)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fps_comparison.png", dpi=300)
    print(f"  Saved: {output_dir / 'fps_comparison.png'}")
    plt.close()


def plot_detections_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot detections per frame comparison."""
    plt.figure(figsize=(12, 6))
    plt.bar(df["Model"], df["Detections/Frame"], color='lightcoral', edgecolor='black')
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Detections per Frame", fontsize=12)
    plt.title("Detection Density Comparison", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "detections_comparison.png", dpi=300)
    print(f"  Saved: {output_dir / 'detections_comparison.png'}")
    plt.close()


def plot_grouped_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot grouped bar chart for multi-metric comparison."""
    # Group by architecture (n/s/m) and dataset (COCO/VisDrone)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # FPS subplot
    models = df["Model"]
    fps = df["Avg FPS"]
    x = range(len(models))
    
    axes[0].bar(x, fps, color='skyblue', edgecolor='black')
    axes[0].set_xlabel("Model", fontsize=12)
    axes[0].set_ylabel("Average FPS", fontsize=12)
    axes[0].set_title("Inference Speed", fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Detections/Frame subplot
    det_per_frame = df["Detections/Frame"]
    
    axes[1].bar(x, det_per_frame, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel("Model", fontsize=12)
    axes[1].set_ylabel("Detections per Frame", fontsize=12)
    axes[1].set_title("Detection Density", fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "grouped_comparison.png", dpi=300)
    print(f"  Saved: {output_dir / 'grouped_comparison.png'}")
    plt.close()


def export_latex_table(df: pd.DataFrame, output_dir: Path):
    """Export results as LaTeX table for papers."""
    latex_str = df.to_latex(index=False, float_format="%.2f")
    latex_path = output_dir / "results_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"  Saved: {latex_path}")


def export_csv(df: pd.DataFrame, output_dir: Path):
    """Export results as CSV."""
    csv_path = output_dir / "results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def main(args: argparse.Namespace):
    """Main analysis function."""
    json_path = Path(args.results_file)
    
    if not json_path.exists():
        print(f"❌ Results file not found: {json_path}")
        return
    
    print(f"Loading results from: {json_path}")
    results = load_results(str(json_path))
    
    # Create output directory
    output_dir = json_path.parent / f"analysis_{json_path.stem}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating analysis in: {output_dir}\n")
    
    # Generate comparison table
    df = generate_comparison_table(results)
    
    if df.empty:
        print("⚠️  No successful results to analyze!")
        return
    
    # Print to console
    print("="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print()
    
    # Generate plots
    print("Generating charts...")
    plot_fps_comparison(df, output_dir)
    plot_detections_comparison(df, output_dir)
    plot_grouped_comparison(df, output_dir)
    
    # Export tables
    print("\nExporting tables...")
    export_csv(df, output_dir)
    export_latex_table(df, output_dir)
    
    print(f"\n✅ Analysis complete! Check {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze batch test results")
    parser.add_argument("results_file", type=str, help="Path to batch test results JSON file")
    
    args = parser.parse_args()
    main(args)
