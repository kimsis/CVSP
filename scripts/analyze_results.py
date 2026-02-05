"""
Results Analysis Script
Generates comparison charts and tables from batch test results JSON.

Usage (run from scripts/ directory):
    python analyze_results.py ../batch_test_results/batch_test_results_20260120_103000.json
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate


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


def generate_batch_style_charts(results: dict, output_dir: Path):
    """Generate plot_batch_test_results style 4-panel charts and table."""
    # Filter successful results
    successful_results = [r for r in results["results"] if r["status"] == "SUCCESS"]
    
    if not successful_results:
        return
    
    # Create DataFrame
    df = pd.DataFrame(successful_results)
    df_plot = df[["model_name", "total_detections", "avg_fps", "avg_detections_per_frame", "total_inference_time"]].copy()
    
    # Create comparison charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Comparison - Batch Test Results\\n{results["timestamp"]}', fontsize=16, fontweight='bold')
    
    models = df_plot["model_name"].values
    x_pos = np.arange(len(models))
    
    # Chart 1: Total Detections
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, df_plot["total_detections"], color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Total Detections', fontweight='bold')
    ax1.set_title('Total Detections per Model')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                 ha='center', va='bottom', fontsize=8)
    
    # Chart 2: Average FPS
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, df_plot["avg_fps"], color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Average FPS', fontweight='bold')
    ax2.set_title('Inference Speed (FPS)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                 ha='center', va='bottom', fontsize=8)
    
    # Chart 3: Average Detections per Frame
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, df_plot["avg_detections_per_frame"], color='salmon', edgecolor='darkred', alpha=0.7)
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Avg Detections per Frame', fontweight='bold')
    ax3.set_title('Average Detections per Frame')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=8)
    
    # Chart 4: Inference Time
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, df_plot["total_inference_time"] * 1000, color='plum', edgecolor='purple', alpha=0.7)
    ax4.set_xlabel('Model', fontweight='bold')
    ax4.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax4.set_title('Average Inference Time per Frame')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    comparison_path = output_dir / "batch_style_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {comparison_path}")
    plt.close()
    
    # Create summary table image
    fig2, ax = plt.subplots(figsize=(14, 0.5 + 0.5 * len(df_plot)))
    ax.axis('off')
    
    table_data = df_plot.copy()
    table_data.columns = ['Model', 'Total Detections', 'Avg FPS', 'Avg Det/Frame', 'Inference Time (s)']
    table_data['Total Detections'] = table_data['Total Detections'].astype(int)
    table_data['Avg FPS'] = table_data['Avg FPS'].apply(lambda x: f'{x:.2f}')
    table_data['Avg Det/Frame'] = table_data['Avg Det/Frame'].apply(lambda x: f'{x:.2f}')
    table_data['Inference Time (s)'] = table_data['Inference Time (s)'].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(cellText=table_data.values, 
                    colLabels=table_data.columns, 
                    loc='center', 
                    cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(len(table_data.columns))))
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    table_path = output_dir / "batch_style_table.png"
    plt.savefig(table_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {table_path}")
    plt.close()


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
    # Generate plot_batch_test_results style charts
    print("\nGenerating 4-panel comparison and table...")
    generate_batch_style_charts(results, output_dir)
    
    export_latex_table(df, output_dir)
    
    print(f"\n✅ Analysis complete! Check {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze batch test results")
    parser.add_argument("--results_file", type=str, help="Path to batch test results JSON file")
    
    args = parser.parse_args()
    main(args)
