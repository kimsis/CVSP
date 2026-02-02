"""
Parse batch test results JSON and generate comparison charts.

Usage:
    python plot_batch_test_results.py batch_test_results/batch_test_results_20260128_071511.json
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python plot_batch_test_results.py <batch_test_results.json>")
    sys.exit(1)

json_path = sys.argv[1]

with open(json_path) as f:
    data = json.load(f)

# Extract results
results = data["results"]

# Filter out interrupted/failed models
successful_results = [r for r in results if r["status"] == "SUCCESS"]

if not successful_results:
    print("No successful results to plot")
    sys.exit(1)

# Create DataFrame
df = pd.DataFrame(successful_results)

# Extract key metrics
df_plot = df[["model_name", "total_detections", "avg_fps", "avg_detections_per_frame", "total_inference_time"]].copy()

# Sort by model name for consistent ordering
# df_plot = df_plot.sort_values("model_name")

print(f"\nBatch Test Results Summary")
print(f"{'='*80}")
print(f"Timestamp: {data['timestamp']}")
print(f"Video(s): {', '.join(data['video_sources'])}")
print(f"Total models tested: {data['total_models_tested']}")
print(f"Successful models: {len(successful_results)}")
print(f"{'='*80}\n")

# Print table
from tabulate import tabulate
print(tabulate(df_plot, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

# Create comparison charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Model Comparison - Batch Test Results\n{data["timestamp"]}', fontsize=16, fontweight='bold')

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
# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=8)

# Chart 2: Average FPS (Inference Speed)
ax2 = axes[0, 1]
bars2 = ax2.bar(x_pos, df_plot["avg_fps"], color='lightgreen', edgecolor='darkgreen', alpha=0.7)
ax2.set_xlabel('Model', fontweight='bold')
ax2.set_ylabel('Average FPS', fontweight='bold')
ax2.set_title('Inference Speed (FPS)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)
# Add value labels on bars
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
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
# Add value labels on bars
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
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
# Add value labels on bars
for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save the chart
output_path = Path(json_path).parent / f"{Path(json_path).stem}_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved comparison chart to: {output_path}")

# Create a summary table image
fig2, ax = plt.subplots(figsize=(14, 0.5 + 0.5 * len(df_plot)))
ax.axis('off')

# Format data for table
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
table_path = Path(json_path).parent / f"{Path(json_path).stem}_table.png"
plt.savefig(table_path, dpi=200, bbox_inches='tight')
print(f"Saved summary table to: {table_path}")

plt.show()
