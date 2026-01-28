import sys
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.table as tbl

if len(sys.argv) < 2:
    print("Usage: python plot_chart.py <results.json>")
    sys.exit(1)

json_path = sys.argv[1]

with open(json_path) as f:
    data = json.load(f)

results = data["results"]
df = pd.DataFrame(results)

# Select and rename columns to match the paper's style
df_table = df[["model_name", "mAP50", "mAP50_95", "precision", "recall"]].copy()
df_table.rename(columns={
    "model_name": "Model",
    "mAP50": "AP$_{0.5}$",
    "mAP50_95": "AP$_{0.5:0.95}$",
    "precision": "Precision",
    "recall": "Recall"
}, inplace=True)

# Format floats for pretty printing
def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else x

df_table = df_table.applymap(fmt)

print(tabulate(df_table, headers="keys", tablefmt="github", showindex=False))

# Save as image using matplotlib table
fig, ax = plt.subplots(figsize=(12, 0.5 + 0.5 * len(df_table)))
ax.axis('off')
table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(df_table.columns))))
plt.tight_layout()
plt.savefig("validation_table.png", bbox_inches='tight', dpi=200)
print("Saved table as validation_table.png")