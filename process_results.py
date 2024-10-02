import pandas as pd

def bold_max_and_format(s):
    # Convert all values in the series to float
    s_float = s.astype(float)
    
    # Find the index of the maximum value in the original series
    max_indices = s_float == s_float.max()
    
    # Format all values in the series to three decimal places
    formatted_s = s_float.map('{:.5f}'.format)
    
    # Apply bold formatting only to the maximum value(s)
    return [f'\\textbf{{{x}}}' if max_indices.iloc[i] else x for i, x in enumerate(formatted_s)]

results = pd.read_csv('./results/metrics_summary.csv', sep=',', header=0)
results.columns = ['Scicheck feat.', 'Frequency feat.','Similarity feat.', 'Embedding feat.' ,'K', 'Precision']

# Group
results = results.groupby(['Scicheck feat.', 'Frequency feat.', 'Similarity feat.', 'Embedding feat.', 'K'])

# Average precision of each group
results = results.mean()

# pivot columns K and Precision so that they become "Precision@K"
results = results.reset_index().pivot(index=['Scicheck feat.', 'Frequency feat.', 'Similarity feat.', 'Embedding feat.'], columns='K', values='Precision')
# Rename columns (currently numbers) to "Precision@K"
results.columns = ['P@' + str(col) for col in results.columns]

# Get as list
# results = results.reset_index()

for col in results.columns:
    if col.startswith('P@'):  # Assuming your precision columns are named like "Precision@5", "Precision@10", etc.
        # set col to numeric
        results[col] = pd.to_numeric(results[col])
        results[col] = bold_max_and_format(results[col])

# Convert to LaTeX
print(results.to_latex(escape=False, index=True))
print(results)

results_data = [
    {"technique": "TransE", "P@5": 0.63000, "P@10": 0.63750, "P@15": 0.63500, "P@20": 0.63375},
    {"technique": "TransH", "P@5": 0.79000, "P@10": 0.77250, "P@15": 0.73667, "P@20": 0.71125},
    {"technique": "TransD", "P@5": 0.80500, "P@10": 0.74250, "P@15": 0.71833, "P@20": 0.71750},
    {"technique": "RotatE", "P@5": 0.72500, "P@10": 0.73250, "P@15": 0.71667, "P@20": 0.70750},
    {"technique": "RGCN",   "P@5": 0.65000, "P@10": 0.65000, "P@15": 0.63333, "P@20": 0.60000},
    {"technique": "ResearchLink (best)",   "P@5": 0.85000, "P@10": 0.77500, "P@15": 0.81667, "P@20": 0.78750}
]

data = pd.DataFrame(results_data)
print(data)

# Bar plot. Bar color for each technique. X axis = different K values. Y axis = Precision@K

import seaborn as sns
import matplotlib.pyplot as plt



custom_params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.color": "black",
    "axes.labelcolor": "black"
}
sns.set_theme(style="whitegrid", font_scale=2, rc=custom_params)  # Sets the style to 'whitegrid' for a clean and minimalistic look


data_melted = data.melt(id_vars=["technique"], var_name="metric", value_name="value")
print(data_melted)

# Plotting
plt.figure(figsize=(12, 8))

sns.barplot(x="metric", y="value", hue="technique", data=data_melted, palette="viridis")
plt.title("Precision Metrics Across Techniques")
plt.xlabel("Metric")
plt.ylabel("Value")
# y axis starts at 0.5 and ends at 0.9
plt.ylim(0.5, 0.9)
# set legend under the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=3, fontsize=16)
plt.tight_layout()
plt.show()

# Save plot as svg
plt.savefig("./results/precision_metrics_across_techniques.svg", format="svg")
plt.savefig("./results/precision_metrics_across_techniques.png", format="png")
# Save as pdf  
plt.savefig("./results/precision_metrics_across_techniques.pdf", format="pdf")