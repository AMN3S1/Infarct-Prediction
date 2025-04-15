import pandas as pd
import matplotlib.pyplot as plt

# Load model results
df = pd.read_csv('model_results.csv')

# Plot precision vs recall for each model
plt.figure(figsize=(8, 6))
for idx, row in df.iterrows():
    plt.scatter(row["recall"], row["precision"], label=row["model"], s=100)

plt.title("Precision vs Recall for Each Model")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend(loc="lower left", bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()

# Save the plot
plt.savefig("precision_vs_recall.png")
print("Saved plot as 'precision_vs_recall.png'")
