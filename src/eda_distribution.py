
import matplotlib
matplotlib.use("Agg")  # to avoid PyCharm errors

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for saving plots
os.makedirs("eda_correlation", exist_ok=True)

# Load the preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")

# Select only numerical columns
numerical_df = df.select_dtypes(include=["int64", "float64"])

# Compute correlation matrix
corr_matrix = numerical_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig("eda_correlation/eda_correlation_heatmap.png")
plt.clf()
