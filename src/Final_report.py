import pandas as pd
import os

# Define metric values for the top 3 models
data = [
    {
        "Model": "CatBoost (Top 15)",
        "Accuracy": 0.7353,
        "Precision": 0.6933,
        "Recall": 0.6157,
        "F1-score": 0.6522,
        "AUC": 0.843
    },
    {
        "Model": "CatBoost (optimized)",
        "Accuracy": 0.7347,
        "Precision": 0.6967,
        "Recall": 0.5994,
        "F1-score": 0.6444,
        "AUC": 0.814
    },
    {
        "Model": "XGBoost (optimized)",
        "Accuracy": 0.7339,
        "Precision": 0.6893,
        "Recall": 0.6122,
        "F1-score": 0.6485,
        "AUC": 0.814
    }
]

# Create DataFrame
df = pd.DataFrame(data)

# Define output file path
output_file = "final_model_report.csv"

# Save to CSV (overwrite if already exists)
df.to_csv(output_file, index=False)

