import pandas as pd
import matplotlib.pyplot as plt

# Load both result files
df_full = pd.read_csv("model_results.csv")
df_top15 = pd.read_csv("model_results_top15.csv", on_bad_lines="skip")

# Ensure f1 column is consistent
df_full.rename(columns={"f1_score": "f1"}, inplace=True)
df_top15.rename(columns={"f1_score": "f1", "f1-score": "f1"}, inplace=True)

# List of models to compare (excluding Dummy)
models_to_compare = [
    "Logistic Regression", "Decision Tree", "KNN_5", "KNN_50", "KNN_150",
    "Naive Bayes", "RandomForest (20 trees)", "RandomForest (100 trees)",
    "XGBoost (100 trees)", "XGBoost (300 trees)", "CatBoost"
]

# Mapping for KNN model name conversion from top15
top15_name_map = {
    "KNN_top15_k=5": "KNN_5_top15",
    "KNN_top15_k=50": "KNN_50_top15",
    "KNN_top15_k=150": "KNN_150_top15"
}

# Filter and rename top15 models
df_top15 = df_top15[~df_top15["model"].str.contains("Dummy")]
df_top15["model"] = df_top15["model"].replace(top15_name_map)

# Append "_top15" to remaining model names in top15
df_top15["model"] = df_top15["model"].apply(lambda x: x if "_top15" in x else f"{x}_top15")

# Filter full dataset models
df_full = df_full[df_full["model"].isin(models_to_compare)]

# Combine
combined_df = pd.concat([df_full[["model", "f1"]], df_top15[["model", "f1"]]], ignore_index=True)

# Plot
plt.figure(figsize=(12, 6))
for model_base in models_to_compare:
    model_top15 = f"{model_base}_top15"
    f1_full = combined_df[combined_df["model"] == model_base]["f1"].values
    f1_top15 = combined_df[combined_df["model"] == model_top15]["f1"].values
    if len(f1_full) > 0 and len(f1_top15) > 0:
        plt.plot(["Full", "Top 15"], [f1_full[0], f1_top15[0]], marker='o', label=model_base)

plt.title("F1-score Comparison: Full Dataset vs Top 15 Features")
plt.ylabel("F1-score")
plt.grid(True)

plt.legend(
    loc="upper left",             # расположение относительно bbox
    bbox_to_anchor=(1.05, 0.15)   # смещаем ниже и правее
)

plt.tight_layout()
plt.savefig("f1_comparison_selected_models.png")


