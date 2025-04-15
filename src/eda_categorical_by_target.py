import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load preprocessed dataset
df = pd.read_csv('preprocessed_data.csv')

# Create folder for saving plots
os.makedirs("eda_plots_by_target", exist_ok=True)

# List of categorical features to analyze
categorical_features = [
    'gender', 'region', 'income_level', 'smoking_status', 'alcohol_consumption',
    'physical_activity', 'dietary_habits', 'air_pollution_exposure',
    'stress_level', 'EKG_results'
]

# Plot barplots for each categorical feature vs target
for col in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=col, hue='heart_attack')
    plt.title(f'Heart Attack Rate by {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"eda_plots_by_target/eda_by_target_{col}.png")
    plt.clf()
