import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("preprocessed_data.csv")

# Create directory to save plots
os.makedirs("numeric_vs_target", exist_ok=True)

# Target variable
target_col = "heart_attack"

# List of numerical features to compare
numeric_cols = [
    'age', 'cholesterol_level', 'waist_circumference', 'sleep_hours',
    'blood_pressure_systolic', 'blood_pressure_diastolic',
    'fasting_blood_sugar', 'cholesterol_hdl', 'cholesterol_ldl', 'triglycerides'
]

# Plot boxplots of numerical features grouped by target
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=target_col, y=col)
    plt.title(f"{col} by Heart Attack")
    plt.xlabel("Heart Attack (0 = No, 1 = Yes)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"numeric_vs_target/{col}_vs_target_boxplot.png")
    plt.clf()
