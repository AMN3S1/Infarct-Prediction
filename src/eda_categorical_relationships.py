
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency

# Create output folder
os.makedirs("eda_cramers_v_categorical_heatmap", exist_ok=True)

# Load preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")

# Select only categorical columns
categorical_cols = [
    'gender', 'region', 'income_level', 'smoking_status', 'alcohol_consumption',
    'physical_activity', 'dietary_habits', 'air_pollution_exposure',
    'stress_level', 'EKG_results'
]

# Function to calculate Cramér's V
def cramers_v(conf_matrix):
    chi2 = chi2_contingency(conf_matrix)[0]
    n = conf_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = conf_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# Calculate Cramér's V matrix
cramers_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

for col1 in categorical_cols:
    for col2 in categorical_cols:
        confusion = pd.crosstab(df[col1], df[col2])
        cramers_matrix.loc[col1, col2] = cramers_v(confusion)

# Convert to float for heatmap plotting
cramers_matrix = cramers_matrix.astype(float)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cramers_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)
plt.title("Cramér's V Correlation Between Categorical Variables")
plt.tight_layout()
plt.savefig("eda_cramers_v_categorical_heatmap/eda_cramers_v_categorical_heatmap.png")
plt.close()
