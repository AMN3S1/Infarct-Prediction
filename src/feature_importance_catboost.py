# feature_importance_catboost.py
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Separate features and target
X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

# Identify categorical features by name
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Create CatBoost Pool with explicit cat_features
train_pool = Pool(data=X, label=y, cat_features=categorical_features)

# Initialize and train the model
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(train_pool)

# Get feature importances
importances = model.get_feature_importance(train_pool)
features = X.columns

# Create and sort DataFrame for visualization
fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
fi_df = fi_df.sort_values(by="Importance", ascending=False)

# Plot and save
plt.figure(figsize=(10, 8))
plt.barh(fi_df["Feature"], fi_df["Importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance (CatBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_catboost.png")
