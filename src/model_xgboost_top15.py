import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import os

# Load top-15 feature dataset
df = pd.read_csv("top15_features_dataset.csv")

# Convert object columns to category
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category")

# Features and target
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Start timing
start = time.time()
n_estimators = 300

# XGBoost with DART and categorical support
model = XGBClassifier(
    booster="dart",
    n_estimators=n_estimators,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    enable_categorical=True,
    verbosity=1
)

model.fit(X_train, y_train)
end = time.time()
train_time = end - start

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Training time: {train_time:.2f} sec")

# Save to results file
results_df = pd.DataFrame([{
    "Model": f"XGBoost ({n_estimators} trees)",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "Train time (s)": round(train_time, 2)
}])

file_path = "model_results_top15.csv"
write_header = not os.path.exists(file_path)
results_df.to_csv(file_path, mode="a", index=False, header=write_header)
