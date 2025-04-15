import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
from joblib import dump

# Load reduced dataset with top 15 features
df = pd.read_csv("top15_features_dataset.csv")

# Separate features and target
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]

# Identify categorical features (CatBoost handles them natively)
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost
model = CatBoostClassifier(verbose=100, cat_features=categorical_cols, random_state=42)

# Train model with time tracking
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# Predict
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Show results
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Training time: {training_time:.2f} sec")

# Save results
results_df = pd.DataFrame([{
    "Model": "CatBoost (Top 15)",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1,
    "Training time (s)": training_time
}])



# Save results to model_results_top15.csv
file_path = "model_results_top15.csv"
write_header = not os.path.exists(file_path)

results_df.to_csv(file_path, mode="a", index=False, header=write_header)


os.makedirs("saved_models", exist_ok=True)
dump(model, "saved_models/catboost_top15_base.joblib")