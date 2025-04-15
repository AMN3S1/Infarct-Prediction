import os

import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("top15_features_dataset.csv")
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]

# Convert object columns to category for XGBoost
for col in X.select_dtypes(include="object"):
    X[col] = X[col].astype("category")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = XGBClassifier(
    booster='gbtree',
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    enable_categorical=True,
    random_state=42
)

# Hyperparameter space
params = {
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "gamma": [0, 1, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Search
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    scoring="f1",
    cv=3,
    n_iter=10,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)

# Evaluate
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print("Best params:", search.best_params_)
print(classification_report(y_test, y_pred))


xgb_optimized_result = pd.DataFrame([{
    "model": "XGBoost (optimized)",
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}])

file_path = "model_results_top15.csv"
write_header = not os.path.exists(file_path)
xgb_optimized_result.to_csv(file_path, mode="a", index=False, header=write_header)


os.makedirs("saved_models", exist_ok=True)
dump(best_model, "saved_models/xgboost_top15_optimized.joblib")