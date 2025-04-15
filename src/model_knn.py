import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
df = pd.read_csv("preprocessed_data.csv")
X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify column types
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Build the full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=150))
])

# Train with progress display
print("Training K-Nearest Neighbors...")
for _ in tqdm(range(1), desc="Fitting model"):
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    duration = time.time() - start_time

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
n_neighbors = pipeline.named_steps['classifier'].n_neighbors

# Display results
print("\nKNN Results:")
print(f"Training time: {duration:.2f} seconds")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"n_neighbors:    {n_neighbors}")
# Save to results file
results_df = pd.DataFrame([{
    "model": f"KNN_{n_neighbors}",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "train_time_sec": duration
}])

results_file = "model_results.csv"
try:
    existing = pd.read_csv(results_file)
    results_df = pd.concat([existing, results_df], ignore_index=True)
except FileNotFoundError:
    pass

results_df.to_csv(results_file, index=False)
print(f"\nResults saved to {results_file}")
