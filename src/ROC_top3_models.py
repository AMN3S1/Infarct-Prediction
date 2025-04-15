import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from joblib import load

# Load dataset
df = pd.read_csv("top15_features_dataset.csv")
X = df.drop("heart_attack", axis=1)
y = df["heart_attack"]


for col in X.select_dtypes(include="object"):
    X[col] = X[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


catboost_base = load("saved_models/catboost_top15_base.joblib")
catboost_opt = load("saved_models/catboost_top15_optimized.joblib")
xgb_opt = load("saved_models/xgboost_top15_optimized.joblib")


y_proba_catboost_base = catboost_base.predict_proba(X_test)[:, 1]
y_proba_catboost_opt = catboost_opt.predict_proba(X_test)[:, 1]
y_proba_xgb_opt = xgb_opt.predict_proba(X_test)[:, 1]

# ROC и AUC
fpr_cb_base, tpr_cb_base, _ = roc_curve(y_test, y_proba_catboost_base)
fpr_cb_opt, tpr_cb_opt, _ = roc_curve(y_test, y_proba_catboost_opt)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb_opt)

auc_cb_base = auc(fpr_cb_base, tpr_cb_base)
auc_cb_opt = auc(fpr_cb_opt, tpr_cb_opt)
auc_xgb = auc(fpr_xgb, tpr_xgb)


plt.figure(figsize=(10, 6))
plt.plot(fpr_cb_base, tpr_cb_base, label=f"CatBoost (Top 15) — AUC: {auc_cb_base:.3f}")
plt.plot(fpr_cb_opt, tpr_cb_opt, label=f"CatBoost (optimized) — AUC: {auc_cb_opt:.3f}")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (optimized) — AUC: {auc_xgb:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые для топ-3 моделей")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curves_top3_models.png")
