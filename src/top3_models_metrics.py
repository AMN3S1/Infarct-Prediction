import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Метрики по трём моделям
data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "CatBoost (Top 15)": [0.7353, 0.6933, 0.6157, 0.6522],
    "CatBoost (optimized)": [0.7347, 0.6967, 0.5994, 0.6444],
    "XGBoost (optimized)": [0.7339, 0.6893, 0.6122, 0.6485]
}

# Создание DataFrame
df = pd.DataFrame(data)

# Построение barplot
metrics = df["Metric"]
x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, df["CatBoost (Top 15)"], width, label="CatBoost (Top 15)")
plt.bar(x, df["CatBoost (optimized)"], width, label="CatBoost (optimized)")
plt.bar(x + width, df["XGBoost (optimized)"], width, label="XGBoost (optimized)")

# Добавление подписей
plt.xticks(x, metrics)
plt.ylim(0.55, 0.75)
plt.ylabel("Score")
plt.title("Top 3 Models Comparison by Metrics")
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Сохранение
plt.savefig("top3_models_all_metrics.png")
