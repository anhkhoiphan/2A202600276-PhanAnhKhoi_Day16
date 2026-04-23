import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

print("=== LightGBM Credit Card Fraud Benchmark ===\n")

t0 = time.time()
df = pd.read_csv("creditcard.csv")
load_time = time.time() - t0
print(f"Load data:  {load_time:.2f}s  ({len(df):,} rows)")

X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "n_jobs": -1,
    "verbose": -1,
}

t1 = time.time()
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])
train_time = time.time() - t1
print(f"Training:   {train_time:.2f}s  (best iteration: {model.best_iteration_})")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc      = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall   = recall_score(y_test, y_pred)

# Inference latency
single_row = X_test.iloc[:1]
timings = [time.time() for _ in range(1001)]
for i in range(1000):
    model.predict_proba(single_row)
    timings[i+1] = time.time()
latency_ms = np.mean([(timings[i+1] - timings[i]) * 1000 for i in range(1000)])

# Throughput
batch = X_test.iloc[:1000]
t2 = time.time()
model.predict_proba(batch)
throughput_time = time.time() - t2

print(f"\n--- Results ---")
print(f"AUC-ROC:               {auc:.6f}")
print(f"Accuracy:              {accuracy:.6f}")
print(f"F1-Score:              {f1:.6f}")
print(f"Precision:             {precision:.6f}")
print(f"Recall:                {recall:.6f}")
print(f"Latency (1 row):       {latency_ms:.3f} ms")
print(f"Throughput (1000 rows):{throughput_time*1000:.1f} ms")

result = {
    "load_time_s": round(load_time, 3),
    "train_time_s": round(train_time, 3),
    "best_iteration": model.best_iteration_,
    "auc_roc": round(auc, 6),
    "accuracy": round(accuracy, 6),
    "f1_score": round(f1, 6),
    "precision": round(precision, 6),
    "recall": round(recall, 6),
    "inference_latency_1row_ms": round(latency_ms, 3),
    "inference_throughput_1000rows_ms": round(throughput_time * 1000, 1),
}

with open("benchmark_result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nSaved to benchmark_result.json")
