import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, plot_importance
from sklearn.metrics import (classification_report, roc_auc_score,
RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay)
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("data/ai_training_edges_full_enhanced.csv")

# Sanity check
real_count = df["is_real_rail"].sum()
fake_count = len(df) - real_count
print(f"[INFO] Real: {real_count}, Fake: {fake_count}, Ratio = {real_count / fake_count:.2f}")

# Features
features = [
    "length_m", "avg_elevation_change", "avg_population_density",
    "elevation_per_meter", "closeness_avg", "betweenness_avg",
    "curvature_deg", "curvature_per_meter"
]
X = df[features]
y = df["is_real_rail"]

# Leave-one-city-out validation
df["city"] = df["city"].fillna("unknown")
target_city = "Berlin, Germany"
train_df = df[df["city"] != target_city]
val_df = df[df["city"] == target_city]

X_train = train_df[features]
y_train = train_df["is_real_rail"]
X_test = val_df[features]
y_test = val_df["is_real_rail"]

# Train model
model = LGBMClassifier(
    force_col_wise=True,
    n_estimators=500,
    max_depth=14,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=30)]
)

# Generate predictions
y_probs = model.predict_proba(X_test)[:, 1]

# Find optimal threshold using F1 score
prec, rec, thresh = precision_recall_curve(y_test, y_probs)
f1 = 2 * (prec * rec) / (prec + rec)
best_idx = np.argmax(f1)
best_threshold = thresh[best_idx]

print(f"[INFO] Best classification threshold (by F1): {best_threshold:.2f}")
y_pred = (y_probs >= best_threshold).astype(int)

# Print metrics
print("\n[RESULT] Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_probs))

# Save model
joblib.dump(model, "models/enhanced_edge_classifier.pkl")
print("Model saved to models/enhanced_edge_classifier.pkl")

# ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.savefig("models/roc_curve.png")
print("ROC curve saved to models/roc_curve.png")

# Plot Precision-Recall Curve
disp = PrecisionRecallDisplay(precision=prec, recall=rec)
disp.plot()
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/precision_recall_curve.png")
print("Precision-Recall curve saved to models/precision_recall_curve.png")

# Plot feature importance
plot_importance(model, importance_type="gain", max_num_features=20, figsize=(10, 6))
plt.title("Feature Importance by Gain")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/feature_importance.png")
print("Feature Importance Plot saved to models/feature_importance.png")
plt.show()

# Save scored edges
df["pred_score"] = model.predict_proba(df[features])[:, 1]
df.to_csv("models/scored_edges.csv", index=False)
print("Scored edges saved to models/scored_edges.csv")
