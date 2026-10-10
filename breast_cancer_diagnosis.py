
import json
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
results = {}

# ---------------------------------------------------------------- 1. Load and inspect
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# In scikit-learn's copy of this dataset 0 = malignant, 1 = benign. We flip it so that 1 = malignant (the class we care about).
df["malignant"] = (df["target"] == 0).astype(int)
df = df.drop(columns="target")

results["rows"], results["columns"] = df.shape[0], df.shape[1] - 1
results["missing_values"] = int(df.isna().sum().sum())
results["duplicate_rows"] = int(df.duplicated().sum())
results["malignant_share_pct"] = round(df["malignant"].mean() * 100, 1)
print(f"Rows: {df.shape[0]}  Features: {df.shape[1] - 1}  Missing: {results['missing_values']}  "
      f"Duplicates: {results['duplicate_rows']}  Malignant: {results['malignant_share_pct']}%")

# ---------------------------------------------------------------- 2. EDA
fig, ax = plt.subplots(figsize=(4, 3.5))
df["malignant"].map({0: "Benign", 1: "Malignant"}).value_counts().plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_title("Class balance"); ax.set_ylabel("Patients"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(f"{OUT}/01_class_balance.png", dpi=150); plt.close()

mean_cols = [c for c in df.columns if c.startswith("mean ")]
corr_with_target = df[mean_cols + ["malignant"]].corr()["malignant"].drop("malignant").sort_values(ascending=False)
results["top_correlated_features"] = {k: round(v, 2) for k, v in corr_with_target.head(3).items()}

plt.figure(figsize=(9, 7))
sns.heatmap(df.drop(columns="malignant").corr(), cmap="coolwarm", center=0, xticklabels=False, yticklabels=False)
plt.title("Feature correlation heatmap"); plt.tight_layout()
plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=150); plt.close()

corr_matrix = df.drop(columns="malignant").corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
results["feature_pairs_with_corr_above_0.9"] = int((upper > 0.9).sum().sum())

# ---------------------------------------------------------------- 3. Stratified split
X, y = df.drop(columns="malignant"), df["malignant"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
results["train_size"], results["test_size"] = len(X_train), len(X_test)

# ---------------------------------------------------------------- 4/5. Pipelines + cross-validated comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=SEED),
    "SVM (RBF)": SVC(probability=True, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
pipes = {name: Pipeline([("scale", StandardScaler()), ("model", m)]) for name, m in models.items()}

cv_rows = []
for name, pipe in pipes.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
    cv_rows.append({"model": name,
                    "cv_accuracy": scores["test_accuracy"].mean(),
                    "cv_recall": scores["test_recall"].mean(),
                    "cv_roc_auc": scores["test_roc_auc"].mean()})
cv_table = pd.DataFrame(cv_rows).sort_values("cv_recall", ascending=False).round(4)
print("\nCross-validation (5-fold, training data only)\n", cv_table.to_string(index=False))
cv_table.to_csv(f"{OUT}/model_comparison_cv.csv", index=False)
results["cv_comparison"] = cv_table.to_dict(orient="records")

# ---------------------------------------------------------------- 6. Hyper-parameter tuning
grids = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)": {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.001]},
}
tuned = {}
for name, grid in grids.items():
    gs = GridSearchCV(pipes[name], grid, cv=cv, scoring="recall", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs
    print(f"Tuned {name}: best CV recall {gs.best_score_:.4f} with {gs.best_params_}")
    results.setdefault("tuning", {})[name] = {"best_cv_recall": round(gs.best_score_, 4),
                                               "best_params": {k.replace('model__', ''): v for k, v in gs.best_params_.items()}}

# pick the tuned model with the best CV recall as the final model
best_name = max(tuned, key=lambda n: tuned[n].best_score_)
final_model = tuned[best_name].best_estimator_
results["final_model"] = best_name

# ---------------------------------------------------------------- 7. Final evaluation on the untouched test set
pred = final_model.predict(X_test)
proba = final_model.predict_proba(X_test)[:, 1]
results["test_metrics"] = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
}
tn, fp, fn, tp = pd.crosstab(y_test, pred).values.ravel()
results["confusion_matrix"] = {"true_negative": int(tn), "false_positive": int(fp),
                               "false_negative": int(fn), "true_positive": int(tp)}
print(f"\nFinal model: {best_name}\n", classification_report(y_test, pred, target_names=["Benign", "Malignant"]))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Benign", "Malignant"], ax=ax[0], colorbar=False)
ax[0].set_title("Confusion matrix (test set)")
RocCurveDisplay.from_predictions(y_test, proba, ax=ax[1]); ax[1].set_title("ROC curve (test set)")
plt.tight_layout(); plt.savefig(f"{OUT}/03_test_evaluation.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 8. Interpretation
perm = permutation_importance(final_model, X_test, y_test, scoring="recall", n_repeats=30, random_state=SEED, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
results["top_features_permutation_importance"] = [k for k in imp.head(5).index]
imp.head(10).sort_values().plot.barh(figsize=(7, 4.5), color="#4C72B0")
plt.title("Top 10 features (permutation importance)"); plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_importance.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 9. Save
joblib.dump(final_model, f"{OUT}/final_model.joblib")
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved model, figures and results.json in ./outputs")
