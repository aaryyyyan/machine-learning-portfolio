
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42
OUT = "outputs_digits"
os.makedirs(OUT, exist_ok=True)
results = {}

# ---------------------------------------------------------------- 1. Load and inspect
digits = load_digits()
X, y = digits.data, digits.target
results["images"] = int(X.shape[0])
results["pixels_per_image"] = int(X.shape[1])
results["classes"] = int(len(np.unique(y)))
results["pixel_min_max"] = [float(X.min()), float(X.max())]
results["missing_values"] = int(np.isnan(X).sum())
results["duplicate_images"] = int(pd.DataFrame(X).duplicated().sum())
counts = pd.Series(y).value_counts()
results["images_per_class_min_max"] = [int(counts.min()), int(counts.max())]
print(f"Images: {X.shape[0]}  Pixels: {X.shape[1]}  Missing: {results['missing_values']}  "
      f"Duplicates: {results['duplicate_images']}  Per class: {counts.min()}-{counts.max()}")

# ---------------------------------------------------------------- 2. EDA
fig, axes = plt.subplots(2, 10, figsize=(12, 3))
for ax, img, label in zip(axes.ravel(), digits.images[:20], y[:20]):
    ax.imshow(img, cmap="gray_r"); ax.set_title(str(label)); ax.axis("off")
plt.suptitle("Sample images"); plt.tight_layout(); plt.savefig(f"{OUT}/01_samples.png", dpi=150); plt.close()

pca = PCA().fit(X / 16.0)
cum = np.cumsum(pca.explained_variance_ratio_)
results["pca_components_for_95pct_variance"] = int(np.argmax(cum >= 0.95) + 1)
plt.figure(figsize=(5, 3.5)); plt.plot(range(1, len(cum) + 1), cum)
plt.axhline(0.95, color="red", ls="--"); plt.xlabel("Components"); plt.ylabel("Cumulative variance")
plt.title("PCA variance explained"); plt.tight_layout(); plt.savefig(f"{OUT}/02_pca.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 3. Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
results["train_size"], results["test_size"] = len(X_train), len(X_test)

# ---------------------------------------------------------------- 4/5. Pipelines + cross-validated comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=SEED),
    "SVM (RBF)": SVC(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=SEED),
}
pipes = {n: Pipeline([("scale", StandardScaler()), ("model", m)]) for n, m in models.items()}
rows = []
for name, pipe in pipes.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    rows.append({"model": name, "cv_accuracy": scores.mean(), "cv_std": scores.std()})
cv_table = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False).round(4)
print("\nCross-validation (5-fold, training data only)\n", cv_table.to_string(index=False))
cv_table.to_csv(f"{OUT}/model_comparison_cv.csv", index=False)
results["cv_comparison"] = cv_table.to_dict(orient="records")

# ---------------------------------------------------------------- 6. Tune the neural network
grid = {
    "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "model__alpha": [1e-4, 1e-3, 1e-2],
}
gs = GridSearchCV(pipes["Neural Network (MLP)"], grid, cv=cv, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)
results["tuning"] = {"candidates_tested": len(gs.cv_results_["params"]),
                     "best_cv_accuracy": round(gs.best_score_, 4),
                     "best_params": {k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v)
                                     for k, v in gs.best_params_.items()}}
print(f"\nTuned neural network: best CV accuracy {gs.best_score_:.4f} with {gs.best_params_}")

# the final model is whichever model has the best cross-validation accuracy (tuned MLP vs best untuned model)
best_untuned_name = cv_table.iloc[0]["model"]
if gs.best_score_ >= cv_table.iloc[0]["cv_accuracy"]:
    final_name, final_model = "Neural Network (MLP), tuned", gs.best_estimator_
else:
    final_name, final_model = best_untuned_name, pipes[best_untuned_name].fit(X_train, y_train)
results["final_model"] = final_name

# ---------------------------------------------------------------- 7. Final evaluation on the untouched test set
pred = final_model.predict(X_test)
results["test_metrics"] = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
    "wrong_predictions": int((pred != y_test).sum()),
    "test_images": int(len(y_test)),
}
mlp_pred = gs.best_estimator_.predict(X_test)
results["tuned_neural_network_test"] = {"accuracy": round(accuracy_score(y_test, mlp_pred), 4),
                                         "wrong_predictions": int((mlp_pred != y_test).sum())}
per_class = classification_report(y_test, pred, output_dict=True)
results["lowest_class_f1"] = {"digit": min((str(d) for d in range(10)), key=lambda d: per_class[d]["f1-score"]),
                              "f1": round(min(per_class[str(d)]["f1-score"] for d in range(10)), 4)}
print(f"\nFinal model: {final_name}\n", classification_report(y_test, pred))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion matrix (test set)"); plt.tight_layout()
plt.savefig(f"{OUT}/03_confusion_matrix.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 8. Error analysis
cm = confusion_matrix(y_test, pred)
np.fill_diagonal(cm, 0)
pairs = sorted(((int(cm[i, j]), i, j) for i in range(10) for j in range(10) if cm[i, j] > 0), reverse=True)
results["most_confused_pairs_true_to_predicted"] = [{"true": i, "predicted": j, "count": c} for c, i, j in pairs[:3]]
wrong_idx = np.where(pred != y_test)[0][:10]
if len(wrong_idx):
    fig, axes = plt.subplots(1, len(wrong_idx), figsize=(1.6 * len(wrong_idx), 2.2))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, wrong_idx):
        ax.imshow(X_test[k].reshape(8, 8), cmap="gray_r"); ax.axis("off")
        ax.set_title(f"true {y_test[k]}\npred {pred[k]}", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/04_misclassified.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 9. Save
joblib.dump(final_model, f"{OUT}/final_model.joblib")
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved model, figures and results.json in ./" + OUT)
