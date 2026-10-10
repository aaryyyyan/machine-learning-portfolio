
import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
OUT = "outputs_segmentation"
os.makedirs(OUT, exist_ok=True)
results = {}


# ---------------------------------------------------------------- 1. Data (synthetic by default)
def generate_transactions(n_customers=3000, seed=SEED):
    """Synthetic 12-month retail log. Customers belong to hidden behaviour groups (the model never sees these)."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
    # group: (share, mean orders, mean order value, how recent the last purchase tends to be: max days since end)
    groups = {
        "frequent_high_spend": (0.15, 22, 95, 30),
        "regular":             (0.35, 8, 55, 90),
        "new":                 (0.15, 2, 40, 45),
        "lapsed_big_spenders": (0.15, 9, 80, 330),
        "occasional_low":      (0.20, 2, 25, 300),
    }
    rows, cid = [], 0
    for share, orders, aov, recency_cap in groups.values():
        for _ in range(int(n_customers * share)):
            cid += 1
            n = max(1, rng.poisson(orders))
            last_day = int(rng.integers(0, recency_cap + 1))            # days before the end of the year
            first_day = int(rng.integers(last_day, 365))                # earliest purchase of this customer
            days_before_end = rng.integers(last_day, first_day + 1, size=n)
            days_before_end[0] = last_day                                # make sure the last purchase is exact
            amounts = rng.lognormal(mean=np.log(aov), sigma=0.45, size=n)
            for d, a in zip(days_before_end, amounts):
                rows.append((f"C{cid:05d}", end - pd.Timedelta(days=int(d)), round(float(a), 2)))
    return pd.DataFrame(rows, columns=["customer_id", "date", "amount"])


if len(sys.argv) > 1:
    tx = pd.read_csv(sys.argv[1], parse_dates=["date"])
    results["data_source"] = f"user CSV: {sys.argv[1]}"
else:
    tx = generate_transactions()
    results["data_source"] = "synthetic retail transactions (seed 42)"
tx.to_csv(f"{OUT}/transactions.csv", index=False)

results["transactions"] = int(len(tx))
results["customers"] = int(tx["customer_id"].nunique())
results["date_range"] = [str(tx["date"].min().date()), str(tx["date"].max().date())]
results["missing_values"] = int(tx.isna().sum().sum())
results["duplicate_rows"] = int(tx.duplicated().sum())
results["negative_or_zero_amounts"] = int((tx["amount"] <= 0).sum())
tx = tx.dropna().drop_duplicates()
tx = tx[tx["amount"] > 0]
print(f"{results['transactions']:,} transactions from {results['customers']:,} customers "
      f"({results['date_range'][0]} to {results['date_range'][1]}); "
      f"missing {results['missing_values']}, duplicates {results['duplicate_rows']}")

# ---------------------------------------------------------------- 2. RFM feature engineering
snapshot = tx["date"].max() + pd.Timedelta(days=1)
rfm = tx.groupby("customer_id").agg(
    recency=("date", lambda d: (snapshot - d.max()).days),
    frequency=("date", "count"),
    monetary=("amount", "sum"),
)
total_revenue = rfm["monetary"].sum()
results["total_revenue"] = round(float(total_revenue), 2)

# ---------------------------------------------------------------- 3. EDA
results["skewness_raw"] = {c: round(float(rfm[c].skew()), 2) for c in rfm}
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for j, c in enumerate(rfm):
    rfm[c].plot.hist(bins=40, ax=axes[0, j], color="#4C72B0"); axes[0, j].set_title(f"{c} (raw)")
    np.log1p(rfm[c]).plot.hist(bins=40, ax=axes[1, j], color="#55A868"); axes[1, j].set_title(f"{c} (log)")
plt.tight_layout(); plt.savefig(f"{OUT}/01_rfm_distributions.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 4. Preprocessing
rfm_log = np.log1p(rfm)
results["skewness_after_log"] = {c: round(float(rfm_log[c].skew()), 2) for c in rfm_log}
scaler = StandardScaler()
X = scaler.fit_transform(rfm_log)

# ---------------------------------------------------------------- 5. Choose k
ks = list(range(2, 11))
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
# Rule (decided before looking at the segments): a business can act on 3-6 segments, so pick the k in that range
# with the best silhouette score. k=2 only splits customers into "high" and "low" value, which is too coarse.
candidates = [k for k in ks if 3 <= k <= 6]
best_k = max(candidates, key=lambda k: sil[ks.index(k)])
results["k_selection_rule"] = "best silhouette among k = 3..6"
results["silhouette_by_k"] = {str(k): round(s, 4) for k, s in zip(ks, sil)}
results["chosen_k"] = best_k
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, "o-", color="#C44E52"); ax[1].axvline(best_k, ls="--", color="grey")
ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(f"{OUT}/02_choose_k.png", dpi=150); plt.close()
print(f"Chosen k (best silhouette among 3-6): {best_k}  (silhouette {sil[ks.index(best_k)]:.3f}; k=2 would be {sil[0]:.3f})")

# ---------------------------------------------------------------- 6. Final model + quality metrics
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
rfm["cluster"] = km.labels_
results["cluster_quality"] = {
    "silhouette": round(silhouette_score(X, km.labels_), 4),
    "davies_bouldin": round(davies_bouldin_score(X, km.labels_), 4),
    "calinski_harabasz": round(calinski_harabasz_score(X, km.labels_), 1),
}

# ---------------------------------------------------------------- 7. Stability
seed_labels = [KMeans(n_clusters=best_k, n_init=10, random_state=s).fit(X).labels_ for s in (1, 2, 3, 4, 5)]
ari_seeds = [adjusted_rand_score(km.labels_, lab) for lab in seed_labels]
rng = np.random.default_rng(SEED)
ari_boot = []
for _ in range(10):
    idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
    boot = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X[idx])
    ari_boot.append(adjusted_rand_score(km.labels_[idx], boot.labels_))
results["stability"] = {"mean_ARI_different_seeds": round(float(np.mean(ari_seeds)), 4),
                        "mean_ARI_80pct_bootstrap_samples": round(float(np.mean(ari_boot)), 4)}

# ---------------------------------------------------------------- 8. Profile and name the segments
profile = rfm.groupby("cluster").agg(customers=("monetary", "size"),
                                     avg_recency_days=("recency", "mean"),
                                     avg_orders=("frequency", "mean"),
                                     avg_spend=("monetary", "mean"),
                                     revenue=("monetary", "sum"))
profile["pct_customers"] = profile["customers"] / len(rfm) * 100
profile["pct_revenue"] = profile["revenue"] / total_revenue * 100


def name_segment(row, med):
    recent, frequent, big = row.avg_recency_days <= med["r"], row.avg_orders >= med["f"], row.avg_spend >= med["m"]
    if recent and frequent and big:
        return "Champions"
    if frequent and big and not recent:
        return "At-risk big spenders"
    if recent and frequent:
        return "Loyal customers"
    if recent and not frequent:
        return "New / recent customers"
    return "Low-value / inactive"


med = {"r": rfm["recency"].median(), "f": rfm["frequency"].median(), "m": rfm["monetary"].median()}
profile["segment"] = profile.apply(name_segment, axis=1, med=med)
profile = profile.round(1).sort_values("pct_revenue", ascending=False)
profile.to_csv(f"{OUT}/segment_profile.csv")
print("\n", profile[["segment", "customers", "pct_customers", "avg_recency_days", "avg_orders", "avg_spend", "pct_revenue"]]
      .to_string())
results["segments"] = profile.reset_index().to_dict(orient="records")
top = profile.iloc[0]
results["top_segment"] = {"name": top["segment"], "pct_customers": float(top["pct_customers"]),
                          "pct_revenue": float(top["pct_revenue"])}

# ---------------------------------------------------------------- 9. Visualise + save
pca = PCA(n_components=2, random_state=SEED).fit(X)
xy = pca.transform(X)
results["pca_2d_variance_explained_pct"] = round(float(pca.explained_variance_ratio_.sum() * 100), 1)
names = rfm["cluster"].map(profile["segment"])
plt.figure(figsize=(7, 5.5))
for seg in names.unique():
    m = (names == seg).values
    plt.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, label=seg)
plt.legend(markerscale=2); plt.title("Customer segments (PCA view)"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig(f"{OUT}/03_segments_pca.png", dpi=150); plt.close()

rfm.assign(segment=names).to_csv(f"{OUT}/customers_with_segments.csv")
joblib.dump({"scaler": scaler, "kmeans": km}, f"{OUT}/kmeans_model.joblib")
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved data, model, figures and results.json in ./{OUT}")
