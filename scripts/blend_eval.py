import pandas as pd
import numpy as np
from pathlib import Path

# MAP@7 scoring
def mapk(actual, predicted, k=7):
    score = 0.0
    for a, p in zip(actual, predicted):
        if len(p) > k:
            p = p[:k]
        score += int(a in p) / (p.index(a) + 1) if a in p else 0.0
    return score / len(actual)

# Paths
DATA_DIR = Path("D:/amexxx/data")
lgb = pd.read_csv(DATA_DIR / "submission_v6.csv")
xgb = pd.read_csv(DATA_DIR / "submission_xgb_v2.csv")
train = pd.read_parquet(DATA_DIR / "train_with_new_feats_v2.parquet")

# Convert id cols to string for safe merge
for df in [lgb, xgb]:
    for col in ["id1", "id2", "id3", "id5"]:
        df[col] = df[col].astype(str)

# Merge predictions
blend = lgb.merge(xgb[["id1", "pred"]], on="id1", suffixes=("_lgb", "_xgb"))

# Create true label lookup from train
true_labels = train[train["y"] == 1][["id3", "id5"]].drop_duplicates()
true_dict = dict(zip(true_labels["id3"], true_labels["id5"]))

# Store results
results = []

for alpha in np.linspace(0, 1, 11):  # 0.0 to 1.0
    blend["blend_pred"] = (1 - alpha) * blend["pred_lgb"] + alpha * blend["pred_xgb"]

    topk = (
        blend.groupby("id3")
        .apply(lambda x: x.sort_values("blend_pred", ascending=False).head(7))
        .reset_index(drop=True)
    )

    pred = topk.groupby("id3")["id5"].apply(list)
    true = pred.index.to_series().map(true_dict)

    score = mapk(true.tolist(), pred.tolist(), k=7)

    print(f"🔀 Blend ratio LGB:{1-alpha:.1f} / XGB:{alpha:.1f} → MAP@7: {score:.5f}")
    results.append((1 - alpha, alpha, score))

# Save results
results_df = pd.DataFrame(results, columns=["lgb_ratio", "xgb_ratio", "map7"])
results_df.to_csv(DATA_DIR / "blend_map7_scores.csv", index=False)
print("\n✅ Blending evaluation complete.")
