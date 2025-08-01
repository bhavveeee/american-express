import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from pathlib import Path
import numpy as np
from xgboost.callback import EarlyStopping
from packaging import version

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
train = pd.read_parquet(DATA_DIR / "train_with_new_feats_v2.parquet")
test = pd.read_parquet(DATA_DIR / "test_with_new_feats_v2.parquet")
meta = pd.read_parquet(DATA_DIR / "offer_metadata.parquet")

# Merge metadata
meta = meta.rename(columns={"id3": "id5"})
for df in [train, test, meta]:
    for col in ["id2", "id3", "id5"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
for col in ["f374", "f378"]:
    if col in meta.columns:
        meta.drop(columns=[col], inplace=True)

train = train.merge(meta, on="id5", how="left")
test = test.merge(meta, on="id5", how="left")

# Load feature list
with open(DATA_DIR / "damnit.txt") as f:
    all_features = f.read().splitlines()
actual_features = [f for f in all_features if f in train.columns and f in test.columns]
print(f"🧮 Using {len(actual_features)} features")

# Prepare data
X = train[actual_features].apply(pd.to_numeric, errors='coerce')
y = train["y"]
X_test = test[actual_features].apply(pd.to_numeric, errors='coerce')
id_train = train[["id1", "id2", "id3", "id5"]].copy()
id_test = test[["id1", "id2", "id3", "id5"]].copy()

# LightGBM parameters
lgb_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "binary_logloss",
    "verbosity": -1,
    "learning_rate": 0.010344433943220351,
    "num_leaves": 151,
    "max_depth": 11,
    "min_child_samples": 91,
    "feature_fraction": 0.6267452541115612,
    "bagging_fraction": 0.8863138717757151,
    "lambda_l1": 0.6326134468561784,
    "lambda_l2": 1.643862056314735,
    "min_split_gain": 0.21507485491413633,
    "seed": 42,
}

# XGBoost parameters (for GPU)
xgb_params = {
    "objective": "binary:logistic",
    "learning_rate": 0.01,
    "max_depth": 11,
    "min_child_weight": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "lambda": 1.5,
    "alpha": 0.6,
    "n_estimators": 10000,
    "tree_method": "hist",
    "device": "cuda",  # change to "auto" if no GPU
    "verbosity": 0,
    "random_state": 42,
}

# Cross-validation
kf = StratifiedKFold(n_splits=16, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
best_iters = []
loglosses, aucs, map7s = [], [], []

print("\n🚀 Starting cross-validation...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n📂 Fold {fold}")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    id_val = id_train.iloc[val_idx].copy()

    # LightGBM training
    model_lgb = lgb.train(
        lgb_params,
        lgb.Dataset(X_tr, y_tr),
        valid_sets=[lgb.Dataset(X_val, y_val)],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(500)
        ]
    )
    lgb_preds = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    best_iters.append(model_lgb.best_iteration)

    # XGBoost training (callbacks handled properly for v3+)
    model_xgb = xgb.XGBClassifier(
        **xgb_params,
        callbacks=[EarlyStopping(rounds=100)]
    )
    model_xgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_preds = model_xgb.predict_proba(X_val)[:, 1]

    # Blend predictions
    final_preds = 0.55 * lgb_preds + 0.45 * xgb_preds
    oof_preds[val_idx] = final_preds

    # Metrics
    ll = log_loss(y_val, final_preds)
    auc = roc_auc_score(y_val, final_preds)
    id_val["pred"] = final_preds
    id_val["y"] = y_val.values
    true = id_val[id_val["y"] == 1].groupby("id3")["id5"].first()
    topk = id_val.groupby("id3").apply(
        lambda x: x.sort_values("pred", ascending=False).head(7)
    ).reset_index(drop=True)
    pred = topk.groupby("id3")["id5"].apply(list)
    true = true[true.index.isin(pred.index)]
    pred = pred[pred.index.isin(true.index)]
    map7 = mapk(true.tolist(), pred.tolist(), k=7)

    print(f"✅ Fold {fold}: LogLoss={ll:.5f} | AUC={auc:.5f} | MAP@7={map7:.5f}")
    loglosses.append(ll)
    aucs.append(auc)
    map7s.append(map7)

# Final validation metrics
print("\n📊 16-Fold CV Results:")
print(f"LogLoss: {np.mean(loglosses):.5f} ± {np.std(loglosses):.5f}")
print(f"AUC:     {np.mean(aucs):.5f} ± {np.std(aucs):.5f}")
print(f"MAP@7:   {np.mean(map7s):.5f} ± {np.std(map7s):.5f}")

# Train final model on all training data
print("\n⚙️  Training final models...")
final_lgb = lgb.train(
    lgb_params,
    lgb.Dataset(X, y),
    num_boost_round=int(np.mean(best_iters))
)

final_xgb = xgb.XGBClassifier(
    **xgb_params
)
final_xgb.fit(X, y)

# Predict test set and blend
lgb_test_preds = final_lgb.predict(X_test, num_iteration=final_lgb.best_iteration)
xgb_test_preds = final_xgb.predict_proba(X_test)[:, 1]
test["pred"] = 0.55 * lgb_test_preds + 0.45 * xgb_test_preds

# Save submission
id_test["pred"] = test["pred"].fillna(0.0)
submission_path = DATA_DIR / "submission_blend_v2"
id_test.to_parquet(submission_path.with_suffix(".parquet"), index=False)
id_test.to_csv(submission_path.with_suffix(".csv"), index=False)

print(f"\n✅ Final submission saved! Rows: {len(id_test)}")
