import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from utils import map7
import warnings
warnings.filterwarnings("ignore")

# --- Load dataset ---
df = pd.read_parquet("D:/amexxx/data/train_data2.parquet")
print("✅ Loaded train_data2.parquet", df.shape)

# --- Fix object types ---
for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = df[col].astype(float)
        except:
            df[col] = df[col].astype("category")

# --- Basic features (quick test) ---
base_features = pd.read_csv("D:/amexxx/data/clean_feature.txt", header=None)[0].tolist()
base_features = [f for f in base_features if f in df.columns]
test_features = base_features[:10]  # just 10 features for debugging

# --- Prepare CV ---
X = df[test_features]
y = df["y"]
meta_cols = df[["id1", "id2", "id3", "id5"]]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = []
oof_true = []
oof_id1, oof_id2, oof_id3, oof_id5 = [], [], [], []

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    meta_val = meta_cols.iloc[val_idx]

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val)

    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.01,
            "num_leaves": 64,
            "verbosity": -1,
        },
        train_set=dtrain,
        valid_sets=[dval],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds.extend(preds)
    oof_true.extend(y_val)
    oof_id1.extend(meta_val["id1"])
    oof_id2.extend(meta_val["id2"])
    oof_id3.extend(meta_val["id3"])
    oof_id5.extend(meta_val["id5"])

# --- Create MAP@7 input ---
df_val = pd.DataFrame({
    "id1": oof_id1,
    "id2": oof_id2,
    "id3": oof_id3,
    "id5": oof_id5,
    "pred": oof_preds
})

# --- Run MAP@7 ---
map_score = map7(df_val)
logloss = log_loss(oof_true, oof_preds)
auc = roc_auc_score(oof_true, oof_preds)
print(f"\n📊 Scores — MAP@7: {map_score:.6f} | Logloss: {logloss:.6f} | AUC: {auc:.6f}")

# --- DEBUG BLOCK ---
print("\n🔍 DEBUGGING MAP@7 INPUT")

print("▶ Total rows in df_val:", len(df_val))
print("▶ Unique id3s (users):", df_val['id3'].nunique())
print("▶ Average offers per id3:", len(df_val) / df_val['id3'].nunique())

id3_counts = df_val['id3'].value_counts()
low_offer_id3s = (id3_counts < 2).sum()
print("▶ id3s with <2 offers:", low_offer_id3s)

print("\n📄 Sample df_val:")
print(df_val.head())

print("\n📈 Prediction stats:")
print(df_val['pred'].describe())

click_match_rate = (df_val['id2'] == df_val['id5']).mean()
print(f"▶ Click match rate (id2 == id5): {click_match_rate:.4f}")

print("\n⭐ Top ranked offers:")
print(df_val.sort_values('pred', ascending=False).head(10))
