import pandas as pd
from pathlib import Path

DATA_DIR = Path("D:/amexxx/data")
OLD_FEATURE_FILE = DATA_DIR / "clean_feature.txt"
NEW_TRAIN_FILE = DATA_DIR / "train_with_new_feats.parquet"
OUTPUT_FILE = DATA_DIR / "clean_feature_v2.txt"

# Load previous feature list
with open(OLD_FEATURE_FILE) as f:
    old_features = f.read().splitlines()

# Load updated train data with new features
df = pd.read_parquet(NEW_TRAIN_FILE)

# Filter numeric columns only
df = df.select_dtypes(include=["number"])

# Final features = intersection of old + numeric columns + engineered
new_features = list(df.columns)
final_features = sorted(set(old_features).union(new_features))

# Save to file
with open(OUTPUT_FILE, "w") as f:
    for feat in final_features:
        f.write(f"{feat}\n")

print(f"✅ Saved updated feature list to: {OUTPUT_FILE}")
print(f"🧮 Total features: {len(final_features)}")
