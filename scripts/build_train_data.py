import pandas as pd
import numpy as np

print("📦 Loading raw data...")
df = pd.read_parquet("D:/amexxx/data/train_data.parquet")

# Fix dtype
df["y"] = df["y"].astype(int)

print("✅ Positive samples (clicked offers)...")
df_pos = df[df["y"] == 1].copy()
print(f"➡️ {len(df_pos)} positive samples")

print("🔁 Generating negative samples...")
# For each user (id2), sample negative offers (y=0)
df_neg = (
    df[df["y"] == 0]
    .groupby("id2")
    .apply(lambda x: x.sample(n=min(len(x), 7), random_state=42))
    .reset_index(drop=True)
)
print(f"✅ Negative samples: {len(df_neg)}")

print("🧱 Combining positives + negatives...")
df_final = pd.concat([df_pos, df_neg], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ y value counts:")
print(df_final["y"].value_counts().to_frame("count"))

print("🔍 Sample clicked rows:")
print(df_final[df_final["y"] == 1].head())

print("💾 Saving to D:/amexxx/data/train_data1.parquet...")
df_final.to_parquet("D:/amexxx/data/train_data1.parquet", index=False)

print("📊 Offer count per user:")
print(df_final.groupby("id2")["id3"].count().describe())

print("✅ Done!")
