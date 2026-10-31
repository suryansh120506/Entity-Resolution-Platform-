"""
deterministic_clean.py — the part of "cleaning 51k rows" that needs no LLM,
no API call, and no chat pasting: whitespace/placeholder normalization and
collapsing EXACT (name, pincode) duplicate groups. Zero hallucination risk
because nothing is invented — only normalized or exactly matched.

Source file is never mutated: this writes new output files alongside it.
"""

import pandas as pd

SRC = r"C:\Academic_data_pipelines\raw_schools_uncleaned.csv"
PLACEHOLDER_PIN = {"000000", "0", "", "nan"}

df = pd.read_csv(SRC, dtype=str)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# 1. whitespace normalization (derived columns, originals untouched)
df["instituteName_norm"] = df["instituteName"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
df["pinCode_norm"] = df["pinCode"].fillna("").str.strip()
df.loc[df["pinCode_norm"].isin(PLACEHOLDER_PIN), "pinCode_norm"] = None

# 2. exact full-row duplicates (safe to drop outright)
exact_dupe_mask = df.duplicated(subset=[c for c in df.columns if c not in ("id",)], keep="first")
print(f"Exact full-row duplicates (all fields identical except id): {exact_dupe_mask.sum()}")

# 3. exact (name, real pincode) duplicate groups — same name AND same
#    non-placeholder pincode is about as close to "definitely the same
#    school" as you get without fuzzy matching or a human.
has_real_pin = df["pinCode_norm"].notna()
dupe_key = df["instituteName_norm"].str.upper() + "||" + df["pinCode_norm"].fillna("")
group_sizes = dupe_key[has_real_pin].value_counts()
dupe_groups = group_sizes[group_sizes > 1]
print(f"Exact (name, real pincode) duplicate groups: {len(dupe_groups)} groups, {dupe_groups.sum()} rows involved")

# build a Dirty_ID -> Master_ID map for just this unambiguous slice
mapping_rows = []
for key, size in dupe_groups.items():
    members = df[has_real_pin & (dupe_key == key)]
    master_id = members.iloc[0]["id"]
    for _, row in members.iterrows():
        mapping_rows.append({"dirty_id": row["id"], "master_id": master_id, "method": "EXACT_NAME_PINCODE"})

mapping_df = pd.DataFrame(mapping_rows)
mapping_df.to_csv(r"C:\Academic_data_pipelines\exact_dedup_mapping.csv", index=False)
df.drop(columns=["instituteName_norm", "pinCode_norm"]).to_csv(
    r"C:\Academic_data_pipelines\raw_schools_normalized.csv", index=False
)

n_placeholder_pin = (df["pinCode_norm"].isna() & df["pinCode"].notna()).sum()
print(f"\nRows with a placeholder pincode (normalized to missing): {n_placeholder_pin}")
print(f"Rows still needing real entity resolution (fuzzy name match, not exact): "
      f"~{len(df) - dupe_groups.sum() - exact_dupe_mask.sum()} at minimum")
print(f"\nWrote exact_dedup_mapping.csv ({len(mapping_df)} rows)")
print(f"Wrote raw_schools_normalized.csv ({len(df)} rows, source untouched)")
