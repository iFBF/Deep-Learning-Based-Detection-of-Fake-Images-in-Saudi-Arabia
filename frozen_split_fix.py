import hashlib

# Load the fixed split shipped with the corrections dataset.
split_file = EVIDENCE / "corrected_development_manifest.csv"
assert hashlib.sha256(split_file.read_bytes()).hexdigest() == "15f17c6e5d7f6c5df0ce4ac0a6f09023b797e6710e3dc64870bfb3ecbf0ad138", "Fixed split file changed"
frozen = pd.read_csv(split_file, dtype={"sha256": str, "phash": str})

# Verify the fixed records against the images checked in this run.
assert frozen.relative_path.is_unique and frozen.sha256.is_unique
assert set(frozen.sha256) == set(clean.sha256), "Retained image population changed"
observed = all_old.set_index("relative_path").reindex(frozen.relative_path)
for field in ["sha256", "label", "group_id"]:
    assert observed[field].tolist() == frozen[field].tolist(), f"Image identity mismatch: {field}"

# Save this exact split for training and holdout preparation.
development = frozen.copy()
assert set(development.split) == {"train", "val"}
assert development.groupby("group_id").split.nunique().max() == 1
assert development.group_id.nunique() == 498
development.to_csv(PREPARED / "development_manifest.csv", index=False)

print(development.groupby("split").agg(images=("label", "size"), groups=("group_id", "nunique")))
print(pd.crosstab(development.split, development.label))
print("Frozen split: OK")
