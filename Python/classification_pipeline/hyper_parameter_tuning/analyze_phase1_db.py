"""Analyze HPT Phase 1 Optuna DB — find best params + problematic trials."""
import optuna
import pandas as pd

DB_PATH = "../performance_reports/hpt_eurobert_210m_phase1.db"
STORAGE_URL = f"sqlite:///{DB_PATH}"
STUDY_NAME = "eurobert_210m_hpt_phase1"

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)

# All trials as DataFrame
df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration"))
df = df.sort_values("number")

print("=" * 80)
print("  HPT PHASE 1 — VOLLSTAENDIGE ANALYSE")
print("=" * 80)

# --- 1. Overview ---
print(f"\nTotal Trials: {len(df)}")
for state in df["state"].unique():
    count = (df["state"] == state).sum()
    print(f"  {state}: {count}")

# --- 2. All trials ---
print("\n" + "=" * 80)
print("  ALLE TRIALS (sortiert nach Nummer)")
print("=" * 80)
param_cols = [c for c in df.columns if c.startswith("params_")]
short_cols = ["number", "value", "state", "duration"] + param_cols
print(df[short_cols].to_string(index=False))

# --- 3. NaN / problematic trials ---
print("\n" + "=" * 80)
print("  PROBLEMATISCHE TRIALS (F1 < 0.5 oder NaN-verdaechtig)")
print("=" * 80)
problematic = df[(df["value"].isna()) | (df["value"] < 0.5)]
if len(problematic) > 0:
    print(problematic[short_cols].to_string(index=False))
    print(f"\n  -> {len(problematic)} problematische Trials gefunden")
else:
    print("  Keine problematischen Trials gefunden.")

# --- 4. Top trials ---
print("\n" + "=" * 80)
print("  TOP 5 TRIALS")
print("=" * 80)
completed = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
top5 = completed.head(5)
print(top5[short_cols].to_string(index=False))

# --- 5. Parameter ranges in top 5 ---
print("\n" + "=" * 80)
print("  PARAMETER-RANGES DER TOP 5 TRIALS")
print("=" * 80)
for col in param_cols:
    param_name = col.replace("params_", "")
    values = top5[col]
    if values.dtype in ["float64", "int64"]:
        print(f"  {param_name:35s}: min={values.min():.6g}  max={values.max():.6g}  "
              f"mean={values.mean():.6g}  std={values.std():.6g}")
    else:
        print(f"  {param_name:35s}: {values.value_counts().to_dict()}")

# --- 6. Parameter ranges in top 10 ---
print("\n" + "=" * 80)
print("  PARAMETER-RANGES DER TOP 10 TRIALS (fuer breiteren Ueberblick)")
print("=" * 80)
top10 = completed.head(10)
for col in param_cols:
    param_name = col.replace("params_", "")
    values = top10[col]
    if values.dtype in ["float64", "int64"]:
        print(f"  {param_name:35s}: min={values.min():.6g}  max={values.max():.6g}  "
              f"mean={values.mean():.6g}")
    else:
        print(f"  {param_name:35s}: {values.value_counts().to_dict()}")

# --- 7. Best trial details ---
print("\n" + "=" * 80)
print(f"  BESTER TRIAL: #{study.best_trial.number}")
print(f"  F1 Macro (CV Mean): {study.best_value:.4f}")
print("=" * 80)
for key, val in study.best_params.items():
    print(f"  {key}: {val}")

# --- 8. Suggested Phase 2 ranges ---
print("\n" + "=" * 80)
print("  VORGESCHLAGENE PHASE 2 SUCHBEREICHE (basierend auf Top 5)")
print("=" * 80)

# Calculate suggested ranges with small buffer
for col in param_cols:
    param_name = col.replace("params_", "")
    values = top5[col]
    if values.dtype == "float64":
        lo = values.min()
        hi = values.max()
        spread = hi - lo
        buffer = spread * 0.2  # 20% buffer
        suggested_lo = max(0, lo - buffer)
        suggested_hi = hi + buffer
        if param_name == "learning_rate":
            print(f"  \"{param_name}\": ({suggested_lo:.2e}, {suggested_hi:.2e}),  # log scale")
        else:
            print(f"  \"{param_name}\": ({suggested_lo:.4f}, {suggested_hi:.4f}),")
    elif values.dtype == "int64":
        lo = int(values.min())
        hi = int(values.max())
        print(f"  \"{param_name}\": ({lo}, {hi}),")
    else:
        unique = values.unique().tolist()
        print(f"  \"{param_name}\": {unique},")
