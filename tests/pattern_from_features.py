# tests/test_add_pattern_from_features.py
from pathlib import Path
import pandas as pd

from tools.api.patterns import add_patterns_to_parquet

def main():
    ROOT = Path(__file__).resolve().parents[1]  # Subcellular_Feature/
    in_parquet = ROOT / "data" / "sim_features.parquet"
    out_parquet = ROOT / "data" / "sim_features_with_pattern.parquet"
    model_path = ROOT / "models" / "multiclass_xgb_8class_prop075_final_from_cv.joblib"

    print("in_parquet:", in_parquet, "exists:", in_parquet.exists())
    print("model_path:", model_path, "exists:", model_path.exists())

    out = add_patterns_to_parquet(
        in_parquet=str(in_parquet),
        out_parquet=str(out_parquet),
        model_path=str(model_path),
    )
    print("saved:", out)

    df = pd.read_parquet(out_parquet)
    print("shape:", df.shape)

    new_cols = [c for c in df.columns if c.startswith("p_")] + ["pattern_top1", "pattern_multi"]
    print("new cols:", new_cols)

    print(df[["cell", "gene", "pattern_top1", "pattern_multi"]].head())

if __name__ == "__main__":
    main()