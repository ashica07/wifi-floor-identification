import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import load_config

def main():
    cfg = load_config()
    raw_path = Path(cfg["paths"]["csv_dir"]) / cfg["datasets"]["full"]
    out_path = Path(cfg["paths"]["csv_dir"]) / cfg["datasets"]["balanced"]
    
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
    df = df.sort_values(["trial", "session", "position_timestamp"]).reset_index(drop=True)
    
    WINDOW = 5
    df["p_med"] = (df.groupby(["trial", "session"])["pressure_hpa"]
                     .transform(lambda s: s.rolling(WINDOW, center=True, min_periods=1).median()))
    df["p_resid"] = df["pressure_hpa"] - df["p_med"]
    df["p_clean"] = df["pressure_hpa"].where(df["p_resid"].abs() <= 1.0, df["p_med"])
    
    df.to_csv(out_path, index=False)
    print(f"✅ Balanced dataset saved to: {out_path}")

if __name__ == "__main__":
    main()