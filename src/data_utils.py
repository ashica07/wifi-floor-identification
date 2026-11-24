import numpy as np
import pandas as pd

def parse_fp(s):
    out = {}
    if not isinstance(s, str):
        return out
    for pair in s.split(";"):
        if ":" not in pair:
            continue
        b, r = pair.rsplit(":", 1)
        try:
            out[b] = float(r)
        except ValueError:
            continue
    return out

def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
    df = df[df["trial"] != 31].reset_index(drop=True)
    
    df = df.sort_values(["trial", "device", "position_timestamp"]).reset_index(drop=True)
    
    # Check if we are using the balanced (clean) dataset
    if "p_clean" in df.columns:
        df["p_smooth"] = (df.groupby(["trial", "device"])["p_clean"]
                          .transform(lambda s: s.rolling(5, center=True, min_periods=1).median()))
    else:
        df["p_smooth"] = (df.groupby(["trial", "device"])["pressure_hpa"]
                          .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))

    # Parse WiFi
    fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
    all_bssids = sorted({b for fp in fps for b in fp})
    bidx = {b: i for i, b in enumerate(all_bssids)}
    W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)
    for i, fp in enumerate(fps):
        for b, r in fp.items():
            W[i, bidx[b]] = r

    # Local coordinates
    lat, lon = df["latitude"].values, df["longitude"].values
    lat0, lon0 = lat.mean(), lon.mean()
    Re = 6371000.0
    df["x_m"] = np.radians(lon - lon0) * Re * np.cos(np.radians(lat0))
    df["y_m"] = np.radians(lat - lat0) * Re
    
    return df, W