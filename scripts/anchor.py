#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd

# =========================================================
# SET THIS to your extracted dataset folder
# Use raw string r"..." so backslashes work on Windows
# =========================================================
ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
# Example:
# ROOT = r"C:\Users\YourName\Downloads\dataset"
# ROOT = "/home/user/dataset"


# =========================================================
# 1. Show the full folder tree
# =========================================================
def show_tree(root, max_files_per_folder=10):
    print("\n" + "=" * 80)
    print("FOLDER STRUCTURE")
    print("=" * 80)

    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        indent = "    " * level
        print(f"{indent}{os.path.basename(dirpath)}/")

        sub_indent = "    " * (level + 1)
        for f in sorted(filenames)[:max_files_per_folder]:
            size_kb = os.path.getsize(os.path.join(dirpath, f)) / 1024
            print(f"{sub_indent}{f}  ({size_kb:.1f} KB)")

        if len(filenames) > max_files_per_folder:
            print(f"{sub_indent}... and {len(filenames) - max_files_per_folder} more files")


# =========================================================
# 2. Preview any text file (auto-detect delimiter)
# =========================================================
def preview_file(path, n_rows=10):
    print("\n" + "=" * 80)
    print(f"PREVIEW: {path}")
    print("=" * 80)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                print(f"{i:03d} | {line.rstrip()}")
                if i >= n_rows - 1:
                    break
    except Exception as e:
        print(f"Could not read file: {e}")
        return

    print("\n--- Attempting to parse with pandas ---")
    for sep in [",", ";", "\t", r"\s+", " "]:
        try:
            df = pd.read_csv(path, sep=sep, nrows=5, engine="python",
                             header=None)
            if df.shape[1] > 1:
                print(f"Detected delimiter: {repr(sep)}  → {df.shape[1]} columns")
                print("Column count:", df.shape[1])
                print(df.head())
                break
        except Exception:
            continue


# =========================================================
# 3. Preview all txt/csv files in a folder (up to N files)
# =========================================================
def preview_folder(folder, max_files=3):
    if not os.path.isdir(folder):
        print(f"[Skipping] Not a folder: {folder}")
        return

    files = [f for f in sorted(os.listdir(folder))
             if f.lower().endswith((".txt", ".csv", ".log", ".dat"))]

    print(f"\n>>> Folder: {folder}")
    print(f">>> {len(files)} text-like files found")

    for f in files[:max_files]:
        preview_file(os.path.join(folder, f))


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    if not os.path.isdir(ROOT):
        print(f"ERROR: Folder does not exist:\n  {ROOT}")
        print("Edit the ROOT variable at the top of this script.")
        raise SystemExit(1)

    # 1. Full tree
    show_tree(ROOT, max_files_per_folder=15)

    # 2. Preview the key folders
    candidates = [
        os.path.join(ROOT, "log files", "training"),
        os.path.join(ROOT, "log files", "testing"),
        os.path.join(ROOT, "log files", "scoring"),
        os.path.join(ROOT, "evaluation"),
        os.path.join(ROOT, "supplementary materials"),
        os.path.join(ROOT, "supplementary materials", "calibration floormaps"),
        os.path.join(ROOT, "supplementary materials", "matlab tools"),
        os.path.join(ROOT, "supplementary materials", "training trails on map"),
    ]

    for folder in candidates:
        preview_folder(folder, max_files=2)

    # 3. Also try to find "log files" wherever it lives
    print("\n" + "=" * 80)
    print("SEARCH: any folder named 'log files'")
    print("=" * 80)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if "log" in os.path.basename(dirpath).lower():
            print(dirpath)

# In[5]:


import os
import pandas as pd

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")

files = [
    "selected_wifi.csv",
    "selected_pressure.csv",
    "selected_position.csv",
    "combined_dataset_v3.csv",
]

for fname in files:
    path = os.path.join(CSV_DIR, fname)
    print("\n" + "=" * 80)
    print("FILE:", fname)
    print("=" * 80)
    if not os.path.exists(path):
        print("NOT FOUND:", path)
        continue

    df = pd.read_csv(path, nrows=5)
    print("Columns:", list(df.columns))
    print("Shape (first 5 rows shown):", df.shape)
    print(df.head())

    # full shape
    full = pd.read_csv(path, usecols=[0])
    print("Total rows:", len(full))

# In[7]:


EXTRA = [
    os.path.join(ROOT, "02 Supplementary Materials", "01 - Calibration Floormaps", "Paatalo_BuildingReference.csv"),
    os.path.join(ROOT, "03 Evaluation", "GT", "GT_IPIN2025_T3_ScoringTrial03.csv"),
    os.path.join(ROOT, "03 Evaluation", "GT", "GT_IPIN2025_T3_ScoringTrial04.csv"),
]

for path in EXTRA:
    print("\n" + "=" * 80)
    print("FILE:", path)
    print("=" * 80)
    if not os.path.exists(path):
        print("NOT FOUND")
        continue
    df = pd.read_csv(path)
    print(df.head(10))
    print("Shape:", df.shape)

# In[23]:


import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
km = KMeans(n_clusters=40, random_state=0, n_init=10)
# =========================================================
ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")
OUT_DIR = os.path.join(ROOT, "my_results")
os.makedirs(OUT_DIR, exist_ok=True)
# =========================================================

# ---------- 1. Load ----------
df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
print(f"Loaded combined_dataset_v3: {df.shape}")
print("Columns:", list(df.columns))

# ---------- 2. Parse WiFi fingerprint -> wide matrix ----------
def parse_fp(s):
    out = {}
    if not isinstance(s, str) or not s:
        return out
    for pair in s.split(";"):
        if ":" not in pair:
            continue
        bssid, rssi = pair.rsplit(":", 1)   # rsplit because BSSID has colons
        try:
            out[bssid] = float(rssi)
        except ValueError:
            continue
    return out

fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
all_bssids = sorted({b for fp in fps for b in fp})
bssid_idx = {b: i for i, b in enumerate(all_bssids)}
W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)  # -100 = not seen
for i, fp in enumerate(fps):
    for b, r in fp.items():
        W[i, bssid_idx[b]] = r
print(f"WiFi matrix: {W.shape}  ({len(all_bssids)} unique BSSIDs)")

# ---------- 3. Lat/lon -> local meters ----------
lat, lon = df["latitude"].values, df["longitude"].values
lat0, lon0 = lat.mean(), lon.mean()
R = 6371000.0
df["x_m"] = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
df["y_m"] = np.radians(lat - lat0) * R

# ---------- 4. Pressure -> relative altitude (per trial) ----------
def p2alt(p, p0):
    return 44330.0 * (1.0 - (p / p0) ** (1.0 / 5.255))

df["altitude"] = 0.0
for trial, g in df.groupby("trial"):
    p0 = g["pressure_hpa"].iloc[0]
    df.loc[g.index, "altitude"] = p2alt(g["pressure_hpa"].values, p0)

# ---------- 5. Train/test split (leave-one-trial-out) ----------
all_trials = sorted(df["trial"].unique())
print("Trials in dataset:", all_trials)
test_trials = all_trials[::4]      # every 4th trial for testing
print("Test trials:", test_trials)

train_mask = ~df["trial"].isin(test_trials).values
test_mask  =  df["trial"].isin(test_trials).values

print(f"Train samples: {train_mask.sum()}  Test samples: {test_mask.sum()}")
print("Train floors:", sorted(df.loc[train_mask, 'floor'].unique()))
print("Test floors: ", sorted(df.loc[test_mask,  'floor'].unique()))

X_train, X_test = W[train_mask], W[test_mask]
xy_train = df.loc[train_mask, ["x_m", "y_m"]].values
xy_test  = df.loc[test_mask,  ["x_m", "y_m"]].values
fl_train = df.loc[train_mask, "floor"].values
fl_test  = df.loc[test_mask,  "floor"].values

# =========================================================
# EXPERIMENT 1: Pressure only
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 1: Pressure only"); print("="*60)

floor_alt = df[train_mask].groupby("floor")["altitude"].mean()
floor_ids = floor_alt.index.values
floor_alts = floor_alt.values
print("Mean relative altitude per floor (m):")
print(floor_alt.round(2))

def floor_from_alt(a):
    d = np.abs(a[:, None] - floor_alts[None, :])
    return floor_ids[np.argmin(d, axis=1)]

fl_pred_p = floor_from_alt(df.loc[test_mask, "altitude"].values)
acc_p = accuracy_score(fl_test, fl_pred_p)
f1_p  = f1_score(fl_test, fl_pred_p, average="macro", zero_division=0)
print(f"Floor Accuracy: {acc_p:.3f}   Macro F1: {f1_p:.3f}")

# =========================================================
# EXPERIMENT 2: WiFi only - conventional kNN
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 2: WiFi only (conventional kNN)"); print("="*60)

knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(X_train)
t0 = time.time()
dist, idx = knn.kneighbors(X_test)
t_knn = time.time() - t0

wts = 1.0 / (dist + 1e-6)
xy_pred_knn = np.einsum("nk,nkd->nd", wts, xy_train[idx]) / wts.sum(axis=1, keepdims=True)
rmse_knn = np.sqrt(np.mean(np.sum((xy_pred_knn - xy_test) ** 2, axis=1)))

fl_pred_knn = mode(fl_train[idx], axis=1, keepdims=False).mode
acc_knn = accuracy_score(fl_test, fl_pred_knn)
f1_knn  = f1_score(fl_test, fl_pred_knn, average="macro", zero_division=0)

print(f"XY RMSE:      {rmse_knn:.2f} m")
print(f"Floor Acc:    {acc_knn:.3f}   Macro F1: {f1_knn:.3f}")
print(f"Search time:  {t_knn*1000:.1f} ms total ({t_knn/len(X_test)*1000:.3f} ms/sample)")

# =========================================================
# EXPERIMENT 3: WiFi only - STH + VP-tree (BallTree)
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 3: WiFi STH + VP-tree"); print("="*60)

def make_hash(fp, thr=-90):
    return tuple(sorted(b for b, r in fp.items() if r > thr))

train_hashes = [make_hash(fps[i]) for i in np.where(train_mask)[0]]
test_hashes  = [make_hash(fps[i]) for i in np.where(test_mask)[0]]

bucket2idx = defaultdict(list)
for i, h in enumerate(train_hashes):
    bucket2idx[h].append(i)

print(f"STH buckets: {len(bucket2idx)}")
sizes = [len(v) for v in bucket2idx.values()]
print(f"Bucket sizes: min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)}")

trees = {h: (BallTree(X_train[idxs]), np.array(idxs))
         for h, idxs in bucket2idx.items() if len(idxs) >= 3}
print(f"Built {len(trees)} BallTrees (VP-tree equivalent)")

t0 = time.time()
xy_pred_sth = np.zeros_like(xy_test)
fl_pred_sth = np.zeros(len(X_test), dtype=int)
k = 5

for i in range(len(X_test)):
    h = test_hashes[i]
    if h in trees:
        tree, idxs = trees[h]
        kk = min(k, len(idxs))
        d, ii = tree.query(X_test[i:i+1], k=kk)
        neigh, dd = idxs[ii[0]], d[0]
    else:
        d, ii = knn.kneighbors(X_test[i:i+1], n_neighbors=k)
        neigh, dd = ii[0], d[0]
    ww = 1.0 / (dd + 1e-6)
    xy_pred_sth[i] = (ww[:, None] * xy_train[neigh]).sum(0) / ww.sum()
    fl_pred_sth[i] = mode(fl_train[neigh], keepdims=False).mode

t_sth = time.time() - t0
rmse_sth = np.sqrt(np.mean(np.sum((xy_pred_sth - xy_test) ** 2, axis=1)))
acc_sth  = accuracy_score(fl_test, fl_pred_sth)
f1_sth   = f1_score(fl_test, fl_pred_sth, average="macro", zero_division=0)

print(f"XY RMSE:      {rmse_sth:.2f} m")
print(f"Floor Acc:    {acc_sth:.3f}   Macro F1: {f1_sth:.3f}")
print(f"Search time:  {t_sth*1000:.1f} ms total ({t_sth/len(X_test)*1000:.3f} ms/sample)")

# =========================================================
# EXPERIMENT 4: Fusion (Bayesian)
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 4: WiFi + Pressure fusion"); print("="*60)

sigma_floor = np.maximum(df[train_mask].groupby("floor")["altitude"].std().values, 1.0)
print("Floor altitude std (m):", dict(zip(floor_ids, sigma_floor.round(2))))

# Pressure floor likelihood (Gaussian on relative altitude)
def pressure_floor_prob(alt, sigma=2.0):
    d = alt[:, None] - floor_alts[None, :]
    logp = -0.5 * (d / sigma) ** 2
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)

P_pressure = pressure_floor_prob(df.loc[test_mask, "altitude"].values, sigma=2.0)

# WiFi floor likelihood from kNN neighbours
def wifi_floor_prob(neigh_floors):
    p = np.zeros(len(floor_ids))
    for f in neigh_floors:
        m = np.where(floor_ids == f)[0]
        if len(m):
            p[m[0]] += 1
    return p / p.sum() if p.sum() > 0 else np.ones(len(floor_ids)) / len(floor_ids)

fl_pred_fused = np.zeros(len(X_test), dtype=int)
for i in range(len(X_test)):
    p_w = wifi_floor_prob(fl_train[idx[i]])
    p_fused = p_w * P_pressure[i]
    fl_pred_fused[i] = floor_ids[np.argmax(p_fused)]

acc_fus = accuracy_score(fl_test, fl_pred_fused)
f1_fus  = f1_score(fl_test, fl_pred_fused, average="macro", zero_division=0)
print(f"Fused Floor Acc: {acc_fus:.3f}   Macro F1: {f1_fus:.3f}")

# =========================================================
# SUMMARY
# =========================================================
summary = pd.DataFrame([
    {"Experiment": "Pressure only",         "Floor_Acc": acc_p,   "Macro_F1": f1_p,   "XY_RMSE_m": np.nan,  "ms_per_sample": np.nan},
    {"Experiment": "WiFi only (kNN)",       "Floor_Acc": acc_knn, "Macro_F1": f1_knn, "XY_RMSE_m": rmse_knn,"ms_per_sample": t_knn/len(X_test)*1000},
    {"Experiment": "WiFi STH + VP-tree",    "Floor_Acc": acc_sth, "Macro_F1": f1_sth, "XY_RMSE_m": rmse_sth,"ms_per_sample": t_sth/len(X_test)*1000},
    {"Experiment": "WiFi + Pressure fusion","Floor_Acc": acc_fus, "Macro_F1": f1_fus, "XY_RMSE_m": np.nan,  "ms_per_sample": np.nan},
])

print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)
print(summary.to_string(index=False))

summary.to_csv(os.path.join(OUT_DIR, "experiment_summary.csv"), index=False)
print(f"\nSaved: {os.path.join(OUT_DIR, 'experiment_summary.csv')}")

# In[11]:


import os
import numpy as np
import pandas as pd

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")

df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))

# Absolute altitude
def p2alt(p, p0=1013.25):
    return 44330.0 * (1.0 - (p / p0) ** (1.0 / 5.255))

df["alt_abs"] = p2alt(df["pressure_hpa"].values)

print("=" * 70)
print("PRESSURE DIAGNOSTIC")
print("=" * 70)

print("\n--- Raw pressure stats ---")
print(df["pressure_hpa"].describe().round(3))

print("\n--- Absolute altitude stats (m) ---")
print(df["alt_abs"].describe().round(2))

print("\n--- Mean pressure per (trial, floor) ---")
piv = df.pivot_table(index="trial", columns="floor",
                     values="pressure_hpa", aggfunc="mean")
print(piv.round(3))

print("\n--- Std pressure per (trial, floor) ---")
piv_std = df.pivot_table(index="trial", columns="floor",
                         values="pressure_hpa", aggfunc="std")
print(piv_std.round(4))

print("\n--- Within-trial pressure range per floor ---")
for trial, g in df.groupby("trial"):
    floors = sorted(g["floor"].unique())
    if len(floors) < 2:
        continue
    print(f"\nTrial {trial}: floors {floors}")
    for f in floors:
        sub = g[g["floor"] == f]["pressure_hpa"]
        print(f"  floor {f}: n={len(sub):3d}  mean={sub.mean():.3f}  "
              f"std={sub.std():.4f}  range=[{sub.min():.3f}, {sub.max():.3f}]")

print("\n--- Per-trial: does pressure decrease monotonically with floor? ---")
for trial, g in df.groupby("trial"):
    means = g.groupby("floor")["pressure_hpa"].mean().sort_index()
    if len(means) < 2:
        continue
    # increasing floor -> decreasing pressure? (higher = lower pressure)
    mono = all(means.iloc[i] > means.iloc[i+1] for i in range(len(means)-1))
    print(f"Trial {trial}: floor→pressure = "
          f"{dict(means.round(3))}  monotonic_decreasing={mono}")

# In[13]:


from sklearn.cluster import MiniBatchKMeans

K = 40  # number of hash buckets
km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5, batch_size=64)

# hash each fingerprint by nearest centroid
km.fit(X_train)
train_bucket = km.predict(X_train)
test_bucket  = km.predict(X_test)

# In[25]:


#pressure diagnosis
import os
import pandas as pd
import numpy as np

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")

df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")

print("Samples per device:", df["device"].value_counts().to_dict())

print("\n--- Pressure std per device ---")
print(df.groupby("device")["pressure_hpa"].std().round(3))

print("\n--- Pressure std per device × trial ---")
piv = df.pivot_table(index="trial", columns="device",
                     values="pressure_hpa", aggfunc="std")
print(piv.round(3))

# In[19]:


#cleaning data
import os
import numpy as np
import pandas as pd

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")

df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")

# Sort by trial & timestamp
df = df.sort_values(["trial", "session", "position_timestamp"]).reset_index(drop=True)

# Rolling median within each trial
WINDOW = 5
df["p_med"] = (df.groupby(["trial", "session"])["pressure_hpa"]
                 .transform(lambda s: s.rolling(WINDOW, center=True, min_periods=1).median()))

# Residual from rolling median
df["p_resid"] = df["pressure_hpa"] - df["p_med"]

# Reject samples where |residual| > threshold
THRESH = 1.0  # hPa
df["p_clean"] = df["pressure_hpa"].where(df["p_resid"].abs() <= THRESH, df["p_med"])

print("Before filtering:")
print(df.groupby("trial")["pressure_hpa"].std().round(3))
print("\nAfter filtering:")
print(df.groupby("trial")["p_clean"].std().round(3))

# Per-floor std after filtering
print("\nPer (trial, floor) std, cleaned:")
piv = df.pivot_table(index="trial", columns="floor",
                     values="p_clean", aggfunc="std")
print(piv.round(3))

# Per-floor mean, cleaned
print("\nPer (trial, floor) mean, cleaned:")
piv2 = df.pivot_table(index="trial", columns="floor",
                      values="p_clean", aggfunc="mean")
print(piv2.round(3))

# Save cleaned version
df.to_csv(os.path.join(CSV_DIR, "combined_dataset_v3_clean.csv"), index=False)
print("\nSaved combined_dataset_v3_clean.csv")

# In[27]:


# Use the CLEANED pressure
df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3_clean.csv"))
df["pressure_hpa"] = df["p_clean"]      # <-- use cleaned values

# In[33]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")
OUT_DIR = os.path.join(ROOT, "my_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load ----------
df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3_clean.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
df = df[df["trial"] != 31].reset_index(drop=True)

# Smooth pressure per trial+device
df = df.sort_values(["trial", "device", "position_timestamp"]).reset_index(drop=True)
df["p_smooth"] = (df.groupby(["trial", "device"])["p_clean"]
                    .transform(lambda s: s.rolling(5, center=True, min_periods=1).median()))

print(f"Loaded: {df.shape}  Devices: {df['device'].value_counts().to_dict()}")

# ---------- Parse WiFi ----------
def parse_fp(s):
    out = {}
    if not isinstance(s, str):
        return out
    for pair in s.split(";"):
        if ":" not in pair:
            continue
        bssid, rssi = pair.rsplit(":", 1)
        try:
            out[bssid] = float(rssi)
        except ValueError:
            continue
    return out

fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
all_bssids = sorted({b for fp in fps for b in fp})
bssid_idx = {b: i for i, b in enumerate(all_bssids)}
W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)
for i, fp in enumerate(fps):
    for b, r in fp.items():
        W[i, bssid_idx[b]] = r

# ---------- Coordinates ----------
lat, lon = df["latitude"].values, df["longitude"].values
lat0, lon0 = lat.mean(), lon.mean()
Re = 6371000.0
df["x_m"] = np.radians(lon - lon0) * Re * np.cos(np.radians(lat0))
df["y_m"] = np.radians(lat - lat0) * Re

# ---------- Split ----------
test_trials = [21, 25, 34, 44, 54]
train_mask = ~df["trial"].isin(test_trials).values
test_mask  =  df["trial"].isin(test_trials).values

X_train, X_test = W[train_mask], W[test_mask]
xy_train = df.loc[train_mask, ["x_m", "y_m"]].values
xy_test  = df.loc[test_mask,  ["x_m", "y_m"]].values
fl_train = df.loc[train_mask, "floor"].values
fl_test  = df.loc[test_mask,  "floor"].values
floors_sorted = sorted(df["floor"].unique())

# =========================================================
# Estimate hPa-per-floor from training data
# =========================================================
diffs = []
tr_df = df[train_mask]
for (trial, device), g in tr_df.groupby(["trial", "device"]):
    means = g.groupby("floor")["p_smooth"].mean().sort_index()
    if len(means) >= 2:
        for i in range(len(means)-1):
            diffs.append(means.iloc[i+1] - means.iloc[i])
hPa_per_floor = -np.mean(diffs) if diffs else 0.4
print(f"\nEstimated hPa per floor (upward): {hPa_per_floor:.3f}")

# =========================================================
# EXPERIMENT 2: WiFi kNN (compute once, reuse)
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 2: WiFi kNN"); print("="*60)

knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=1).fit(X_train)
t0 = time.time()
dist_knn, idx_knn = knn.kneighbors(X_test)   # idx_knn: (n_test, 5) train indices
t_knn = time.time() - t0

wts = 1.0 / (dist_knn + 1e-6)
xy_pred_knn = np.einsum("nk,nkd->nd", wts, xy_train[idx_knn]) / wts.sum(axis=1, keepdims=True)
rmse_knn = np.sqrt(np.mean(np.sum((xy_pred_knn - xy_test) ** 2, axis=1)))
fl_pred_knn = mode(fl_train[idx_knn], axis=1, keepdims=False).mode
acc_knn = accuracy_score(fl_test, fl_pred_knn)
f1_knn  = f1_score(fl_test, fl_pred_knn, average="macro", zero_division=0)
print(f"XY RMSE: {rmse_knn:.2f} m   Floor Acc: {acc_knn:.3f}   F1: {f1_knn:.3f}")
print(f"Search: {t_knn/len(X_test)*1000:.3f} ms/sample")

# =========================================================
# EXPERIMENT 1: Pressure only (anchor-based)
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 1: Pressure only (anchor-based)"); print("="*60)

test_df = df.loc[test_mask].copy().reset_index(drop=True)
preds_pressure = np.zeros(len(test_df), dtype=int)

for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values

    # Anchor using WiFi majority vote on first 10 samples
    first_idx = idx[:10]
    anchor_floor = int(mode(fl_train[idx_knn[first_idx].flatten()], keepdims=False).mode)
    anchor_p = p[:10].mean()

    # Track pressure change per sample
    delta_p = p - anchor_p
    delta_floor = np.round(-delta_p / hPa_per_floor).astype(int)
    pred = np.clip(anchor_floor + delta_floor,
                   min(floors_sorted), max(floors_sorted))
    preds_pressure[idx] = pred

acc_p = accuracy_score(fl_test, preds_pressure)
f1_p  = f1_score(fl_test, preds_pressure, average="macro", zero_division=0)
print(f"Floor Acc: {acc_p:.3f}   F1: {f1_p:.3f}")

# =========================================================
# EXPERIMENT 3: WiFi STH (k-means)
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 3: WiFi STH via k-means"); print("="*60)

K = 40
km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5, batch_size=3072)
km.fit(X_train)
train_bucket = km.predict(X_train)
test_bucket  = km.predict(X_test)

bucket2idx = defaultdict(list)
for i, b in enumerate(train_bucket):
    bucket2idx[b].append(i)

sizes = [len(v) for v in bucket2idx.values()]
print(f"Buckets: {len(bucket2idx)}, sizes min={min(sizes)}, med={int(np.median(sizes))}, max={max(sizes)}")
bucket_trees = {b: BallTree(X_train[idxs]) for b, idxs in bucket2idx.items() if len(idxs) >= 3}

t0 = time.time()
xy_pred_sth = np.zeros_like(xy_test)
fl_pred_sth = np.zeros(len(X_test), dtype=int)
k = 5

for i in range(len(X_test)):
    b = int(test_bucket[i])
    cands = [b]
    if len(bucket2idx.get(b, [])) < k:
        d2c = np.linalg.norm(km.cluster_centers_ - X_test[i], axis=1)
        cands = [int(x) for x in np.argsort(d2c)[:3]]

    neigh_global, neigh_dist = [], []
    for cb in cands:
        if cb in bucket_trees:
            kk = min(k, len(bucket2idx[cb]))
            d, ii = bucket_trees[cb].query(X_test[i:i+1], k=kk)
            for j in ii[0]:
                neigh_global.append(bucket2idx[cb][int(j)])
            neigh_dist.extend(d[0].tolist())

    if len(neigh_global) < k:
        d, ii = knn.kneighbors(X_test[i:i+1], n_neighbors=k)
        neigh_global = ii[0].tolist()
        neigh_dist  = d[0].tolist()

    order = np.argsort(neigh_dist)[:k]
    neigh_global = [neigh_global[o] for o in order]
    neigh_dist = np.array([neigh_dist[o] for o in order])

    ww = 1.0 / (neigh_dist + 1e-6)
    xy_pred_sth[i] = (ww[:, None] * xy_train[neigh_global]).sum(0) / ww.sum()
    fl_pred_sth[i] = mode(fl_train[neigh_global], keepdims=False).mode

t_sth = time.time() - t0
rmse_sth = np.sqrt(np.mean(np.sum((xy_pred_sth - xy_test) ** 2, axis=1)))
acc_sth  = accuracy_score(fl_test, fl_pred_sth)
f1_sth   = f1_score(fl_test, fl_pred_sth, average="macro", zero_division=0)
print(f"XY RMSE: {rmse_sth:.2f} m   Floor Acc: {acc_sth:.3f}   F1: {f1_sth:.3f}")
print(f"Search: {t_sth/len(X_test)*1000:.3f} ms/sample")

# =========================================================
# EXPERIMENT 4: Fusion (fixed)
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 4: Fusion"); print("="*60)

def wifi_floor_prob(neigh_floors_arr):
    """neigh_floors_arr: 1D array of floor labels from kNN neighbors"""
    p = np.zeros(len(floors_sorted))
    for f in neigh_floors_arr:
        if f in floors_sorted:
            p[floors_sorted.index(int(f))] += 1
    return p / p.sum() if p.sum() > 0 else np.ones(len(floors_sorted)) / len(floors_sorted)

def pressure_floor_prob(pred_floor_from_pressure, sigma=1.0):
    """Gaussian around the anchor-derived floor estimate"""
    centers = np.arange(1, 8)  # possible floor values
    d = centers - pred_floor_from_pressure
    logp = -0.5 * (d / sigma) ** 2
    logp -= logp.max()
    p = np.exp(logp)
    # keep only valid floor labels
    p_valid = np.zeros(len(floors_sorted))
    for i, f in enumerate(floors_sorted):
        if f in centers:
            p_valid[i] = p[np.where(centers == f)[0][0]]
    return p_valid / p_valid.sum() if p_valid.sum() > 0 else p_valid

preds_fused = np.zeros(len(X_test), dtype=int)

for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values

    # Anchor (same as Experiment 1)
    first_idx = idx[:10]
    anchor_floor = int(mode(fl_train[idx_knn[first_idx].flatten()], keepdims=False).mode)
    anchor_p = p[:10].mean()

    for j, row_i in enumerate(idx):
        # WiFi prob (uses idx_knn)
        p_w = wifi_floor_prob(fl_train[idx_knn[row_i]])

        # Pressure prob
        delta_p = p[j] - anchor_p
        p_floor_pred = anchor_floor + (-delta_p / hPa_per_floor)
        p_p = pressure_floor_prob(p_floor_pred, sigma=1.0)

        p_fused = p_w * p_p
        preds_fused[row_i] = floors_sorted[int(np.argmax(p_fused))]

acc_fus = accuracy_score(fl_test, preds_fused)
f1_fus  = f1_score(fl_test, preds_fused, average="macro", zero_division=0)
print(f"Fused Floor Acc: {acc_fus:.3f}   F1: {f1_fus:.3f}")

# =========================================================
# Summary
# =========================================================
summary = pd.DataFrame([
    {"Experiment": "Pressure only (anchor)",     "Floor_Acc": acc_p,   "Macro_F1": f1_p,   "XY_RMSE_m": np.nan,  "ms_per_sample": np.nan},
    {"Experiment": "WiFi kNN",                    "Floor_Acc": acc_knn, "Macro_F1": f1_knn, "XY_RMSE_m": rmse_knn,"ms_per_sample": t_knn/len(X_test)*1000},
    {"Experiment": "WiFi STH (k-means)",          "Floor_Acc": acc_sth, "Macro_F1": f1_sth, "XY_RMSE_m": rmse_sth,"ms_per_sample": t_sth/len(X_test)*1000},
    {"Experiment": "WiFi + Pressure fusion",      "Floor_Acc": acc_fus, "Macro_F1": f1_fus, "XY_RMSE_m": np.nan,  "ms_per_sample": np.nan},
])
print("\n" + "="*70); print("SUMMARY"); print("="*70)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT_DIR, "experiment_summary_final.csv"), index=False)

# In[35]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")
OUT_DIR = os.path.join(ROOT, "my_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load RAW (not _clean!) ----------
df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
df = df[df["trial"] != 31].reset_index(drop=True)

# Sort so per-group rolling works
df = df.sort_values(["trial", "device", "position_timestamp"]).reset_index(drop=True)

# Smooth ONLY within (trial, device)
df["p_smooth"] = (df.groupby(["trial", "device"])["pressure_hpa"]
                    .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))

print(f"Loaded: {df.shape}")

# Per (trial, device) pressure std — sanity check
print("\nPer-(trial,device) pressure std:")
piv = df.pivot_table(index="trial", columns="device",
                     values="p_smooth", aggfunc="std")
print(piv.round(3))

# ---------- WiFi matrix ----------
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

fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
all_bssids = sorted({b for fp in fps for b in fp})
bidx = {b: i for i, b in enumerate(all_bssids)}
W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)
for i, fp in enumerate(fps):
    for b, r in fp.items():
        W[i, bidx[b]] = r
print(f"WiFi matrix: {W.shape}")

# ---------- Local coordinates ----------
lat, lon = df["latitude"].values, df["longitude"].values
lat0, lon0 = lat.mean(), lon.mean()
Re = 6371000.0
df["x_m"] = np.radians(lon - lon0) * Re * np.cos(np.radians(lat0))
df["y_m"] = np.radians(lat - lat0) * Re

# ---------- Split ----------
test_trials = [21, 25, 34, 44, 54]
train_mask = ~df["trial"].isin(test_trials).values
test_mask  =  df["trial"].isin(test_trials).values
X_train, X_test = W[train_mask], W[test_mask]
xy_train = df.loc[train_mask, ["x_m", "y_m"]].values
xy_test  = df.loc[test_mask,  ["x_m", "y_m"]].values
fl_train = df.loc[train_mask, "floor"].values
fl_test  = df.loc[test_mask,  "floor"].values
floors_sorted = sorted(df["floor"].unique())
print(f"Train: {train_mask.sum()}  Test: {test_mask.sum()}")

# =========================================================
# Estimate hPa per floor from training (per trial+device, then average)
# =========================================================
diffs = []
for (trial, device), g in df[train_mask].groupby(["trial", "device"]):
    means = g.groupby("floor")["p_smooth"].mean().sort_index()
    if len(means) >= 2:
        for i in range(len(means)-1):
            # higher floor has LOWER pressure → means[i+1] < means[i]
            # we want "hPa drop per floor upward" as a positive number
            drop = means.iloc[i] - means.iloc[i+1]
            diffs.append(drop)

hPa_per_floor = float(np.mean(diffs))
print(f"\nEstimated hPa DROP per floor upward: {hPa_per_floor:.3f} hPa "
      f"(expected ~0.4)")

# =========================================================
# EXPERIMENT 2: WiFi kNN
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 2: WiFi kNN"); print("="*60)

knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=1).fit(X_train)
t0 = time.time()
dist_knn, idx_knn = knn.kneighbors(X_test)
t_knn = time.time() - t0

wts = 1.0 / (dist_knn + 1e-6)
xy_pred_knn = np.einsum("nk,nkd->nd", wts, xy_train[idx_knn]) / wts.sum(axis=1, keepdims=True)
rmse_knn = np.sqrt(np.mean(np.sum((xy_pred_knn - xy_test) ** 2, axis=1)))
fl_pred_knn = mode(fl_train[idx_knn], axis=1, keepdims=False).mode
acc_knn = accuracy_score(fl_test, fl_pred_knn)
f1_knn  = f1_score(fl_test, fl_pred_knn, average="macro", zero_division=0)
print(f"XY RMSE: {rmse_knn:.2f} m   Floor Acc: {acc_knn:.3f}   F1: {f1_knn:.3f}")
print(f"Search: {t_knn/len(X_test)*1000:.3f} ms/sample")

# =========================================================
# EXPERIMENT 1: Pressure only (anchor + tracked delta)
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 1: Pressure only"); print("="*60)

test_df = df.loc[test_mask].copy().reset_index(drop=True)
preds_pressure = np.zeros(len(test_df), dtype=int)

print("Anchor diagnostics (first 10 samples per trial/device):")
for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values
    first_idx = idx[:10]

    # Anchor from WiFi's majority floor vote on first 10 samples
    neigh_flat = idx_knn[first_idx].flatten()
    anchor_floor = int(mode(fl_train[neigh_flat], keepdims=False).mode)
    true_start = int(g["floor"].iloc[0])

    # Anchor pressure: mean of first 10 samples
    anchor_p = p[:10].mean()

    # Track delta from anchor
    delta_p = p - anchor_p
    delta_floor = np.round(delta_p / hPa_per_floor).astype(int)  # NEGATIVE delta_p → positive delta_floor (up)
    pred = np.clip(anchor_floor + delta_floor,
                   min(floors_sorted), max(floors_sorted))
    preds_pressure[idx] = pred

    print(f"  {trial}/{device}: true_start={true_start}  "
          f"anchor_floor={anchor_floor}  anchor_p={anchor_p:.2f}  "
          f"Δp_range=[{delta_p.min():+.2f}, {delta_p.max():+.2f}]")

acc_p = accuracy_score(fl_test, preds_pressure)
f1_p  = f1_score(fl_test, preds_pressure, average="macro", zero_division=0)
print(f"\nPressure-only Floor Acc: {acc_p:.3f}   F1: {f1_p:.3f}")

# =========================================================
# EXPERIMENT 3: WiFi STH via k-means
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 3: WiFi STH via k-means"); print("="*60)

K = 40
km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5, batch_size=3072)
km.fit(X_train)
train_bucket = km.predict(X_train)
test_bucket  = km.predict(X_test)

bucket2idx = defaultdict(list)
for i, b in enumerate(train_bucket):
    bucket2idx[b].append(i)

sizes = [len(v) for v in bucket2idx.values()]
print(f"Buckets: {len(bucket2idx)}, sizes min={min(sizes)}, med={int(np.median(sizes))}, max={max(sizes)}")
bucket_trees = {b: BallTree(X_train[idxs]) for b, idxs in bucket2idx.items() if len(idxs) >= 3}

t0 = time.time()
xy_pred_sth = np.zeros_like(xy_test)
fl_pred_sth = np.zeros(len(X_test), dtype=int)
k = 5
cand_counts = []

for i in range(len(X_test)):
    b = int(test_bucket[i])
    cands = [b]
    if len(bucket2idx.get(b, [])) < k:
        d2c = np.linalg.norm(km.cluster_centers_ - X_test[i], axis=1)
        cands = [int(x) for x in np.argsort(d2c)[:3]]

    neigh_global, neigh_dist = [], []
    for cb in cands:
        if cb in bucket_trees:
            kk = min(k, len(bucket2idx[cb]))
            d, ii = bucket_trees[cb].query(X_test[i:i+1], k=kk)
            for j in ii[0]:
                neigh_global.append(bucket2idx[cb][int(j)])
            neigh_dist.extend(d[0].tolist())
    cand_counts.append(len(neigh_global))

    if len(neigh_global) < k:
        d, ii = knn.kneighbors(X_test[i:i+1], n_neighbors=k)
        neigh_global = ii[0].tolist()
        neigh_dist  = d[0].tolist()

    order = np.argsort(neigh_dist)[:k]
    neigh_global = [neigh_global[o] for o in order]
    neigh_dist = np.array([neigh_dist[o] for o in order])

    ww = 1.0 / (neigh_dist + 1e-6)
    xy_pred_sth[i] = (ww[:, None] * xy_train[neigh_global]).sum(0) / ww.sum()
    fl_pred_sth[i] = mode(fl_train[neigh_global], keepdims=False).mode

t_sth = time.time() - t0
rmse_sth = np.sqrt(np.mean(np.sum((xy_pred_sth - xy_test) ** 2, axis=1)))
acc_sth  = accuracy_score(fl_test, fl_pred_sth)
f1_sth   = f1_score(fl_test, fl_pred_sth, average="macro", zero_division=0)
print(f"XY RMSE: {rmse_sth:.2f} m   Floor Acc: {acc_sth:.3f}   F1: {f1_sth:.3f}")
print(f"Search: {t_sth/len(X_test)*1000:.3f} ms/sample")
print(f"Avg candidate set size: {np.mean(cand_counts):.1f} vs {len(X_train)} for kNN "
      f"→ {len(X_train)/np.mean(cand_counts):.1f}× reduction")

# =========================================================
# EXPERIMENT 4: Fusion
# =========================================================
print("\n" + "="*60); print("EXPERIMENT 4: Fusion"); print("="*60)

def wifi_floor_prob(neigh_floors_arr):
    p = np.zeros(len(floors_sorted))
    for f in neigh_floors_arr:
        if f in floors_sorted:
            p[floors_sorted.index(int(f))] += 1
    return p / p.sum() if p.sum() > 0 else np.ones(len(floors_sorted)) / len(floors_sorted)

def pressure_floor_prob(pred_floor_float, sigma=0.7):
    """Gaussian over floors centered at a fractional floor estimate"""
    d = np.array(floors_sorted) - pred_floor_float
    logp = -0.5 * (d / sigma) ** 2
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()

preds_fused = np.zeros(len(X_test), dtype=int)

for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values
    first_idx = idx[:10]
    neigh_flat = idx_knn[first_idx].flatten()
    anchor_floor = int(mode(fl_train[neigh_flat], keepdims=False).mode)
    anchor_p = p[:10].mean()

    for j, row_i in enumerate(idx):
        # WiFi distribution
        p_w = wifi_floor_prob(fl_train[idx_knn[row_i]])

        # Pressure estimate of floor (fractional)
        delta_p = p[j] - anchor_p
        p_floor_float = anchor_floor + delta_p / hPa_per_floor
        p_p = pressure_floor_prob(p_floor_float, sigma=0.7)

        p_fused = p_w * p_p
        preds_fused[row_i] = floors_sorted[int(np.argmax(p_fused))]

acc_fus = accuracy_score(fl_test, preds_fused)
f1_fus  = f1_score(fl_test, preds_fused, average="macro", zero_division=0)
print(f"Fused Floor Acc: {acc_fus:.3f}   F1: {f1_fus:.3f}")

# =========================================================
# Summary
# =========================================================
summary = pd.DataFrame([
    {"Experiment": "Pressure only (anchor)", "Floor_Acc": acc_p,   "Macro_F1": f1_p,   "XY_RMSE_m": np.nan,  "ms_per_sample": np.nan},
    {"Experiment": "WiFi kNN",               "Floor_Acc": acc_knn, "Macro_F1": f1_knn, "XY_RMSE_m": rmse_knn,"ms_per_sample": t_knn/len(X_test)*1000},
    {"Experiment": "WiFi STH (k-means)",     "Floor_Acc": acc_sth, "Macro_F1": f1_sth, "XY_RMSE_m": rmse_sth,"ms_per_sample": t_sth/len(X_test)*1000},
    {"Experiment": "WiFi + Pressure fusion", "Floor_Acc": acc_fus, "Macro_F1": f1_fus, "XY_RMSE_m": np.nan,  "ms_per_sample": np.nan},
])
print("\n" + "="*70); print("SUMMARY"); print("="*70)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT_DIR, "experiment_summary_final.csv"), index=False)

# In[37]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")
OUT_DIR = os.path.join(ROOT, "my_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load raw data ----------
df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
df = df[df["trial"] != 31].reset_index(drop=True)
df = df.sort_values(["trial", "device", "position_timestamp"]).reset_index(drop=True)

# Per-(trial,device) smoothing
df["p_smooth"] = (df.groupby(["trial", "device"])["pressure_hpa"]
                    .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))

print(f"Loaded: {df.shape}")

# ---------- WiFi matrix ----------
def parse_fp(s):
    out = {}
    if not isinstance(s, str): return out
    for pair in s.split(";"):
        if ":" not in pair: continue
        b, r = pair.rsplit(":", 1)
        try: out[b] = float(r)
        except ValueError: continue
    return out

fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
all_bssids = sorted({b for fp in fps for b in fp})
bidx = {b: i for i, b in enumerate(all_bssids)}
W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)
for i, fp in enumerate(fps):
    for b, r in fp.items():
        W[i, bidx[b]] = r

# ---------- Local coordinates ----------
lat, lon = df["latitude"].values, df["longitude"].values
lat0, lon0 = lat.mean(), lon.mean()
Re = 6371000.0
df["x_m"] = np.radians(lon - lon0) * Re * np.cos(np.radians(lat0))
df["y_m"] = np.radians(lat - lat0) * Re

# ---------- Split ----------
test_trials = [21, 25, 34, 44, 54]
train_mask = ~df["trial"].isin(test_trials).values
test_mask  =  df["trial"].isin(test_trials).values
X_train, X_test = W[train_mask], W[test_mask]
xy_train = df.loc[train_mask, ["x_m", "y_m"]].values
xy_test  = df.loc[test_mask,  ["x_m", "y_m"]].values
fl_train = df.loc[train_mask, "floor"].values
fl_test  = df.loc[test_mask,  "floor"].values
floors_sorted = sorted(df["floor"].unique())

print(f"Train: {train_mask.sum()}  Test: {test_mask.sum()}")

# =========================================================
# FIX: robust per-floor drop estimate
# =========================================================
print("\n--- Per-trial pressure drop (floor i → floor i+1) ---")
all_drops = []
for (trial, device), g in df[train_mask].groupby(["trial", "device"]):
    medians = g.groupby("floor")["p_smooth"].median().sort_index()
    if len(medians) < 2:
        continue
    for i in range(len(medians) - 1):
        drop = float(medians.iloc[i] - medians.iloc[i+1])   # up = pressure drop
        print(f"  trial {trial} {device}: floor {medians.index[i]}→"
              f"{medians.index[i+1]}  drop={drop:+.3f} hPa")
        if drop > 0:
            all_drops.append(drop)

hPa_per_floor = float(np.median(all_drops)) if all_drops else 0.4
print(f"\nhPa DROP per floor (median of positives): {hPa_per_floor:.3f}")

# =========================================================
# WiFi kNN (compute once)
# =========================================================
knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=1).fit(X_train)
t0 = time.time()
dist_knn, idx_knn = knn.kneighbors(X_test)
t_knn = time.time() - t0

wts = 1.0 / (dist_knn + 1e-6)
xy_pred_knn = np.einsum("nk,nkd->nd", wts, xy_train[idx_knn]) / wts.sum(axis=1, keepdims=True)
rmse_knn = np.sqrt(np.mean(np.sum((xy_pred_knn - xy_test) ** 2, axis=1)))
fl_pred_knn = mode(fl_train[idx_knn], axis=1, keepdims=False).mode
acc_knn = accuracy_score(fl_test, fl_pred_knn)
f1_knn = f1_score(fl_test, fl_pred_knn, average="macro", zero_division=0)
print(f"\nWiFi kNN: XY RMSE {rmse_knn:.2f} m   Floor Acc {acc_knn:.3f}   F1 {f1_knn:.3f}")
print(f"Search: {t_knn/len(X_test)*1000:.3f} ms/sample")

# =========================================================
# Pressure-only, TWO anchor variants
# =========================================================
test_df = df.loc[test_mask].copy().reset_index(drop=True)
ANCHOR_N = 30      # longer anchor window

def pressure_predict(anchor_floor):
    preds = np.zeros(len(test_df), dtype=int)
    for (trial, device), g in test_df.groupby(["trial", "device"]):
        idx = g.index.values
        p = g["p_smooth"].values
        anchor_p = p[:ANCHOR_N].mean()
        delta_p = p - anchor_p
        delta_floor = np.round(delta_p / hPa_per_floor).astype(int)
        preds[idx] = np.clip(anchor_floor + delta_floor,
                             min(floors_sorted), max(floors_sorted))
    return preds

# --- Anchor A: WiFi majority vote on first N samples ---
anchor_wifi = {}
for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values[:ANCHOR_N]
    votes = fl_train[idx_knn[idx].flatten()]
    anchor_wifi[(trial, device)] = int(mode(votes, keepdims=False).mode)

# We need a per-row anchor, so rebuild:
preds_p_wifi = np.zeros(len(test_df), dtype=int)
preds_p_gt   = np.zeros(len(test_df), dtype=int)

print("\n--- Anchor comparison ---")
for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values

    # WiFi anchor
    a_wifi = anchor_wifi[(trial, device)]
    # GT anchor (upper bound)
    a_gt = int(g["floor"].iloc[:ANCHOR_N].mode().iloc[0])

    true_start = int(g["floor"].iloc[0])
    print(f"  {trial}/{device}:  true={true_start}  WiFi_anchor={a_wifi}  GT_anchor={a_gt}")

    anchor_p = p[:ANCHOR_N].mean()
    delta_p = p - anchor_p
    d_floor = np.round(delta_p / hPa_per_floor).astype(int)

    preds_p_wifi[idx] = np.clip(a_wifi + d_floor, min(floors_sorted), max(floors_sorted))
    preds_p_gt[idx]   = np.clip(a_gt   + d_floor, min(floors_sorted), max(floors_sorted))

acc_p_wifi = accuracy_score(fl_test, preds_p_wifi)
acc_p_gt   = accuracy_score(fl_test, preds_p_gt)
f1_p_wifi  = f1_score(fl_test, preds_p_wifi, average="macro", zero_division=0)
f1_p_gt    = f1_score(fl_test, preds_p_gt,   average="macro", zero_division=0)

print(f"\nPressure-only (WiFi anchor)  Floor Acc: {acc_p_wifi:.3f}   F1: {f1_p_wifi:.3f}")
print(f"Pressure-only (GT anchor)    Floor Acc: {acc_p_gt:.3f}   F1: {f1_p_gt:.3f}")

# =========================================================
# WiFi STH via k-means
# =========================================================
K = 40
km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5, batch_size=3072)
km.fit(X_train)
train_bucket = km.predict(X_train)
test_bucket  = km.predict(X_test)
bucket2idx = defaultdict(list)
for i, b in enumerate(train_bucket):
    bucket2idx[b].append(i)
bucket_trees = {b: BallTree(X_train[idxs]) for b, idxs in bucket2idx.items() if len(idxs) >= 3}

t0 = time.time()
xy_pred_sth = np.zeros_like(xy_test)
fl_pred_sth = np.zeros(len(X_test), dtype=int)
k = 5
for i in range(len(X_test)):
    b = int(test_bucket[i])
    cands = [b]
    if len(bucket2idx.get(b, [])) < k:
        d2c = np.linalg.norm(km.cluster_centers_ - X_test[i], axis=1)
        cands = [int(x) for x in np.argsort(d2c)[:3]]
    ng, nd = [], []
    for cb in cands:
        if cb in bucket_trees:
            kk = min(k, len(bucket2idx[cb]))
            d, ii = bucket_trees[cb].query(X_test[i:i+1], k=kk)
            for j in ii[0]:
                ng.append(bucket2idx[cb][int(j)])
            nd.extend(d[0].tolist())
    if len(ng) < k:
        d, ii = knn.kneighbors(X_test[i:i+1], n_neighbors=k)
        ng, nd = ii[0].tolist(), d[0].tolist()
    order = np.argsort(nd)[:k]
    ng = [ng[o] for o in order]
    nd = np.array([nd[o] for o in order])
    ww = 1.0 / (nd + 1e-6)
    xy_pred_sth[i] = (ww[:, None] * xy_train[ng]).sum(0) / ww.sum()
    fl_pred_sth[i] = mode(fl_train[ng], keepdims=False).mode

t_sth = time.time() - t0
rmse_sth = np.sqrt(np.mean(np.sum((xy_pred_sth - xy_test) ** 2, axis=1)))
acc_sth  = accuracy_score(fl_test, fl_pred_sth)
f1_sth   = f1_score(fl_test, fl_pred_sth, average="macro", zero_division=0)
print(f"\nWiFi STH: XY RMSE {rmse_sth:.2f} m   Floor Acc {acc_sth:.3f}   F1 {f1_sth:.3f}")
print(f"Search: {t_sth/len(X_test)*1000:.3f} ms/sample")

# =========================================================
# Fusion (WiFi-anchor + WiFi kNN probabilities)
# =========================================================
def wifi_floor_prob(neigh_floors_arr):
    p = np.zeros(len(floors_sorted))
    for f in neigh_floors_arr:
        if f in floors_sorted:
            p[floors_sorted.index(int(f))] += 1
    return p / p.sum() if p.sum() > 0 else np.ones(len(floors_sorted)) / len(floors_sorted)

def pressure_floor_prob(pred_floor_float, sigma=1.0):
    d = np.array(floors_sorted) - pred_floor_float
    logp = -0.5 * (d / sigma) ** 2
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()

preds_fused = np.zeros(len(X_test), dtype=int)

for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values
    a_wifi = anchor_wifi[(trial, device)]
    anchor_p = p[:ANCHOR_N].mean()

    for j, row_i in enumerate(idx):
        p_w = wifi_floor_prob(fl_train[idx_knn[row_i]])
        delta_p = p[j] - anchor_p
        p_floor_float = a_wifi + delta_p / hPa_per_floor
        p_p = pressure_floor_prob(p_floor_float, sigma=1.0)
        p_fused = p_w * p_p
        preds_fused[row_i] = floors_sorted[int(np.argmax(p_fused))]

acc_fus = accuracy_score(fl_test, preds_fused)
f1_fus  = f1_score(fl_test, preds_fused, average="macro", zero_division=0)
print(f"\nFusion Floor Acc: {acc_fus:.3f}   F1: {f1_fus:.3f}")

# =========================================================
# Summary
# =========================================================
summary = pd.DataFrame([
    {"Experiment": "Pressure only (WiFi anchor)", "Floor_Acc": acc_p_wifi, "Macro_F1": f1_p_wifi},
    {"Experiment": "Pressure only (GT anchor)",   "Floor_Acc": acc_p_gt,   "Macro_F1": f1_p_gt},
    {"Experiment": "WiFi kNN",                    "Floor_Acc": acc_knn,    "Macro_F1": f1_knn},
    {"Experiment": "WiFi STH (k-means)",          "Floor_Acc": acc_sth,    "Macro_F1": f1_sth},
    {"Experiment": "WiFi + Pressure fusion",      "Floor_Acc": acc_fus,    "Macro_F1": f1_fus},
])
print("\n" + "="*70); print("SUMMARY"); print("="*70)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT_DIR, "experiment_summary_final.csv"), index=False)

# In[39]:


# =========================================================
# FINAL RUN: physics-corrected pressure + gated fusion
# =========================================================
import os, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")
OUT_DIR = os.path.join(ROOT, "my_results")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
df = df[df["trial"] != 31].reset_index(drop=True)
df = df.sort_values(["trial", "device", "position_timestamp"]).reset_index(drop=True)
df["p_smooth"] = (df.groupby(["trial", "device"])["pressure_hpa"]
                    .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))

def parse_fp(s):
    out = {}
    if not isinstance(s, str): return out
    for pair in s.split(";"):
        if ":" not in pair: continue
        b, r = pair.rsplit(":", 1)
        try: out[b] = float(r)
        except ValueError: continue
    return out

fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
all_bssids = sorted({b for fp in fps for b in fp})
bidx = {b: i for i, b in enumerate(all_bssids)}
W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)
for i, fp in enumerate(fps):
    for b, r in fp.items():
        W[i, bidx[b]] = r

lat, lon = df["latitude"].values, df["longitude"].values
lat0, lon0 = lat.mean(), lon.mean()
Re = 6371000.0
df["x_m"] = np.radians(lon - lon0) * Re * np.cos(np.radians(lat0))
df["y_m"] = np.radians(lat - lat0) * Re

test_trials = [21, 25, 34, 44, 54]
train_mask = ~df["trial"].isin(test_trials).values
test_mask  =  df["trial"].isin(test_trials).values
X_train, X_test = W[train_mask], W[test_mask]
xy_train = df.loc[train_mask, ["x_m", "y_m"]].values
xy_test  = df.loc[test_mask,  ["x_m", "y_m"]].values
fl_train = df.loc[train_mask, "floor"].values
fl_test  = df.loc[test_mask,  "floor"].values
floors_sorted = sorted(df["floor"].unique())

# WiFi kNN
knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=1).fit(X_train)
t0 = time.time()
dist_knn, idx_knn = knn.kneighbors(X_test)
t_knn = time.time() - t0
wts = 1.0 / (dist_knn + 1e-6)
xy_pred_knn = np.einsum("nk,nkd->nd", wts, xy_train[idx_knn]) / wts.sum(axis=1, keepdims=True)
rmse_knn = np.sqrt(np.mean(np.sum((xy_pred_knn - xy_test) ** 2, axis=1)))
fl_pred_knn = mode(fl_train[idx_knn], axis=1, keepdims=False).mode
acc_knn = accuracy_score(fl_test, fl_pred_knn)
f1_knn  = f1_score(fl_test, fl_pred_knn, average="macro", zero_division=0)

# =========================================================
# Physics-corrected pressure slope
# =========================================================
# Use trial 41's clean transition
g41 = df[(df.trial == 41) & (df.device == "SA52")]
m41 = g41.groupby("floor")["p_smooth"].mean()
HPA_PER_FLOOR = float(m41.iloc[0] - m41.iloc[1])   # ~0.33 hPa
print(f"\nPhysics-corrected hPa per floor (from trial 41): {HPA_PER_FLOOR:.3f}")

# =========================================================
# Pressure-only with WiFi anchor
# =========================================================
test_df = df.loc[test_mask].copy().reset_index(drop=True)
N_ANCHOR = 5
preds_p = np.zeros(len(test_df), dtype=int)

for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values
    neigh_flat = idx_knn[idx[:N_ANCHOR]].flatten()
    anchor_floor = int(mode(fl_train[neigh_flat], keepdims=False).mode)
    anchor_p = p[:N_ANCHOR].mean()
    delta_p = p - anchor_p
    d_floor = np.round(delta_p / HPA_PER_FLOOR).astype(int)
    preds_p[idx] = np.clip(anchor_floor + d_floor, min(floors_sorted), max(floors_sorted))

acc_p = accuracy_score(fl_test, preds_p)
f1_p  = f1_score(fl_test, preds_p, average="macro", zero_division=0)
print(f"Pressure-only (WiFi anchor, physics): {acc_p:.3f}  F1: {f1_p:.3f}")

# =========================================================
# Gated fusion: pressure only votes when transition detected
# =========================================================
TRANS_THRESH = 0.15  # hPa — half a floor

def wifi_floor_prob(neigh_floors_arr):
    p = np.zeros(len(floors_sorted))
    for f in neigh_floors_arr:
        if f in floors_sorted:
            p[floors_sorted.index(int(f))] += 1
    return p / p.sum() if p.sum() > 0 else np.ones(len(floors_sorted)) / len(floors_sorted)

preds_fused_gated = np.zeros(len(X_test), dtype=int)
n_pressure_fired = 0

for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values
    neigh_flat = idx_knn[idx[:N_ANCHOR]].flatten()
    anchor_floor = int(mode(fl_train[neigh_flat], keepdims=False).mode)
    anchor_p = p[:N_ANCHOR].mean()

    for j, row_i in enumerate(idx):
        p_w = wifi_floor_prob(fl_train[idx_knn[row_i]])
        delta_p = p[j] - anchor_p

        if abs(delta_p) >= TRANS_THRESH:
            n_pressure_fired += 1
            p_floor_float = anchor_floor + delta_p / HPA_PER_FLOOR
            d = np.array(floors_sorted) - p_floor_float
            logp = -0.5 * (d / 1.0) ** 2
            logp -= logp.max()
            p_p = np.exp(logp) / np.exp(logp).sum()
            p_fused = p_w * p_p
        else:
            p_fused = p_w

        preds_fused_gated[row_i] = floors_sorted[int(np.argmax(p_fused))]

acc_fg = accuracy_score(fl_test, preds_fused_gated)
f1_fg  = f1_score(fl_test, preds_fused_gated, average="macro", zero_division=0)
print(f"Gated fusion: {acc_fg:.3f}  F1: {f1_fg:.3f}  "
      f"(pressure fired on {n_pressure_fired}/{len(test_df)} samples)")

# =========================================================
# Summary
# =========================================================
summary = pd.DataFrame([
    {"Experiment": "WiFi kNN",                   "Floor_Acc": acc_knn, "Macro_F1": f1_knn, "XY_RMSE_m": rmse_knn},
    {"Experiment": "Pressure only (physics)",    "Floor_Acc": acc_p,   "Macro_F1": f1_p,   "XY_RMSE_m": np.nan},
    {"Experiment": "Gated fusion",               "Floor_Acc": acc_fg,  "Macro_F1": f1_fg,  "XY_RMSE_m": np.nan},
])
print("\n" + "="*70)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT_DIR, "experiment_summary_final.csv"), index=False)

# CHANGING IT

# In[2]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")

df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
df = df[df["trial"] != 31].reset_index(drop=True)
df = df.sort_values(["trial", "device", "position_timestamp"]).reset_index(drop=True)
df["p_smooth"] = (df.groupby(["trial", "device"])["pressure_hpa"]
                    .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))

def parse_fp(s):
    out = {}
    if not isinstance(s, str): return out
    for pair in s.split(";"):
        if ":" not in pair: continue
        b, r = pair.rsplit(":", 1)
        try: out[b] = float(r)
        except ValueError: continue
    return out

fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
all_bssids = sorted({b for fp in fps for b in fp})
bidx = {b: i for i, b in enumerate(all_bssids)}
W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)
for i, fp in enumerate(fps):
    for b, r in fp.items():
        W[i, bidx[b]] = r

lat, lon = df["latitude"].values, df["longitude"].values
lat0, lon0 = lat.mean(), lon.mean()
Re = 6371000.0
df["x_m"] = np.radians(lon - lon0) * Re * np.cos(np.radians(lat0))
df["y_m"] = np.radians(lat - lat0) * Re

test_trials = [21, 25, 34, 44, 54]
train_mask = ~df["trial"].isin(test_trials).values
test_mask  =  df["trial"].isin(test_trials).values
X_train, X_test = W[train_mask], W[test_mask]
fl_train = df.loc[train_mask, "floor"].values
fl_test  = df.loc[test_mask,  "floor"].values
floors_sorted = sorted(df["floor"].unique())

knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=1).fit(X_train)
dist_knn, idx_knn = knn.kneighbors(X_test)

HPA_PER_FLOOR = 0.301
test_df = df.loc[test_mask].copy().reset_index(drop=True)

# =========================================================
# Helper: WiFi floor distribution for a test sample
# =========================================================
def wifi_floor_dist(test_i):
    p = np.zeros(len(floors_sorted))
    for f in fl_train[idx_knn[test_i]]:
        if f in floors_sorted:
            p[floors_sorted.index(int(f))] += 1
    return p / p.sum()

# Precompute all WiFi distributions
wifi_probs_all = np.array([wifi_floor_dist(i) for i in range(len(X_test))])

# =========================================================
# Anchor strategies
# =========================================================

def anchor_majority(idx, N=5):
    """Current baseline: hard majority vote over first N samples."""
    votes = fl_train[idx_knn[idx[:N]].flatten()]
    return int(mode(votes, keepdims=False).mode)

def anchor_soft(idx, N=5):
    """Soft vote: average WiFi distributions over first N samples."""
    probs = np.zeros(len(floors_sorted))
    for row_i in idx[:N]:
        probs += wifi_probs_all[row_i]
    return floors_sorted[int(np.argmax(probs))]

def anchor_soft_long(idx, N=30):
    """Soft vote over a longer window."""
    return anchor_soft(idx, N=N)

def anchor_ml_search(idx, p, prior_anchor=None, lambda_prior=0.3):
    """
    Try every possible starting floor. For each candidate:
      - Run pressure tracking for the whole trial.
      - Score = mean log P_wifi(predicted floor) over all samples.
    Pick the anchor with the best score.
    Optionally bias toward a prior anchor (from majority or soft).
    """
    N_ANCHOR = 5
    anchor_p = p[:N_ANCHOR].mean()
    delta_p = p - anchor_p
    d_floor = np.round(delta_p / HPA_PER_FLOOR).astype(int)

    best_anchor = None
    best_score = -np.inf
    for a in floors_sorted:
        pred = np.clip(a + d_floor, min(floors_sorted), max(floors_sorted))
        log_scores = []
        for j, pf in enumerate(pred):
            fi = floors_sorted.index(int(pf))
            log_scores.append(np.log(wifi_probs_all[idx[j], fi] + 1e-9))
        score = np.mean(log_scores)
        if prior_anchor is not None:
            score -= lambda_prior * abs(a - prior_anchor)
        if score > best_score:
            best_score = score
            best_anchor = a
    return int(best_anchor)

# =========================================================
# Apply each strategy and evaluate
# =========================================================

print("=" * 90)
print("ANCHOR STRATEGY COMPARISON")
print("=" * 90)
print(f"{'Trial/Dev':<14}{'True':<6}{'Majority':<10}{'Soft5':<8}"
      f"{'Soft30':<8}{'ML':<6}{'ML+prior':<10}")
print("-" * 90)

results = {}
for strategy_name in ["majority", "soft5", "soft30", "ml", "ml_prior"]:
    results[strategy_name] = np.zeros(len(test_df), dtype=int)

for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values
    true_start = int(g["floor"].iloc[0])

    a_maj = anchor_majority(idx, N=5)
    a_s5  = anchor_soft(idx, N=5)
    a_s30 = anchor_soft_long(idx, N=30)
    a_ml  = anchor_ml_search(idx, p, prior_anchor=None)
    a_mlp = anchor_ml_search(idx, p, prior_anchor=a_s5, lambda_prior=0.3)

    print(f"{trial}/{device:<8}{true_start:<6}{a_maj:<10}{a_s5:<8}"
          f"{a_s30:<8}{a_ml:<6}{a_mlp:<10}")

    anchor_p = p[:5].mean()
    delta_p = p - anchor_p
    d_floor = np.round(delta_p / HPA_PER_FLOOR).astype(int)

    for name, a in [("majority", a_maj), ("soft5", a_s5),
                    ("soft30", a_s30), ("ml", a_ml), ("ml_prior", a_mlp)]:
        pred = np.clip(a + d_floor, min(floors_sorted), max(floors_sorted))
        results[name][idx] = pred

print("\n" + "=" * 90)
print("FLOOR ACCURACY BY ANCHOR STRATEGY")
print("=" * 90)
for name in results:
    acc = accuracy_score(fl_test, results[name])
    f1  = f1_score(fl_test, results[name], average="macro", zero_division=0)
    print(f"  {name:<14}  Acc = {acc:.3f}   F1 = {f1:.3f}")

# =========================================================
# Now try fusion with each anchor strategy
# =========================================================
print("\n" + "=" * 90)
print("FUSION WITH EACH ANCHOR STRATEGY")
print("=" * 90)

def pressure_floor_prob(pred_floor_float, sigma=1.0):
    d = np.array(floors_sorted) - pred_floor_float
    logp = -0.5 * (d / sigma) ** 2
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()

TRANS_THRESH = 0.15

for name in ["majority", "soft5", "soft30", "ml", "ml_prior"]:
    preds_naive = np.zeros(len(test_df), dtype=int)
    preds_gated = np.zeros(len(test_df), dtype=int)

    for (trial, device), g in test_df.groupby(["trial", "device"]):
        idx = g.index.values
        p = g["p_smooth"].values
        true_start = int(g["floor"].iloc[0])

        # Recompute anchor for this strategy
        if name == "majority":
            a = anchor_majority(idx, N=5)
        elif name == "soft5":
            a = anchor_soft(idx, N=5)
        elif name == "soft30":
            a = anchor_soft_long(idx, N=30)
        elif name == "ml":
            a = anchor_ml_search(idx, p)
        else:
            a_s5 = anchor_soft(idx, N=5)
            a = anchor_ml_search(idx, p, prior_anchor=a_s5, lambda_prior=0.3)

        anchor_p = p[:5].mean()

        for j, row_i in enumerate(idx):
            p_w = wifi_probs_all[row_i]
            delta_p = p[j] - anchor_p
            p_floor_float = a + delta_p / HPA_PER_FLOOR
            p_p = pressure_floor_prob(p_floor_float)

            # naive: always fuse
            p_fused_n = p_w * p_p
            preds_naive[row_i] = floors_sorted[int(np.argmax(p_fused_n))]

            # gated: only fuse when transition detected
            if abs(delta_p) >= TRANS_THRESH:
                p_fused_g = p_w * p_p
            else:
                p_fused_g = p_w
            preds_gated[row_i] = floors_sorted[int(np.argmax(p_fused_g))]

    acc_n = accuracy_score(fl_test, preds_naive)
    f1_n  = f1_score(fl_test, preds_naive, average="macro", zero_division=0)
    acc_g = accuracy_score(fl_test, preds_gated)
    f1_g  = f1_score(fl_test, preds_gated, average="macro", zero_division=0)

    print(f"  {name:<14}  Naive: {acc_n:.3f} (F1 {f1_n:.3f})   "
          f"Gated: {acc_g:.3f} (F1 {f1_g:.3f})")

print("\nFor reference:")
print(f"  WiFi-only kNN:  {accuracy_score(fl_test, mode(fl_train[idx_knn], axis=1, keepdims=False).mode):.3f}")

# In[6]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")

df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
df = df[df["trial"] != 31].reset_index(drop=True)
df = df.sort_values(["trial", "device", "position_timestamp"]).reset_index(drop=True)
df["p_smooth"] = (df.groupby(["trial", "device"])["pressure_hpa"]
                    .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))

def parse_fp(s):
    out = {}
    if not isinstance(s, str): return out
    for pair in s.split(";"):
        if ":" not in pair: continue
        b, r = pair.rsplit(":", 1)
        try: out[b] = float(r)
        except ValueError: continue
    return out

fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
all_bssids = sorted({b for fp in fps for b in fp})
bidx = {b: i for i, b in enumerate(all_bssids)}
W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)
for i, fp in enumerate(fps):
    for b, r in fp.items():
        W[i, bidx[b]] = r

test_trials = [21, 25, 34, 44, 54]
train_mask = ~df["trial"].isin(test_trials).values
test_mask  =  df["trial"].isin(test_trials).values
fl_test = df.loc[test_mask, "floor"].values
floors_sorted = sorted(df["floor"].unique())

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy().reset_index(drop=True)

# =========================================================
# Fixed Strategy 2: baseline estimated from FIRST N samples only
# =========================================================
print("=" * 70)
print("FIXED STRATEGY 2: Baseline from first N samples only")
print("=" * 70)

for N in [5, 10, 20, 30, 50]:
    preds = np.zeros(len(test_df), dtype=int)

    for (trial, device), g in test_df.groupby(["trial", "device"]):
        idx = g.index.values
        p = g["p_smooth"].values

        # ONLY the first N samples (realistic)
        p_baseline_test = p[:N].mean()

        # Build per-trial training references corrected to this baseline
        candidate_refs = []
        for tr_trial, tr in train_df[train_df["device"] == device].groupby("trial"):
            p_baseline_tr = tr["p_smooth"].iloc[:N].mean()
            offset = p_baseline_test - p_baseline_tr
            tr_ref = tr.groupby("floor")["p_smooth"].mean() + offset
            candidate_refs.append(tr_ref)

        # Average all corrected references
        combined = defaultdict(list)
        for ref in candidate_refs:
            for f, v in ref.items():
                combined[f].append(v)
        final_ref = {f: np.mean(v) for f, v in combined.items()}
        ref_pressures = np.array([final_ref.get(f, 9999) for f in floors_sorted])

        d = np.abs(p[:, None] - ref_pressures[None, :])
        preds[idx] = np.array(floors_sorted)[np.argmin(d, axis=1)]

    acc = accuracy_score(fl_test, preds)
    f1  = f1_score(fl_test, preds, average="macro", zero_division=0)
    print(f"  N={N:>3}  → Floor Acc: {acc:.3f}   F1: {f1:.3f}")

# =========================================================
# Reference values
# =========================================================
print("\n" + "=" * 70)
print("REFERENCE")
print("=" * 70)
print(f"  WiFi-only kNN:                0.690")
print(f"  Pressure-only (GT anchor):    0.472")
print(f"  Pressure-only (WiFi anchor):  0.255")
print(f"  Strategy 2 with full-trial mean (LEAKY): 0.347")

# DROPPING 2 and 6

# In[9]:


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Ashica\Downloads\2025_IPIN_Competition_Track03\2025_IPIN_Competition_Track03"
CSV_DIR = os.path.join(ROOT, "01 Logfiles", "01 - Training", "converted_csv")

# ---------- Load ----------
df = pd.read_csv(os.path.join(CSV_DIR, "combined_dataset_v3.csv"))
df["device"] = df["file"].str.extract(r"_(SA52|SG24)\.txt$")
df = df[df["trial"] != 31].reset_index(drop=True)

# ---------- FILTER: keep only floors 3, 4, 5 ----------
KEEP_FLOORS = [3, 4, 5]
df = df[df["floor"].isin(KEEP_FLOORS)].reset_index(drop=True)
print(f"After keeping only floors {KEEP_FLOORS}: {df.shape}")
print(f"Floor distribution: {df['floor'].value_counts().sort_index().to_dict()}")
print(f"Trial distribution: {df['trial'].value_counts().sort_index().to_dict()}")

df = df.sort_values(["trial", "device", "position_timestamp"]).reset_index(drop=True)
df["p_smooth"] = (df.groupby(["trial", "device"])["pressure_hpa"]
                    .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()))

def parse_fp(s):
    out = {}
    if not isinstance(s, str): return out
    for pair in s.split(";"):
        if ":" not in pair: continue
        b, r = pair.rsplit(":", 1)
        try: out[b] = float(r)
        except ValueError: continue
    return out

fps = df["wifi_fingerprint"].apply(parse_fp).tolist()
all_bssids = sorted({b for fp in fps for b in fp})
bidx = {b: i for i, b in enumerate(all_bssids)}
W = np.full((len(df), len(all_bssids)), -100.0, dtype=np.float32)
for i, fp in enumerate(fps):
    for b, r in fp.items():
        W[i, bidx[b]] = r

lat, lon = df["latitude"].values, df["longitude"].values
lat0, lon0 = lat.mean(), lon.mean()
Re = 6371000.0
df["x_m"] = np.radians(lon - lon0) * Re * np.cos(np.radians(lat0))
df["y_m"] = np.radians(lat - lat0) * Re

# ---------- Split ----------
test_trials = [21, 25, 34, 44, 54]
train_mask = ~df["trial"].isin(test_trials).values
test_mask  =  df["trial"].isin(test_trials).values
X_train, X_test = W[train_mask], W[test_mask]
xy_train = df.loc[train_mask, ["x_m", "y_m"]].values
xy_test  = df.loc[test_mask,  ["x_m", "y_m"]].values
fl_train = df.loc[train_mask, "floor"].values
fl_test  = df.loc[test_mask,  "floor"].values
floors_sorted = sorted(df["floor"].unique())

print(f"\nTrain: {train_mask.sum()}  Test: {test_mask.sum()}")
print(f"Train floor distribution: {dict(zip(*np.unique(fl_train, return_counts=True)))}")
print(f"Test floor distribution:  {dict(zip(*np.unique(fl_test,  return_counts=True)))}")

# ---------- WiFi kNN ----------
knn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=1).fit(X_train)
t0 = time.time()
dist_knn, idx_knn = knn.kneighbors(X_test)
t_knn = time.time() - t0

wts = 1.0 / (dist_knn + 1e-6)
xy_pred_knn = np.einsum("nk,nkd->nd", wts, xy_train[idx_knn]) / wts.sum(axis=1, keepdims=True)
rmse_knn = np.sqrt(np.mean(np.sum((xy_pred_knn - xy_test) ** 2, axis=1)))
fl_pred_knn = mode(fl_train[idx_knn], axis=1, keepdims=False).mode
acc_knn = accuracy_score(fl_test, fl_pred_knn)
f1_knn  = f1_score(fl_test, fl_pred_knn, average="macro", zero_division=0)

print(f"\nWiFi kNN: XY RMSE {rmse_knn:.2f} m   Floor Acc {acc_knn:.3f}   F1 {f1_knn:.3f}")

# ---------- WiFi STH ----------
K = 30
km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5, batch_size=3072)
km.fit(X_train)
train_bucket = km.predict(X_train)
test_bucket  = km.predict(X_test)
bucket2idx = defaultdict(list)
for i, b in enumerate(train_bucket):
    bucket2idx[b].append(i)
bucket_trees = {b: BallTree(X_train[idxs]) for b, idxs in bucket2idx.items() if len(idxs) >= 3}

t0 = time.time()
xy_pred_sth = np.zeros_like(xy_test)
fl_pred_sth = np.zeros(len(X_test), dtype=int)
for i in range(len(X_test)):
    b = int(test_bucket[i])
    cands = [b]
    if len(bucket2idx.get(b, [])) < 5:
        d2c = np.linalg.norm(km.cluster_centers_ - X_test[i], axis=1)
        cands = [int(x) for x in np.argsort(d2c)[:3]]
    ng, nd = [], []
    for cb in cands:
        if cb in bucket_trees:
            kk = min(5, len(bucket2idx[cb]))
            d, ii = bucket_trees[cb].query(X_test[i:i+1], k=kk)
            for j in ii[0]:
                ng.append(bucket2idx[cb][int(j)])
            nd.extend(d[0].tolist())
    if len(ng) < 5:
        d, ii = knn.kneighbors(X_test[i:i+1], n_neighbors=5)
        ng, nd = ii[0].tolist(), d[0].tolist()
    order = np.argsort(nd)[:5]
    ng = [ng[o] for o in order]
    nd = np.array([nd[o] for o in order])
    ww = 1.0 / (nd + 1e-6)
    xy_pred_sth[i] = (ww[:, None] * xy_train[ng]).sum(0) / ww.sum()
    fl_pred_sth[i] = mode(fl_train[ng], keepdims=False).mode

t_sth = time.time() - t0
rmse_sth = np.sqrt(np.mean(np.sum((xy_pred_sth - xy_test) ** 2, axis=1)))
acc_sth  = accuracy_score(fl_test, fl_pred_sth)
f1_sth   = f1_score(fl_test, fl_pred_sth, average="macro", zero_division=0)

print(f"WiFi STH: XY RMSE {rmse_sth:.2f} m   Floor Acc {acc_sth:.3f}   F1 {f1_sth:.3f}")

# ---------- Pressure slope ----------
all_drops = []
for (trial, device), g in df[train_mask].groupby(["trial", "device"]):
    medians = g.groupby("floor")["p_smooth"].median().sort_index()
    if len(medians) < 2: continue
    for i in range(len(medians)-1):
        drop = float(medians.iloc[i] - medians.iloc[i+1])
        if drop > 0: all_drops.append(drop)
HPA_PER_FLOOR = float(np.median(all_drops)) if all_drops else 0.3
print(f"\nhPa per floor: {HPA_PER_FLOOR:.3f}")

# ---------- Pressure-only (WiFi anchor) ----------
test_df = df.loc[test_mask].copy().reset_index(drop=True)
preds_p = np.zeros(len(test_df), dtype=int)
N_ANCHOR = 5

print("\nAnchor diagnostics:")
for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values
    neigh = idx_knn[idx[:N_ANCHOR]].flatten()
    anchor_floor = int(mode(fl_train[neigh], keepdims=False).mode)
    anchor_p = p[:N_ANCHOR].mean()
    delta_p = p - anchor_p
    d_floor = np.round(delta_p / HPA_PER_FLOOR).astype(int)
    preds_p[idx] = np.clip(anchor_floor + d_floor, min(floors_sorted), max(floors_sorted))

    true_start = int(g["floor"].iloc[0])
    print(f"  {trial}/{device}: true={true_start}  anchor={anchor_floor}  "
          f"Δp=[{delta_p.min():+.2f}, {delta_p.max():+.2f}]")

acc_p = accuracy_score(fl_test, preds_p)
f1_p  = f1_score(fl_test, preds_p, average="macro", zero_division=0)
print(f"\nPressure-only: Floor Acc {acc_p:.3f}   F1 {f1_p:.3f}")

# ---------- Naive Bayesian fusion ----------
def wifi_floor_prob(neigh_floors_arr):
    p = np.zeros(len(floors_sorted))
    for f in neigh_floors_arr:
        if f in floors_sorted:
            p[floors_sorted.index(int(f))] += 1
    return p / p.sum() if p.sum() > 0 else np.ones(len(floors_sorted))/len(floors_sorted)

def pressure_floor_prob(pred_floor_float, sigma=1.0):
    d = np.array(floors_sorted) - pred_floor_float
    logp = -0.5 * (d / sigma) ** 2
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()

preds_naive = np.zeros(len(X_test), dtype=int)
preds_gated = np.zeros(len(X_test), dtype=int)
TRANS_THRESH = 0.15

for (trial, device), g in test_df.groupby(["trial", "device"]):
    idx = g.index.values
    p = g["p_smooth"].values
    neigh = idx_knn[idx[:N_ANCHOR]].flatten()
    anchor_floor = int(mode(fl_train[neigh], keepdims=False).mode)
    anchor_p = p[:N_ANCHOR].mean()

    for j, row_i in enumerate(idx):
        p_w = wifi_floor_prob(fl_train[idx_knn[row_i]])
        delta_p = p[j] - anchor_p
        p_floor_float = anchor_floor + delta_p / HPA_PER_FLOOR
        p_p = pressure_floor_prob(p_floor_float)

        preds_naive[row_i] = floors_sorted[int(np.argmax(p_w * p_p))]

        if abs(delta_p) >= TRANS_THRESH:
            preds_gated[row_i] = floors_sorted[int(np.argmax(p_w * p_p))]
        else:
            preds_gated[row_i] = floors_sorted[int(np.argmax(p_w))]

acc_n = accuracy_score(fl_test, preds_naive)
acc_g = accuracy_score(fl_test, preds_gated)
f1_n  = f1_score(fl_test, preds_naive, average="macro", zero_division=0)
f1_g  = f1_score(fl_test, preds_gated, average="macro", zero_division=0)

print(f"\nNaive fusion:  Floor Acc {acc_n:.3f}   F1 {f1_n:.3f}")
print(f"Gated fusion:  Floor Acc {acc_g:.3f}   F1 {f1_g:.3f}")

# ---------- Summary ----------
print("\n" + "="*70)
print("SUMMARY — FLOORS 3, 4, 5 ONLY")
print("="*70)
summary = pd.DataFrame([
    {"Experiment": "WiFi kNN",              "Floor_Acc": acc_knn, "Macro_F1": f1_knn, "XY_RMSE": rmse_knn},
    {"Experiment": "WiFi STH",              "Floor_Acc": acc_sth, "Macro_F1": f1_sth, "XY_RMSE": rmse_sth},
    {"Experiment": "Pressure only",         "Floor_Acc": acc_p,   "Macro_F1": f1_p,   "XY_RMSE": np.nan},
    {"Experiment": "Naive fusion",          "Floor_Acc": acc_n,   "Macro_F1": f1_n,   "XY_RMSE": np.nan},
    {"Experiment": "Gated fusion",          "Floor_Acc": acc_g,   "Macro_F1": f1_g,   "XY_RMSE": np.nan},
])
print(summary.to_string(index=False))

# In[ ]:



