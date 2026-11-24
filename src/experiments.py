import time
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score

def run_all_experiments(df, W, test_trials=[21, 25, 34, 44, 54]):
    train_mask = ~df["trial"].isin(test_trials).values
    test_mask  =  df["trial"].isin(test_trials).values
    
    X_train, X_test = W[train_mask], W[test_mask]
    xy_train = df.loc[train_mask, ["x_m", "y_m"]].values
    xy_test  = df.loc[test_mask,  ["x_m", "y_m"]].values
    fl_train = df.loc[train_mask, "floor"].values
    fl_test  = df.loc[test_mask,  "floor"].values
    floors_sorted = sorted(df["floor"].unique())

    # --- Calculate hPa per floor ---
    diffs = []
    for (trial, device), g in df[train_mask].groupby(["trial", "device"]):
        means = g.groupby("floor")["p_smooth"].mean().sort_index()
        if len(means) >= 2:
            for i in range(len(means)-1):
                drop = means.iloc[i] - means.iloc[i+1]
                diffs.append(drop)
    hPa_per_floor = float(np.mean(diffs)) if diffs else 0.4

    # --- EXP 2: WiFi kNN ---
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

    # --- EXP 1: Pressure Only ---
    test_df = df.loc[test_mask].copy().reset_index(drop=True)
    preds_pressure = np.zeros(len(test_df), dtype=int)
    for (trial, device), g in test_df.groupby(["trial", "device"]):
        idx = g.index.values
        p = g["p_smooth"].values
        first_idx = idx[:10]
        neigh_flat = idx_knn[first_idx].flatten()
        anchor_floor = int(mode(fl_train[neigh_flat], keepdims=False).mode)
        anchor_p = p[:10].mean()
        
        delta_p = p - anchor_p
        delta_floor = np.round(delta_p / hPa_per_floor).astype(int)
        pred = np.clip(anchor_floor + delta_floor, min(floors_sorted), max(floors_sorted))
        preds_pressure[idx] = pred
        
    acc_p = accuracy_score(fl_test, preds_pressure)
    f1_p  = f1_score(fl_test, preds_pressure, average="macro", zero_division=0)

    # --- EXP 3: WiFi STH ---
    K = 40
    km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=5, batch_size=3072).fit(X_train)
    test_bucket = km.predict(X_test)
    
    bucket2idx = defaultdict(list)
    for i, b in enumerate(km.predict(X_train)):
        bucket2idx[b].append(i)
    bucket_trees = {b: BallTree(X_train[idxs]) for b, idxs in bucket2idx.items() if len(idxs) >= 3}

    t0 = time.time()
    xy_pred_sth = np.zeros_like(xy_test)
    fl_pred_sth = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        b = int(test_bucket[i])
        cands = [b] if len(bucket2idx.get(b, [])) >= 5 else [int(x) for x in np.argsort(np.linalg.norm(km.cluster_centers_ - X_test[i], axis=1))[:3]]
        
        neigh_global, neigh_dist = [], []
        for cb in cands:
            if cb in bucket_trees:
                d, ii = bucket_trees[cb].query(X_test[i:i+1], k=min(5, len(bucket2idx[cb])))
                for j in ii[0]: neigh_global.append(bucket2idx[cb][int(j)])
                neigh_dist.extend(d[0].tolist())

        if len(neigh_global) < 5:
            d, ii = knn.kneighbors(X_test[i:i+1], n_neighbors=5)
            neigh_global, neigh_dist = ii[0].tolist(), d[0].tolist()
            
        ww = 1.0 / (np.array(neigh_dist[:5]) + 1e-6)
        xy_pred_sth[i] = (ww[:, None] * xy_train[neigh_global[:5]]).sum(0) / ww.sum()
        fl_pred_sth[i] = mode(fl_train[neigh_global[:5]], keepdims=False).mode

    t_sth = time.time() - t0
    rmse_sth = np.sqrt(np.mean(np.sum((xy_pred_sth - xy_test) ** 2, axis=1)))
    acc_sth  = accuracy_score(fl_test, fl_pred_sth)
    f1_sth   = f1_score(fl_test, fl_pred_sth, average="macro", zero_division=0)

    # --- EXP 4: Fusion ---
    preds_fused = np.zeros(len(X_test), dtype=int)
    def wifi_floor_prob(arr):
        p = np.zeros(len(floors_sorted))
        for f in arr:
            if f in floors_sorted: p[floors_sorted.index(int(f))] += 1
        return p / p.sum() if p.sum() > 0 else np.ones(len(floors_sorted)) / len(floors_sorted)

    def pressure_floor_prob(pred_f, sigma=0.7):
        d = np.array(floors_sorted) - pred_f
        p = np.exp(-0.5 * (d / sigma) ** 2)
        return p / p.sum()

    for (trial, device), g in test_df.groupby(["trial", "device"]):
        idx = g.index.values
        p = g["p_smooth"].values
        anchor_floor = int(mode(fl_train[idx_knn[idx[:10]].flatten()], keepdims=False).mode)
        anchor_p = p[:10].mean()
        for j, row_i in enumerate(idx):
            p_w = wifi_floor_prob(fl_train[idx_knn[row_i]])
            p_f = anchor_floor + (p[j] - anchor_p) / hPa_per_floor
            preds_fused[row_i] = floors_sorted[int(np.argmax(p_w * pressure_floor_prob(p_f)))]

    acc_fus = accuracy_score(fl_test, preds_fused)
    f1_fus  = f1_score(fl_test, preds_fused, average="macro", zero_division=0)

    # --- Summary ---
    return pd.DataFrame([
        {"Experiment": "Pressure only", "Floor_Acc": acc_p,   "Macro_F1": f1_p,   "XY_RMSE_m": np.nan,  "ms_per_sample": np.nan},
        {"Experiment": "WiFi kNN",      "Floor_Acc": acc_knn, "Macro_F1": f1_knn, "XY_RMSE_m": rmse_knn,"ms_per_sample": t_knn/len(X_test)*1000},
        {"Experiment": "WiFi STH",      "Floor_Acc": acc_sth, "Macro_F1": f1_sth, "XY_RMSE_m": rmse_sth,"ms_per_sample": t_sth/len(X_test)*1000},
        {"Experiment": "Fusion",        "Floor_Acc": acc_fus, "Macro_F1": f1_fus, "XY_RMSE_m": np.nan,  "ms_per_sample": np.nan},
    ])