"""
preprocessing_a12.py
────────────────────
Preprocessing pipeline built specifically for the SWaT.A12 (Mar 2026)
schema, which differs substantially from SWaT.A1/A2 (Dec 2015).

Key schema differences vs A1/A2:
  - Timestamp column: 't_stamp'  (not 'Timestamp')
  - Continuous sensors: 'XXX.Pv' suffix  (not raw names)
  - Actuators:          'XXX.Status' suffix, already 0/1 int (no dict-strings)
  - Alarms:             'XXX.Alarm' suffix, strings: 'Active'/'Inactive'/'Bad Input'
  - Process state:      'P1_STATE' … 'P6_STATE' (new — useful for conditional GAN/VAE)
  - Speed:              'P501.Speed', 'P502.Speed' (continuous)
  - NO label column     → clean dataset, no attacks

Strategy for GAN/VAE research:
  Option A (recommended): combine A12 normal data with SWaT.A1 attack windows
                          mapped by shared sensor IDs (LIT101, FIT101, …)
  Option B:               VAE anomaly detection — train on A12 normal only,
                          score reconstruction error against A1 attacks
  Option C:               inject synthetic attacks via known iTrust attack patterns
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS  (derived from inspecting 11-Mar-2026_0900_1700.csv)
# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP_COL = "t_stamp"

# 33 continuous process-value sensors
PV_SENSORS = [
    "LIT101.Pv", "FIT101.Pv",
    "FIT201.Pv", "AIT201.Pv", "AIT202.Pv", "AIT203.Pv",
    "AIT301.Pv", "AIT302.Pv", "AIT303.Pv", "LIT301.Pv",
    "FIT301.Pv", "DPIT301.Pv",
    "LIT401.Pv", "FIT401.Pv", "AIT401.Pv", "AIT402.Pv",
    "FIT501.Pv", "FIT502.Pv", "FIT503.Pv", "FIT504.Pv",
    "AIT501.Pv", "AIT502.Pv", "AIT503.Pv", "AIT504.Pv",
    "PIT501.Pv", "PIT502.Pv", "PIT503.Pv",
    "P501.Speed", "P502.Speed",          # continuous speed — treated like .Pv
    "LIT601.Pv", "LIT602.Pv", "FIT601.Pv", "FIT602.Pv",
]

# 32 binary actuator/status columns  (0 or 1 integer — no parsing needed)
STATUS_COLS = [
    "MV101.Status", "P101.Status", "P102.Status",
    "MV201.Status", "P201.Status", "P202.Status", "P203.Status",
    "P204.Status",  "P205.Status", "P206.Status", "P207.Status", "P208.Status",
    "MV301.Status", "MV302.Status", "MV303.Status", "MV304.Status",
    "P301.Status",  "P302.Status",
    "P401.Status",  "P402.Status", "P403.Status",  "P404.Status",
    "UV401.Status",
    "P501.Status",  "P502.Status",
    "MV501.Status", "MV502.Status", "MV503.Status", "MV504.Status",
    "P601.Status",  "P602.Status",  "P603.Status",
]

# 15 alarm columns — strings: "Active" / "Inactive" / "Bad Input"
ALARM_COLS = [
    "LS201.Alarm",   "LS202.Alarm",  "LSL203.Alarm", "LSLL203.Alarm",
    "PSH301.Alarm",  "DPSH301.Alarm",
    "LS401.Alarm",
    "PSH501.Alarm",  "PSL501.Alarm",
    "LSH601.Alarm",  "LSL601.Alarm",
    "LSH602.Alarm",  "LSL602.Alarm",
    "LSH603.Alarm",  "LSL603.Alarm",
]

# Alarm encoding: Inactive=0, Active=1, Bad Input=2 (sensor fault — keep separate)
ALARM_MAP = {"Inactive": 0, "Active": 1, "Bad Input": 2}

# 6 process-state columns (integer stage indicator; useful as conditioning signal)
STATE_COLS = [
    "P1_STATE", "P2_STATE", "P3_STATE",
    "P4_STATE", "P5_STATE", "P6_STATE",
]

# Canonical feature order fed to the model:
# [PV_SENSORS | STATUS_COLS | ALARM_COLS (encoded) | STATE_COLS]
ALL_FEATURES = PV_SENSORS + STATUS_COLS + ALARM_COLS + STATE_COLS
N_FEATURES   = len(ALL_FEATURES)   # 86
N_CONTINUOUS = len(PV_SENSORS)     # 33  (only these get MinMax-scaled)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN  (handles multiple session files automatically)
# ─────────────────────────────────────────────────────────────────────────────

def load_swat_a12(paths, label: int = 0) -> pd.DataFrame:
    """
    Load one or more SWaT.A12 CSV files and return a clean DataFrame.

    Args:
        paths:  str, Path, or list of str/Path.
                Pass a list to concatenate multiple session files
                (e.g. all CSVs in SWaT.A12_OTDataset_Mar_26/).
        label:  0 = normal (default), 1 = attack.
                A12 has no label column so you supply it manually.

    Returns:
        DataFrame with columns in ALL_FEATURES order + 'label'.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # ── drop timestamp ───────────────────────────────────────────────────────
    df = df.drop(columns=[TIMESTAMP_COL], errors="ignore")

    # ── encode alarm strings → 0/1/2 ─────────────────────────────────────────
    for col in ALARM_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip()
                .map(ALARM_MAP)
                .fillna(0)          # unknown → treat as Inactive
                .astype(np.int8)
            )

    # ── coerce everything to numeric, forward-fill dropouts ──────────────────
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0           # column missing in this session file
            print(f"  [warn] column '{col}' not found — filled with 0")

    df = df[ALL_FEATURES].ffill().fillna(0)
    df["label"] = label

    n = len(df)
    print(f"[SWaT.A12] Loaded {n:,} rows from {len(paths)} file(s)  "
          f"| label={label} | features={N_FEATURES}")
    return df


def load_wadi_a3(paths, label: int = 0) -> pd.DataFrame:
    """
    Load WaDi.A3 Dec 2023 Historian CSV (clean, no attacks).
    WaDi.A3 uses a different column naming convention — auto-detected.

    Args:
        paths:  str, Path, or list.
        label:  0 (always for this clean dataset).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # drop known metadata columns
    drop_cols = ["Row", "Date", "Time", "t_stamp",
                 "Attack LABLE (1:No Attack, -1:Attack)"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                 errors="ignore")

    # keep only numeric columns
    df = df.select_dtypes(include=[np.number]).ffill().fillna(0)
    df["label"] = label

    print(f"[WaDi.A3]  Loaded {len(df):,} rows from {len(paths)} file(s)  "
          f"| features={df.shape[1]-1}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SENSOR SCALER  (fit on normal-only, MinMax continuous columns only)
# ─────────────────────────────────────────────────────────────────────────────

class A12Scaler:
    """
    MinMax scaler for SWaT.A12 that:
      - clips continuous sensors to [p1, p99] of NORMAL data before fitting
      - scales only PV_SENSORS + STATE_COLS (continuous/ordinal)
      - leaves STATUS_COLS (0/1) and ALARM_COLS (0/1/2) unchanged
    """

    SCALE_COLS    = PV_SENSORS + STATE_COLS   # 39 columns to scale
    PASSTHRU_COLS = STATUS_COLS + ALARM_COLS  # 47 columns left unchanged

    def __init__(self, clip: tuple[float, float] = (1.0, 99.0)):
        self.clip        = clip
        self.scaler      = MinMaxScaler(feature_range=(0.0, 1.0))
        self.clip_lo_    = {}
        self.clip_hi_    = {}
        self._fitted     = False

    def fit(self, df: pd.DataFrame) -> "A12Scaler":
        """Fit on normal rows only (label == 0)."""
        normal = df[df["label"] == 0][self.SCALE_COLS]
        for col in self.SCALE_COLS:
            lo = np.percentile(normal[col].dropna(), self.clip[0])
            hi = np.percentile(normal[col].dropna(), self.clip[1])
            self.clip_lo_[col] = lo
            self.clip_hi_[col] = hi
        clipped = normal.copy()
        for col in self.SCALE_COLS:
            clipped[col] = clipped[col].clip(self.clip_lo_[col],
                                              self.clip_hi_[col])
        self.scaler.fit(clipped.values)
        self._fitted = True
        print(f"[A12Scaler] Fitted on {len(normal):,} normal rows  "
              f"| scale_cols={len(self.SCALE_COLS)}  "
              f"| passthru_cols={len(self.PASSTHRU_COLS)}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns float32 array (N, 86) in canonical feature order:
            [PV_SENSORS scaled | STATUS_COLS raw | ALARM_COLS raw | STATE_COLS scaled]

        Wait — that mixes ordering. We return in ALL_FEATURES order to keep
        it consistent with the column definition at the top of this file.
        Scaled features: PV_SENSORS (0..32) + STATE_COLS (placed last).
        """
        assert self._fitted, "Call fit() before transform()"

        # scale continuous + state
        cont = df[self.SCALE_COLS].copy()
        for col in self.SCALE_COLS:
            cont[col] = cont[col].clip(self.clip_lo_[col], self.clip_hi_[col])
        scaled = self.scaler.transform(cont.values).astype(np.float32)

        # passthrough
        passthru = df[self.PASSTHRU_COLS].values.astype(np.float32)

        # reassemble in ALL_FEATURES order
        # ALL_FEATURES = PV_SENSORS + STATUS_COLS + ALARM_COLS + STATE_COLS
        pv_arr     = scaled[:, :len(PV_SENSORS)]                     # 33
        state_arr  = scaled[:, len(PV_SENSORS):]                     #  6
        status_arr = passthru[:, :len(STATUS_COLS)]                  # 32
        alarm_arr  = passthru[:, len(STATUS_COLS):]                  # 15
        return np.concatenate(
            [pv_arr, status_arr, alarm_arr, state_arr], axis=1
        ).astype(np.float32)

    def inverse_transform_pv(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-scale the first 33 columns (PV sensors) only."""
        dummy = np.zeros((len(arr), len(self.SCALE_COLS)), dtype=np.float32)
        dummy[:, :len(PV_SENSORS)] = arr[:, :len(PV_SENSORS)]
        orig = self.scaler.inverse_transform(dummy)
        return orig[:, :len(PV_SENSORS)]


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIONAL SMOOTHING  (Savitzky-Golay on PV sensors only)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_pv(X: np.ndarray,
              window_length: int = 5,
              polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to the first N_CONTINUOUS columns.
    Call AFTER scaling, BEFORE windowing.
    """
    X = X.copy()
    for i in range(N_CONTINUOUS):
        X[:, i] = savgol_filter(X[:, i], window_length, polyorder)
    X[:, :N_CONTINUOUS] = np.clip(X[:, :N_CONTINUOUS], 0.0, 1.0)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 4. SLIDING WINDOWS
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(X: np.ndarray,
                 y: np.ndarray,
                 window: int,
                 stride: int,
                 attack_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    seqs, labels = [], []
    for start in range(0, len(X) - window + 1, stride):
        seqs.append(X[start : start + window])
        labels.append(1 if y[start : start + window].mean() >= attack_threshold
                      else 0)
    seqs   = np.stack(seqs).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    atk    = labels.sum()
    print(f"[windows] total={len(seqs):,}  normal={( labels==0).sum():,}  "
          f"attack={atk:,} ({atk/max(len(labels),1):.2%})")
    return seqs, labels


def temporal_split(seqs, labels,
                   train_frac=0.70, val_frac=0.15):
    N      = len(seqs)
    n_tr   = int(N * train_frac)
    n_val  = int(N * val_frac)
    splits = {
        "train": (seqs[:n_tr],              labels[:n_tr]),
        "val":   (seqs[n_tr:n_tr+n_val],    labels[n_tr:n_tr+n_val]),
        "test":  (seqs[n_tr+n_val:],        labels[n_tr+n_val:]),
    }
    for name, (s, l) in splits.items():
        a = l.sum()
        print(f"  [{name:>5}] {len(s):>6,} windows | "
              f"attack: {a:>5,} ({a/max(len(l),1):.2%})")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 5. STRATEGY A — merge A12 normal + A1 attacks
#    (recommended for GAN/VAE augmentation research)
# ─────────────────────────────────────────────────────────────────────────────

# Shared sensor IDs between SWaT.A12 and SWaT.A1/A2
# A12 name          →  A1/A2 name
A12_TO_A1_MAP = {
    "LIT101.Pv": "LIT101", "FIT101.Pv": "FIT101",
    "FIT201.Pv": "FIT201", "AIT201.Pv": "AIT201",
    "AIT202.Pv": "AIT202", "AIT203.Pv": "AIT203",
    "AIT301.Pv": "AIT301", "AIT302.Pv": "AIT302",
    "AIT303.Pv": "AIT303", "LIT301.Pv": "LIT301",
    "FIT301.Pv": "FIT301", "DPIT301.Pv": "DPIT301",
    "LIT401.Pv": "LIT401", "FIT401.Pv": "FIT401",
    "AIT401.Pv": "AIT401", "AIT402.Pv": "AIT402",
    "FIT501.Pv": "FIT501", "FIT502.Pv": "FIT502",
    "FIT503.Pv": "FIT503", "FIT504.Pv": "FIT504",
    "AIT501.Pv": "AIT501", "AIT502.Pv": "AIT502",
    "AIT503.Pv": "AIT503", "AIT504.Pv": "AIT504",
    "PIT501.Pv": "PIT501", "PIT502.Pv": "PIT502",
    "PIT503.Pv": "PIT503",
    "LIT601.Pv": "LIT601", "LIT602.Pv": "LIT602",
    "FIT601.Pv": "FIT601", "FIT602.Pv": "FIT602",
}
SHARED_PV_COLS_A12 = list(A12_TO_A1_MAP.keys())   # 31 cols shared
SHARED_PV_COLS_A1  = list(A12_TO_A1_MAP.values())  # 31 cols in A1


def load_swat_a1_attacks(attack_csv: str) -> pd.DataFrame:
    """
    Load SWaT.A1/A2 (Dec 2015) attack CSV, keep only shared sensor columns,
    and return with label=1. The A1 format uses raw column names without suffix.
    """
    df = pd.read_csv(attack_csv, low_memory=False)
    df.columns = df.columns.str.strip()

    # parse label
    if "Normal/Attack" in df.columns:
        df["label"] = (
            df["Normal/Attack"].astype(str).str.strip()
            .map({"Normal": 0, "Attack": 1, "A ttack": 1})
            .fillna(0).astype(int)
        )
    else:
        df["label"] = 1

    df = df.drop(columns=["Timestamp", "Normal/Attack"], errors="ignore")

    # keep only shared columns + label
    keep = [c for c in SHARED_PV_COLS_A1 if c in df.columns]
    df   = df[keep + ["label"]]
    df   = df.select_dtypes(include=[np.number, "object"]).copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.ffill().fillna(0)

    attacks_only = df[df["label"] == 1].copy()
    print(f"[SWaT.A1]  Loaded {len(df):,} rows  |  "
          f"attack rows: {len(attacks_only):,} "
          f"({len(attacks_only)/len(df):.2%})")
    return attacks_only


def align_a1_to_a12_schema(df_a1: pd.DataFrame,
                            scaler: A12Scaler,
                            window: int,
                            stride: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Rename A1 columns to A12 naming (.Pv suffix), zero-fill missing A12
    columns, then scale + window the attack rows to produce attack windows
    in the same (N, W, 86) shape as A12 windows.
    """
    # rename A1 → A12 column names
    df = df_a1.rename(columns={v: k for k, v in A12_TO_A1_MAP.items()})

    # add missing A12 columns with zeros
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df[ALL_FEATURES + ["label"]].copy()

    X = scaler.transform(df)
    y = df["label"].values.astype(np.int64)
    return make_windows(X, y, window, stride, attack_threshold=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PYTORCH DATASETS + DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────

class CPSDataset(torch.utils.data.Dataset):
    def __init__(self, seqs: np.ndarray, labels: np.ndarray):
        self.seqs   = torch.from_numpy(seqs)
        self.labels = torch.from_numpy(labels)

    def __len__(self):          return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx], self.labels[idx]


def make_attack_loader(seqs, labels, batch_size=64) -> DataLoader:
    """Attack-only loader — for GAN / VAE training."""
    mask = labels == 1
    if mask.sum() == 0:
        raise ValueError("No attack windows. Load A1 attack data first.")
    ds = CPSDataset(seqs[mask], labels[mask])
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def make_balanced_loader(real_seqs, real_labels,
                         syn_atk_seqs=None,
                         target_attack_frac=0.40,
                         batch_size=64) -> DataLoader:
    """Balanced loader (real normal + real attacks + optional synthetic attacks)."""
    normal_idx = np.where(real_labels == 0)[0]
    attack_idx = np.where(real_labels == 1)[0]
    n_normal   = len(normal_idx)
    n_need     = int(n_normal / (1.0 - target_attack_frac)) - n_normal
    n_real_atk = len(attack_idx)

    seqs_parts   = [real_seqs[normal_idx], real_seqs[attack_idx]]
    labels_parts = [real_labels[normal_idx], real_labels[attack_idx]]

    if syn_atk_seqs is not None and n_need > n_real_atk:
        n_syn = min(n_need - n_real_atk, len(syn_atk_seqs))
        idx   = np.random.choice(len(syn_atk_seqs), n_syn, replace=False)
        seqs_parts.append(syn_atk_seqs[idx])
        labels_parts.append(np.ones(n_syn, dtype=np.int64))
        print(f"[balanced] added {n_syn:,} synthetic attack windows")

    all_seqs   = np.concatenate(seqs_parts)
    all_labels = np.concatenate(labels_parts)
    perm       = np.random.permutation(len(all_seqs))
    ds = CPSDataset(all_seqs[perm], all_labels[perm])
    print(f"[balanced] total={len(ds):,}  "
          f"attack_frac={all_labels[perm].mean():.2%}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL PIPELINE  (Strategy A: A12 normal + A1 attacks)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_strategy_a(
    a12_normal_paths,          # list of A12 CSV paths (normal sessions)
    a1_attack_csv:   str,      # SWaT.A1 attack CSV path
    window_size:     int = 60,
    stride:          int = 10,
    apply_smoothing: bool = True,
    batch_size:      int = 64,
) -> dict:
    """
    Full preprocessing pipeline — Strategy A.

    Returns:
        scaler, splits (train/val/test), attack_loader, n_features
    """
    print("\n" + "─" * 60)
    print("Strategy A: SWaT.A12 normal  +  SWaT.A1 attack windows")
    print("─" * 60)

    # 1. Load A12 normal sessions
    df_normal = load_swat_a12(a12_normal_paths, label=0)

    # 2. Fit scaler on A12 normal
    scaler = A12Scaler().fit(df_normal)

    # 3. Scale + window A12 normal
    X_normal = scaler.transform(df_normal)
    if apply_smoothing:
        X_normal = smooth_pv(X_normal)
    y_normal = df_normal["label"].values.astype(np.int64)
    normal_seqs, normal_labels = make_windows(X_normal, y_normal,
                                              window_size, stride)

    # 4. Load + align A1 attack windows
    df_a1_attacks = load_swat_a1_attacks(a1_attack_csv)
    atk_seqs, atk_labels = align_a1_to_a12_schema(
        df_a1_attacks, scaler, window_size, stride
    )

    # 5. Merge into single pool (keep temporal order of normal; attacks appended)
    all_seqs   = np.concatenate([normal_seqs, atk_seqs])
    all_labels = np.concatenate([normal_labels, atk_labels])

    print(f"\nCombined pool: {len(all_seqs):,} windows  "
          f"| attack: {all_labels.sum():,} "
          f"({all_labels.mean():.2%})")

    # 6. Temporal split (normal portion in time order; attacks split randomly)
    n_tr  = int(len(normal_seqs) * 0.70)
    n_val = int(len(normal_seqs) * 0.15)
    n_atk_tr  = int(len(atk_seqs) * 0.70)
    n_atk_val = int(len(atk_seqs) * 0.15)

    splits = {
        "train": (
            np.concatenate([normal_seqs[:n_tr],         atk_seqs[:n_atk_tr]]),
            np.concatenate([normal_labels[:n_tr],        atk_labels[:n_atk_tr]]),
        ),
        "val": (
            np.concatenate([normal_seqs[n_tr:n_tr+n_val],
                            atk_seqs[n_atk_tr:n_atk_tr+n_atk_val]]),
            np.concatenate([normal_labels[n_tr:n_tr+n_val],
                            atk_labels[n_atk_tr:n_atk_tr+n_atk_val]]),
        ),
        "test": (
            np.concatenate([normal_seqs[n_tr+n_val:],
                            atk_seqs[n_atk_tr+n_atk_val:]]),
            np.concatenate([normal_labels[n_tr+n_val:],
                            atk_labels[n_atk_tr+n_atk_val:]]),
        ),
    }
    print("\nSplit summary:")
    for name, (s, l) in splits.items():
        print(f"  [{name:>5}] {len(s):>6,} windows | "
              f"attack: {l.sum():>5,} ({l.mean():.2%})")

    # 7. Attack-only loader for generative model training
    tr_seqs, tr_labels = splits["train"]
    attack_loader = make_attack_loader(tr_seqs, tr_labels, batch_size)

    return {
        "scaler":        scaler,
        "splits":        splits,
        "attack_loader": attack_loader,
        "n_features":    N_FEATURES,
        "n_continuous":  N_CONTINUOUS,
        "feature_names": ALL_FEATURES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    # All session CSVs in the SWaT.A12 OT dataset folder
    a12_files = sorted(glob.glob("data/swat_a12/*.csv"))

    result = run_pipeline_strategy_a(
        a12_normal_paths = a12_files,
        a1_attack_csv    = "data/swat_a1/SWaT_Dataset_Attack_v0.csv",
        window_size      = 60,
        stride           = 10,
        apply_smoothing  = True,
        batch_size       = 64,
    )

    train_seqs,  train_labels  = result["splits"]["train"]
    val_seqs,    val_labels    = result["splits"]["val"]
    test_seqs,   test_labels   = result["splits"]["test"]
    attack_loader = result["attack_loader"]
    n_feat        = result["n_features"]      # 86
    n_cont        = result["n_continuous"]    # 33

    print(f"\nReady for model training:")
    print(f"  n_features : {n_feat}")
    print(f"  n_continuous (scaled): {n_cont}")
    print(f"  attack_loader batches: {len(attack_loader)}")
    print(f"  train shape: {train_seqs.shape}")

    # Pass attack_loader to your WGAN-GP / β-VAE training functions
    # (from q1_cps_augmentation.py)
