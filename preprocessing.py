"""
preprocessing.py
────────────────
Full preprocessing pipeline for iTrust SWaT and WaDi datasets,
designed for GAN / VAE generative model training.

Covers:
  - Dataset-specific CSV cleaning (SWaT A1/A2, WaDi A1/A2)
  - Outlier clipping + MinMax normalization (fit on normal only)
  - Binary actuator separation
  - Optional Savitzky-Golay smoothing
  - Temporal sliding windows with majority-vote labeling
  - Temporal train/val/test split (no leakage)
  - Attack-only DataLoader for generative model training
  - Balanced augmented DataLoader for downstream IDS training
"""

import re
import ast
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  Dataset-specific CSV cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _parse_swat_actuator_cell(val):
    """
    Older SWaT files store actuator states as dict-like strings:
        "{u'IsSystem': False, u'Name': u'Inactive', u'Value': 0}"
    This extracts the numeric 'Value' field.
    """
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ("Active", "1"):
        return 1.0
    if s in ("Inactive", "0"):
        return 0.0
    # try ast parse
    try:
        s_clean = s.replace("u'", "'")
        d = ast.literal_eval(s_clean)
        if isinstance(d, dict) and "Value" in d:
            return float(d["Value"])
    except Exception:
        pass
    # last resort: find first integer
    m = re.search(r"\d+", s)
    return float(m.group()) if m else np.nan


def load_swat(normal_path: str, attack_path: str) -> pd.DataFrame:
    """
    Load and clean both SWaT CSVs. Returns a single DataFrame with columns:
        [sensor/actuator features …]  +  'label' (0 = normal, 1 = attack)
    """
    def _read(path, label_val):
        df = pd.read_csv(path, low_memory=False)

        # strip column-name whitespace
        df.columns = df.columns.str.strip()

        # drop Timestamp (data is 1 Hz; position = time)
        df = df.drop(columns=["Timestamp"], errors="ignore")

        # parse label column
        if "Normal/Attack" in df.columns:
            df["label"] = (
                df["Normal/Attack"]
                .astype(str).str.strip()
                .map({"Normal": 0, "Attack": 1, "A ttack": 1})
                .fillna(label_val)
                .astype(int)
            )
            df = df.drop(columns=["Normal/Attack"])
        else:
            df["label"] = label_val

        # fix actuator columns stored as dict-strings
        for col in df.columns:
            if col == "label":
                continue
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) else ""
            if isinstance(sample, str) and ("Value" in sample or
                                            "Active" in str(sample)):
                df[col] = df[col].apply(_parse_swat_actuator_cell)

        # drop any remaining non-numeric columns
        label_series = df["label"].copy()
        df = df.select_dtypes(include=[np.number])
        df["label"] = label_series

        # forward-fill then zero-fill sensor dropouts
        df = df.ffill().fillna(0)
        return df

    normal = _read(normal_path, label_val=0)
    attack = _read(attack_path, label_val=1)
    df = pd.concat([normal, attack], ignore_index=True)
    print(f"[SWaT] Loaded  {len(normal):>7,} normal rows  +  "
          f"{len(attack):>6,} rows (inc. attacks)")
    print(f"       Features: {df.shape[1] - 1}  |  "
          f"Attack ratio: {df['label'].mean():.3%}")
    return df


def load_wadi(normal_path: str, attack_path: str) -> pd.DataFrame:
    """
    Load and clean both WaDi CSVs. Returns the same schema as load_swat.
    Uses the updated normal file (WADI_14days_new.csv) that has unstable
    readings removed by iTrust.
    """
    LABEL_COL_PATTERNS = [
        "Attack LABLE",        # original file spelling
        "Attack LABEL",
        "Attack_LABEL",
    ]

    def _find_label_col(df):
        for pat in LABEL_COL_PATTERNS:
            matches = [c for c in df.columns if pat.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    def _read(path, label_val):
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()

        # drop metadata columns
        df = df.drop(columns=["Row", "Date", "Time"], errors="ignore")

        lbl_col = _find_label_col(df)
        if lbl_col:
            # WaDi uses 1 = normal, -1 = attack
            df["label"] = (
                pd.to_numeric(df[lbl_col], errors="coerce")
                .map({1: 0, -1: 1})
                .fillna(label_val)
                .astype(int)
            )
            df = df.drop(columns=[lbl_col])
        else:
            df["label"] = label_val

        label_series = df["label"].copy()
        df = df.select_dtypes(include=[np.number])
        df["label"] = label_series
        df = df.ffill().fillna(0)
        return df

    normal = _read(normal_path, label_val=0)
    attack = _read(attack_path, label_val=1)
    df = pd.concat([normal, attack], ignore_index=True)
    print(f"[WaDi] Loaded  {len(normal):>7,} normal rows  +  "
          f"{len(attack):>6,} rows (inc. attacks)")
    print(f"       Features: {df.shape[1] - 1}  |  "
          f"Attack ratio: {df['label'].mean():.3%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Feature classification (continuous vs binary actuators)
# ─────────────────────────────────────────────────────────────────────────────

def identify_binary_columns(df: pd.DataFrame,
                             exclude: list[str] = ("label",),
                             threshold: int = 3) -> tuple[list[str], list[str]]:
    """
    Splits feature columns into:
      binary_cols     — columns with ≤ threshold unique values (actuators)
      continuous_cols — everything else (sensor readings)

    Returns (continuous_cols, binary_cols).
    """
    feature_cols = [c for c in df.columns if c not in exclude]
    binary_cols     = [c for c in feature_cols
                       if df[c].nunique() <= threshold]
    continuous_cols = [c for c in feature_cols
                       if c not in binary_cols]
    print(f"[features] continuous: {len(continuous_cols)}  "
          f"binary/actuator: {len(binary_cols)}")
    return continuous_cols, binary_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Outlier clipping + normalization (fit on normal only)
# ─────────────────────────────────────────────────────────────────────────────

class SensorScaler:
    """
    MinMax scaler that:
      - clips outliers to [p_low, p_high] percentile computed on NORMAL data
      - fits and scales continuous columns only
      - passes binary columns through unchanged
    """

    def __init__(self, clip_low: float = 1.0, clip_high: float = 99.0):
        self.clip_low  = clip_low
        self.clip_high = clip_high
        self.scaler         = MinMaxScaler(feature_range=(0.0, 1.0))
        self.clip_bounds_   = {}   # col → (lo, hi)
        self.continuous_cols: list[str] = []
        self.binary_cols:     list[str] = []

    def fit(self, df: pd.DataFrame,
            continuous_cols: list[str],
            binary_cols:     list[str]) -> "SensorScaler":
        """Fit on NORMAL-only rows."""
        self.continuous_cols = continuous_cols
        self.binary_cols     = binary_cols

        normal_df = df[df["label"] == 0][continuous_cols]

        # compute per-column clip bounds on normal data
        for col in continuous_cols:
            lo = np.percentile(normal_df[col].dropna(), self.clip_low)
            hi = np.percentile(normal_df[col].dropna(), self.clip_high)
            self.clip_bounds_[col] = (lo, hi)

        # clip then fit scaler
        clipped = normal_df.copy()
        for col in continuous_cols:
            lo, hi = self.clip_bounds_[col]
            clipped[col] = clipped[col].clip(lo, hi)
        self.scaler.fit(clipped.values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns scaled array (N, n_features) with column order:
            continuous_cols (scaled)  +  binary_cols (unchanged)
        """
        cont = df[self.continuous_cols].copy()
        for col in self.continuous_cols:
            lo, hi = self.clip_bounds_[col]
            cont[col] = cont[col].clip(lo, hi)
        scaled_cont = self.scaler.transform(cont.values)

        binary_arr  = df[self.binary_cols].values.astype(np.float32)
        return np.concatenate([scaled_cont, binary_arr], axis=1).astype(np.float32)

    def inverse_transform_continuous(self, arr: np.ndarray) -> np.ndarray:
        """Invert scaling on the continuous portion only (first N columns)."""
        n = len(self.continuous_cols)
        return self.scaler.inverse_transform(arr[:, :n])

    @property
    def feature_names(self) -> list[str]:
        return self.continuous_cols + self.binary_cols

    @property
    def n_features(self) -> int:
        return len(self.continuous_cols) + len(self.binary_cols)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Optional Savitzky-Golay smoothing (continuous sensors only)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_continuous(X_scaled: np.ndarray,
                      n_continuous: int,
                      window_length: int = 5,
                      polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to each continuous sensor column.
    Only the first n_continuous columns are smoothed; binary columns are left intact.
    Call AFTER scaling but BEFORE windowing.
    """
    X = X_scaled.copy()
    for i in range(n_continuous):
        X[:, i] = savgol_filter(X[:, i], window_length, polyorder)
    # clip back to [0, 1] in case filtering slightly overshoots
    X[:, :n_continuous] = np.clip(X[:, :n_continuous], 0.0, 1.0)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Sliding windows with majority-vote labeling
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(X: np.ndarray,
                 y: np.ndarray,
                 window: int,
                 stride: int,
                 attack_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert flat (T, F) arrays into overlapping windows.

    Args:
        X:                 scaled sensor array  (T, F)
        y:                 binary labels        (T,)
        window:            window length in timesteps
        stride:            step between consecutive windows
        attack_threshold:  fraction of attack timesteps needed to label
                           a window as attack (majority-vote default = 0.5)

    Returns:
        seqs   (N, window, F)
        labels (N,)  — 0 normal, 1 attack
    """
    seqs, labels = [], []
    for start in range(0, len(X) - window + 1, stride):
        s = X[start : start + window]
        l = y[start : start + window]
        seqs.append(s)
        labels.append(1 if l.mean() >= attack_threshold else 0)

    seqs   = np.stack(seqs,   axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    n_atk  = labels.sum()
    print(f"[windows] total: {len(seqs):,}  "
          f"normal: {(labels==0).sum():,}  "
          f"attack: {n_atk:,}  "
          f"({n_atk/len(labels):.2%} attack)")
    return seqs, labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  Temporal train / val / test split (no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(seqs:   np.ndarray,
                   labels: np.ndarray,
                   train_frac: float = 0.70,
                   val_frac:   float = 0.15,
                   ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Chronological split — does NOT shuffle.
    Windows are already in temporal order from make_windows.

    train_frac + val_frac must be < 1.0; the remainder becomes test.

    Returns:
        {
          "train": (seqs, labels),
          "val":   (seqs, labels),
          "test":  (seqs, labels),
        }
    """
    N  = len(seqs)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    splits = {
        "train": (seqs[:n_train],              labels[:n_train]),
        "val":   (seqs[n_train:n_train+n_val], labels[n_train:n_train+n_val]),
        "test":  (seqs[n_train+n_val:],        labels[n_train+n_val:]),
    }

    for name, (s, l) in splits.items():
        a = l.sum()
        print(f"  [{name:>5}] {len(s):>6,} windows  |  "
              f"attack: {a:>5,} ({a/max(len(l),1):.2%})")

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  PyTorch Datasets + DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

class CPSWindowDataset(Dataset):
    def __init__(self, seqs: np.ndarray, labels: np.ndarray):
        self.seqs   = torch.from_numpy(seqs)
        self.labels = torch.from_numpy(labels)

    def __len__(self):            return len(self.seqs)
    def __getitem__(self, idx):   return self.seqs[idx], self.labels[idx]


def make_attack_loader(seqs: np.ndarray, labels: np.ndarray,
                       batch_size: int, num_workers: int = 0) -> DataLoader:
    """
    DataLoader containing ONLY attack windows.
    Used to train the generative model (GAN / VAE).
    """
    mask = labels == 1
    if mask.sum() == 0:
        raise ValueError("No attack windows found. Check label mapping or "
                         "attack_threshold in make_windows.")
    ds = CWSWindowDataset(seqs[mask], labels[mask])
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      drop_last=True, num_workers=num_workers)


class CWSWindowDataset(CPSWindowDataset):
    pass   # alias for clarity in make_attack_loader


def make_balanced_loader(real_seqs:   np.ndarray,
                         real_labels: np.ndarray,
                         syn_atk_seqs: np.ndarray,
                         target_attack_frac: float = 0.40,
                         batch_size: int = 64,
                         num_workers: int = 0) -> DataLoader:
    """
    Build a balanced DataLoader for downstream IDS training.

    Combines:
      - all real normal windows
      - all real attack windows
      - enough synthetic attack windows to reach target_attack_frac

    target_attack_frac=0.40 means attacks = 40% of training set.

    NOTE: NEVER include synthetic samples in the test set.
    """
    real_normal_idx = np.where(real_labels == 0)[0]
    real_attack_idx = np.where(real_labels == 1)[0]
    n_normal  = len(real_normal_idx)
    n_r_atk   = len(real_attack_idx)

    # how many synthetic attack samples do we need?
    # total = n_normal / (1 - target_attack_frac)
    total_target  = int(n_normal / (1.0 - target_attack_frac))
    n_attack_need = total_target - n_normal
    n_syn_needed  = max(0, n_attack_need - n_r_atk)
    n_syn_used    = min(n_syn_needed, len(syn_atk_seqs))

    seqs_list   = [real_seqs[real_normal_idx],
                   real_seqs[real_attack_idx]]
    labels_list = [real_labels[real_normal_idx],
                   real_labels[real_attack_idx]]

    if n_syn_used > 0:
        idx = np.random.choice(len(syn_atk_seqs), n_syn_used, replace=False)
        seqs_list.append(syn_atk_seqs[idx])
        labels_list.append(np.ones(n_syn_used, dtype=np.int64))

    all_seqs   = np.concatenate(seqs_list,   axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # shuffle the combined set
    perm = np.random.permutation(len(all_seqs))
    all_seqs, all_labels = all_seqs[perm], all_labels[perm]

    final_atk_frac = all_labels.mean()
    print(f"[balanced loader] {len(all_seqs):,} windows  |  "
          f"synthetic added: {n_syn_used:,}  |  "
          f"attack fraction: {final_atk_frac:.2%}")

    ds = CPSWindowDataset(all_seqs, all_labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      drop_last=False, num_workers=num_workers)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  Convenience: full preprocessing pipeline in one call
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(
    normal_path:       str,
    attack_path:       str,
    dataset:           str   = "swat",    # "swat" | "wadi"
    window_size:       int   = 60,
    stride:            int   = 10,
    train_frac:        float = 0.70,
    val_frac:          float = 0.15,
    apply_smoothing:   bool  = True,
    clip_percentiles:  tuple = (1.0, 99.0),
) -> dict:
    """
    End-to-end preprocessing.

    Returns dict with keys:
        scaler      — fitted SensorScaler (save this; needed for inverse transform)
        splits      — {"train": (seqs, labels), "val": ..., "test": ...}
        feature_names
        n_features
        n_continuous
    """
    print(f"\n{'─'*55}")
    print(f"Preprocessing {dataset.upper()} dataset")
    print(f"{'─'*55}")

    # 1. Load
    if dataset.lower() == "swat":
        df = load_swat(normal_path, attack_path)
    elif dataset.lower() == "wadi":
        df = load_wadi(normal_path, attack_path)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'swat' or 'wadi'.")

    # 2. Identify feature types
    cont_cols, bin_cols = identify_binary_columns(df)

    # 3. Fit scaler on normal data only
    scaler = SensorScaler(clip_low=clip_percentiles[0],
                          clip_high=clip_percentiles[1])
    scaler.fit(df, cont_cols, bin_cols)

    # 4. Transform full dataset
    y = df["label"].values.astype(np.int64)
    X = scaler.transform(df)

    # 5. Optional smoothing on continuous columns
    if apply_smoothing:
        print("[smooth] Applying Savitzky-Golay to continuous sensors...")
        X = smooth_continuous(X, n_continuous=len(cont_cols))

    # 6. Sliding windows
    print(f"[windows] size={window_size}  stride={stride}")
    seqs, labels = make_windows(X, y, window_size, stride)

    # 7. Temporal split
    print("[split] Temporal train/val/test (no shuffle):")
    splits = temporal_split(seqs, labels, train_frac, val_frac)

    print(f"\n✓ Preprocessing complete. n_features={scaler.n_features} "
          f"(continuous={len(cont_cols)}, binary={len(bin_cols)})")

    return {
        "scaler":        scaler,
        "splits":        splits,
        "feature_names": scaler.feature_names,
        "n_features":    scaler.n_features,
        "n_continuous":  len(cont_cols),
    }


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── SWaT ─────────────────────────────────────────────────────────────────
    result = run_preprocessing(
        normal_path    = "data/swat/SWaT_Dataset_Normal_v1.csv",
        attack_path    = "data/swat/SWaT_Dataset_Attack_v0.csv",
        dataset        = "swat",
        window_size    = 60,
        stride         = 10,
        apply_smoothing= True,
    )

    train_seqs,  train_labels  = result["splits"]["train"]
    val_seqs,    val_labels    = result["splits"]["val"]
    test_seqs,   test_labels   = result["splits"]["test"]
    scaler       = result["scaler"]
    n_feat       = result["n_features"]

    # DataLoader for generative model training (attack windows only)
    atk_loader = make_attack_loader(train_seqs, train_labels, batch_size=64)

    print(f"\nAttack loader: {len(atk_loader)} batches of 64")
    print(f"n_features: {n_feat}")
    print(f"Feature names (first 5): {result['feature_names'][:5]}")

    # ── WaDi (same call, different paths and dataset flag) ────────────────────
    # result_wadi = run_preprocessing(
    #     normal_path  = "data/wadi/WADI_14days_new.csv",
    #     attack_path  = "data/wadi/WADI_attackdataLABLE.csv",
    #     dataset      = "wadi",
    #     window_size  = 120,   # WaDi has slower dynamics
    #     stride       = 20,
    # )
