"""
StageGenX — Stage-Conditioned Generative Attack Synthesis
for Coupled Industrial Control Systems

Paper: "StageGenX: Process-Stage-Aware Generative Augmentation for
        Cyber-Physical Attack Detection in Water Critical Infrastructure"
Target: IEEE Transactions on Information Forensics and Security

Architecture:
  - Stage embedding layer (P1–P6 → 16-dim vector)
  - Stage-Conditioned VAE  (SCVAE):  LSTM encoder + decoder, stage injected at both ends
  - Stage-Conditioned WGAN-GP (SCWGAN-GP): LSTM generator + critic, stage injected
  - CrossOver normal model (Topic 2 folded): VAE trained on coupled SWaT+WaDi A12/A3
  - Downstream IDS: LSTM-AE for anomaly scoring (paper-ready evaluation)

Datasets:
  - SWaT.A12 (Mar 2026)  → normal windows (86 features, .Pv / .Status / .Alarm / P_STATE)
  - SWaT.A1  (Dec 2015)  → attack windows mapped to A12 schema (31 shared sensors)
  - WaDi.A3  (Dec 2023)  → CrossOver normal (for Topic 2 section)

Run:
  python stagegenx.py --mode train_all   # trains SCVAE + SCWGAN-GP + CrossOver VAE
  python stagegenx.py --mode eval        # runs full experiment table
  python stagegenx.py --mode ablation    # stage-conditioned vs unconditioned comparison
"""

import os, math, argparse, random, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── global config ───────────────────────────────────────────────────────────
CFG = dict(
    # paths — update to your local paths
    a12_normal_glob  = "data/swat_a12/*.csv",
    a1_attack_csv    = "data/swat_a1/SWaT_Dataset_Attack_v0.csv",
    wadi_a3_glob     = "data/wadi_a3/*.csv",

    # windowing
    window_size      = 60,
    stride           = 10,

    # model dims
    n_features       = 86,     # from preprocessing_a12.py
    n_stages         = 6,      # P1–P6
    stage_emb_dim    = 16,
    latent_dim       = 32,
    hidden_dim       = 128,
    n_layers         = 2,

    # training
    batch_size       = 64,
    lr               = 1e-3,
    epochs_vae       = 200,
    epochs_gan       = 300,
    beta             = 4.0,    # β-VAE KL weight
    n_critic         = 5,      # WGAN-GP critic steps per G step
    lam_gp           = 10,     # gradient penalty coefficient

    # generation
    n_synthetic      = 1000,   # per stage

    # output
    out_dir          = "results/stagegenx",
)
os.makedirs(CFG["out_dir"], exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET HELPERS  (mirrors preprocessing_a12.py — self-contained here)
# ══════════════════════════════════════════════════════════════════════════════

# Column groups from SWaT.A12 schema
PV_COLS = [
    "LIT101.Pv","FIT101.Pv","FIT201.Pv","AIT201.Pv","AIT202.Pv","AIT203.Pv",
    "AIT301.Pv","AIT302.Pv","AIT303.Pv","LIT301.Pv","FIT301.Pv","DPIT301.Pv",
    "LIT401.Pv","FIT401.Pv","AIT401.Pv","AIT402.Pv","FIT501.Pv","FIT502.Pv",
    "FIT503.Pv","FIT504.Pv","AIT501.Pv","AIT502.Pv","AIT503.Pv","AIT504.Pv",
    "PIT501.Pv","PIT502.Pv","PIT503.Pv","P501.Speed","P502.Speed",
    "LIT601.Pv","LIT602.Pv","FIT601.Pv","FIT602.Pv",
]
STATUS_COLS = [
    "MV101.Status","P101.Status","P102.Status","MV201.Status","P201.Status",
    "P202.Status","P203.Status","P204.Status","P205.Status","P206.Status",
    "P207.Status","P208.Status","MV301.Status","MV302.Status","MV303.Status",
    "MV304.Status","P301.Status","P302.Status","P401.Status","P402.Status",
    "P403.Status","P404.Status","UV401.Status","P501.Status","P502.Status",
    "MV501.Status","MV502.Status","MV503.Status","MV504.Status","P601.Status",
    "P602.Status","P603.Status",
]
ALARM_COLS = [
    "LS201.Alarm","LS202.Alarm","LSL203.Alarm","LSLL203.Alarm","PSH301.Alarm",
    "DPSH301.Alarm","LS401.Alarm","PSH501.Alarm","PSL501.Alarm","LSH601.Alarm",
    "LSL601.Alarm","LSH602.Alarm","LSL602.Alarm","LSH603.Alarm","LSL603.Alarm",
]
STATE_COLS  = ["P1_STATE","P2_STATE","P3_STATE","P4_STATE","P5_STATE","P6_STATE"]
ALL_FEATS   = PV_COLS + STATUS_COLS + ALARM_COLS + STATE_COLS  # 86 cols
ALARM_MAP   = {"Inactive": 0, "Active": 1, "Bad Input": 2}
A12_TO_A1   = {f.replace(".Pv",""):f for f in PV_COLS if ".Pv" in f}  # A1→A12


def _load_csvs(paths, label=0):
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["t_stamp","Timestamp","Row","Date","Time"], errors="ignore")
    for col in ALARM_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(ALARM_MAP).fillna(0).astype(np.int8)
    if "Normal/Attack" in df.columns:
        df["label"] = df["Normal/Attack"].astype(str).str.strip().map(
            {"Normal":0,"Attack":1,"A ttack":1}).fillna(label).astype(int)
        df = df.drop(columns=["Normal/Attack"])
    else:
        df["label"] = label
    for col in ALL_FEATS:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[ALL_FEATS + ["label"]].ffill().fillna(0)
    return df


def load_a12_normal(glob_pattern):
    paths = sorted(glob.glob(glob_pattern))
    if not paths: raise FileNotFoundError(f"No files: {glob_pattern}")
    return _load_csvs(paths, label=0)


def load_a1_attacks(csv_path):
    """Load SWaT.A1 attack CSV and remap column names to A12 schema."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Timestamp"], errors="ignore")
    if "Normal/Attack" in df.columns:
        df["label"] = df["Normal/Attack"].astype(str).str.strip().map(
            {"Normal":0,"Attack":1,"A ttack":1}).fillna(0).astype(int)
        df = df.drop(columns=["Normal/Attack"])
    # rename A1 column names → A12 .Pv names
    rename = {v.replace(".Pv",""):v for v in PV_COLS if ".Pv" in v}
    df = df.rename(columns=rename)
    for col in ALL_FEATS:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[ALL_FEATS + ["label"]].ffill().fillna(0)
    return df[df["label"] == 1].copy()


def fit_scaler(df_normal):
    scaler = MinMaxScaler()
    pv_and_state = PV_COLS + STATE_COLS
    normal_vals  = df_normal[pv_and_state].copy()
    for col in pv_and_state:
        lo = np.percentile(normal_vals[col].dropna(), 1)
        hi = np.percentile(normal_vals[col].dropna(), 99)
        normal_vals[col] = normal_vals[col].clip(lo, hi)
    scaler.fit(normal_vals.values)
    return scaler, pv_and_state


def transform(df, scaler, scale_cols):
    cont = df[scale_cols].copy()
    X_scaled = scaler.transform(cont.values)
    status_arr = df[STATUS_COLS].values.astype(np.float32)
    alarm_arr  = df[ALARM_COLS].values.astype(np.float32) / 2.0  # 0/0.5/1
    # reassemble: PV(33) + Speed(2 embedded in PV) + Status(32) + Alarm(15) + State(6) = 86
    # scale_cols = PV_COLS + STATE_COLS  (39 total)
    pv_arr    = X_scaled[:, :len(PV_COLS)]
    state_arr = X_scaled[:, len(PV_COLS):]
    return np.concatenate([pv_arr, status_arr, alarm_arr, state_arr], axis=1).astype(np.float32)


def make_windows(X, y, stage_col_idx, window, stride):
    """Returns (seqs, labels, stages) with stage = majority stage in window."""
    seqs, labels, stages = [], [], []
    for s in range(0, len(X) - window + 1, stride):
        seqs.append(X[s:s+window])
        labels.append(1 if y[s:s+window].mean() >= 0.5 else 0)
        # dominant stage in window (0-indexed: P1=0 … P6=5)
        raw_stage = X[s:s+window, stage_col_idx]
        dominant  = int(np.round(raw_stage.mean()).clip(1, 6)) - 1
        stages.append(dominant)
    return (np.stack(seqs).astype(np.float32),
            np.array(labels, dtype=np.int64),
            np.array(stages, dtype=np.int64))


STAGE_COL_IDX = len(PV_COLS) + len(STATUS_COLS) + len(ALARM_COLS)  # first STATE col index


# ── PyTorch dataset ───────────────────────────────────────────────────────────

class CPSDataset(Dataset):
    def __init__(self, seqs, labels, stages):
        self.seqs   = torch.from_numpy(seqs)
        self.labels = torch.from_numpy(labels)
        self.stages = torch.from_numpy(stages)

    def __len__(self):          return len(self.seqs)
    def __getitem__(self, i):   return self.seqs[i], self.labels[i], self.stages[i]


def attack_loader(seqs, labels, stages, batch_size):
    mask = labels == 1
    ds   = CPSDataset(seqs[mask], labels[mask], stages[mask])
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STAGE EMBEDDING  (shared by all models)
# ══════════════════════════════════════════════════════════════════════════════

class StageEmbedding(nn.Module):
    """Learnable lookup table: stage index (0–5) → dense vector."""
    def __init__(self, n_stages=6, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_stages, emb_dim)

    def forward(self, stage_idx):
        return self.emb(stage_idx)   # (B, emb_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SCVAE  — Stage-Conditioned Variational Autoencoder
# ══════════════════════════════════════════════════════════════════════════════

class SCVAEEncoder(nn.Module):
    def __init__(self, n_feat, hidden, n_layers, latent, emb_dim):
        super().__init__()
        self.lstm   = nn.LSTM(n_feat + emb_dim, hidden, n_layers, batch_first=True)
        self.fc_mu  = nn.Linear(hidden, latent)
        self.fc_lv  = nn.Linear(hidden, latent)

    def forward(self, x, stage_emb):
        # inject stage embedding at every timestep
        B, W, F = x.shape
        s = stage_emb.unsqueeze(1).expand(B, W, -1)   # (B, W, emb_dim)
        inp = torch.cat([x, s], dim=-1)                # (B, W, F+emb_dim)
        _, (h, _) = self.lstm(inp)
        h = h[-1]                                       # (B, hidden)
        return self.fc_mu(h), self.fc_lv(h)


class SCVAEDecoder(nn.Module):
    def __init__(self, latent, hidden, n_layers, n_feat, window, emb_dim):
        super().__init__()
        self.window  = window
        self.fc_in   = nn.Linear(latent + emb_dim, hidden)
        self.lstm    = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.fc_out  = nn.Linear(hidden, n_feat)

    def forward(self, z, stage_emb):
        inp = torch.cat([z, stage_emb], dim=-1)         # (B, latent+emb_dim)
        h   = self.fc_in(inp).unsqueeze(1).expand(-1, self.window, -1)
        out, _ = self.lstm(h)
        return torch.sigmoid(self.fc_out(out))           # (B, W, F) in [0,1]


class SCVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        F, H, L, Z, E, W = (cfg["n_features"], cfg["hidden_dim"],
                             cfg["n_layers"], cfg["latent_dim"],
                             cfg["stage_emb_dim"], cfg["window_size"])
        self.stage_emb = StageEmbedding(cfg["n_stages"], E)
        self.encoder   = SCVAEEncoder(F, H, L, Z, E)
        self.decoder   = SCVAEDecoder(Z, H, L, F, W, E)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def forward(self, x, stage):
        emb        = self.stage_emb(stage)
        mu, lv     = self.encoder(x, emb)
        z          = self.reparameterize(mu, lv)
        recon      = self.decoder(z, emb)
        return recon, mu, lv

    def sample(self, n, stage_idx, device):
        """Generate n synthetic windows for a given stage (0-indexed)."""
        stages = torch.full((n,), stage_idx, dtype=torch.long, device=device)
        emb    = self.stage_emb(stages)
        z      = torch.randn(n, self.decoder.fc_in.in_features - emb.shape[-1],
                             device=device)
        with torch.no_grad():
            return self.decoder(z, emb).cpu().numpy()


def scvae_loss(recon, x, mu, lv, beta):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_scvae(loader, cfg, device):
    model = SCVAE(cfg).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg["lr"])
    model.train()
    for epoch in range(cfg["epochs_vae"]):
        total = 0.0
        for x, _, stage in loader:
            x, stage = x.to(device), stage.to(device)
            recon, mu, lv = model(x, stage)
            loss, rl, kl  = scvae_loss(recon, x, mu, lv, cfg["beta"])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"[SCVAE] ep {epoch+1:>3} | loss={total/len(loader):.4f} "
                  f"recon={rl.item():.4f} kl={kl.item():.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SCWGAN-GP  — Stage-Conditioned Wasserstein GAN with Gradient Penalty
# ══════════════════════════════════════════════════════════════════════════════

class SCGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        Z, H, L, F, W, E = (cfg["latent_dim"], cfg["hidden_dim"],
                             cfg["n_layers"], cfg["n_features"],
                             cfg["window_size"], cfg["stage_emb_dim"])
        self.stage_emb = StageEmbedding(cfg["n_stages"], E)
        self.fc_in     = nn.Linear(Z + E, H)
        self.lstm      = nn.LSTM(H, H, L, batch_first=True)
        self.fc_out    = nn.Linear(H, F)
        self.window    = W

    def forward(self, z, stage):
        emb = self.stage_emb(stage)
        inp = torch.cat([z, emb], dim=-1)
        h   = self.fc_in(inp).unsqueeze(1).expand(-1, self.window, -1)
        out, _ = self.lstm(h)
        return torch.sigmoid(self.fc_out(out))          # (B, W, F)


class SCCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        F, H, L, E = (cfg["n_features"], cfg["hidden_dim"],
                      cfg["n_layers"], cfg["stage_emb_dim"])
        self.stage_emb = StageEmbedding(cfg["n_stages"], E)
        self.lstm      = nn.LSTM(F + E, H, L, batch_first=True)
        self.fc_out    = nn.Linear(H, 1)

    def forward(self, x, stage):
        B, W, _ = x.shape
        emb = self.stage_emb(stage).unsqueeze(1).expand(B, W, -1)
        inp = torch.cat([x, emb], dim=-1)
        _, (h, _) = self.lstm(inp)
        return self.fc_out(h[-1])                        # (B, 1)


def _gradient_penalty(critic, real, fake, stage, device):
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_i    = critic(interp, stage)
    grads  = torch.autograd.grad(d_i, interp,
                                  grad_outputs=torch.ones_like(d_i),
                                  create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=[1,2]) - 1) ** 2).mean()


def train_scwgan(loader, cfg, device):
    G   = SCGenerator(cfg).to(device)
    C   = SCCritic(cfg).to(device)
    optG = optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
    optC = optim.Adam(C.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))

    G.train(); C.train()
    for epoch in range(cfg["epochs_gan"]):
        for real_x, _, stage in loader:
            real_x, stage = real_x.to(device), stage.to(device)
            B = real_x.size(0)

            # critic updates
            for _ in range(cfg["n_critic"]):
                z    = torch.randn(B, cfg["latent_dim"], device=device)
                fake = G(z, stage).detach()
                gp   = _gradient_penalty(C, real_x, fake, stage, device)
                lC   = C(fake, stage).mean() - C(real_x, stage).mean() + cfg["lam_gp"] * gp
                optC.zero_grad(); lC.backward(); optC.step()

            # generator update
            z    = torch.randn(B, cfg["latent_dim"], device=device)
            fake = G(z, stage)
            lG   = -C(fake, stage).mean()
            optG.zero_grad(); lG.backward(); optG.step()

        if (epoch + 1) % 100 == 0:
            print(f"[SCWGAN] ep {epoch+1:>3} | C={lC.item():.4f}  G={lG.item():.4f}")

    return G


def generate_stage(model, stage_idx, n, cfg, device, model_type="scvae"):
    """Generate n windows for stage_idx (0-based). Works for SCVAE or SCGenerator."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, n, cfg["batch_size"]):
            bs    = min(cfg["batch_size"], n - start)
            stage = torch.full((bs,), stage_idx, dtype=torch.long, device=device)
            if model_type == "scvae":
                emb  = model.stage_emb(stage)
                z    = torch.randn(bs, cfg["latent_dim"], device=device)
                x    = model.decoder(z, emb)
            else:  # scwgan
                z    = torch.randn(bs, cfg["latent_dim"], device=device)
                x    = model(z, stage)
            out.append(x.cpu().numpy())
    return np.concatenate(out, axis=0)[:n]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TOPIC 2 FOLDED IN — CrossOver Normal VAE
#     Trains a standard β-VAE on the coupled SWaT.A12 + WaDi.A3 normal data.
#     Reconstruction error is used as anomaly score for the CrossOver section.
# ══════════════════════════════════════════════════════════════════════════════

class CrossOverVAE(nn.Module):
    """
    Unconditioned β-VAE trained on CrossOver-mode normal data.
    At inference, reconstruction error = anomaly score.
    """
    def __init__(self, n_feat, hidden, n_layers, latent, window):
        super().__init__()
        self.window  = window
        # encoder
        self.enc_lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True)
        self.fc_mu    = nn.Linear(hidden, latent)
        self.fc_lv    = nn.Linear(hidden, latent)
        # decoder
        self.fc_dec   = nn.Linear(latent, hidden)
        self.dec_lstm = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.fc_out   = nn.Linear(hidden, n_feat)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)
        h = h[-1]
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        h   = self.fc_dec(z).unsqueeze(1).expand(-1, self.window, -1)
        out, _ = self.dec_lstm(h)
        return torch.sigmoid(self.fc_out(out))

    def forward(self, x):
        mu, lv = self.encode(x)
        z      = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decode(z), mu, lv

    def anomaly_score(self, x):
        """Per-sample MSE reconstruction error (lower = more normal)."""
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(x)
            return F.mse_loss(recon, x, reduction="none").mean(dim=[1,2])


def train_crossover_vae(normal_loader, cfg, device):
    """Train VAE on combined SWaT.A12 + WaDi.A3 CrossOver normal data."""
    model = CrossOverVAE(
        cfg["n_features"], cfg["hidden_dim"], cfg["n_layers"],
        cfg["latent_dim"],  cfg["window_size"]
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    model.train()
    for epoch in range(cfg["epochs_vae"]):
        total = 0.0
        for x, *_ in normal_loader:
            x = x.to(device)
            recon, mu, lv = model(x)
            loss = (F.mse_loss(recon, x) +
                    cfg["beta"] * (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"[CrossOverVAE] ep {epoch+1:>3} | loss={total/len(normal_loader):.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DOWNSTREAM IDS  — LSTM Autoencoder for anomaly detection
#     Trained on augmented (real normal + synthetic attacks) data.
#     Anomaly score = reconstruction error threshold.
# ══════════════════════════════════════════════════════════════════════════════

class LSTMAE(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layers=2):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, n_layers, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        B, W, _ = x.shape
        _, (h, c) = self.enc(x)
        dec_inp   = torch.zeros(B, W, h.shape[-1], device=x.device)
        out, _    = self.dec(dec_inp, (h, c))
        return torch.sigmoid(self.out(out))

    def anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            return F.mse_loss(self(x), x, reduction="none").mean(dim=[1,2])


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION METRICS  (paper-ready)
# ══════════════════════════════════════════════════════════════════════════════

def mmd_rbf(X, Y, gamma=1.0):
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(X, X, gamma); YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def tstr_evaluation(syn_atk, real_seqs, real_labels, seed=42):
    """
    Train-on-Synthetic-Test-on-Real.
    Returns dict with precision, recall, F1, AUC-ROC for attack class.
    """
    flat = lambda a: a.reshape(len(a), -1)
    norm_mask = real_labels == 0
    X_train   = np.concatenate([flat(real_seqs[norm_mask]), flat(syn_atk)])
    y_train   = np.concatenate([np.zeros(norm_mask.sum()), np.ones(len(syn_atk))])
    X_test, y_test = flat(real_seqs), real_labels

    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    return {
        "precision": report.get("1", {}).get("precision", 0),
        "recall":    report.get("1", {}).get("recall",    0),
        "f1":        report.get("1", {}).get("f1-score",  0),
        "auc_roc":   auc_roc,
        "auc_pr":    auc_pr,
    }


def per_stage_mmd(syn_seqs, syn_stages, real_seqs, real_stages):
    """Compute MMD between real and synthetic windows, broken down per stage."""
    results = {}
    flat    = lambda a: a.reshape(len(a), -1)
    for s in range(6):
        rm = real_stages == s; sm = syn_stages == s
        if rm.sum() < 2 or sm.sum() < 2:
            results[f"P{s+1}"] = float("nan")
            continue
        results[f"P{s+1}"] = mmd_rbf(flat(real_seqs[rm]), flat(syn_seqs[sm]))
    return results


def run_full_evaluation(scvae, scwgan, real_seqs, real_labels, real_stages, cfg, device):
    """Paper Table 1: per-stage MMD + TSTR for SCVAE and SCWGAN."""
    print("\n" + "="*65)
    print("FULL EVALUATION")
    print("="*65)

    # generate per-stage synthetic attacks from both models
    vae_syn_parts, gan_syn_parts, syn_stage_list = [], [], []
    for s in range(6):
        n = cfg["n_synthetic"]
        vae_part = generate_stage(scvae, s, n, cfg, device, "scvae")
        gan_part = generate_stage(scwgan, s, n, cfg, device, "scwgan")
        vae_syn_parts.append(vae_part)
        gan_syn_parts.append(gan_part)
        syn_stage_list.extend([s] * n)

    vae_syn    = np.concatenate(vae_syn_parts)
    gan_syn    = np.concatenate(gan_syn_parts)
    syn_stages = np.array(syn_stage_list)

    # MMD per stage
    vae_mmd = per_stage_mmd(vae_syn, syn_stages, real_seqs, real_stages)
    gan_mmd = per_stage_mmd(gan_syn, syn_stages, real_seqs, real_stages)

    print("\nPer-stage MMD (lower = more realistic):")
    print(f"{'Stage':<8} {'SCVAE':>10} {'SCWGAN-GP':>12}")
    for k in vae_mmd:
        print(f"{k:<8} {vae_mmd[k]:>10.6f} {gan_mmd[k]:>12.6f}")

    # TSTR
    print("\nTSTR evaluation (train synthetic / test real):")
    vae_tstr = tstr_evaluation(vae_syn, real_seqs, real_labels)
    gan_tstr = tstr_evaluation(gan_syn, real_seqs, real_labels)

    print(f"\n{'Metric':<12} {'SCVAE':>10} {'SCWGAN-GP':>12}")
    for metric in ["precision", "recall", "f1", "auc_roc", "auc_pr"]:
        print(f"{metric:<12} {vae_tstr[metric]:>10.4f} {gan_tstr[metric]:>12.4f}")

    # save
    results = {"vae_mmd": vae_mmd, "gan_mmd": gan_mmd,
               "vae_tstr": vae_tstr, "gan_tstr": gan_tstr}
    with open(f"{cfg['out_dir']}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {cfg['out_dir']}/results.json")
    return results, vae_syn, gan_syn, syn_stages


def run_ablation(scvae_conditioned, real_seqs, real_labels, real_stages, cfg, device):
    """
    Paper Table 2 — Ablation: stage-conditioned vs unconditioned VAE.
    Trains an unconditioned VAE on the same attack data as ablation baseline.
    """
    print("\nAblation: stage-conditioned vs unconditioned")

    # unconditioned baseline: plain β-VAE (reuse CrossOverVAE class, no stage input)
    uncond_vae = CrossOverVAE(
        cfg["n_features"], cfg["hidden_dim"], cfg["n_layers"],
        cfg["latent_dim"], cfg["window_size"]
    ).to(device)

    # use the attack loader from the main pipeline
    atk_mask   = real_labels == 1
    atk_seqs   = real_seqs[atk_mask]
    atk_stages = real_stages[atk_mask]
    ds         = CPSDataset(atk_seqs, np.ones(len(atk_seqs), dtype=np.int64), atk_stages)
    loader     = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    opt = optim.Adam(uncond_vae.parameters(), lr=cfg["lr"])
    uncond_vae.train()
    for epoch in range(cfg["epochs_vae"]):
        for x, *_ in loader:
            x = x.to(device)
            recon, mu, lv = uncond_vae(x)
            loss = (F.mse_loss(recon, x) +
                    cfg["beta"] * (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())))
            opt.zero_grad(); loss.backward(); opt.step()

    # generate unconditioned synthetic attacks
    uncond_vae.eval()
    with torch.no_grad():
        z    = torch.randn(cfg["n_synthetic"] * 6, cfg["latent_dim"], device=device)
        syn_uncond = torch.sigmoid(uncond_vae.decode(z)).cpu().numpy()

    # generate conditioned synthetic attacks
    syn_cond = np.concatenate([
        generate_stage(scvae_conditioned, s, cfg["n_synthetic"], cfg, device, "scvae")
        for s in range(6)
    ])

    cond_tstr   = tstr_evaluation(syn_cond,   real_seqs, real_labels)
    uncond_tstr = tstr_evaluation(syn_uncond, real_seqs, real_labels)

    print(f"\n{'Metric':<12} {'Unconditioned':>14} {'Stage-cond (ours)':>18}")
    for m in ["precision", "recall", "f1", "auc_roc"]:
        delta = cond_tstr[m] - uncond_tstr[m]
        sign  = "+" if delta >= 0 else ""
        print(f"{m:<12} {uncond_tstr[m]:>14.4f} {cond_tstr[m]:>14.4f}  ({sign}{delta:.4f})")

    return {"conditioned": cond_tstr, "unconditioned": uncond_tstr}


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    W  = CFG["window_size"]
    ST = CFG["stride"]

    # ── Data loading ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df_normal  = load_a12_normal(CFG["a12_normal_glob"])
    df_attacks = load_a1_attacks(CFG["a1_attack_csv"])
    df_all     = pd.concat([df_normal, df_attacks], ignore_index=True)

    scaler, scale_cols = fit_scaler(df_normal)
    X_all = transform(df_all, scaler, scale_cols)
    y_all = df_all["label"].values.astype(np.int64)

    seqs, labels, stages = make_windows(X_all, y_all, STAGE_COL_IDX, W, ST)
    print(f"  Windows: {len(seqs):,}  attack: {labels.sum():,}  "
          f"stages: {np.unique(stages)}")

    # temporal split
    N     = len(seqs)
    n_tr  = int(N * 0.70); n_val = int(N * 0.15)
    tr_s, tr_l, tr_st  = seqs[:n_tr],   labels[:n_tr],   stages[:n_tr]
    val_s,val_l,val_st = seqs[n_tr:n_tr+n_val], labels[n_tr:n_tr+n_val], stages[n_tr:n_tr+n_val]
    te_s, te_l, te_st  = seqs[n_tr+n_val:], labels[n_tr+n_val:], stages[n_tr+n_val:]
    print(f"  Train={len(tr_s):,}  Val={len(val_s):,}  Test={len(te_s):,}")

    atk_load = attack_loader(tr_s, tr_l, tr_st, CFG["batch_size"])

    if args.mode in ("train_all", "train_vae"):
        # ── Train SCVAE ───────────────────────────────────────────────────────
        print("\n[2/6] Training SCVAE...")
        scvae = train_scvae(atk_load, CFG, DEVICE)
        torch.save(scvae.state_dict(), "checkpoints/scvae.pt")
        print("  Saved → checkpoints/scvae.pt")

    if args.mode in ("train_all", "train_gan"):
        # ── Train SCWGAN-GP ───────────────────────────────────────────────────
        print("\n[3/6] Training SCWGAN-GP...")
        scwgan = train_scwgan(atk_load, CFG, DEVICE)
        torch.save(scwgan.state_dict(), "checkpoints/scwgan.pt")
        print("  Saved → checkpoints/scwgan.pt")

    if args.mode in ("train_all", "train_crossover"):
        # ── Topic 2: CrossOver VAE ────────────────────────────────────────────
        print("\n[4/6] Training CrossOver VAE (Topic 2)...")
        # Load WaDi.A3 and merge with A12 normal for the coupled normal model
        wadi_paths = sorted(glob.glob(CFG["wadi_a3_glob"]))
        if wadi_paths:
            df_wadi  = _load_csvs(wadi_paths, label=0)
            # WaDi has different features — keep only shared columns, zero-fill rest
            for col in ALL_FEATS:
                if col not in df_wadi.columns: df_wadi[col] = 0.0
            df_crossover = pd.concat([df_normal, df_wadi[ALL_FEATS + ["label"]]], ignore_index=True)
        else:
            print("  WaDi.A3 files not found — using A12 normal only for CrossOver VAE")
            df_crossover = df_normal

        X_co   = transform(df_crossover, scaler, scale_cols)
        y_co   = df_crossover["label"].values.astype(np.int64)
        s_co, l_co, st_co = make_windows(X_co, y_co, STAGE_COL_IDX, W, ST)
        co_ds  = CPSDataset(s_co[l_co==0], l_co[l_co==0], st_co[l_co==0])
        co_ldr = DataLoader(co_ds, batch_size=CFG["batch_size"], shuffle=True, drop_last=True)
        co_vae = train_crossover_vae(co_ldr, CFG, DEVICE)
        torch.save(co_vae.state_dict(), "checkpoints/crossover_vae.pt")
        print("  Saved → checkpoints/crossover_vae.pt")

    if args.mode == "eval":
        # ── Load saved models ─────────────────────────────────────────────────
        print("\n[5/6] Loading saved models for evaluation...")
        scvae  = SCVAE(CFG).to(DEVICE)
        scwgan = SCGenerator(CFG).to(DEVICE)
        scvae.load_state_dict(torch.load("checkpoints/scvae.pt",  map_location=DEVICE))
        scwgan.load_state_dict(torch.load("checkpoints/scwgan.pt", map_location=DEVICE))

        # ── Evaluation ────────────────────────────────────────────────────────
        print("\n[6/6] Running evaluation...")
        results, vae_syn, gan_syn, syn_stages = run_full_evaluation(
            scvae, scwgan, te_s, te_l, te_st, CFG, DEVICE
        )
        ablation = run_ablation(scvae, te_s, te_l, te_st, CFG, DEVICE)

    if args.mode == "ablation":
        scvae = SCVAE(CFG).to(DEVICE)
        scvae.load_state_dict(torch.load("checkpoints/scvae.pt", map_location=DEVICE))
        run_ablation(scvae, te_s, te_l, te_st, CFG, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train_all","train_vae","train_gan",
                                            "train_crossover","eval","ablation"],
                        default="train_all")
    main(parser.parse_args())


# ══════════════════════════════════════════════════════════════════════════════
# PAPER BASELINES TO IMPLEMENT (for comparison table in Section 6)
# ══════════════════════════════════════════════════════════════════════════════
#
# Implement each as a function train_<name>(loader, cfg, device) → model
# then call tstr_evaluation(generate(...), real_seqs, real_labels)
#
# 1. MAD-GAN (Li et al., 2019) — LSTM GAN, no conditioning
#    → arXiv:1901.04997
#
# 2. LSTM-VAE (Park et al., 2018) — LSTM encoder-decoder VAE
#    → unconditioned version of your SCVAE (already in run_ablation)
#
# 3. USAD (Audibert et al., KDD 2020) — dual AE with adversarial training
#    → github.com/manigalati/usad
#
# 4. TimeGAN (Yoon et al., NeurIPS 2019) — sequential GAN with supervisor
#    → github.com/jsyoon0823/TimeGAN
#
# 5. Unconditioned WGAN-GP  ← already in run_ablation as ablation baseline
#
# ══════════════════════════════════════════════════════════════════════════════
# Q1 WEEKLY TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
#
# Week 1   Data pipeline:  preprocessing_a12.py verified end-to-end on A12
# Week 2   Request + receive SWaT.A1 attack CSV; run align_a1_to_a12_schema()
# Week 3   SCVAE training + hyperparameter sweep (latent_dim, beta, window)
# Week 4   SCWGAN-GP training + stability check (critic loss convergence)
# Week 5   CrossOver VAE on A12+WaDi.A3 combined normal; anomaly score calibration
# Week 6   Implement MAD-GAN + LSTM-VAE baselines; run TSTR table
# Week 7   Per-stage MMD analysis; t-SNE visualisation of latent space per stage
# Week 8   Ablation: unconditioned vs stage-conditioned (Table 2)
# Week 9   CrossOver vs isolated normal VAE comparison (Section 5 results)
# Week 10  Downstream LSTM-AE IDS trained on augmented data; F1/AUC-ROC
# Week 11  Paper draft: S1–S4 (intro, related, dataset, architecture)
# Week 12  Paper draft: S5–S7 (CrossOver section, experiments, conclusion)
# Week 13  Revision + submission to IEEE TIFS (or Computers & Security)
