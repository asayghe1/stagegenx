"""
Q1 Research: Generative Models for CPS Dataset Augmentation
Datasets: SWaT / WaDi / EPIC (iTrust, SUTD)
Models:   GAN track  → TimeGAN / WGAN-GP
          VAE track  → LSTM-VAE / β-VAE
Author:   <your name>
"""

import os, math, random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    # paths — update to where your datasets live
    "swat_normal":  "data/swat/SWaT_Dataset_Normal_v1.csv",
    "swat_attack":  "data/swat/SWaT_Dataset_Attack_v0.csv",
    "wadi_normal":  "data/wadi/WADI_14days_new.csv",
    "wadi_attack":  "data/wadi/WADI_attackdataLABLE.csv",

    # windowing
    "window_size":  60,      # timesteps per sample (1 min @ 1 Hz for SWaT/WaDi)
    "stride":       10,      # sliding window stride

    # model
    "latent_dim":   32,
    "hidden_dim":   64,
    "num_layers":   2,
    "batch_size":   64,
    "lr":           1e-3,
    "epochs_gan":   200,
    "epochs_vae":   150,
    "beta":         4.0,     # β-VAE weight on KL term

    # generation
    "n_synthetic":  500,     # synthetic attack samples to generate per model

    # reproducibility
    "seed":         42,
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

class ITrustLoader:
    """Loads and preprocesses SWaT or WaDi CSVs from iTrust."""

    def __init__(self, normal_path: str, attack_path: str, dataset: str = "swat"):
        self.dataset = dataset.lower()
        self.scaler  = MinMaxScaler()
        self.feature_cols: list[str] = []

        df_normal = self._read(normal_path, label_col_value=0)
        df_attack = self._read(attack_path, label_col_value=1)
        self.df = pd.concat([df_normal, df_attack], ignore_index=True)

    def _read(self, path: str, label_col_value: int) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)

        # ── SWaT: drop first row (column descriptions), fix dtypes ──
        if self.dataset == "swat":
            df.columns = df.columns.str.strip()
            # last column is "Normal/Attack" label in raw file
            if "Normal/Attack" in df.columns:
                df["label"] = df["Normal/Attack"].str.strip().map(
                    {"Normal": 0, "Attack": 1}
                ).fillna(label_col_value)
                df = df.drop(columns=["Normal/Attack"])
            else:
                df["label"] = label_col_value
            df = df.drop(columns=["Timestamp"], errors="ignore")

        # ── WaDi: similar cleanup ──
        elif self.dataset == "wadi":
            df.columns = df.columns.str.strip()
            if "Attack LABLE (1:No Attack, -1:Attack)" in df.columns:
                df["label"] = df["Attack LABLE (1:No Attack, -1:Attack)"].map(
                    {1: 0, -1: 1}
                ).fillna(label_col_value)
                df = df.drop(
                    columns=["Attack LABLE (1:No Attack, -1:Attack)"], errors="ignore"
                )
            else:
                df["label"] = label_col_value
            df = df.drop(columns=["Row", "Date", "Time"], errors="ignore")

        # keep only numeric columns
        label = df["label"].copy()
        df = df.select_dtypes(include=[np.number])
        df["label"] = label
        df = df.fillna(method="ffill").fillna(0)
        return df

    def get_features(self) -> list[str]:
        return [c for c in self.df.columns if c != "label"]

    def fit_scale(self) -> "ITrustLoader":
        feat = self.get_features()
        self.feature_cols = feat
        self.scaler.fit(self.df[feat].values)
        return self

    def scaled_arrays(self):
        """Returns X (scaled), y (binary labels)."""
        X = self.scaler.transform(self.df[self.feature_cols].values).astype(np.float32)
        y = self.df["label"].values.astype(np.int64)
        return X, y


def sliding_windows(X: np.ndarray, y: np.ndarray,
                    window: int, stride: int):
    """
    Convert flat (T, F) arrays into overlapping windows (N, W, F).
    Label per window = majority vote.
    """
    seqs, labels = [], []
    for start in range(0, len(X) - window + 1, stride):
        seqs.append(X[start : start + window])
        labels.append(int(y[start : start + window].mean() >= 0.5))
    return np.stack(seqs), np.array(labels)


class WindowDataset(Dataset):
    def __init__(self, seqs: np.ndarray, labels: np.ndarray):
        self.seqs   = torch.tensor(seqs,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):  return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labels[i]


def make_attack_loader(seqs: np.ndarray, labels: np.ndarray,
                       batch_size: int) -> DataLoader:
    """DataLoader containing only attack-class windows."""
    mask   = labels == 1
    ds     = WindowDataset(seqs[mask], labels[mask])
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. GAN TRACK
# ──────────────────────────────────────────────────────────────────────────────

# 2a. WGAN-GP (practical, stable; good baseline for time-series)
# ─────────────────────────────────────────────────────────────

class LSTMGenerator(nn.Module):
    """Generates (batch, window, n_feat) from Gaussian noise."""
    def __init__(self, latent_dim: int, hidden_dim: int,
                 n_layers: int, n_feat: int, window: int):
        super().__init__()
        self.window = window
        self.n_feat = n_feat
        self.fc_in  = nn.Linear(latent_dim, hidden_dim)
        self.lstm   = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                              batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, n_feat)

    def forward(self, z):                       # z: (B, latent_dim)
        h = self.fc_in(z).unsqueeze(1)          # (B, 1, H)
        h = h.expand(-1, self.window, -1)       # (B, W, H)
        out, _ = self.lstm(h)                   # (B, W, H)
        return torch.sigmoid(self.fc_out(out))  # (B, W, F)  → [0,1]


class LSTMCritic(nn.Module):
    """Critic (no sigmoid) for WGAN-GP."""
    def __init__(self, n_feat: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.lstm   = nn.LSTM(n_feat, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):                       # x: (B, W, F)
        out, (h, _) = self.lstm(x)
        return self.fc_out(h[-1])               # (B, 1)


def gradient_penalty(critic, real, fake, device):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((grads.norm(2, dim=[1, 2]) - 1) ** 2).mean()
    return gp


def train_wgan_gp(attack_loader: DataLoader, n_feat: int,
                  cfg: dict, device: torch.device):
    """
    Train WGAN-GP on attack sequences.
    Returns trained generator.
    """
    W  = cfg["window_size"]
    ld = cfg["latent_dim"]
    hd = cfg["hidden_dim"]
    nl = cfg["num_layers"]

    G = LSTMGenerator(ld, hd, nl, n_feat, W).to(device)
    C = LSTMCritic(n_feat, hd, nl).to(device)

    opt_G = optim.Adam(G.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))

    n_critic = 5   # critic steps per generator step
    lam      = 10  # GP coefficient

    G.train(); C.train()
    for epoch in range(cfg["epochs_gan"]):
        for real_x, _ in attack_loader:
            real_x = real_x.to(device)
            B = real_x.size(0)

            # ── critic update ──
            for _ in range(n_critic):
                z    = torch.randn(B, ld, device=device)
                fake = G(z).detach()
                gp   = gradient_penalty(C, real_x, fake, device)
                loss_C = C(fake).mean() - C(real_x).mean() + lam * gp
                opt_C.zero_grad(); loss_C.backward(); opt_C.step()

            # ── generator update ──
            z      = torch.randn(B, ld, device=device)
            fake   = G(z)
            loss_G = -C(fake).mean()
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if (epoch + 1) % 50 == 0:
            print(f"[WGAN-GP] epoch {epoch+1:>3} | "
                  f"C: {loss_C.item():.4f}  G: {loss_G.item():.4f}")

    return G


def generate_gan_samples(G: LSTMGenerator, n: int,
                         cfg: dict, device: torch.device) -> np.ndarray:
    """Generate n synthetic attack windows. Returns (n, W, F)."""
    G.eval()
    samples = []
    with torch.no_grad():
        for _ in range(0, n, cfg["batch_size"]):
            bs = min(cfg["batch_size"], n - len(samples) * cfg["batch_size"])
            z  = torch.randn(bs, cfg["latent_dim"], device=device)
            samples.append(G(z).cpu().numpy())
    return np.concatenate(samples, axis=0)[:n]


# ──────────────────────────────────────────────────────────────────────────────
# 3. VAE TRACK
# ──────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, n_feat: int, hidden_dim: int,
                 n_layers: int, latent_dim: int):
        super().__init__()
        self.lstm   = nn.LSTM(n_feat, hidden_dim, n_layers, batch_first=True)
        self.fc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv  = nn.Linear(hidden_dim, latent_dim)  # log-variance

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]                               # last layer hidden state
        return self.fc_mu(h), self.fc_lv(h)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int,
                 n_layers: int, n_feat: int, window: int):
        super().__init__()
        self.window = window
        self.fc_in  = nn.Linear(latent_dim, hidden_dim)
        self.lstm   = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, n_feat)

    def forward(self, z):
        h = self.fc_in(z).unsqueeze(1).expand(-1, self.window, -1)
        out, _ = self.lstm(h)
        return torch.sigmoid(self.fc_out(out))  # (B, W, F)


class LSTMVAE(nn.Module):
    def __init__(self, n_feat: int, hidden_dim: int, n_layers: int,
                 latent_dim: int, window: int):
        super().__init__()
        self.encoder = LSTMEncoder(n_feat, hidden_dim, n_layers, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, n_layers, n_feat, window)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.encoder.fc_mu.out_features, device=device)
        with torch.no_grad():
            return self.decoder(z)


def elbo_loss(recon, x, mu, log_var, beta: float = 1.0):
    """β-VAE ELBO: MSE reconstruction + β * KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(attack_loader: DataLoader, n_feat: int,
              cfg: dict, device: torch.device) -> LSTMVAE:
    """
    Train β-VAE on attack sequences.
    Returns trained model.
    """
    model = LSTMVAE(
        n_feat, cfg["hidden_dim"], cfg["num_layers"],
        cfg["latent_dim"], cfg["window_size"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    model.train()
    for epoch in range(cfg["epochs_vae"]):
        total_loss = 0.0
        for x, _ in attack_loader:
            x     = x.to(device)
            recon, mu, log_var = model(x)
            loss, rl, kl = elbo_loss(recon, x, mu, log_var, beta=cfg["beta"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            avg = total_loss / len(attack_loader)
            print(f"[β-VAE] epoch {epoch+1:>3} | loss: {avg:.4f} "
                  f"| recon: {rl.item():.4f}  kl: {kl.item():.4f}")

    return model


def generate_vae_samples(vae: LSTMVAE, n: int,
                         device: torch.device) -> np.ndarray:
    """Generate n synthetic attack windows. Returns (n, W, F)."""
    vae.eval()
    with torch.no_grad():
        out = vae.sample(n, device).cpu().numpy()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    Measures distance between real and synthetic distributions.
    Lower = better.
    X, Y: (N, D) — flatten windows before passing.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def train_on_synthetic_test_on_real(
    syn_seqs:   np.ndarray,   # (N_syn, W, F)  synthetic attack
    real_seqs:  np.ndarray,   # (N_real, W, F) real (mix attack + normal)
    real_labels: np.ndarray,  # (N_real,)
) -> dict:
    """
    TSTR evaluation: train a classifier on synthetic attack + real normal,
    test on real held-out data.
    Returns classification report dict.
    """
    n_feat = real_seqs.shape[2]
    W      = real_seqs.shape[1]

    # flatten (N, W, F) → (N, W*F)
    def flat(arr): return arr.reshape(len(arr), -1)

    normal_mask  = real_labels == 0
    normal_seqs  = real_seqs[normal_mask]
    real_attacks = real_seqs[~normal_mask]

    # training set: all synthetic attacks + equal normal
    n = min(len(syn_seqs), len(normal_seqs))
    X_train = np.concatenate([flat(syn_seqs[:n]), flat(normal_seqs[:n])])
    y_train = np.concatenate([np.ones(n), np.zeros(n)])

    # test set: held-out real attacks + normal
    X_test  = flat(real_seqs)
    y_test  = real_labels

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    print("\n[TSTR] Classification report (train-on-synthetic / test-on-real)")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Attack"]))
    return report


def evaluate_models(
    gan_samples: np.ndarray,   # (N, W, F)
    vae_samples: np.ndarray,
    real_attack_seqs: np.ndarray,
    all_seqs: np.ndarray,
    all_labels: np.ndarray,
):
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    def flat(arr): return arr.reshape(len(arr), -1)
    R = flat(real_attack_seqs)

    # ── MMD ──
    gan_mmd = mmd_rbf(R, flat(gan_samples))
    vae_mmd = mmd_rbf(R, flat(vae_samples))
    print(f"\nMMD (lower = more realistic):")
    print(f"  GAN  : {gan_mmd:.6f}")
    print(f"  VAE  : {vae_mmd:.6f}")

    # ── TSTR ──
    print("\n--- GAN synthetic ---")
    gan_report = train_on_synthetic_test_on_real(
        gan_samples, all_seqs, all_labels
    )
    print("\n--- VAE synthetic ---")
    vae_report = train_on_synthetic_test_on_real(
        vae_samples, all_seqs, all_labels
    )

    print("\nAttack F1 summary:")
    print(f"  GAN  F1: {gan_report.get('1',{}).get('f1-score', 0):.4f}")
    print(f"  VAE  F1: {vae_report.get('1',{}).get('f1-score', 0):.4f}")

    return {"gan_mmd": gan_mmd, "vae_mmd": vae_mmd,
            "gan_report": gan_report, "vae_report": vae_report}


# ──────────────────────────────────────────────────────────────────────────────
# 5. SAVE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def save_synthetic(samples: np.ndarray, path: str, feature_names: list[str]):
    """Flatten windows and save as CSV for downstream use."""
    N, W, F = samples.shape
    flat = samples.reshape(N * W, F)
    df   = pd.DataFrame(flat, columns=feature_names)
    df["label"] = 1
    df.to_csv(path, index=False)
    print(f"Saved {N} synthetic attack sequences → {path}")


def save_checkpoint(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # ── Load data ──
    print("Loading SWaT dataset...")
    loader = ITrustLoader(
        CFG["swat_normal"],
        CFG["swat_attack"],
        dataset="swat"
    ).fit_scale()
    X, y = loader.scaled_arrays()
    n_feat = X.shape[1]
    feature_names = loader.feature_cols
    print(f"  Features: {n_feat}  |  Samples: {len(X)}"
          f"  |  Attack ratio: {y.mean():.3f}")

    # ── Sliding windows ──
    seqs, labels = sliding_windows(X, y, CFG["window_size"], CFG["stride"])
    print(f"  Windows: {len(seqs)}  |  Attack windows: {labels.sum()}")

    # ── Attack-only dataloader (for generative model training) ──
    atk_loader = make_attack_loader(seqs, labels, CFG["batch_size"])

    # ── Train WGAN-GP ──
    print("\nTraining WGAN-GP...")
    G = train_wgan_gp(atk_loader, n_feat, CFG, DEVICE)
    save_checkpoint(G, "checkpoints/wgan_gp_generator.pt")
    gan_samples = generate_gan_samples(G, CFG["n_synthetic"], CFG, DEVICE)
    save_synthetic(gan_samples, "outputs/synthetic_attack_gan.csv", feature_names)

    # ── Train β-VAE ──
    print("\nTraining β-VAE...")
    vae = train_vae(atk_loader, n_feat, CFG, DEVICE)
    save_checkpoint(vae, "checkpoints/lstmvae.pt")
    vae_samples = generate_vae_samples(vae, CFG["n_synthetic"], DEVICE)
    save_synthetic(vae_samples, "outputs/synthetic_attack_vae.csv", feature_names)

    # ── Evaluate ──
    real_attack_seqs = seqs[labels == 1]
    results = evaluate_models(
        gan_samples, vae_samples,
        real_attack_seqs,
        seqs, labels
    )

    # ── Save results ──
    summary = pd.DataFrame({
        "metric": ["MMD", "Attack F1 (TSTR)"],
        "GAN":    [results["gan_mmd"],
                   results["gan_report"].get("1", {}).get("f1-score", 0)],
        "VAE":    [results["vae_mmd"],
                   results["vae_report"].get("1", {}).get("f1-score", 0)],
    })
    summary.to_csv("outputs/q1_results_summary.csv", index=False)
    print("\nResults summary saved → outputs/q1_results_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# QUICK-START NOTES
# ──────────────────────────────────────────────────────────────────────────────
#
# 1. INSTALL DEPENDENCIES
#    pip install torch pandas numpy scikit-learn
#
# 2. UPDATE PATHS in CFG to point to your iTrust CSV files.
#    SWaT normal:  SWaT_Dataset_Normal_v1.csv (or v0)
#    SWaT attack:  SWaT_Dataset_Attack_v0.csv
#    WaDi normal:  WADI_14days_new.csv
#    WaDi attack:  WADI_attackdataLABLE.csv
#    To switch to WaDi, change the ITrustLoader call: dataset="wadi"
#
# 3. KEY HYPERPARAMETERS TO TUNE (CFG dict above)
#    window_size  → how many timesteps per sample (try 30, 60, 120)
#    latent_dim   → bottleneck size (try 16, 32, 64)
#    beta         → KL weight in β-VAE (try 1, 4, 8 — higher = more disentangled)
#    epochs_gan   → WGAN-GP needs ~200+ epochs to stabilize
#    n_critic     → critic steps per G step; 5 is standard for WGAN-GP
#
# 4. RECOMMENDED NEXT STEPS (extend this file)
#    a. Add TimeGAN (embed-recover + supervised + joint training)
#    b. Add conditional GAN/VAE (conditioned on attack type label)
#    c. Replace RandomForest with LSTM classifier for TSTR eval
#    d. Add FID score (adapt standard FID to time-series with PCA reduction)
#    e. Test augmented data on known IDS baselines (IsolationForest, USAD, etc.)
#    f. Cross-dataset: train on SWaT synthetic, test IDS on WaDi
#
# 5. CITATION (required by iTrust terms of use)
#    "iTrust, Centre for Research in Cyber Security,
#     Singapore University of Technology and Design"
#    SWaT paper: Goh et al., CRITIS 2016
#    WaDi paper: Ahmed et al., CySWATER 2017
