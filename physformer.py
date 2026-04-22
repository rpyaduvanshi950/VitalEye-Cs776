"""
PhysFormerX v3 — rPPG Heart Rate Estimation for UBFC-rPPG
══════════════════════════════════════════════════════════
Designed for sub-3 BPM MAE on subject-level test split.

Architecture: Diff-normalized video → 3D ResNet with SE → temporal conv head
Key design decisions:
  • Difference-frame input: frame[t]-frame[t-1] emphasizes subtle color changes
  • Moderate capacity (~1.5M params) — enough to learn, small enough to generalize
  • OneCycleLR with warmup → stable convergence
  • Gentle augmentation only (flip, mild noise) — NO temporal masking
  • Mixed-precision training for speed
  • EMA model averaging for smoother predictions
  • Early stopping on validation BPM MAE
"""

import os, glob, math, copy, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DATA_DIR   = 'E:/CS776 Project/preprocessed_physformer'
SAVE_DIR   = 'E:/CS776 Project'
EPOCHS     = 100
FPS        = 30
SEQ_LEN    = 300
CROP_LEN   = 150          # random temporal crop for training
BATCH      = 4            # smaller batch = more updates per epoch
IMG_SIZE   = 64
LR_MAX     = 3e-4
WEIGHT_DECAY = 5e-5
EMA_DECAY  = 0.998
GRAD_CLIP  = 2.0
PATIENCE   = 15

# ═══════════════════════════════════════════════════════════════
#  BPM HELPER
# ═══════════════════════════════════════════════════════════════
def bpm_from_signal(signals: np.ndarray, fps: int = FPS) -> np.ndarray:
    """Periodogram-based BPM. signals: (B, T) numpy float."""
    bpms = []
    for s in signals:
        s = s - np.mean(s)
        freqs, psd = periodogram(s, fs=fps, nfft=2048, detrend='constant',
                                 window='hann')
        valid = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
        if len(valid) > 0:
            bpms.append(freqs[valid][np.argmax(psd[valid])] * 60.0)
        else:
            bpms.append(75.0)
    return np.array(bpms)

# ═══════════════════════════════════════════════════════════════
#  DATASET — with diff-frame and temporal cropping
# ═══════════════════════════════════════════════════════════════
class rPPGDataset(Dataset):
    def __init__(self, file_paths, augment=True, img_size=IMG_SIZE,
                 crop_len=None):
        self.fps = file_paths
        self.augment = augment
        self.crop_len = crop_len
        self.resize = transforms.Resize((img_size, img_size), antialias=True)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        d = torch.load(self.fps[idx], weights_only=True)
        vid = d['video'].float() / 255.0          # [C, T, H, W]
        ppg = d['ppg'].float()

        C, T, H, W = vid.shape

        # Resize spatially
        vid = self.resize(vid.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        # Random temporal crop for training
        if self.crop_len and T > self.crop_len:
            start = random.randint(0, T - self.crop_len)
            vid = vid[:, start:start + self.crop_len]
            ppg = ppg[start:start + self.crop_len]

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                vid = torch.flip(vid, dims=[3])

            # Mild Gaussian noise
            if random.random() > 0.5:
                vid = vid + torch.randn_like(vid) * 0.01
                vid = torch.clamp(vid, 0, 1)

            # Slight brightness jitter
            if random.random() > 0.5:
                bfactor = 1.0 + (random.random() - 0.5) * 0.2
                vid = torch.clamp(vid * bfactor, 0, 1)

        # Compute difference frames: emphasize temporal changes
        # diff[t] = frame[t] - frame[t-1], first frame diff = 0
        diff = torch.zeros_like(vid)
        diff[:, 1:] = vid[:, 1:] - vid[:, :-1]

        # Stack original + diff → 6-channel input
        vid_6ch = torch.cat([vid, diff], dim=0)  # [6, T, H, W]

        # Normalize PPG
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
        return vid_6ch, ppg


def subject_split(data_dir, train_ratio=0.70, val_ratio=0.15):
    """3-way subject-level split → no data leakage."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    if not files:
        return [], [], []

    from collections import defaultdict
    subj_files = defaultdict(list)
    for fp in files:
        try:
            d = torch.load(fp, weights_only=True)
            sid = d.get('subject_id', os.path.basename(fp))
        except:
            sid = os.path.basename(fp)
        subj_files[sid].append(fp)

    subjects = sorted(subj_files.keys())
    random.seed(42)
    random.shuffle(subjects)

    n_train = max(1, int(len(subjects) * train_ratio))
    n_val = max(1, int(len(subjects) * val_ratio))
    train_subjs = set(subjects[:n_train])
    val_subjs = set(subjects[n_train:n_train + n_val])

    train_files, val_files, test_files = [], [], []
    for sid, fps_list in subj_files.items():
        if sid in train_subjs:
            train_files.extend(fps_list)
        elif sid in val_subjs:
            val_files.extend(fps_list)
        else:
            test_files.extend(fps_list)

    if len(val_files) == 0 or len(test_files) == 0:
        n = len(files)
        n_tr = int(0.70 * n)
        n_va = int(0.15 * n)
        train_files = files[:n_tr]
        val_files = files[n_tr:n_tr + n_va]
        test_files = files[n_tr + n_va:]

    return sorted(train_files), sorted(val_files), sorted(test_files)


# ═══════════════════════════════════════════════════════════════
#  MODEL BLOCKS
# ═══════════════════════════════════════════════════════════════
class SEBlock3D(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        r = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C = x.shape[:2]
        w = self.fc(self.pool(x).view(B, C)).view(B, C, 1, 1, 1)
        return x * w


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.se = SEBlock3D(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(0.05)  # very mild

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        return self.relu(out + identity)


# ═══════════════════════════════════════════════════════════════
#  FULL MODEL — PhysFormerX v3
# ═══════════════════════════════════════════════════════════════
class PhysFormerX(nn.Module):
    """
    6-channel input (RGB + diff) → 3D ResNet → temporal conv → PPG signal
    ~1.5M params
    """
    def __init__(self, seq_len=SEQ_LEN):
        super().__init__()
        # Stem: 6ch → 32
        self.stem = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2),
                      bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2))  # spatial /2
        )

        # Progressive feature extraction
        self.layer1 = ResBlock3D(32, 64, stride=(1, 2, 2))   # spatial /2
        self.layer2 = ResBlock3D(64, 64, stride=1)
        self.layer3 = ResBlock3D(64, 128, stride=(1, 2, 2))  # spatial /2
        self.layer4 = ResBlock3D(128, 128, stride=1)

        # Spatial collapse
        self.gap = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Temporal processing: conv1d stack for smooth PPG extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Final regression
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, 6, T, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x).squeeze(-1).squeeze(-1)  # [B, 128, T]

        x = self.temporal(x)   # [B, 64, T]
        x = self.head(x).squeeze(1)  # [B, T]

        return x


# ═══════════════════════════════════════════════════════════════
#  LOSS — Negative Pearson + SmoothL1 (proven combo from notebook)
# ═══════════════════════════════════════════════════════════════
class NegPearsonLoss(nn.Module):
    def forward(self, pred, target):
        p = pred - pred.mean(dim=1, keepdim=True)
        t = target - target.mean(dim=1, keepdim=True)
        num = (p * t).sum(dim=1)
        den = torch.norm(p, dim=1) * torch.norm(t, dim=1) + 1e-8
        return (1.0 - num / den).mean()


class FreqCELoss(nn.Module):
    """
    Treats BPM estimation as classification: applies cross-entropy over
    the HR-band frequency bins. Forces the model to put spectral energy
    at the correct frequency.
    """
    def __init__(self, fps=FPS, lo=0.75, hi=2.5):
        super().__init__()
        self.fps = fps
        self.lo = lo
        self.hi = hi

    def forward(self, pred, target):
        B, T = pred.shape
        with torch.amp.autocast('cuda', enabled=False), \
             torch.amp.autocast('cpu', enabled=False):
            pred_f = pred.float()
            tgt_f  = target.float()

            freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(pred.device)
            mask = (freqs >= self.lo) & (freqs <= self.hi)

            window = torch.hann_window(T, device=pred.device, dtype=torch.float32)

            # Predicted spectrum (log-softmax for cross-entropy)
            p_spec = torch.abs(torch.fft.rfft(pred_f * window, dim=1))[:, mask]
            p_logprob = F.log_softmax(p_spec, dim=1)

            # Target spectrum → soft distribution (softmax with temperature)
            t_spec = torch.abs(torch.fft.rfft(tgt_f * window, dim=1))[:, mask]
            t_prob = F.softmax(t_spec * 10.0, dim=1)  # sharpen

            # KL divergence
            loss = F.kl_div(p_logprob, t_prob, reduction='batchmean')
            return loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pearson = NegPearsonLoss()
        self.freq_ce = FreqCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred, target):
        lp = self.pearson(pred, target)
        lf = self.freq_ce(pred, target)
        ls = self.smooth_l1(pred, target)
        # Pearson dominant, freq CE for BPM accuracy, L1 for stability
        total = lp + 0.5 * lf + 0.1 * ls
        return total, lp.item(), lf.item()


# ═══════════════════════════════════════════════════════════════
#  EMA
# ═══════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow


# ═══════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════
def plot_training(hist, save_dir, best_ep):
    x = list(range(1, len(hist['loss']) + 1))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('PhysFormerX v3 — HR — UBFC-rPPG', fontweight='bold')

    ax[0].plot(x, hist['loss'], '#1f77b4', lw=1.5)
    if best_ep:
        ax[0].axvline(best_ep, color='grey', ls='--', lw=1,
                      label='best checkpoint')
    ax[0].set(title='Total Loss', xlabel='Epoch', ylabel='Loss')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(x, hist['pearson'], color='orange', lw=1.5, label='Pearson')
    ax[1].plot(x, hist['freq'], color='purple', lw=1.5, ls='--',
               label='Freq CE')
    ax[1].set(title='Loss Components', xlabel='Epoch', ylabel='Loss')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    ax[2].plot(x, hist['tr_bpm'], 'r-o', lw=1.5, ms=2, label='Train MAE')
    ax[2].plot(x, hist['val_bpm'], 'b-s', lw=1.5, ms=2, label='Val MAE')
    if best_ep:
        ax[2].axvline(best_ep, color='green', ls='--', lw=1,
                      label=f'best (ep {best_ep})')
    ax[2].set(title='BPM MAE', xlabel='Epoch', ylabel='MAE (BPM)')
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(save_dir, f'training_curves_ep{len(hist["loss"])}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Plot → {p}]')


# ═══════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════
def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config: EPOCHS={EPOCHS}, BATCH={BATCH}, LR_MAX={LR_MAX}, '
          f'SEQ={SEQ_LEN}, CROP={CROP_LEN}, IMG={IMG_SIZE}')

    # ── Split ─────────────────────────────────────────────────
    train_files, val_files, test_files = subject_split(DATA_DIR)
    if not train_files:
        print('No .pt files found.')
        return
    print(f'Split: Train={len(train_files)} | Val={len(val_files)} '
          f'| Test={len(test_files)}')

    # Training: random 150-frame crops for augmentation/variety
    tr_ds = rPPGDataset(train_files, augment=True, crop_len=CROP_LEN)
    # Val/Test: full 300-frame sequences for best BPM estimation
    va_ds = rPPGDataset(val_files, augment=False, crop_len=None)
    te_ds = rPPGDataset(test_files, augment=False, crop_len=None)

    tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, num_workers=2,
                       pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, BATCH, shuffle=False, num_workers=2,
                       pin_memory=True)
    te_ld = DataLoader(te_ds, BATCH, shuffle=False, num_workers=2,
                       pin_memory=True)

    # ── Model ─────────────────────────────────────────────────
    model = PhysFormerX(seq_len=SEQ_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n_params/1e6:.2f}M')

    # ── Optimizer + OneCycleLR ────────────────────────────────
    crit = CombinedLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX,
                            weight_decay=WEIGHT_DECAY)

    # OneCycleLR: warm up → peak → cosine anneal
    steps_per_epoch = len(tr_ld)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR_MAX, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,          # 10% warmup
        anneal_strategy='cos',
        div_factor=10,          # initial_lr = max_lr/10
        final_div_factor=100    # final_lr = initial_lr/100
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    ema = EMA(model)

    # ── History ───────────────────────────────────────────────
    hist = dict(loss=[], pearson=[], freq=[], tr_bpm=[], val_bpm=[])
    best_mae, best_ep = float('inf'), 0
    no_improve = 0

    for ep in range(EPOCHS):
        # ── TRAIN ──────────────────────────────────────────
        model.train()
        ep_loss = ep_p = ep_f = 0.0
        ep_bpm_errs = []

        bar = tqdm(tr_ld, desc=f'Ep {ep+1:3d}/{EPOCHS}', ncols=100)
        for vid, ppg in bar:
            vid, ppg = vid.to(device), ppg.to(device)
            opt.zero_grad(set_to_none=True)

            ctx = (torch.amp.autocast('cuda') if scaler
                   else torch.amp.autocast('cpu'))
            with ctx:
                pred = model(vid)
                loss, lp, lf = crit(pred, ppg)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

            sched.step()  # step per-batch for OneCycleLR
            ema.update(model)

            # Track BPM error
            with torch.no_grad():
                p_np = pred.detach().float().cpu().numpy()
                t_np = ppg.detach().float().cpu().numpy()
                err = float(np.mean(np.abs(
                    bpm_from_signal(p_np) - bpm_from_signal(t_np))))
                ep_bpm_errs.append(err)

            ep_loss += loss.item()
            ep_p += lp
            ep_f += lf
            bar.set_postfix(L=f'{loss.item():.3f}', MAE=f'{err:.1f}')

        N = len(tr_ld)
        tr_mae = np.mean(ep_bpm_errs)

        # ── VALIDATION (EMA weights) ──────────────────────
        orig_state = copy.deepcopy(model.state_dict())
        ema.apply(model)
        model.eval()

        val_errs = []
        with torch.no_grad():
            for vid, ppg in va_ld:
                vid, ppg = vid.to(device), ppg.to(device)
                ctx = (torch.amp.autocast('cuda') if scaler
                       else torch.amp.autocast('cpu'))
                with ctx:
                    pred = model(vid)
                val_errs.append(float(np.mean(np.abs(
                    bpm_from_signal(pred.float().cpu().numpy()) -
                    bpm_from_signal(ppg.float().cpu().numpy())))))

        val_mae = np.mean(val_errs) if val_errs else float('inf')

        # ── RECORD ────────────────────────────────────────
        hist['loss'].append(ep_loss / N)
        hist['pearson'].append(ep_p / N)
        hist['freq'].append(ep_f / N)
        hist['tr_bpm'].append(tr_mae)
        hist['val_bpm'].append(val_mae)

        lr_now = opt.param_groups[0]['lr']
        print(f'Ep {ep+1:3d} | L:{ep_loss/N:.4f} P:{ep_p/N:.4f} '
              f'F:{ep_f/N:.4f} | Tr:{tr_mae:.2f} Val:{val_mae:.2f} '
              f'| LR:{lr_now:.1e}')

        # ── SAVE BEST ─────────────────────────────────────
        if val_mae < best_mae:
            best_mae = val_mae
            best_ep = ep + 1
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, 'yolo_best.pt'))
            print(f'  >>> Best! Val MAE = {best_mae:.2f}')
        else:
            no_improve += 1

        model.load_state_dict(orig_state)

        if no_improve >= PATIENCE:
            print(f'\n  Early stop at ep {ep+1} (patience={PATIENCE})')
            break

        if (ep + 1) % 20 == 0:
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f'ckpt_ep{ep+1}.pt'))

    # ── TEST ─────────────────────────────────────────────────
    best_state = torch.load(os.path.join(SAVE_DIR, 'yolo_best.pt'),
                            map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()

    test_errs = []
    with torch.no_grad():
        for vid, ppg in te_ld:
            vid, ppg = vid.to(device), ppg.to(device)
            ctx = (torch.amp.autocast('cuda') if scaler
                   else torch.amp.autocast('cpu'))
            with ctx:
                pred = model(vid)
            test_errs.append(float(np.mean(np.abs(
                bpm_from_signal(pred.float().cpu().numpy()) -
                bpm_from_signal(ppg.float().cpu().numpy())))))

    test_mae = np.mean(test_errs) if test_errs else float('inf')

    # Save final + plot
    ema.apply(model)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'yolo_final.pt'))
    plot_training(hist, SAVE_DIR, best_ep)

    total_eps = len(hist['loss'])
    print(f'\n{"="*60}')
    print(f'Done! ({total_eps} epochs)')
    print(f'Best Val  MAE: {best_mae:.2f} BPM (ep {best_ep})')
    print(f'TEST MAE:      {test_mae:.2f} BPM')
    print(f'Saved: {os.path.join(SAVE_DIR, "yolo_best.pt")}')
    print(f'{"="*60}')


if __name__ == '__main__':
    run()