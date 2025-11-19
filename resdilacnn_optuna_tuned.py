# resdilacnn_optuna_tuned.py
import time, os, sys
import h5py, scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.utils.parametrize as P
from torch.nn.utils import weight_norm as legacy_weight_norm

# =========================
# Config / toggles
# =========================
SNR_index = np.arange(0, 45, 5)
train_rate = 0.75
USE_MASKED_LOSS = False  # False = paper-style (all 52); True = data-only (48 tones)

# Data subcarrier mask (Python 0-based)
DSC_IDX = np.array(
    [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,
     21,22,23,24,25,26,27,28,29,30,
     32,33,34,35,36,37,38,39,40,41,42,43,44,
     46,47,48,49,50,51],
    dtype=int
)

# =========================
# OPTIMAL hyperparameters from Optuna
# Trial 159 | Val Loss: 0.002441 | 28.62 hours (163 trials)
# =========================
BEST_BATCH_SIZE = 128
BEST_LR = 0.00047723253464726804
BEST_N_BLOCKS = 5
BEST_KERNEL_SIZE = 4
BEST_DROPOUT = 0.22791870093014122
BEST_DILATION_DEPTH = 5
BEST_HIDDEN_CHANNELS = 256
BEST_STEP_SIZE = 11
BEST_GAMMA = 0.9477107033183908

# Derived
BEST_DILATIONS = tuple(2 ** i for i in range(BEST_DILATION_DEPTH))  # (1, 2, 4, 8, 16)

print(f"\n{'='*70}")
print(f"OPTUNA-TUNED TCN CONFIGURATION")
print(f"{'='*70}")
print(f"Trial: 159 (Best of 163 trials)")
print(f"Validation Loss: 0.002441")
print(f"Optimization Time: 28.62 hours")
print(f"\nHyperparameters:")
print(f"  Batch Size: {BEST_BATCH_SIZE}")
print(f"  Learning Rate: {BEST_LR:.6f}")
print(f"  Blocks: {BEST_N_BLOCKS}")
print(f"  Kernel Size: {BEST_KERNEL_SIZE}")
print(f"  Dropout: {BEST_DROPOUT:.4f}")
print(f"  Hidden Channels: {BEST_HIDDEN_CHANNELS}")
print(f"  Dilations: {BEST_DILATIONS}")
print(f"  Step Size: {BEST_STEP_SIZE}")
print(f"  Gamma: {BEST_GAMMA:.4f}")
print(f"{'='*70}\n")

# =========================
# Instrumentation helpers
# =========================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_conv1d_macs(model: nn.Module, input_len: int) -> int:
    macs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            Cout, Cin, K = m.weight.shape[0], m.weight.shape[1], m.weight.shape[2]
            macs += Cout * Cin * K * input_len
    return macs

def calc_macs_flops(model: nn.Module, input_len: int) -> tuple[int, int]:
    macs = compute_conv1d_macs(model, input_len)
    flops = 2 * macs
    return macs, flops

def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# =========================
# Model: ResDilaCNN / TCN-style
# =========================
def wnorm(module, name='weight', dim=0):
    try:
        return P.weight_norm(module, name=name, dim=dim)
    except Exception:
        return legacy_weight_norm(module, name=name, dim=dim)

class ResidualKLayerCNN(nn.Module):
    """One residual dilated CNN block (non-causal). I/O: (B, Cin, L) -> (B, Chid, L)."""
    def __init__(self, num_inputs=100, hidden_channels=100, kernel_size=2,
                 dilations=(1,2,4,8), dropout=0.05):
        super().__init__()
        assert len(dilations) >= 2

        self.kernel_size = kernel_size
        self.dilations   = tuple(dilations)
        self.relu        = nn.ReLU()

        convs, drops = [], []
        in_c = num_inputs
        for d in self.dilations:
            conv = nn.Conv1d(
                in_c,
                hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                dilation=d,
                bias=True
            )
            conv = wnorm(conv)
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            convs.append(conv)
            drops.append(nn.Dropout(dropout))
            in_c = hidden_channels
        self.convs = nn.ModuleList(convs)
        self.drops = nn.ModuleList(drops)

        self.downsample = None
        if num_inputs != hidden_channels:
            self.downsample = nn.Conv1d(num_inputs, hidden_channels, kernel_size=1)
            self.downsample = wnorm(self.downsample)
            nn.init.kaiming_uniform_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        y = x
        for conv, d, drop in zip(self.convs, self.dilations, self.drops):
            total = d * (self.kernel_size - 1)
            left  = total // 2
            right = total - left
            y = F.pad(y, (left, right))
            y = conv(y)
            y = self.relu(y)
            y = drop(y)

        res = x if self.downsample is None else self.downsample(x)
        y = self.relu(y + res)
        return y

class ResidualKStack(nn.Module):
    """
    Stacked residual dilated CNNs with global skip and final 1x1 head.
    I/O: (B, 100, 52) -> (B, 100, 52)
    """
    def __init__(self, num_inputs=100, hidden_channels=100, kernel_size=2,
                 dilations=(1,2,4,8), dropout=0.05, n_blocks=4, output_size=100):
        super().__init__()
        assert n_blocks >= 1

        blocks = []
        in_c = num_inputs
        for _ in range(n_blocks):
            blocks.append(
                ResidualKLayerCNN(
                    num_inputs=in_c,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    dropout=dropout
                )
            )
            in_c = hidden_channels
        self.blocks = nn.ModuleList(blocks)

        if num_inputs != hidden_channels:
            self.global_skip = nn.Conv1d(num_inputs, hidden_channels, kernel_size=1)
            self.global_skip = wnorm(self.global_skip)
            nn.init.kaiming_uniform_(self.global_skip.weight, nonlinearity='relu')
        else:
            self.global_skip = nn.Identity()

        self.head = nn.Conv1d(hidden_channels, output_size, kernel_size=1)
        nn.init.kaiming_uniform_(self.head.weight, nonlinearity='relu')

        self.relu = nn.ReLU()

    def forward(self, x):
        g = self.global_skip(x)
        y = x
        for blk in self.blocks:
            y = blk(y)
        y = self.relu(y + g)
        y = self.head(y)
        return y


# =========================
# Utils (data & scaling)
# =========================
def ensure_bcl(x):
    """Ensure array is (batch, channels=100, length=52)."""
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {x.shape}")
    if x.shape[1]==100 and x.shape[2]==52:
        return x
    if x.shape[0]==100 and x.shape[1]==52:
        return np.transpose(x, (2,0,1))
    if x.shape[0]==52 and x.shape[1]==100:
        return np.transpose(x, (2,1,0))
    raise ValueError(f"Unrecognized shape {x.shape}, wanted (*,100,52).")

def fit_scale_XY(X, Y):
    """Channel-wise StandardScaler on X and Y."""
    N,C,L = X.shape
    Xf = X.transpose(1,0,2).reshape(C, -1).T
    Yf = Y.transpose(1,0,2).reshape(C, -1).T
    sx = StandardScaler().fit(Xf)
    sy = StandardScaler().fit(Yf)
    Xn = sx.transform(Xf).T.reshape(C, N, L).transpose(1,0,2)
    Yn = sy.transform(Yf).T.reshape(C, N, L).transpose(1,0,2)
    return Xn, Yn, sx, sy

def apply_scale(X, scaler):
    N,C,L = X.shape
    Xf = X.transpose(1,0,2).reshape(C, -1).T
    Xn = scaler.transform(Xf).T.reshape(C, N, L).transpose(1,0,2)
    return Xn

def invert_scale(Xn, scaler):
    N,C,L = Xn.shape
    Xf = Xn.transpose(1,0,2).reshape(C, -1).T
    X  = scaler.inverse_transform(Xf).T.reshape(C, N, L).transpose(1,0,2)
    return X

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, device) -> torch.Tensor:
    """MSE only over data subcarriers."""
    B, C, L = pred.shape
    mask_L = torch.zeros(L, device=device)
    mask_L[DSC_IDX] = 1.0
    mask = mask_L.view(1,1,L).expand(B,C,L)
    diff2 = (pred - target) ** 2
    num = (diff2 * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den

def loss_fn(pred, target, device):
    if USE_MASKED_LOSS:
        return masked_mse_loss(pred, target, device)
    else:
        return F.mse_loss(pred, target)

def to_numpy_complex(arr):
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    dt = getattr(arr, "dtype", None)
    if dt is not None and dt.names is not None and "real" in dt.names and "imag" in dt.names:
        return (arr["real"] + 1j * arr["imag"]).astype(np.complex64, copy=False)
    return arr.astype(np.float64).astype(np.complex64)

def norm_XY(arr, nTime, nSC):
    if arr.ndim != 3:
        raise AssertionError(f"Expected 3D array, got {arr.ndim}D")
    s = arr.shape
    if s[1] == nTime and s[2] == nSC:
        return arr
    if s[0] == nSC and s[1] == nTime:
        return np.transpose(arr, (2, 1, 0))
    if s[0] == nTime and s[1] == nSC:
        return np.transpose(arr, (2, 0, 1))
    raise AssertionError(f"Cannot normalize X/Y with shape {s}")

def norm_yfd(arr, N_expected, nSym_expected):
    if arr.ndim != 3:
        raise AssertionError(f"Expected 3D array, got {arr.ndim}D")
    s = tuple(arr.shape)
    dims = np.array(s)
    idxN   = np.where(dims == N_expected)[0]
    idxSym = np.where(dims == nSym_expected)[0]
    if len(idxN) != 1 or len(idxSym) != 1:
        raise AssertionError(f"Cannot disambiguate Y_DataSubCarriers shape {s}")
    idxDSC = ({0,1,2} - {int(idxN[0]), int(idxSym[0])}).pop()
    nDSC = int(dims[idxDSC])
    yfd_norm = np.transpose(arr, (int(idxN[0]), int(idxSym[0]), int(idxDSC)))
    return yfd_norm, nDSC

# =========================
# Main
# =========================
# =========================
# Main - SIMPLIFIED ARGUMENT PARSING
# =========================
if len(sys.argv) < 5:
    print("Usage:")
    print("  Train: python resdilacnn_optuna_tuned.py <snr_label> <modu> <scheme> train <epochs>")
    print("  Test : python resdilacnn_optuna_tuned.py <snr_label> <modu> <scheme> <test_channel> test")
    sys.exit(1)

# Parse common arguments
snr_label = sys.argv[1]         # 'MixedSNR' or 'HighSNR'
modulation_order = sys.argv[2]  # '16QAM'
scheme = sys.argv[3]            # 'DPA'
mode_or_channel = sys.argv[4]   # Either 'train' or channel name like 'VTV_SDWW'

# Determine mode
if mode_or_channel == 'train':
    mode = 'train'
    if len(sys.argv) < 6:
        print("Error: Training requires <epochs> argument")
        sys.exit(1)
    EPOCHS = int(sys.argv[5])
    BATCH = BEST_BATCH_SIZE
else:
    # Assume it's a test channel name, check for 'test' keyword
    mode = 'test'
    test_channel = mode_or_channel
    if len(sys.argv) < 6 or sys.argv[5] != 'test':
        print("Error: Testing requires format: <snr_label> <modu> <scheme> <test_channel> test")
        sys.exit(1)

# =========================
# TRAINING MODE
# =========================
if mode == 'train':
    print(f"\n{'='*70}")
    print(f"TRAINING: Optuna-Tuned TCN")
    print(f"{'='*70}")
    print(f"SNR Label: {snr_label}")
    print(f"Modulation: {modulation_order}")
    print(f"Scheme: {scheme}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH} (fixed)")
    print(f"{'='*70}\n")

    # Load data
    mat_path = f'./MixedChannel_{snr_label}_{modulation_order}_{scheme}_TCN_training_dataset.mat'
    
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Training dataset not found: {mat_path}")
    
    print(f"Loading: {mat_path}")
    with h5py.File(mat_path, 'r') as f:
        X = np.array(f['TCN_Datasets']['Train_X'])
        Y = np.array(f['TCN_Datasets']['Train_Y'])
    print("Loaded (raw): X", X.shape, " Y", Y.shape)

    # Force shapes -> (N,100,52)
    X = ensure_bcl(X)
    Y = ensure_bcl(Y)
    print("As (N,100,52): X", X.shape, " Y", Y.shape)

    # Fit scalers and normalize
    Xn, Yn, sx, sy = fit_scale_XY(X, Y)
    scaler_file = f'./MixedChannel_{snr_label}_{modulation_order}_{scheme}_TCN_scalers.npz'
    np.savez(scaler_file,
             sx_mean=sx.mean_, sx_scale=sx.scale_, 
             sy_mean=sy.mean_, sy_scale=sy.scale_)
    print(f"Saved scalers: {scaler_file}")

    # Split by SNR blocks
    N = Xn.shape[0]
    blocks = len(SNR_index)
    per = N // blocks
    idx = np.arange(N)
    train_idx, val_idx = [], []
    
    for b in range(blocks):
        blk = np.random.permutation(idx[b*per:(b+1)*per])
        cut = int(len(blk)*train_rate)
        train_idx += blk[:cut].tolist()
        val_idx   += blk[cut:].tolist()
    
    Train_X, Train_Y = Xn[train_idx], Yn[train_idx]
    Val_X,   Val_Y   = Xn[val_idx],   Yn[val_idx]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Data loaders
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    tr_loader = data.DataLoader(
        data.TensorDataset(
            torch.from_numpy(Train_X).float(), 
            torch.from_numpy(Train_Y).float()
        ),
        batch_size=BATCH, shuffle=True, num_workers=4
    )
    va_loader = data.DataLoader(
        data.TensorDataset(
            torch.from_numpy(Val_X).float(), 
            torch.from_numpy(Val_Y).float()
        ),
        batch_size=BATCH, shuffle=False, num_workers=4
    )

    # Build model with OPTIMAL hyperparameters
    model = ResidualKStack(
        num_inputs=100,
        hidden_channels=BEST_HIDDEN_CHANNELS,
        kernel_size=BEST_KERNEL_SIZE,
        dilations=BEST_DILATIONS,
        dropout=BEST_DROPOUT,
        n_blocks=BEST_N_BLOCKS,
        output_size=100
    ).to(device)

    print(f"Model Architecture:")
    print(f"  Blocks: {BEST_N_BLOCKS}")
    print(f"  Hidden Channels: {BEST_HIDDEN_CHANNELS}")
    print(f"  Kernel Size: {BEST_KERNEL_SIZE}")
    print(f"  Dilations: {BEST_DILATIONS}")
    print(f"  Dropout: {BEST_DROPOUT:.4f}\n")

    opt = optim.Adam(model.parameters(), lr=BEST_LR)
    sch = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=BEST_STEP_SIZE,
        gamma=BEST_GAMMA
    )

    # Model complexity
    n_params = count_parameters(model)
    macs, flops = calc_macs_flops(model, input_len=52)
    print(f'Parameters: {n_params:,}')
    print(f'MACs/forward: {macs:,}')
    print(f'FLOPs/forward: {flops:,}')
    print(f'Loss mode: {"MASKED (48 data SCs)" if USE_MASKED_LOSS else "FULL (52 active SCs)"}\n')

    # Early stopping
    patience  = 20
    min_delta = 1e-5
    no_improve = 0
    best = np.inf
    best_state = None

    tr_hist, va_hist = [], []
    total_train_wall = 0.0

    print("Starting training...\n")

    for ep in range(EPOCHS):
        model.train()
        tl = 0.0
        ep_start = time.perf_counter()
        
        for step, (xb, yb) in enumerate(tr_loader):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            sync_if_cuda()
            t0 = time.perf_counter()
            pred = model(xb)
            sync_if_cuda()
            fwd_ms = (time.perf_counter() - t0) * 1000.0

            loss = loss_fn(pred, yb, device)
            loss.backward()
            opt.step()
            tl += float(loss)

            if step % 200 == 0:
                print(f"Epoch {ep+1}/{EPOCHS} | Step {step} | train_loss {float(loss):.6f} | batch fwd {fwd_ms:.2f} ms")

        total_train_wall += time.perf_counter() - ep_start
        tr_hist.append(tl/len(tr_loader))

        # Validation
        model.eval()
        vl = 0.0
        with torch.no_grad():
            sync_if_cuda()
            tv0 = time.perf_counter()
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vl += float(loss_fn(pred, yb, device))
            sync_if_cuda()
            val_ms = (time.perf_counter() - tv0) * 1000.0

        va = vl/len(va_loader)
        va_hist.append(va)
        print(f"Epoch {ep+1}/{EPOCHS} | train {tr_hist[-1]:.6f} | val {va:.6f} | val fwd {val_ms:.2f} ms")

        # Early stopping
        if va < best - min_delta:
            best = va
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            model_path = f'./MixedChannel_{snr_label}_{modulation_order}_{scheme}_TCN_optimized.pt'
            torch.save(best_state, model_path)
            print(f"  → Saved best model (val loss: {va:.6f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {ep+1} (no val improvement in {patience} epochs).")
                break

        sch.step()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Total training time: {total_train_wall/60:.2f} minutes")
    print(f"Best validation loss: {best:.6f}")
    print(f"Model saved: {model_path}")
    print(f"{'='*70}\n")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(tr_hist, label='Train', linewidth=2)
    plt.plot(va_hist, label='Validation', linewidth=2)
    plt.axhline(y=0.002441, color='r', linestyle='--', label='Optuna Best (0.002441)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss' if not USE_MASKED_LOSS else 'Masked MSE Loss', fontsize=12)
    plt.title(f'TCN Training: {snr_label} {modulation_order} {scheme}', fontsize=14)
    plt.tight_layout()
    plot_file = f'./MixedChannel_{snr_label}_{modulation_order}_{scheme}_TCN_loss.png'
    plt.savefig(plot_file, dpi=300)
    print(f"Loss plot saved: {plot_file}\n")

# =========================
# TESTING MODE
# =========================
elif mode == 'test':
    print(f"\n{'='*70}")
    print(f"TESTING: Optuna-Tuned TCN")
    print(f"{'='*70}")
    print(f"Model trained with: {snr_label}")
    print(f"Testing channel: {test_channel}")
    print(f"Modulation: {modulation_order}")
    print(f"Scheme: {scheme}")
    print(f"{'='*70}\n")

    modu_way = 1 if modulation_order == 'QPSK' else 2 if modulation_order == '16QAM' else None
    if modu_way is None:
        raise ValueError("Unsupported modulation_order")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load scalers
    scaler_path = f'./MixedChannel_{snr_label}_{modulation_order}_{scheme}_TCN_scalers.npz'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}\nTrain first!")
    
    sc = np.load(scaler_path)
    sx = StandardScaler()
    sx.mean_ = sc['sx_mean']
    sx.scale_ = sc['sx_scale']
    sy = StandardScaler()
    sy.mean_ = sc['sy_mean']
    sy.scale_ = sc['sy_scale']
    print(f"Loaded scalers: {scaler_path}")

    # Load model
    model_path = f'./MixedChannel_{snr_label}_{modulation_order}_{scheme}_TCN_optimized.pt'
    
    model = ResidualKStack(
        num_inputs=100,
        hidden_channels=BEST_HIDDEN_CHANNELS,
        kernel_size=BEST_KERNEL_SIZE,
        dilations=BEST_DILATIONS,
        dropout=BEST_DROPOUT,
        n_blocks=BEST_N_BLOCKS,
        output_size=100
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {model_path}\n")

    n_params = count_parameters(model)
    macs, flops = calc_macs_flops(model, input_len=52)
    print(f'Parameters: {n_params:,}')
    print(f'MACs/forward: {macs:,}')
    print(f'FLOPs/forward: {flops:,}\n')

    # Determine mobility
    if 'RTV' in test_channel:
        mobility = 'High'
    else:
        mobility = 'Very_High'

    ppositions_mat = np.array([7, 21, 32, 46], dtype=int)
    PILOT_IDX = ppositions_mat - 1

    dpositions_mat = np.array([1,2,3,4,5,6,
                               8,9,10,11,12,13,14,15,16,17,18,19,20,
                               22,23,24,25,26,27,28,29,30,31,
                               33,34,35,36,37,38,39,40,41,42,43,44,
                               47,48,49,50,51,52], dtype=int)
    DSC_IDX_local = dpositions_mat - 1

    nSym  = 50
    nTime = 2 * nSym
    nSC   = 52

    eps = 1e-8
    clamp_min_mag = 1e-3

    # Test on all SNRs
    for n_snr in SNR_index:
        print(f"\n{'='*70}")
        print(f"Testing SNR = {n_snr} dB")
        print(f"{'='*70}")

        test_file = f'./{mobility}_{test_channel}_{modulation_order}_{scheme}_TCN_testing_dataset_{n_snr}.mat'
        
        if not os.path.exists(test_file):
            print(f"Warning: Test file not found: {test_file}")
            print("Skipping this SNR...\n")
            continue
        
        with h5py.File(test_file, 'r') as file:
            D     = file['TCN_Datasets']
            X     = np.array(D['Test_X'])
            Y     = np.array(D['Test_Y'])
            yf_d  = np.array(D['Y_DataSubCarriers'])

        print('Loaded (raw): X', X.shape, ' Y', Y.shape, ' yf_d', yf_d.shape)

        X    = norm_XY(X, nTime=nTime, nSC=nSC).astype(np.float32, copy=False)
        Y    = norm_XY(Y, nTime=nTime, nSC=nSC).astype(np.float32, copy=False)
        N    = X.shape[0]
        yf_d = to_numpy_complex(yf_d)
        yf_d, nDSC = norm_yfd(yf_d, N_expected=N, nSym_expected=nSym)
        yf_d = np.array(yf_d, dtype=np.complex64, copy=False)
        print('Normalized: X', X.shape, ' Y', Y.shape, ' yf_d', yf_d.shape)

        DSC_use = DSC_IDX

        # Normalize with training scaler
        X_normalized = apply_scale(X, sx)

        # Pre-DD forward pass
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_normalized).float().to(device)
            sync_if_cuda()
            t0 = time.perf_counter()
            Y_hat_tensor = model(X_tensor)
            sync_if_cuda()
            fwd_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Pre-DD forward time: {fwd_ms:.2f} ms")

            Y_hat_normalized = Y_hat_tensor.cpu().numpy()
            Y_hat_den = invert_scale(Y_hat_normalized, sy)
            H_pre = (Y_hat_den[:, 0::2, :] + 1j * Y_hat_den[:, 1::2, :]).astype(np.complex64)

        # Decision-Directed refinement
        nDSC = len(DSC_use)
        hf_dd = np.zeros((N, nSym, nDSC), dtype=np.complex64)

        t_all_start = time.perf_counter()
        for i in range(N):
            X_frame_norm = X_normalized[i].copy()

            for j in range(nSym):
                r_idx = 2*j
                i_idx = r_idx + 1

                with torch.no_grad():
                    X_single = torch.from_numpy(X_frame_norm[np.newaxis, :, :]).float().to(device)
                    Y_pred_norm = model(X_single)[0].cpu().numpy()
                    Y_pred = invert_scale(Y_pred_norm[np.newaxis, :, :], sy)[0]

                h_pred = (Y_pred[r_idx, :] + 1j * Y_pred[i_idx, :]).astype(np.complex64)

                y_data = yf_d[i, j, DSC_use].astype(np.complex64, copy=False)
                h_data = h_pred[DSC_use]

                # Clamp tiny magnitudes
                h_abs = np.abs(h_data)
                h_ang = np.angle(h_data + eps)
                h_data = np.where(h_abs < clamp_min_mag,
                                  clamp_min_mag * np.exp(1j*h_ang),
                                  h_data).astype(np.complex64, copy=False)

                # Hard decisions
                from functions import demap, map as map_sym
                s_eq  = y_data / h_data
                x_hat = demap(s_eq, modu_way)
                x_mod = map_sym(x_hat, modu_way).astype(np.complex64, copy=False)

                h_refined = y_data / (x_mod + eps)
                hf_dd[i, j, :] = h_refined

                # Feedback for next symbol
                if j < nSym - 1:
                    next_r = 2*(j+1)
                    next_i = next_r + 1
                    h_update = np.zeros((1, 100, 52), dtype=np.float32)
                    h_update[0, next_r, DSC_use] = np.real(h_refined).astype(np.float32)
                    h_update[0, next_i, DSC_use] = np.imag(h_refined).astype(np.float32)
                    h_update_norm = apply_scale(h_update, sx)[0]
                    X_frame_norm[next_r, DSC_use] = h_update_norm[next_r, DSC_use]
                    X_frame_norm[next_i, DSC_use] = h_update_norm[next_i, DSC_use]

            if i % 500 == 0:
                print(f"  Processed {i}/{N} frames")

        dd_time = (time.perf_counter() - t_all_start) * 1000.0
        print(f"Total DD loop time: {dd_time:.2f} ms")

        # Save results
        dest_name = f'./MixedChannel_{snr_label}_{modulation_order}_{scheme}_TCN_{test_channel}_Results_{n_snr}.mat'
        scipy.io.savemat(dest_name, {
            f'{scheme}_TCN_test_x_{n_snr}': X.astype(np.float32),
            f'{scheme}_TCN_test_y_{n_snr}': Y.astype(np.float32),
            f'{scheme}_TCN_pred_preDD_{n_snr}': H_pre,
            f'{scheme}_TCN_pred_postDD_{n_snr}': hf_dd
        })
        print(f"Saved: {dest_name}")
        print(f"{'='*70}\n")
    
    print("Testing complete!\n")

else:
    print("Invalid mode. Use 'train' or 'test'")
    sys.exit(1)
