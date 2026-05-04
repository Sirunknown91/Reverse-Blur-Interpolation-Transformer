import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob


class RBIDataset(Dataset):
    """
    Real-world Blur Interpolation (RBI) dataset loader.
    Loads aligned sharp (500fps) and blurred (25fps) video frame pairs.
    Each blurred frame corresponds to a window of 9 consecutive sharp frames.
    """
    def __init__(self, root_dir, split='train', num_sharp_frames=9, patch_size=256):
        self.root_dir = os.path.join(root_dir, split)
        self.num_sharp_frames = num_sharp_frames
        self.patch_size = patch_size
        self.samples = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self._build_sample_list()

    def _build_sample_list(self):
        video_dirs = sorted(os.listdir(self.root_dir))
        for video in video_dirs:
            sharp_dir = os.path.join(self.root_dir, video, 'sharp')
            blur_dir  = os.path.join(self.root_dir, video, 'blur')

            if not os.path.exists(sharp_dir) or not os.path.exists(blur_dir):
                continue

            sharp_frames = sorted(glob.glob(os.path.join(sharp_dir, '*.png')))
            blur_frames  = sorted(glob.glob(os.path.join(blur_dir, '*.png')))

            half = self.num_sharp_frames // 2

            for b_idx, blur_path in enumerate(blur_frames):
                centre = b_idx
                start  = centre - half
                end    = centre + half + 1

                if start < 0 or end > len(sharp_frames):
                    continue

                window = sharp_frames[start:end]

                if len(window) == self.num_sharp_frames:
                    self.samples.append((window, blur_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sharp_paths, blur_path = self.samples[idx]
        sharp_frames = torch.stack([
            self.transform(Image.open(p).convert('RGB')) for p in sharp_paths
        ])                                   # (9, 3, H, W)
        blur_frame = self.transform(Image.open(blur_path).convert('RGB'))  # (3, H, W)

        # Random crop
        _, H, W = blur_frame.shape
        if self.patch_size:
            top  = torch.randint(0, H - self.patch_size, (1,)).item()
            left = torch.randint(0, W - self.patch_size, (1,)).item()
            sharp_frames = sharp_frames[:, :, top:top+self.patch_size, left:left+self.patch_size]
            blur_frame   = blur_frame[:,    top:top+self.patch_size, left:left+self.patch_size]

        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            sharp_frames = torch.flip(sharp_frames, dims=[-1])
            blur_frame   = torch.flip(blur_frame,   dims=[-1])

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            sharp_frames = torch.flip(sharp_frames, dims=[-2])
            blur_frame   = torch.flip(blur_frame,   dims=[-2])

        return sharp_frames, blur_frame


# ── MS-RSTB Building Blocks ───────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self Attention (W-MSA) as used in Swin Transformer.
    Operates on non-overlapping local windows of size window_size x window_size.
    This is the core attention mechanism inside each RSTB block.
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads   = num_heads
        self.scale       = (dim // num_heads) ** -0.5

        # Learnable relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index for each token pair inside a window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)                                   # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)                           # (N, N)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv   = nn.Linear(dim, dim * 3)
        self.proj  = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (num_windows*B, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                # each: (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)          # (B_, num_heads, N, N)

        # Add relative position bias
        rel_pos_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + rel_pos_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x    = self.proj(x)
        return x


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    x: (B, H, W, C)
    Returns: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partitioning back to feature map.
    windows: (num_windows*B, window_size, window_size, C)
    Returns: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """
    One Swin Transformer Block: LayerNorm -> W-MSA -> residual -> LayerNorm -> MLP -> residual.
    This is the fundamental building block of RSTB.
    """
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2  = nn.LayerNorm(dim)
        mlp_hidden  = int(dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x, H, W):
        # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Partition into windows, apply attention, reverse
        x_windows    = window_partition(x, self.window_size)           # (nW*B, ws, ws, C)
        x_windows    = x_windows.view(-1, self.window_size ** 2, C)    # (nW*B, ws*ws, C)
        attn_windows = self.attn(x_windows)                             # (nW*B, ws*ws, C)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)     # (B, Hp, Wp, C)

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB) as described in the BIT paper.
    Contains a sequence of Swin Transformer Blocks followed by a Conv layer,
    with a residual connection around the whole thing.
    """
    def __init__(self, dim, num_heads, num_swin_blocks=2, window_size=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size=window_size)
            for _ in range(num_swin_blocks)
        ])
        # Conv layer at the end of RSTB (as in the paper)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x, H, W):
        # x: (B, H*W, C)
        residual = x
        for blk in self.blocks:
            x = blk(x, H, W)
        # Reshape to spatial, apply conv, reshape back
        B, _, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)                           # (B, H*W, C)
        return x + residual


class MS_RSTB(nn.Module):
    """
    Multi-Scale RSTB (MS-RSTB) as used in the BIT paper.
    Processes the input at multiple scales (full, 1/2, 1/4) independently,
    then fuses the multi-scale features back together.
    This helps capture both fine-grained local details and coarse global motion.
    """
    def __init__(self, dim, num_heads, num_rstb_blocks=2, window_size=8):
        super().__init__()
        # Full-scale RSTB
        self.rstb_full  = nn.ModuleList([
            RSTB(dim, num_heads, window_size=window_size) for _ in range(num_rstb_blocks)
        ])
        # Half-scale RSTB
        self.rstb_half  = nn.ModuleList([
            RSTB(dim, num_heads, window_size=window_size) for _ in range(num_rstb_blocks)
        ])
        # Quarter-scale RSTB
        self.rstb_qtr   = nn.ModuleList([
            RSTB(dim, num_heads, window_size=window_size) for _ in range(num_rstb_blocks)
        ])

        # Downsample convolutions
        self.down2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.down4 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)

        # Upsample back to full scale
        self.up2   = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.up4   = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=4)

        # Fusion: combine 3 scales back into one
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # ── Full scale ──────────────────────────────────────────
        x_full = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        for rstb in self.rstb_full:
            x_full = rstb(x_full, H, W)
        x_full = x_full.transpose(1, 2).view(B, C, H, W)

        # ── Half scale ──────────────────────────────────────────
        x_h  = self.down2(x)                            # (B, C, H/2, W/2)
        H2, W2 = x_h.shape[2], x_h.shape[3]
        x_h  = x_h.flatten(2).transpose(1, 2)
        for rstb in self.rstb_half:
            x_h = rstb(x_h, H2, W2)
        x_h  = x_h.transpose(1, 2).view(B, C, H2, W2)
        x_h  = self.up2(x_h)                            # back to (B, C, H, W)

        # ── Quarter scale ───────────────────────────────────────
        x_q  = self.down4(x)                            # (B, C, H/4, W/4)
        H4, W4 = x_q.shape[2], x_q.shape[3]
        x_q  = x_q.flatten(2).transpose(1, 2)
        for rstb in self.rstb_qtr:
            x_q = rstb(x_q, H4, W4)
        x_q  = x_q.transpose(1, 2).view(B, C, H4, W4)
        x_q  = self.up4(x_q)                            # back to (B, C, H, W)

        # ── Fuse all scales ─────────────────────────────────────
        x_fused = torch.cat([x_full, x_h, x_q], dim=1) # (B, 3C, H, W)
        x_fused = self.fusion(x_fused)                  # (B, C, H, W)
        return x_fused


# ── Full ReverseBIT Model with MS-RSTB ───────────────────────────────────────

class ReverseBIT(nn.Module):
    """
    ReverseBIT: takes N sharp frames and synthesizes a motion-blurred frame.
    Replaces the simple CNN backbone with MS-RSTB as used in the original BIT paper.

    Pipeline:
        1. Shallow feature extraction (Conv)
        2. MS-RSTB for deep feature learning (multi-scale transformer)
        3. Reconstruction head (Conv -> output image)
    """
    def __init__(self, in_channels=3, num_frames=9, dim=32, num_heads=4,
                 num_rstb_blocks=2, window_size=8, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.dropout = nn.Dropout2d(dropout)

        # Step 1: Shallow feature extractor — maps stacked frames to feature space
        self.shallow_feat = nn.Conv2d(in_channels * num_frames, dim, kernel_size=3, padding=1)

        # Step 2: MS-RSTB — the core of the BIT paper architecture
        self.ms_rstb = MS_RSTB(
            dim=dim,
            num_heads=num_heads,
            num_rstb_blocks=num_rstb_blocks,
            window_size=window_size,
        )

        # Step 3: Reconstruction head — maps features back to RGB image
        self.recon = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, sharp_frames):
        # sharp_frames: (B, N, 3, H, W)
        B, N, C, H, W = sharp_frames.shape

        # Use the center frame as the base (anchors color correctly from the start)
        center_frame = sharp_frames[:, N // 2, :, :, :]  # (B, 3, H, W)

        # Flatten frames into channels: (B, N*3, H, W)
        x = sharp_frames.view(B, N * C, H, W)

        # Shallow features: (B, dim, H, W)
        x = self.shallow_feat(x)

        x = self.dropout(x)

        # MS-RSTB deep features: (B, dim, H, W)
        x = self.ms_rstb(x)

        # Reconstruct residual: (B, 3, H, W)
        residual = self.recon(x)

        # Add center frame as skip connection — fixes color cast
        return center_frame + residual


def create_model(dim=32, num_heads=4, num_rstb_blocks=2, window_size=8):
    """
    Creates a ReverseBIT model with MS-RSTB backbone.

    CPU-friendly defaults:
        dim=32            (use 64 if you have a GPU)
        num_heads=4
        num_rstb_blocks=2 (increase for more capacity)
        window_size=8
    """
    return ReverseBIT(
        dim=dim,
        num_heads=num_heads,
        num_rstb_blocks=num_rstb_blocks,
        window_size=window_size,
    )


# ── SSIM Loss ─────────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    Used as an additional metric-aware loss on top of L1.
    SSIM ranges from 0 to 1; higher is better.
    As a loss we use (1 - SSIM) so lower is better.
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window', self._gaussian_window(window_size, sigma))

    def _gaussian_window(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        window = g[:, None] * g[None, :]          # (size, size)
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)
        return window

    def forward(self, pred, target):
        C = pred.shape[1]
        window = self.window.expand(C, 1, -1, -1).to(pred.device)
        pad    = self.window_size // 2

        mu1    = F.conv2d(pred,   window, padding=pad, groups=C)
        mu2    = F.conv2d(target, window, padding=pad, groups=C)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred   * pred,   window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
        sigma12   = F.conv2d(pred   * target, window, padding=pad, groups=C) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def compute_psnr(pred, target, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
    Higher is better. Typical good values: > 30 dB.
    pred, target: tensors in [0, 1]
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * torch.log10(mse).item()


def compute_ssim(pred, target):
    """
    Compute mean SSIM score between predicted and target images.
    Returns a float between 0 and 1. Higher is better.
    pred, target: tensors in [0, 1], shape (B, C, H, W)
    """
    ssim_loss_fn = SSIMLoss()
    with torch.no_grad():
        loss = ssim_loss_fn(pred, target)
    return (1 - loss.item())  # convert back from loss to score


# ── Training Loop ──────────────────────────────────────

def main():
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root  = 'dataset/'
    epochs     = 10
    lr         = 1e-4
    batch_size = 2

    print(f"Training on: {device}")
    if device.type == 'cpu':
        print("Note: MS-RSTB on CPU will be slow. Consider reducing dim or num_rstb_blocks.")

    # Load dataset
    train_dataset = RBIDataset(root_dir=data_root, split='train', patch_size=256)
    val_dataset   = RBIDataset(root_dir=data_root, split='test',   patch_size=256)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model — CPU-friendly dims (increase dim to 64 on GPU)
    model     = create_model(dim=32, num_heads=4, num_rstb_blocks=2, window_size=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) #decay changed
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Loss: L1 + SSIM combined (as commonly used in image restoration)
    l1_loss   = nn.L1Loss()
    ssim_loss = SSIMLoss().to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training...\n")

    best_psnr = 0.0

    for epoch in range(epochs):

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for batch_idx, (sharp_frames, blur_target) in enumerate(train_loader):
            sharp_frames = sharp_frames.to(device)
            blur_target  = blur_target.to(device)

            optimizer.zero_grad()
            blur_pred = model(sharp_frames)
            blur_pred = blur_pred.clamp(0, 1)

            # Combined L1 + SSIM loss (0.8 * L1 + 0.2 * SSIM_loss)
            loss = 0.8 * l1_loss(blur_pred, blur_target) + \
                   0.2 * ssim_loss(blur_pred, blur_target)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validation with PSNR & SSIM ───────────────────────────────────────
        model.eval()
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        val_loss_total = 0.0

        with torch.no_grad():
            for sharp_frames, blur_target in val_loader:
                sharp_frames = sharp_frames.to(device)
                blur_target  = blur_target.to(device)

                blur_pred = model(sharp_frames)
                blur_pred_clamped = blur_pred.clamp(0, 1)

                # Loss
                val_loss = 0.8 * l1_loss(blur_pred, blur_target) + \
                           0.2 * ssim_loss(blur_pred, blur_target)
                val_loss_total += val_loss.item()

                # PSNR and SSIM per batch
                val_psnr_total += compute_psnr(blur_pred_clamped, blur_target)
                val_ssim_total += compute_ssim(blur_pred_clamped, blur_target)

        avg_val_loss = val_loss_total / len(val_loader)
        avg_psnr     = val_psnr_total / len(val_loader)
        avg_ssim     = val_ssim_total / len(val_loader)

        print(f"\nEpoch [{epoch}/{epochs}] Summary:")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}")
        print(f"  Val PSNR   : {avg_psnr:.2f} dB")
        print(f"  Val SSIM   : {avg_ssim:.4f}")
        print(f"  LR         : {scheduler.get_last_lr()[0]:.2e}\n")

        # ── Save checkpoint every epoch ───────────────────────────────────────
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_psnr': avg_psnr,
            'val_ssim': avg_ssim,
        }, f'checkpoints/epoch_{epoch}.pth')

        # Save best model separately based on PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_psnr': avg_psnr,
                'val_ssim': avg_ssim,
            }, 'checkpoints/best_model.pth')
            print(f"  ✓ New best model saved (PSNR: {best_psnr:.2f} dB)")

        scheduler.step()

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()