import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

# ── Dataset ───────────────────────────────────────────────────────────────────

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

    # def _build_sample_list(self):
    #     video_dirs = sorted(os.listdir(self.root_dir))
    #     for video in video_dirs:
    #         sharp_dir = os.path.join(self.root_dir, video, 'sharp')
    #         blur_dir  = os.path.join(self.root_dir, video, 'blur')
    #         if not os.path.exists(sharp_dir) or not os.path.exists(blur_dir):
    #             continue
    #         sharp_frames = sorted(glob.glob(os.path.join(sharp_dir, '*.png')))
    #         blur_frames  = sorted(glob.glob(os.path.join(blur_dir,  '*.png')))
    #         half = self.num_sharp_frames // 2
    #         for b_idx, blur_path in enumerate(blur_frames):
    #             centre = b_idx * 20          # 500fps / 25fps = 20 sharp per blur
    #             start  = centre - half
    #             end    = centre + half + 1
    #             if start < 0 or end > len(sharp_frames):
    #                 continue
    #             window = sharp_frames[start:end]
    #             if len(window) == self.num_sharp_frames:
    #                 self.samples.append((window, blur_path))

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

        return sharp_frames, blur_frame


# ── Model ─────────────────────────────────────────────────────────────────────

class ReverseBIT(nn.Module):
    # Reverse Blur Interpolation Transformer

    def __init__(self, in_channels=3, base_channels=64, num_frames=9):
        super().__init__()
        self.num_frames = num_frames

        # Shallow feature extractor (per-frame)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels * num_frames, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, sharp_frames):
        # sharp_frames: (B, N, 3, H, W)
        B, N, C, H, W = sharp_frames.shape
        x = sharp_frames.view(B, N * C, H, W)   # flatten frames into channels
        x = self.feature_extractor(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x                                  # (B, 3, H, W)


def create_model():
    """
    Creates a fresh ReverseBIT model
    """
    return ReverseBIT()


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = 'dataset/'
    epochs    = 5
    lr        = 1e-4
    batch_size = 2

    # Load dataset
    train_dataset = RBIDataset(root_dir=data_root, split='train', patch_size=256)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Dataset loaded: {len(train_dataset)} training samples")

    model     = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (sharp_frames, blur_target) in enumerate(train_loader):
            sharp_frames = sharp_frames.to(device)
            blur_target  = blur_target.to(device)

            optimizer.zero_grad()
            blur_pred = model(sharp_frames)
            loss      = criterion(blur_pred, blur_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        if epoch % 1 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, f'checkpoints/epoch_{epoch}.pth')

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] Avg Loss: {avg_loss:.4f} "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # TODO save model checkpoint every N epochs

        # TODO run validation loop and log PSNR/SSIM


if __name__ == "__main__":
    main()