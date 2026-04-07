import os
import torch
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import glob

from reverse_blur_interpolation_transformer_training import create_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path):
    """Load an RGB image as a (3, H, W) float tensor in [0, 1]."""
    return transforms.ToTensor()(Image.open(path).convert('RGB'))


def load_sharp_window(frame_dir, num_frames=9):
    """
    Load a consecutive window of sharp frames from a directory.
    Returns a (1, N, 3, H, W) batch tensor ready for the model.
    """
    paths = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
    if len(paths) < num_frames:
        raise ValueError(f"Need at least {num_frames} frames, found {len(paths)} in {frame_dir}")

    # Take the first full window
    window = paths[:num_frames]
    frames = torch.stack([load_image(p) for p in window])   # (N, 3, H, W)
    return frames.unsqueeze(0)                               # (1, N, 3, H, W)


def naive_average_baseline(sharp_frames):
    """
    Traditional baseline: simple temporal average of sharp frames.
    sharp_frames: (1, N, 3, H, W)
    Returns: (3, H, W)
    """
    return sharp_frames.squeeze(0).mean(dim=0)


# ── Main inference ────────────────────────────────────────────────────────────

def main():
    device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoints/epoch_4.pth'
    input_dir       = 'dataset/train/scene_1/sharp'
    output_dir      = 'results/'
    num_frames      = 9

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = create_model().to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    model.eval()
    print(f"Model loaded on {device}")

    # Load sharp frames
    sharp_frames = load_sharp_window(input_dir, num_frames=num_frames).to(device)
    print(f"Loaded {num_frames} sharp frames from {input_dir}")

    # Run model
    with torch.no_grad():
        blur_pred = model(sharp_frames)                      # (1, 3, H, W)
        blur_pred = blur_pred.squeeze(0).clamp(0, 1).cpu()  # (3, H, W)

    # Naive baseline for comparison
    baseline = naive_average_baseline(sharp_frames.cpu())

    # Save outputs
    vutils.save_image(blur_pred, os.path.join(output_dir, 'rbit_output.png'))
    vutils.save_image(baseline,  os.path.join(output_dir, 'naive_baseline.png'))

    # Side-by-side comparison
    comparison = torch.stack([baseline, blur_pred], dim=0)
    vutils.save_image(comparison, os.path.join(output_dir, 'comparison.png'), nrow=2, padding=4)

    print(f"Results saved to {output_dir}")

    # TODO run model on blurry images and save inferred data


if __name__ == "__main__":
    main()