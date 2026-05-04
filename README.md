# Reverse Blur Interpolation Transformer (RBIT)

This repository implements a transformer-based model for generating **realistic motion blur from a sequence of sharp video frames**. The method draws inspiration from the paper [Blur Interpolation Transformer for Real-World Motion from Blur](https://doi.org/10.48550/arXiv.2211.11423), but _reverses_ the task: while the original generates sharp intermediary images from blurred inputs, **this work synthesizes the corresponding blurred image from a stack of sharp frames**.

## Overview

- **Goal:** Learn to produce a realistic, temporally consistent blurred frame given a sequence of sharp frames as input.
- **Approach:** Transformer-based architecture adapted from the BIT paper.
- **Main scripts:**
  - `reverse_blur_interpolation_transformer_training.py` — Model, training loop.
  - `reverse_blur_interpolation_transformer_inferer.py` — Inference/demo script.

## Getting Started

### Requirements

- Python 3.8+
- PyTorch (tested on >=1.10)
- torchvision
- PIL (Pillow)

Install dependencies:

```bash
pip install torch torchvision pillow
```

### Dataset Structure

The dataset directory should be organized **per scene** as follows:

```
dataset/
  train/
    scene_1/
      blur/
        0000.png
        0001.png
        ...
      sharp/
        0000.png
        0001.png
        ...
    scene_2/
      blur/
      sharp/
    ...
  test/
    scene_Y/
      blur/
      sharp/
```

**Key points:**

- `blur/` contains multiple blurry images, typically the temporal average (or real captured) for each scene/shot.
- `sharp/` contains multiple _sequential_ sharp frames that correspond temporally to the blurred image.
- By default, the code expects **sharp frames as PNGs named `0000.png`, `0001.png`, ...`**, and similar for blurred images.
- Adjust the paths in the scripts as needed.

### Training

Edit hyperparameters and dataset paths in `reverse_blur_interpolation_transformer_training.py` as needed.

Start training:

```bash
python reverse_blur_interpolation_transformer_training.py
```

### Inference

After training, run:

```bash
python reverse_blur_interpolation_transformer_inferer.py
```

Edit the script to set:

- `input_dir` — directory of test sharp frames (see above)
- `checkpoint_path` — path to your trained model weights
- `output_dir` — where to save results

This will produce:

- Predicted blurred image (`rbit_output.png`)
- Naive baseline (temporal average, `naive_baseline.png`)
- Comparison image (`comparison.png`)

### Citation

If you use this code or dataset file structure, please cite the original BIT paper as above.

## Notes

- Make sure each scene in your dataset has at least as many sharp frames as required by the model window (default: 9).
- The sample scripts expect images as PNG. Modify as necessary if using another format.
- The code is written for clarity and simplicity for research/experimentation.
