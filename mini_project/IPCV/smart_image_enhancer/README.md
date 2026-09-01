# Smart Image Enhancement and Quality Improvement System

A mini computer-vision project that automatically analyzes an image's
quality (brightness, contrast, sharpness, noise, colorfulness) and
applies only the corrections it actually needs — instead of running a
fixed filter stack on every photo. Built with Python, OpenCV, and NumPy.
Includes a CLI, a Tkinter GUI, and a manual mode for full control.

## Features

- **Quality analysis** — measures brightness, contrast, sharpness
  (Laplacian variance), noise level, and colorfulness before and after
  processing.
- **Smart "Auto" pipeline** — decides which of the following to apply,
  and how strongly, based on the measured metrics:
  - Denoising (Non-Local Means)
  - Exposure correction (Gamma correction)
  - Local contrast enhancement (CLAHE on the LAB lightness channel)
  - Colour cast correction (Gray-World white balance)
  - Sharpening (Unsharp Masking)
- **Manual mode** — apply any operation individually with your own
  parameters via CLI flags.
- **Resolution boost** — Lanczos upscaling with a light sharpen pass.
- **Before/after comparison image** — side-by-side output for quick
  visual verification.
- **Desktop GUI** — load an image, click "Auto Enhance", preview, save.

## Project structure

```
smart_image_enhancer/
├── input/              # Drop images here - reference by filename only
├── output/             # Enhanced results are written here automatically
├── enhancer.py         # Core ImageEnhancer class + auto-decision pipeline
├── utils.py            # Image quality metric functions
├── main.py             # Command-line interface
├── gui.py               # Tkinter desktop GUI
├── requirements.txt
└── README.md
```

## Installation

```bash
cd smart_image_enhancer
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

### 1. Auto mode (recommended)

Drop your image into the `input/` folder, then just reference it by
filename - the enhanced result is written to `output/` automatically:

```bash
python main.py --input photo.jpg --auto
```

This reads `input/photo.jpg` and writes `output/photo_enhanced.jpg`.
Full or relative paths still work if you'd rather not use the
`input`/`output` folders:

```bash
python main.py --input /some/where/photo.jpg --output result.jpg --auto
```

Auto mode + 2x resolution boost + a before/after comparison image:

```bash
python main.py --input photo.jpg --auto --upscale 2 --compare
```

### 2. Manual mode

Apply only the operations you choose:

```bash
python main.py --input photo.jpg \
    --denoise 8 --gamma 0.8 --clahe --sharpen 0.6 --white-balance
```

Available manual flags: `--denoise`, `--brightness`, `--contrast`,
`--gamma`, `--clahe`, `--white-balance`, `--sharpen`, `--upscale`.

### 3. Just check image quality (no processing)

```bash
python main.py --input photo.jpg --analyze-only
```

### 4. Desktop GUI

```bash
python gui.py
```

Click **Open Image**, then **Auto Enhance**, then **Save Result**.

## How the "smart" decision-making works

`ImageEnhancer.auto_enhance()` in `enhancer.py` reads the metrics from
`utils.analyze()` and applies a small rule set, e.g.:

| Metric              | Condition          | Action                          |
|----------------------|---------------------|----------------------------------|
| `noise_level`        | > 2.5               | Denoise, strength scales with noise |
| `brightness`         | < 90 or > 175       | Gamma correction                |
| `contrast`           | < 45                | CLAHE local contrast enhancement |
| `colorfulness`       | < 15                | Gray-world white balance        |
| `sharpness`          | < 100               | Unsharp mask, amount scales with blurriness |

Operations are ordered deliberately: denoise → exposure → contrast →
colour → sharpen, so that later steps don't amplify artifacts left by
earlier ones (e.g. sharpening noisy pixels).

## Notes

- All processing uses classical, deterministic CV techniques (no
  pretrained deep learning model), so the project runs fully offline
  with no GPU or downloads required.
- The upscale step uses Lanczos interpolation, not a neural
  super-resolution model — it improves size/perceived sharpness but
  does not hallucinate new detail the way an ML super-resolution model
  would.
- Every processing step is logged (`enhancer.log`) so you can see
  exactly what was applied and why.

## Possible extensions

- Swap the upscaler for OpenCV's `dnn_superres` module or a pretrained
  ESRGAN model for true super-resolution.
- Add a batch-processing mode for entire folders.
- Add face-aware enhancement (detect faces, protect skin tones during
  sharpening/contrast steps).
