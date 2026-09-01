"""
main.py
-------
Command line interface for the Smart Image Enhancement and Quality
Improvement System.

Drop images into the input/ folder (next to this script) and just pass
their filename - results are written to the output/ folder
automatically. Full/relative paths still work too if you prefer them.

Usage examples
--------------
Auto mode (recommended - the system decides what the image needs):
    python main.py --input photo.jpg --auto

    (reads input/photo.jpg, writes output/photo_enhanced.jpg)

Auto mode + 2x resolution boost + side-by-side comparison image:
    python main.py --input photo.jpg --auto --upscale 2 --compare

Manual mode (apply only the operations you choose, with your own strength):
    python main.py --input photo.jpg \
        --denoise 8 --gamma 0.8 --clahe --sharpen 0.6 --white-balance

Just inspect quality metrics without writing any file:
    python main.py --input photo.jpg --analyze-only

You can still give explicit paths / a custom output name if you want:
    python main.py --input /some/where/photo.jpg --output result.jpg --auto
"""

import argparse
import os
import sys

import cv2

import utils
from enhancer import ImageEnhancer

# Folders that live next to this script. Input images can just be placed
# in INPUT_DIR and referenced by filename; enhanced results are written
# to OUTPUT_DIR automatically unless a different path is given.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart Image Enhancement and Quality Improvement System"
    )
    p.add_argument("--input", "-i", required=True, help="Filename in the input/ folder, or a full/relative path")
    p.add_argument("--output", "-o", default=None, help="Output filename (saved into output/) or a full path. Defaults to <name>_enhanced.jpg in output/")
    p.add_argument("--compare", action="store_true", help="Also save a side-by-side before/after image")
    p.add_argument("--analyze-only", action="store_true", help="Only print quality metrics, no enhancement")

    p.add_argument("--auto", action="store_true", help="Let the system automatically decide the enhancements")

    manual = p.add_argument_group("manual mode (ignored if --auto is used)")
    manual.add_argument("--denoise", type=float, default=None, metavar="STRENGTH", help="Apply denoising, e.g. 7")
    manual.add_argument("--brightness", type=float, default=None, metavar="BETA", help="Brightness offset -255..255")
    manual.add_argument("--contrast", type=float, default=None, metavar="ALPHA", help="Contrast gain, e.g. 1.2")
    manual.add_argument("--gamma", type=float, default=None, metavar="GAMMA", help="Gamma correction, e.g. 0.8")
    manual.add_argument("--clahe", action="store_true", help="Apply CLAHE local contrast enhancement")
    manual.add_argument("--white-balance", action="store_true", help="Apply gray-world white balance")
    manual.add_argument("--sharpen", type=float, default=None, metavar="AMOUNT", help="Unsharp mask amount, e.g. 0.6")

    p.add_argument("--upscale", type=float, default=None, metavar="FACTOR", help="Resolution boost factor, e.g. 2")

    return p


def main() -> None:
    args = build_parser().parse_args()

    input_path = _resolve_input_path(args.input)
    if not os.path.exists(input_path):
        print(f"Error: input file not found -> {args.input}")
        print(f"(looked in current directory and in: {INPUT_DIR})")
        sys.exit(1)

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: could not read image (unsupported format?) -> {input_path}")
        sys.exit(1)

    before_metrics = utils.analyze(image)
    utils.print_report("BEFORE", before_metrics)

    if args.analyze_only:
        return

    enhancer = ImageEnhancer(image)

    if args.auto:
        enhancer.auto_enhance(upscale_factor=args.upscale)
    else:
        if args.denoise is not None:
            enhancer.denoise(strength=args.denoise)
        if args.brightness is not None or args.contrast is not None:
            enhancer.adjust_brightness_contrast(
                alpha=args.contrast if args.contrast is not None else 1.0,
                beta=args.brightness if args.brightness is not None else 0.0,
            )
        if args.gamma is not None:
            enhancer.gamma_correction(gamma=args.gamma)
        if args.clahe:
            enhancer.equalize_clahe()
        if args.white_balance:
            enhancer.white_balance()
        if args.sharpen is not None:
            enhancer.sharpen(amount=args.sharpen)
        if args.upscale is not None:
            enhancer.upscale(factor=args.upscale)

    result = enhancer.result()
    after_metrics = utils.analyze(result)
    utils.print_report("AFTER", after_metrics)

    print("\nOperations applied:")
    for step in enhancer.log:
        print(f"  - {step}")

    output_path = _resolve_output_path(args.output, input_path)
    enhancer.save(output_path)
    print(f"\nSaved enhanced image to: {output_path}")

    if args.compare:
        compare_path = _compare_path(output_path)
        cv2.imwrite(compare_path, enhancer.side_by_side())
        print(f"Saved before/after comparison to: {compare_path}")


def _resolve_input_path(input_arg: str) -> str:
    """Accepts either a direct/relative path or just a filename.
    A bare filename is looked up inside INPUT_DIR if it isn't found
    in the current directory first."""
    if os.path.exists(input_arg):
        return input_arg
    candidate = os.path.join(INPUT_DIR, input_arg)
    if os.path.exists(candidate):
        return candidate
    return input_arg  # let the caller report the not-found error


def _resolve_output_path(output_arg: str | None, input_path: str) -> str:
    """If the user gave an explicit --output, honour it as-is. Otherwise,
    write the result into OUTPUT_DIR using the input's filename."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_arg:
        # A bare filename (no directory component) still lands in OUTPUT_DIR.
        if os.path.dirname(output_arg) == "":
            return os.path.join(OUTPUT_DIR, output_arg)
        return output_arg
    base, ext = os.path.splitext(os.path.basename(input_path))
    return os.path.join(OUTPUT_DIR, f"{base}_enhanced{ext or '.jpg'}")


def _compare_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return f"{base}_compare{ext or '.jpg'}"


if __name__ == "__main__":
    main()
