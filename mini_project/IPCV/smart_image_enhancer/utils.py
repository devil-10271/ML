"""
utils.py
--------
Helper functions that measure objective quality metrics of an image.
These metrics are used by the ImageEnhancer's "auto" mode to decide
which corrections an image actually needs, instead of applying every
filter blindly.

Metrics implemented:
    * brightness        -> mean pixel intensity (0-255)
    * contrast           -> standard deviation of pixel intensity
    * sharpness          -> variance of the Laplacian (higher = sharper)
    * noise_level        -> estimated noise using a high-pass residual
    * colorfulness        -> Hasler & Susstrunk colorfulness metric
"""

import cv2
import numpy as np


def brightness(img_bgr: np.ndarray) -> float:
    """Average brightness on a 0-255 scale."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def contrast(img_bgr: np.ndarray) -> float:
    """Standard deviation of intensities - a simple contrast proxy."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def sharpness(img_bgr: np.ndarray) -> float:
    """
    Variance of the Laplacian. Blurry images have a narrow response
    (low variance); sharp / detailed images have a wide response.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def noise_level(img_bgr: np.ndarray) -> float:
    """
    Rough noise estimate: subtract a heavily blurred version of the
    image from the original and measure the standard deviation of the
    residual. Real detail also produces residual, so this is only a
    proxy, not a scientific measurement.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    residual = gray - blurred
    return float(np.std(residual))


def colorfulness(img_bgr: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness metric."""
    b, g, r = cv2.split(img_bgr.astype("float"))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return float(std_root + 0.3 * mean_root)


def analyze(img_bgr: np.ndarray) -> dict:
    """Return every metric in one dictionary."""
    return {
        "brightness": round(brightness(img_bgr), 2),
        "contrast": round(contrast(img_bgr), 2),
        "sharpness": round(sharpness(img_bgr), 2),
        "noise_level": round(noise_level(img_bgr), 2),
        "colorfulness": round(colorfulness(img_bgr), 2),
    }


def print_report(title: str, metrics: dict) -> None:
    print(f"\n--- {title} ---")
    for key, value in metrics.items():
        print(f"  {key:<14}: {value}")
