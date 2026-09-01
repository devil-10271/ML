"""
enhancer.py
-----------
Core engine of the Smart Image Enhancement and Quality Improvement System.

The ImageEnhancer class wraps a set of classic, well understood computer
vision techniques (no external pretrained models needed, so it runs fully
offline) and exposes both:

    1. Individual operations  (denoise, sharpen, gamma_correction, ...)
    2. An "auto" pipeline that inspects the image's own quality metrics
       (see utils.py) and decides which operations are actually needed
       and by how much - this is the "smart" part of the system.

All internal processing is done with OpenCV in BGR format.
"""

from __future__ import annotations

import cv2
import numpy as np

import utils


class ImageEnhancer:
    def __init__(self, image: np.ndarray):
        """image: a BGR numpy array (as read by cv2.imread)."""
        if image is None:
            raise ValueError("ImageEnhancer received an empty image.")
        self.original = image.copy()
        self.image = image.copy()
        self.log: list[str] = []

    # ------------------------------------------------------------------
    # Basic building blocks
    # ------------------------------------------------------------------

    def denoise(self, strength: int = 7) -> "ImageEnhancer":
        """Non-local means denoising - good at removing grain/noise
        while keeping edges reasonably intact."""
        self.image = cv2.fastNlMeansDenoisingColored(
            self.image, None, h=strength, hColor=strength,
            templateWindowSize=7, searchWindowSize=21,
        )
        self.log.append(f"denoise(strength={strength})")
        return self

    def adjust_brightness_contrast(self, alpha: float = 1.0, beta: float = 0.0) -> "ImageEnhancer":
        """alpha = contrast gain (1.0 = no change), beta = brightness offset (-255..255)."""
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        self.log.append(f"brightness_contrast(alpha={alpha:.2f}, beta={beta:.1f})")
        return self

    def gamma_correction(self, gamma: float = 1.0) -> "ImageEnhancer":
        """gamma < 1 brightens shadows, gamma > 1 darkens the image."""
        inv_gamma = 1.0 / max(gamma, 1e-6)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)
        self.log.append(f"gamma_correction(gamma={gamma:.2f})")
        return self

    def equalize_clahe(self, clip_limit: float = 2.0, tile_grid_size: int = 8) -> "ImageEnhancer":
        """Contrast Limited Adaptive Histogram Equalization on the L channel
        of LAB colour space - improves local contrast without blowing out
        colours the way global histogram equalization does."""
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        self.image = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        self.log.append(f"clahe(clip_limit={clip_limit}, tile={tile_grid_size})")
        return self

    def sharpen(self, amount: float = 1.0, radius: int = 5) -> "ImageEnhancer":
        """Unsharp masking: blur the image, then push the original away
        from the blur to exaggerate edges."""
        blurred = cv2.GaussianBlur(self.image, (0, 0), sigmaX=radius)
        self.image = cv2.addWeighted(self.image, 1 + amount, blurred, -amount, 0)
        self.log.append(f"sharpen(amount={amount:.2f}, radius={radius})")
        return self

    def enhance_details(self, sigma_s: float = 12.0, sigma_r: float = 0.20) -> "ImageEnhancer":
        """Edge-aware detail/texture boost (OpenCV's detailEnhance).
        Complements unsharp masking - unsharp mask punches up edge
        contrast, this pulls out fine mid-frequency texture (skin
        pores, fabric weave, foliage, brick, hair) that a simple
        unsharp mask tends to miss on moderately soft images."""
        self.image = cv2.detailEnhance(self.image, sigma_s=sigma_s, sigma_r=sigma_r)
        self.log.append(f"enhance_details(sigma_s={sigma_s}, sigma_r={sigma_r})")
        return self

    def white_balance(self) -> "ImageEnhancer":
        """Simple 'gray world' white balance: assumes the average scene
        colour should be neutral gray and rescales channels accordingly."""
        img = self.image.astype(np.float32)
        b, g, r = cv2.split(img)
        b_avg, g_avg, r_avg = b.mean(), g.mean(), r.mean()
        gray_avg = (b_avg + g_avg + r_avg) / 3
        b = np.clip(b * (gray_avg / (b_avg + 1e-6)), 0, 255)
        g = np.clip(g * (gray_avg / (g_avg + 1e-6)), 0, 255)
        r = np.clip(r * (gray_avg / (r_avg + 1e-6)), 0, 255)
        self.image = cv2.merge([b, g, r]).astype(np.uint8)
        self.log.append("white_balance()")
        return self

    def upscale(self, factor: float = 2.0) -> "ImageEnhancer":
        """Resolution boost using high quality Lanczos interpolation,
        followed by a light sharpen pass to counter the softness that
        interpolation introduces."""
        h, w = self.image.shape[:2]
        new_size = (int(w * factor), int(h * factor))
        self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_LANCZOS4)
        self.sharpen(amount=0.4, radius=3)
        self.log.append(f"upscale(factor={factor})")
        return self

    # ------------------------------------------------------------------
    # The "smart" auto pipeline
    # ------------------------------------------------------------------

    def auto_enhance(self, upscale_factor: float | None = None) -> "ImageEnhancer":
        """
        Inspects the current image's metrics and decides which
        corrections to apply, and how strongly, instead of running every
        filter at a fixed strength. Order matters: denoise before
        sharpening, colour/exposure fixes before contrast enhancement.
        """
        metrics = utils.analyze(self.image)

        # 1. Denoise first if the image is noisy, so we don't sharpen noise later.
        #    Even mild grain gets a light pass, since sharpening below would
        #    otherwise amplify it into visible noise.
        if metrics["noise_level"] > 1.2:
            strength = float(np.clip(metrics["noise_level"] * 1.8, 3, 14))
            self.denoise(strength=strength)

        # 2. Fix exposure. Too dark or too bright -> gamma correction.
        if metrics["brightness"] < 90:
            gamma = np.clip(0.6 + (metrics["brightness"] / 90) * 0.4, 0.5, 0.95)
            self.gamma_correction(gamma=gamma)
        elif metrics["brightness"] > 175:
            gamma = np.clip(1.0 + (metrics["brightness"] - 175) / 80, 1.05, 1.6)
            self.gamma_correction(gamma=gamma)

        # 3. Improve local contrast if the image looks flat/washed out.
        #    (raised from 45 -> 65 so mildly hazy/flat images get caught too)
        if metrics["contrast"] < 65:
            clip = float(np.clip((65 - metrics["contrast"]) / 20, 1.5, 4.0))
            self.equalize_clahe(clip_limit=clip)

        # 4. Correct colour cast if colours look weak/unbalanced.
        if metrics["colorfulness"] < 20:
            self.white_balance()

        # 5. Sharpen / restore detail. Uses a continuous curve instead of a
        #    hard on/off switch, so mild softness still gets a real fix
        #    rather than a token 0.2 polish - but capped so it never tips
        #    into halos/ringing or amplifies grain into visible noise.
        sharp = metrics["sharpness"]
        if sharp < 400:
            amount = float(np.clip(1.0 - sharp / 500, 0.45, 1.0))
            self.sharpen(amount=amount, radius=3)
            # A light, edge-aware texture pass for moderately soft images.
            # Kept mild on purpose - a strong detailEnhance pass looks
            # "painterly"/noisy rather than genuinely sharper.
            if sharp < 200:
                self.enhance_details(sigma_s=8, sigma_r=0.10)
        else:
            # Already sharp - light polish only, no detail pass needed.
            self.sharpen(amount=0.25, radius=3)

        # 6. Optional resolution boost.
        if upscale_factor and upscale_factor > 1.0:
            self.upscale(factor=upscale_factor)

        return self

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def reset(self) -> "ImageEnhancer":
        self.image = self.original.copy()
        self.log.clear()
        return self

    def result(self) -> np.ndarray:
        return self.image

    def side_by_side(self) -> np.ndarray:
        """Original and enhanced image placed next to each other,
        resized to match heights so they line up cleanly."""
        h = min(self.original.shape[0], self.image.shape[0])

        def resize_h(img, target_h):
            scale = target_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), target_h))

        left = resize_h(self.original, h)
        right = resize_h(self.image, h)
        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        return np.hstack([left, gap, right])

    def save(self, path: str) -> None:
        cv2.imwrite(path, self.image)
