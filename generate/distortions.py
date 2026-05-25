import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from typing import Callable

IMG_WIDTH = 250
IMG_HEIGHT = 70


def apply_line(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_lines = rng.integers(1, 3)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for _ in range(n_lines):
        angle = rng.uniform(-30, 30)
        angle_rad = np.deg2rad(angle)
        cx = IMG_WIDTH / 2
        cy = IMG_HEIGHT / 2
        half_diag = np.hypot(IMG_WIDTH, IMG_HEIGHT) / 2
        x0 = int(cx - half_diag * np.cos(angle_rad))
        y0 = int(cy - half_diag * np.sin(angle_rad))
        x1 = int(cx + half_diag * np.cos(angle_rad))
        y1 = int(cy + half_diag * np.sin(angle_rad))
        draw.line([(x0, y0), (x1, y1)], fill=0, width=1)
    return np.array(pil)


def apply_dots(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    n_dots = rng.integers(30, 61)
    xs = rng.integers(0, IMG_WIDTH - 2, size=n_dots)
    ys = rng.integers(0, IMG_HEIGHT - 2, size=n_dots)
    for x, y in zip(xs, ys):
        out[y:y+3, x:x+3] = 0
    return out


def apply_wave(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    amplitude = rng.uniform(3, 6)
    wavelength = rng.uniform(30, 60)
    out = np.full_like(img, 255)
    for col in range(IMG_WIDTH):
        shift = int(amplitude * np.sin(2 * np.pi * col / wavelength))
        out[:, col] = np.roll(img[:, col], shift)
    return out


def apply_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(1.5, 2.5)
    blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def apply_salt_pepper(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    density = rng.uniform(0.02, 0.04)
    n_pixels = int(IMG_WIDTH * IMG_HEIGHT * density)
    xs = rng.integers(0, IMG_WIDTH, size=n_pixels)
    ys = rng.integers(0, IMG_HEIGHT, size=n_pixels)
    values = rng.choice([0, 255], size=n_pixels)
    for x, y, v in zip(xs, ys, values):
        out[y, x] = v
    return out


def apply_pixelate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    factor = rng.integers(4, 9)
    pil = Image.fromarray(img)
    small = pil.resize((IMG_WIDTH // factor, IMG_HEIGHT // factor), Image.NEAREST)
    return np.array(small.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST))


def apply_rotation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-30, 30)
    pil = Image.fromarray(img)
    rotated = pil.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
    return np.array(rotated)


DISTORTIONS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "line":        apply_line,
    "dots":        apply_dots,
    "wave":        apply_wave,
    "blur":        apply_blur,
    "salt_pepper": apply_salt_pepper,
    "pixelate":    apply_pixelate,
    "rotation":    apply_rotation,
}
