import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from typing import Callable

from .renderer import IMG_WIDTH, IMG_HEIGHT


def _draw_straight_line(draw, cx, cy, angle_rad):
    half_diag = np.hypot(IMG_WIDTH, IMG_HEIGHT) / 2
    x0 = int(cx - half_diag * np.cos(angle_rad))
    y0 = int(cy - half_diag * np.sin(angle_rad))
    x1 = int(cx + half_diag * np.cos(angle_rad))
    y1 = int(cy + half_diag * np.sin(angle_rad))
    draw.line([(x0, y0), (x1, y1)], fill=0, width=1)


def _draw_wavy_line(draw, cx, cy, angle_rad, amplitude, wavelength):
    half_diag = np.hypot(IMG_WIDTH, IMG_HEIGHT) / 2
    perp_rad = angle_rad + np.pi / 2
    ts = np.linspace(-half_diag, half_diag, int(half_diag * 4))
    pts = []
    for t in ts:
        offset = amplitude * np.sin(2 * np.pi * t / wavelength)
        x = cx + t * np.cos(angle_rad) + offset * np.cos(perp_rad)
        y = cy + t * np.sin(angle_rad) + offset * np.sin(perp_rad)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=0, width=1)


def apply_easy_line(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Always identical pixels: fixed horizontal line at vertical center
    out = img.copy()
    out[IMG_HEIGHT // 2, :] = 0
    return out


def apply_hard_line(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # One line at a random angle; 50% chance of sinusoidal waviness
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    angle_rad = np.deg2rad(rng.uniform(-30, 30))
    cx, cy = IMG_WIDTH / 2, IMG_HEIGHT / 2
    if rng.random() < 0.5:
        amplitude = rng.uniform(2, 5)
        wavelength = rng.uniform(20, 50)
        _draw_wavy_line(draw, cx, cy, angle_rad, amplitude, wavelength)
    else:
        _draw_straight_line(draw, cx, cy, angle_rad)
    return np.array(pil)


def apply_two_lines(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Two lines that cross; angles differ by 25–75°
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    cx, cy = IMG_WIDTH / 2, IMG_HEIGHT / 2
    angle1 = rng.uniform(-40, 40)
    delta = rng.uniform(25, 75) * (1 if rng.random() < 0.5 else -1)
    for angle in [angle1, angle1 + delta]:
        _draw_straight_line(draw, cx, cy, np.deg2rad(angle))
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
    amplitude = rng.uniform(1.5, 3.5)
    wavelength = rng.uniform(20, 45)
    out = np.full_like(img, 255)
    for col in range(IMG_WIDTH):
        shift = int(amplitude * np.sin(2 * np.pi * col / wavelength))
        out[:, col] = np.roll(img[:, col], shift)
    return out


def apply_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(0.6, 1.2)
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
    factor = rng.integers(3, 6)
    pil = Image.fromarray(img)
    small = pil.resize((IMG_WIDTH // factor, IMG_HEIGHT // factor), Image.NEAREST)
    return np.array(small.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST))


def apply_rotation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-10, 10)
    pil = Image.fromarray(img)
    rotated = pil.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
    return np.array(rotated)


def apply_italic(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Horizontal shear: top of letters lean right
    angle = rng.uniform(8, 15)
    s = np.tan(np.deg2rad(angle))
    h, w = img.shape
    pil = Image.fromarray(img)
    # PIL affine: x_src = a*x_dst + b*y_dst + c; center shear around mid-height
    transform = (1, s, -s * (h - 1) / 2, 0, 1, 0)
    sheared = pil.transform((w, h), Image.AFFINE, transform,
                             resample=Image.BILINEAR, fillcolor=255)
    return np.array(sheared)


def apply_bold(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    radius = rng.integers(1, 3)
    black_mask = img < 128
    dilated = binary_dilation(black_mask, iterations=int(radius))
    out = img.copy()
    out[dilated] = 0
    return out


DISTORTIONS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "easy_line":   apply_easy_line,
    "hard_line":   apply_hard_line,
    "two_lines":   apply_two_lines,
    "dots":        apply_dots,
    "wave":        apply_wave,
    "blur":        apply_blur,
    "salt_pepper": apply_salt_pepper,
    "rotation":    apply_rotation,
    "italic":      apply_italic,
    "bold":        apply_bold,
}
