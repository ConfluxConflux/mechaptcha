from pathlib import Path
import requests
from fontTools.ttLib import TTFont
from PIL import ImageFont

BASE = "https://raw.githubusercontent.com/google/fonts/main"

FONT_URLS = {
    "roboto":            f"{BASE}/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
    "roboto_mono":       f"{BASE}/ofl/robotomono/RobotoMono%5Bwght%5D.ttf",
    "open_sans":         f"{BASE}/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
    "lato":              f"{BASE}/ofl/lato/Lato-Regular.ttf",
    "merriweather":      f"{BASE}/ofl/merriweather/Merriweather%5Bopsz%2Cwdth%2Cwght%5D.ttf",
    "playfair_display":  f"{BASE}/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "oswald":            f"{BASE}/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "source_code_pro":   f"{BASE}/ofl/sourcecodepro/SourceCodePro%5Bwght%5D.ttf",
    "ubuntu":            f"{BASE}/ufl/ubuntu/Ubuntu-Regular.ttf",
    "pt_serif":          f"{BASE}/ofl/ptserif/PT_Serif-Web-Bold.ttf",
    "nunito":            f"{BASE}/ofl/nunito/Nunito%5Bwght%5D.ttf",
    "raleway":           f"{BASE}/ofl/raleway/Raleway%5Bwght%5D.ttf",
    "inconsolata":       f"{BASE}/ofl/inconsolata/Inconsolata%5Bwdth%2Cwght%5D.ttf",
    "libre_baskerville": f"{BASE}/ofl/librebaskerville/LibreBaskerville%5Bwght%5D.ttf",
    "cabin":             f"{BASE}/ofl/cabin/Cabin%5Bwdth%2Cwght%5D.ttf",
    "arvo":              f"{BASE}/ofl/arvo/Arvo-Regular.ttf",
    "quicksand":         f"{BASE}/ofl/quicksand/Quicksand%5Bwght%5D.ttf",
    "courier_prime":     f"{BASE}/ofl/courierprime/CourierPrime-Regular.ttf",
    "pt_mono":           f"{BASE}/ofl/ptmono/PTM55FT.ttf",
    "noto_sans":         f"{BASE}/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
}

VALIDATION_CHARSET = "abcdefghijklmnopqrstuvwxyz"


def download_fonts(font_dir: Path) -> list[Path]:
    font_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name, url in FONT_URLS.items():
        dest = font_dir / f"{name}.ttf"
        if dest.exists():
            downloaded.append(dest)
            continue
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            downloaded.append(dest)
            print(f"  Downloaded {name}")
        except Exception as e:
            print(f"  Skipped {name}: {e}")
    return downloaded


def validate_font(path: Path, charset: str = VALIDATION_CHARSET) -> bool:
    try:
        tt = TTFont(path)
        cmap = tt.getBestCmap()
        if cmap is None:
            return False
        for ch in charset:
            if ord(ch) not in cmap:
                return False
        return True
    except Exception:
        return False


def load_fonts(font_dir: Path, font_size: int = 44) -> dict[str, ImageFont.FreeTypeFont]:
    fonts = {}
    for path in sorted(font_dir.glob("*.ttf")):
        name = path.stem
        if not validate_font(path):
            print(f"  Skipping {name} (failed glyph validation)")
            continue
        try:
            fonts[name] = ImageFont.truetype(str(path), font_size)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    return fonts
