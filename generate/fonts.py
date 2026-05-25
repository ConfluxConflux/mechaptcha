from pathlib import Path
import requests
from fontTools.ttLib import TTFont
from PIL import ImageFont

FONT_URLS = {
    "roboto":            "https://raw.githubusercontent.com/google/fonts/main/apache/roboto/Roboto-Regular.ttf",
    "roboto_mono":       "https://raw.githubusercontent.com/google/fonts/main/apache/robotomono/RobotoMono-Regular.ttf",
    "open_sans":         "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans-Regular.ttf",
    "lato":              "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Regular.ttf",
    "merriweather":      "https://raw.githubusercontent.com/google/fonts/main/ofl/merriweather/Merriweather-Regular.ttf",
    "playfair_display":  "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay-Regular.ttf",
    "oswald":            "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald-Regular.ttf",
    "source_code_pro":   "https://raw.githubusercontent.com/google/fonts/main/ofl/sourcecodepro/SourceCodePro-Regular.ttf",
    "ubuntu":            "https://raw.githubusercontent.com/google/fonts/main/ofl/ubuntu/Ubuntu-Regular.ttf",
    "pt_serif":          "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PTSerif-Regular.ttf",
    "nunito":            "https://raw.githubusercontent.com/google/fonts/main/ofl/nunito/Nunito-Regular.ttf",
    "raleway":           "https://raw.githubusercontent.com/google/fonts/main/ofl/raleway/Raleway-Regular.ttf",
    "inconsolata":       "https://raw.githubusercontent.com/google/fonts/main/ofl/inconsolata/Inconsolata-Regular.ttf",
    "libre_baskerville": "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville-Regular.ttf",
    "cabin":             "https://raw.githubusercontent.com/google/fonts/main/ofl/cabin/Cabin-Regular.ttf",
    "arvo":              "https://raw.githubusercontent.com/google/fonts/main/ofl/arvo/Arvo-Regular.ttf",
    "quicksand":         "https://raw.githubusercontent.com/google/fonts/main/ofl/quicksand/Quicksand-Regular.ttf",
    "courier_prime":     "https://raw.githubusercontent.com/google/fonts/main/ofl/courierprime/CourierPrime-Regular.ttf",
    "pt_mono":           "https://raw.githubusercontent.com/google/fonts/main/ofl/ptmono/PTMono-Regular.ttf",
    "noto_sans":         "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans-Regular.ttf",
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
