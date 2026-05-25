from PIL import Image, ImageDraw, ImageFont

IMG_WIDTH = 250
IMG_HEIGHT = 70
SLOT_WIDTH = IMG_WIDTH // 5  # 50px per character slot


def render_captcha(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    assert len(text) == 5, f"Expected 5 characters, got {len(text)}"

    img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)

    for i, ch in enumerate(text):
        slot_x = i * SLOT_WIDTH
        bbox = draw.textbbox((0, 0), ch, font=font)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]

        x = slot_x + (SLOT_WIDTH - char_w) // 2 - bbox[0]
        y = (IMG_HEIGHT - char_h) // 2 - bbox[1]

        draw.text((x, y), ch, fill=0, font=font)

    return img
