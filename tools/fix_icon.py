"""
VoiceTTSr Icon Transparency Fixer
Converts icon.ico and icon.png to clean, 32-bit RGBA icons with true alpha transparency in the corners.
Includes standard Windows multi-resolution icon sizes: 256x256, 128x128, 64x64, 48x48, 32x32, 16x16.
"""

import os
from PIL import Image, ImageDraw, ImageFilter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON_ICO_PATH = os.path.join(ROOT_DIR, "icon.ico")
ICON_PNG_PATH = os.path.join(ROOT_DIR, "icon.png")


def make_transparent_icon():
    print("[ICON] Loading base icon...")
    img = Image.open(ICON_ICO_PATH).convert("RGBA")
    w, h = img.size

    # Floodfill the 4 corners to convert white background to pure transparent (0, 0, 0, 0)
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        ImageDraw.floodfill(img, (x, y), (0, 0, 0, 0), thresh=60)

    # Clean any slight halo around the corners
    # Extract alpha channel
    r, g, b, a = img.split()

    # Smooth the alpha mask edge slightly for anti-aliasing
    a = a.filter(ImageFilter.SMOOTH)
    img.putalpha(a)

    # Save clean 32-bit RGBA PNG
    img.save(ICON_PNG_PATH, format="PNG")
    print(f"[SUCCESS] Saved transparent PNG icon: {ICON_PNG_PATH}")

    # Save multi-resolution Windows ICO (256, 128, 64, 48, 32, 16) with 32-bit alpha
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    img.save(ICON_ICO_PATH, format="ICO", sizes=sizes)
    print(f"[SUCCESS] Saved multi-resolution transparent ICO icon: {ICON_ICO_PATH}")


if __name__ == "__main__":
    make_transparent_icon()
