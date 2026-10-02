"""
Compose a grimoire / bestiary cover for *The Visibilities*.

The page is one engraved field: a mini-fort of the Heptabrot (z^7+c) or
Octabrot (z^8+c) sits below the title, its (d-1) bastions in red lead, its
ring of d arms reaching out to the margins.  Around it:

  * a seal: two rings drawn around the fort with d marks set on the d arms
    and (d-1) marks set on the bastions, so the clash between the two counts
    is written into the border of the seal as well as the mathematics;
  * a double-ruled frame with a star polygon {d/3} in each corner, with the
    fort's (d-1)-gon inside it;
  * the title in Fell type, as in an early-modern book of marvels.

Run  python -m visibilities.cover --help  for options.  Specimens (named
mini-forts) live in specimens.py.
"""
import argparse
import math
import os
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import (render_fields, find_nucleus, fort_geometry, arm_phase,
                     bastion_phase)
from .grimoire import PALETTES, plate, downsample, to_uint8, _hex
from .specimens import SPECIMENS

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONTS = {
    # Fell types: cut in the 1670s for John Fell's Oxford press; SIL OFL.
    "fell": ("IM+Fell+English", "IMFellEnglish-Regular.ttf"),
    "fell-italic": ("IM+Fell+English:ital@1", "IMFellEnglish-Italic.ttf"),
    "fell-sc": ("IM+Fell+English+SC", "IMFellEnglishSC-Regular.ttf"),
    "fraktur": ("UnifrakturMaguntia", "UnifrakturMaguntia.ttf"),
}
FALLBACK = ["/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]


def font(name, size):
    """Load a Google font (downloaded once into visibilities/fonts/)."""
    family, fname = FONTS[name]
    path = os.path.join(FONT_DIR, fname)
    if not os.path.exists(path):
        try:
            os.makedirs(FONT_DIR, exist_ok=True)
            req = urllib.request.Request(
                f"https://fonts.googleapis.com/css2?family={family}",
                headers={"User-Agent": "Wget/1.0"})  # old UA -> TTF urls
            css = urllib.request.urlopen(req, timeout=20).read().decode()
            url = css.split("src: url(")[1].split(")")[0]
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # offline: fall back to a system serif
            print(f"  (could not fetch {name}: {e}; using a system serif)")
            for fb in FALLBACK:
                if os.path.exists(fb):
                    return ImageFont.truetype(fb, size)
            return ImageFont.load_default()
    return ImageFont.truetype(path, size)


# ---------------------------------------------------------------------------
# Drawing helpers (on a supersampled RGBA overlay)
# ---------------------------------------------------------------------------

def star_points(cx, cy, r, n, step, rot=-math.pi / 2):
    pts = [(cx + r * math.cos(rot + 2 * math.pi * k / n),
            cy + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]
    order, k = [], 0
    for _ in range(n):
        order.append(pts[k])
        k = (k + step) % n
    return order, pts


def draw_star(dr, cx, cy, r, n, step, col, width, rot=-math.pi / 2):
    """Star polygon {n/step}; draws all components when gcd(n,step) > 1."""
    g = math.gcd(n, step)
    for s in range(g):
        order, _ = star_points(cx, cy, r, n // g, step // g,
                               rot + 2 * math.pi * s / n)
        dr.line(order + [order[0]], fill=col, width=width, joint="curve")


def draw_polygon(dr, cx, cy, r, n, col, width, rot=-math.pi / 2, fill=None):
    pts = [(cx + r * math.cos(rot + 2 * math.pi * k / n),
            cy + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]
    dr.polygon(pts, outline=col, fill=fill, width=width)


def spaced_text(dr, xy, text, fnt, fill, spacing=0.0, anchor="mm"):
    """Letter-spaced text centred on xy."""
    widths = [dr.textlength(ch, font=fnt) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = xy[0] - total / 2
    for ch, w in zip(text, widths):
        dr.text((x + w / 2, xy[1]), ch, font=fnt, fill=fill, anchor=anchor)
        x += w + spacing
    return total


def ornament(dr, cx, cy, r, d, ink, red, lw):
    """Corner roundel: circle, {d/3} star, (d-1)-gon fort inside."""
    dr.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ink, width=lw)
    dr.ellipse((cx - r * 0.86, cy - r * 0.86, cx + r * 0.86, cy + r * 0.86),
               outline=ink, width=max(1, lw // 2))
    draw_star(dr, cx, cy, r * 0.86, d, 3 if d > 6 else 2, ink, lw)
    draw_polygon(dr, cx, cy, r * 0.30, d - 1, ink, max(1, lw // 2), fill=red)


# ---------------------------------------------------------------------------
# The cover
# ---------------------------------------------------------------------------

def make_cover(specimen, trim=(6.0, 9.0), bleed=0.125, dpi=300, ss=2,
               palette=None, zoom=None, title="The Visibilities",
               subtitle=None, author=None, seal=True, fort_y=0.56,
               orient=True, turn=0.0, max_iter=None, out="cover.png",
               plate_only=None):
    sp = SPECIMENS[specimen]
    d = sp["d"]
    palette = palette or sp.get("palette", "vellum")
    pal = PALETTES[palette]
    zoom = zoom or sp["zoom"]
    max_iter = max_iter or max(6000, 80 * sp["period"])

    W = int(round((trim[0] + 2 * bleed) * dpi))
    H = int(round((trim[1] + 2 * bleed) * dpi))
    c0 = find_nucleus(d, complex(*sp["c"]), sp["period"])
    size, ang = fort_geometry(d, c0, sp["period"])
    if orient:
        # stand one of the d arms straight up the page
        phi = arm_phase(d, c0, size, ang)
        ang += phi - math.pi / 2
    ang += math.radians(turn + sp.get("turn", 0.0))

    # 'zoom' is the frame half-width in fort sizes; place the fort at fort_y
    half_w = zoom * size
    pix = 2 * half_w / W
    dy = (fort_y - 0.5) * H * pix              # shift fort down the page
    off = complex(-dy * math.sin(ang), dy * math.cos(ang))
    centre = c0 + off
    print(f"{specimen}: d={d} p={sp['period']} size={size:.3e} "
          f"canvas {W}x{H} ss={ss} max_iter={max_iter}")

    fields = render_fields(d, centre.real, centre.imag, half_w, H * ss, W * ss,
                           ang, max_iter, float(d), 1000.0)
    img = plate(fields, pix / ss, palette, line=sp.get("line", 1.1) * ss,
                hatch=sp.get("hatch", 0.35), wash=sp.get("wash", 0.35),
                contour_alpha=sp.get("contour", 0.5))
    img = downsample(img, ss)
    base = Image.fromarray(to_uint8(img))
    if plate_only:
        base.save(plate_only, dpi=(dpi, dpi))

    # --- overlay drawn at 2x for anti-aliasing
    S = 2
    ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    ink = tuple(int(255 * x) for x in _hex(pal["ink"]) * (0.9 if pal.get("light_ink") else 1)) + (255,)
    red = tuple(int(255 * x) for x in _hex(pal["fill"][1])) + (255,)
    gold = tuple(int(255 * x) for x in _hex(pal["rim"])) + (255,)
    paper = tuple(int(255 * x) for x in _hex(pal["paper"][0])) + (255,)
    # on dark grounds the rubric is gilt rather than red
    rubric = gold if pal.get("light_ink") else red
    u = dpi * S * trim[0] / 6.0      # layout unit: an inch on a 6-inch-wide cover
    bl = bleed * dpi * S
    lw = max(2, int(0.012 * u))
    WS, HS = W * S, H * S

    # seal around the fort: the d arms cross a band at d gates (stars); the
    # d-1 bastions point at d-1 gilt studs on the inner rule
    fx, fy = WS / 2, fort_y * HS
    if seal:
        R = min(sp.get("seal", 2.4) * size / pix * S, 0.42 * WS)
        band = 0.05 * R
        dr.ellipse((fx - R - band, fy - R - band, fx + R + band, fy + R + band),
                   outline=paper[:3] + (150,), width=int(2 * band))
        for rr, wdt in [(R - band, lw), (R + band, lw)]:
            dr.ellipse((fx - rr, fy - rr, fx + rr, fy + rr), outline=ink, width=wdt)
        arm0 = arm_phase(d, c0, size, ang)
        bas0 = bastion_phase(d, c0, size, ang)
        for k in range(d):
            t = arm0 + 2 * math.pi * k / d
            x, y = fx + R * math.cos(t), fy - R * math.sin(t)
            draw_star(dr, x, y, band * 0.9, d, 3 if d > 6 else 2, rubric, max(1, lw), rot=-t)
        for k in range(d - 1):
            t = bas0 + 2 * math.pi * k / (d - 1)
            x, y = fx + (R - band * 2.2) * math.cos(t), fy - (R - band * 2.2) * math.sin(t)
            rs = band * 0.38
            dr.ellipse((x - rs, y - rs, x + rs, y + rs), fill=gold, outline=ink, width=max(1, lw // 2))

    # frame: inset from the trim edge, roundels on the corners
    m = int(bl + 0.30 * u)
    m2 = m + int(0.07 * u)
    dr.rectangle((m, m, WS - m, HS - m), outline=ink, width=lw * 2)
    dr.rectangle((m2, m2, WS - m2, HS - m2), outline=ink, width=lw)
    r_orn = int(0.30 * u)
    for (x, y) in [(m, m), (WS - m, m), (m, HS - m), (WS - m, HS - m)]:
        dr.ellipse((x - r_orn, y - r_orn, x + r_orn, y + r_orn), fill=paper)
        ornament(dr, x, y, r_orn, d, ink, red, lw)

    # title cartouche
    ty = int(bl + 1.25 * u)
    cw, ch = int(4.3 * u), int(1.30 * u)
    box = (WS / 2 - cw / 2, ty - ch / 2, WS / 2 + cw / 2, ty + ch / 2)
    _cartouche(ov, box, pal, ink, lw)
    words = title.split(" ", 1)
    if len(words) == 2 and words[0].lower() == "the":
        dr.text((WS / 2, ty - 0.34 * u), words[0], font=font("fell-italic", int(0.30 * u)),
                fill=ink, anchor="mm")
        big = words[1].upper()
        fs = int(0.60 * u)
        f_big = font("fell-sc", fs)
        while dr.textlength(big, font=f_big) + 0.03 * u * len(big) > cw * 0.86:
            fs -= 2
            f_big = font("fell-sc", fs)
        spaced_text(dr, (WS / 2, ty + 0.15 * u), big, f_big, rubric, spacing=0.03 * u)
    else:
        spaced_text(dr, (WS / 2, ty), title.upper(), font("fell-sc", int(0.50 * u)),
                    rubric, spacing=0.03 * u)

    by = HS - int(bl + 0.95 * u)
    if subtitle or author:
        bw, bh = int(3.6 * u), int(0.85 * u if (subtitle and author) else 0.55 * u)
        _cartouche(ov, (WS / 2 - bw / 2, by - bh / 2, WS / 2 + bw / 2, by + bh / 2), pal, ink, lw)
    if subtitle:
        dr.text((WS / 2, by - (0.16 * u if author else 0)), subtitle,
                font=font("fell-italic", int(0.21 * u)), fill=ink, anchor="mm")
    if author:
        spaced_text(dr, (WS / 2, by + (0.18 * u if subtitle else 0)), author.upper(),
                    font("fell-sc", int(0.22 * u)), ink, spacing=0.04 * u)

    ov = ov.resize((W, H), Image.LANCZOS)
    base = base.convert("RGBA")
    base.alpha_composite(ov)
    base = base.convert("RGB")
    base.save(out, dpi=(dpi, dpi))
    print(f"  wrote {out}")
    return out


def _cartouche(ov, box, pal, ink, lw):
    """A vellum panel with clipped (bastioned) corners and a double rule."""
    x0, y0, x1, y1 = box
    k = (y1 - y0) * 0.22
    pts = [(x0 + k, y0), (x1 - k, y0), (x1, y0 + k), (x1, y1 - k),
           (x1 - k, y1), (x0 + k, y1), (x0, y1 - k), (x0, y0 + k)]
    col = tuple(int(255 * x) for x in _hex(pal["paper"][0])) + (235,)
    dr = ImageDraw.Draw(ov)
    dr.polygon(pts, fill=col, outline=ink, width=lw * 2)
    g = lw * 4
    inner = [(x0 + k + g * 0.4, y0 + g), (x1 - k - g * 0.4, y0 + g), (x1 - g, y0 + k + g * 0.4),
             (x1 - g, y1 - k - g * 0.4), (x1 - k - g * 0.4, y1 - g), (x0 + k + g * 0.4, y1 - g),
             (x0 + g, y1 - k - g * 0.4), (x0 + g, y0 + k + g * 0.4)]
    dr.polygon(inner, outline=ink, width=lw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specimen", choices=sorted(SPECIMENS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--palette", choices=sorted(PALETTES))
    ap.add_argument("--zoom", type=float, help="frame half-width in fort sizes")
    ap.add_argument("--trim", default="6x9", help="trim size in inches, WxH")
    ap.add_argument("--bleed", type=float, default=0.125)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--title", default="The Visibilities")
    ap.add_argument("--subtitle")
    ap.add_argument("--author")
    ap.add_argument("--no-seal", action="store_true")
    ap.add_argument("--fort-y", type=float, default=0.56)
    ap.add_argument("--turn", type=float, default=0.0, help="extra rotation, degrees")
    ap.add_argument("--plate", help="also save the bare plate (no type) here")
    a = ap.parse_args()
    tw, th = (float(x) for x in a.trim.lower().split("x"))
    make_cover(a.specimen, trim=(tw, th), bleed=a.bleed, dpi=a.dpi, ss=a.ss,
               palette=a.palette, zoom=a.zoom, title=a.title, subtitle=a.subtitle,
               author=a.author, seal=not a.no_seal, fort_y=a.fort_y, turn=a.turn,
               out=a.out or f"cover_{a.specimen}.png", plate_only=a.plate)


if __name__ == "__main__":
    main()
