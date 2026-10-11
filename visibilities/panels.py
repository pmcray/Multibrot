"""
Front and back board panels in the luminous style: a big central fort,
black, wrapped in nested rings of baroque outworks.

  front   Heptabrot (z^7 + c): hexagonal keep, 7-fold outworks, amaranth
  back    Octabrot  (z^8 + c): heptagonal keep, 8-fold outworks, trifle

The forts are deep (around 1e-24); they were built by iterated Julia
morphing (deep.morph_path) and are stored to full precision in
data/deep_specimens.json.

    python -m visibilities.panels front --dpi 100 --ar --out front.png
    python -m visibilities.panels back  --dpi 300 --out back.png

Sizes default to the dust-wrapper panel (164 x 241 mm) with the fort on the
board's centre: 3.5 mm towards the fore-edge from the panel's centre,
where the AR device is blind-blocked.  --ar overlays the device for
proofs only.
"""
import argparse
import json
import math
import os
import subprocess
import tempfile
import time

import numpy as np
import mpmath as mp
from PIL import Image, ImageChops, ImageFilter

from .deep import DeepFort, window_offsets, render_deep
from .luminous import colour

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
MM = 1 / 25.4

PANELS = {
    "front": dict(specimen="hepta-citadel-deep", palette="amaranth", cycles=5.0, phase=0.0),
    "back": dict(specimen="octa-star-deep", palette="trifle", cycles=4.0, phase=0.15),
}


def load_specimen(name):
    with open(os.path.join(DATA, "deep_specimens.json")) as f:
        sp = json.load(f)[name]
    mp.mp.dps = len(sp["re"]) + 5
    c = mp.mpc(mp.mpf(sp["re"]), mp.mpf(sp["im"]))
    return DeepFort(sp["d"], c, sp["period"], sp["size"])


def upright(fort, turn_deg=0.0):
    """Frame angle that stands one of the fort's bastions straight up."""
    d = fort.d
    sector = 2 * math.pi / (d - 1)
    # in the fort's canonical frame the bastions sit at sector/2 + k*sector
    k = round((math.pi / 2 - sector / 2) / sector)
    lobe = sector / 2 + k * sector
    return fort.angle + (lobe - math.pi / 2) + math.radians(turn_deg)


def render_panel(fort, width_mm=164.0, height_mm=241.0, fort_dx_mm=3.5,
                 fort_dy_mm=0.0, radius_mm=26.0, dpi=150, ss=2, turn_deg=0.0,
                 palette="amaranth", cycles=5.0, phase=0.0, strip_rows=384,
                 verbose=True):
    """
    The panel as RGB uint8.  radius_mm is the fort's radius on the page
    (its size); fort_dx_mm / fort_dy_mm place the fort relative to the
    panel's centre (+x towards the fore-edge of the front board).
    """
    W, H = int(round(width_mm * MM * dpi)), int(round(height_mm * MM * dpi))
    WS, HS = W * ss, H * ss
    px_mm = 25.4 / (dpi * ss)
    pix_c = fort.size / radius_mm * px_mm            # c per supersampled pixel
    ang = upright(fort, turn_deg)
    rot = complex(np.exp(1j * ang))
    # pixel (i, j) -> offset from the nucleus in c
    xs = ((np.arange(WS) + 0.5) - WS / 2) * px_mm - fort_dx_mm
    it = max(20000, 60 * fort.p)
    mu = np.empty((HS, WS), np.float32)
    de = np.empty((HS, WS), np.float32)
    t0 = time.time()
    for r0 in range(0, HS, strip_rows):
        r1 = min(HS, r0 + strip_rows)
        ys = (HS / 2 - (np.arange(r0, r1) + 0.5)) * px_mm - fort_dy_mm
        w = (xs[None, :] + 1j * ys[:, None]) * (pix_c / px_mm) * rot
        mu[r0:r1], de[r0:r1] = render_deep(fort.d, fort.ref, fort.coef,
                                           np.ascontiguousarray(w.real),
                                           np.ascontiguousarray(w.imag), it, 1e3)
        if verbose:
            print(f"\r  {r1}/{HS} rows  {time.time() - t0:.0f}s", end="", flush=True)
    if verbose:
        print()
    img = colour(mu, de / pix_c / ss * 1.0, palette, cycles=cycles, phase=phase)
    # colour() wants the distance in output pixels: de/pix_c is in
    # supersampled pixels, so divide by ss
    if ss > 1:
        img = img.reshape(H, ss, W, ss, 3).mean(axis=(1, 3))
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def ar_device(px):
    """The AR device rasterised to a px-wide alpha mask (black = ink)."""
    pdf = os.path.join(DATA, "atomic_razor.pdf")
    with tempfile.TemporaryDirectory() as tmp:
        dpi = int(px / (216 / 72)) + 1           # device is ~216 pt wide
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", pdf,
                        os.path.join(tmp, "ar")], check=True)
        im = Image.open(os.path.join(tmp, "ar.png")).convert("L")
    im = im.resize((px, int(round(px * im.height / im.width))), Image.LANCZOS)
    return ImageChops.invert(im)                 # alpha: 255 where inked


def overlay_ar(arr, dpi, size_mm, fort_dx_mm=3.5, fort_dy_mm=0.0):
    """Proof only: show the blind-blocked device as a faint deboss."""
    im = Image.fromarray(arr).convert("RGB")
    a = ar_device(int(round(size_mm * MM * dpi)))
    cx = im.width / 2 + fort_dx_mm * MM * dpi
    cy = im.height / 2 - fort_dy_mm * MM * dpi
    x0, y0 = int(cx - a.width / 2), int(cy - a.height / 2)
    off = max(1, int(0.35 * MM * dpi))
    hi = Image.new("RGB", a.size, (90, 70, 100))
    lo = Image.new("RGB", a.size, (0, 0, 0))
    soft = a.filter(ImageFilter.GaussianBlur(off / 2)).point(lambda v: v * 0.55)
    im.paste(hi, (x0 + off, y0 + off), soft)
    im.paste(lo, (x0, y0), a.point(lambda v: v * 0.7))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("side", choices=sorted(PANELS))
    ap.add_argument("--specimen")
    ap.add_argument("--palette")
    ap.add_argument("--cycles", type=float)
    ap.add_argument("--phase", type=float)
    ap.add_argument("--radius", type=float, default=26.0, help="fort radius on the page, mm")
    ap.add_argument("--turn", type=float, default=0.0, help="extra rotation, degrees")
    ap.add_argument("--width", type=float, default=164.0)
    ap.add_argument("--height", type=float, default=241.0)
    ap.add_argument("--fort-dx", type=float, default=None,
                    help="fort offset from panel centre, mm (default: board centre)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--ar", type=float, nargs="?", const=40.0, default=None,
                    help="overlay the AR device (proofs), width in mm (default 40)")
    ap.add_argument("--out")
    a = ap.parse_args()
    P = dict(PANELS[a.side])
    for k in ("specimen", "palette", "cycles", "phase"):
        if getattr(a, k) is not None:
            P[k] = getattr(a, k)
    # board centre: 3.5 mm towards the fore-edge (right on the front, left on the back)
    dx = a.fort_dx if a.fort_dx is not None else (3.5 if a.side == "front" else -3.5)
    fort = load_specimen(P["specimen"])
    print(f"  {a.side}: {fort}")
    arr = render_panel(fort, a.width, a.height, dx, 0.0, a.radius, a.dpi, a.ss,
                       a.turn, P["palette"], P["cycles"], P["phase"])
    if a.ar:
        arr = overlay_ar(arr, a.dpi, a.ar, dx)
    out = a.out or f"panel_{a.side}.png"
    Image.fromarray(arr).save(out, dpi=(a.dpi, a.dpi))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
