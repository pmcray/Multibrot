"""
The dust-wrapper for The Visibilities: Octabrot on the back, Heptabrot on
the front, joined seamlessly across the spine and carried on round both
flaps.

Wrapper laid flat, outside up, left to right (all sizes in mm):

  back flap | turn | back panel | spine | front panel | turn | front flap

The defaults (WrapperSpec) are for the 784-page Royal Octavo hardback:
156 x 234 mm pages; 70 gsm, volume 1.5 paper (a 41.6 mm text block);
2.5 mm boards with 3 mm squares; a 48 mm spine; 150 mm double-width flaps.
Have the binder confirm the spine width from a bulking dummy, then pass
--spine.

Two things vary across the wrapper.

1. The family parameter t (bridge.py): t = 1 (z^8 + c) at the back fort,
   t = 0 (z^7 + c) at the front fort, changing smoothly between them and
   held constant beyond each fort (so the flaps are pure Octabrot and pure
   Heptabrot).

2. Where each point looks in the parameter plane.  The default, mode
   "pair", is one flat plate with no singular point.  The front shows fort
   A (Heptabrot, hexagonal, 7 arms); the back shows fort B, a heptagonal
   Octabrot fort (8 arms) in whose arms A sits once carried to t = 1.  The
   map

        w = zoom (1 - exp(-lam (zeta - zeta_front))) / lam

   is conformal everywhere (shapes undistorted), with a gentle steady zoom
   and twist across the wrapper; complex lam puts B at the centre of the
   back board with an arm upright.

   If both boards showed the same fort, a smooth map would need a pole or
   a branch point between them (Riemann-Hurwitz).  Those layouts are kept
   for comparison as modes "infinity" (the spine is the point at infinity)
   and "branch" (the spine is a branch point); both streak at the spine.

Each fort is centred on its board, not its wrapper panel.  The board starts
a joint's width from the spine fold, so its centre lies a few mm towards the
fore-edge.  The front fort therefore sits over the AR device blind-blocked
into the board.

Run  python -m visibilities.jacket --help.
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageDraw

from .bridge import TrackedFort, render_family, stabilise
from .engine import find_nucleus, fort_geometry, fold_phase
from .grimoire import plate, downsample, to_uint8, vellum, fbm, PALETTES

HERE = os.path.dirname(__file__)
MM = 1 / 25.4                        # inches per mm (the maps work in inches)


@dataclass
class WrapperSpec:
    """Dust-wrapper geometry in mm."""
    height: float = 241.0       # board height 240 (234 + 2 x 3 squares) + 1
    panel: float = 164.0        # board width 157 + joint ~7
    spine: float = 48.0         # case spine 46.6 (41.6 block + 2 x 2.5 board) + ease
    flap: float = 150.0         # double width
    turn: float = 4.0           # wrap round the board's fore-edge
    bleed: float = 5.0
    joint: float = 7.0          # spine fold to the board's spine edge
    board: float = 157.0        # board width (its centre is where the fort goes)

    @property
    def trim_width(self):
        return 2 * (self.flap + self.turn + self.panel) + self.spine

    def folds(self):
        """x positions (mm from the left trim edge) of the four folds."""
        a = self.flap
        b = a + self.turn
        c = b + self.panel
        d = c + self.spine
        e = d + self.panel
        f = e + self.turn
        return dict(back_flap=a, back_fore_edge=b, back_spine=c,
                    front_spine=d, front_fore_edge=e, front_flap=f)

    def fort_centres(self):
        """x of the back and front fort centres: the centres of the boards."""
        f = self.folds()
        back = f["back_spine"] - self.joint - self.board / 2
        front = f["front_spine"] + self.joint + self.board / 2
        return back, front


def smoothstep(e0, e1, x):
    s = np.clip((x - e0) / (e1 - e0), 0, 1)
    return s * s * (3 - 2 * s)


def solve_pair(fort, B, pB, Dl, zoom, rB_in, grid=241):
    """
    Choose lam (complex, per inch) and the t = 1 frame F1 so that fort B,
    of period pB, sits Dl inches left of the front fort with radius rB_in
    inches and one of its 8 arms pointing up.  Returns (lam, F1).
    """
    A1 = fort.centre(1.0)
    sB, _ = fort_geometry(8, B, pB)
    psiB = fold_phase(8, B, sB, 0.0, 8, (2.0, 3.5))      # arm direction in c
    gap = B - A1
    best = None
    for lr in np.linspace(1e-4, 0.6, grid):
        for li in np.linspace(-0.4, 0.4, grid):
            lam = complex(lr, li)
            e = np.exp(lam * Dl)
            r = sB * abs(1 - e) / (abs(gap) * abs(lam) * abs(e))
            F1 = gap * lam / (zoom * (1 - e))
            rot = psiB - math.pi / 2 - np.angle(F1 * e)
            sector = math.pi / 4
            rot = (rot + sector / 2) % sector - sector / 2
            err = (math.log(r / rB_in)) ** 2 + 4 * rot ** 2 + 0.01 * li ** 2
            if best is None or err < best[0]:
                best = (err, lam, F1)
    return best[1], best[2]


def wrapper(fort, spec=WrapperSpec(), dpi=150, ss=2, front_radius=21.0,
            back_radius=40.0, mode="pair", fort_y=0.5, palette="vellum",
            max_iter=6000, B=None, pB=7, strip_rows=512, verbose=True):
    """
    Render the whole dust-wrapper, bleed included, as an RGB array.

    front_radius / back_radius are the forts' radii on the wrapper in mm.
    fort_y is the forts' height as a fraction of the trimmed height from the
    top.  The picture is computed in horizontal strips of strip_rows
    (supersampled) rows to bound memory; paper, texture and normalisation
    are shared so the strips join invisibly.
    """
    Wmm = spec.trim_width + 2 * spec.bleed
    Hmm = spec.height + 2 * spec.bleed
    W, H = int(round(Wmm * MM * dpi)), int(round(Hmm * MM * dpi))
    WS, HS = W * ss, H * ss
    px_mm = 25.4 / (dpi * ss)

    # wrapper coordinates in inches: x from the left trim edge, y up from
    # the forts' line
    x = ((np.arange(WS) + 0.5) * px_mm - spec.bleed) * MM
    y0 = ((fort_y * spec.height) + spec.bleed) * MM
    y = y0 - (np.arange(HS) + 0.5) * px_mm * MM
    xb_mm, xf_mm = spec.fort_centres()
    xb, xf = xb_mm * MM, xf_mm * MM
    Dl = xf - xb
    zoom = 1.0 / (front_radius * MM)            # fort sizes per inch, front

    # t: 1 at the back fort, 0 at the front fort, constant beyond each; the
    # spacing follows the measured rate of change of the picture
    t_col = fort.t_of(1.0 - smoothstep(xb, xf, x))

    lam = None
    if mode == "pair":
        lam, F1 = solve_pair(fort, B, pB, Dl, zoom, back_radius * MM)
        fort.set_end_frame(F1)
        if verbose:
            print(f"  pair: lam={lam:.4f}/in  zoom ratio across wrapper "
                  f"{abs(np.exp(lam * Dl)):.2f}  twist {math.degrees(lam.imag * Dl):.1f} deg")

    tu, inv = np.unique(np.round(t_col, 6), return_inverse=True)
    cen = np.array([fort.centre(t) for t in tu])[inv]
    frm = np.array([fort.frame(t) for t in tu])[inv]

    def strip_fields(r0, r1):
        zeta = x[None, :] + 1j * y[r0:r1, None]
        if mode == "pair":
            e = np.exp(-lam * (zeta - xf))
            w, dw = zoom * (1 - e) / lam, zoom * e
        elif mode == "infinity":
            w = zoom * (Dl / math.pi) * np.tan(math.pi * (zeta - xb) / Dl)
            dw = zoom / np.cos(math.pi * (zeta - xb) / Dl) ** 2
        elif mode == "branch":
            w = zoom * (zeta - xb) * (zeta - xf) / Dl
            dw = zoom * (2 * zeta - xb - xf) / Dl
        else:
            raise ValueError(mode)
        C = cen[None, :] + frm[None, :] * w
        T = np.ascontiguousarray(np.broadcast_to(t_col[None, :], C.shape), dtype=np.float64)
        mu, de, s7, s8, tr = render_family(np.ascontiguousarray(C.real),
                                           np.ascontiguousarray(C.imag),
                                           T, max_iter, 1000.0)
        pix = (np.abs(frm[None, :] * dw) / (dpi * ss)).astype(np.float32)
        st = ((1 - T) * s7 + T * s8).astype(np.float32)
        return mu, (de / pix).astype(np.float32), st, tr

    # pass 1: the escape-time fields, strip by strip
    t0 = time.time()
    strip_rows -= strip_rows % ss
    mu = np.empty((HS, WS), np.float32)
    dpx = np.empty_like(mu)
    st = np.empty_like(mu)
    tr = np.empty_like(mu)
    for r0 in range(0, HS, strip_rows):
        r1 = min(HS, r0 + strip_rows)
        mu[r0:r1], dpx[r0:r1], st[r0:r1], tr[r0:r1] = strip_fields(r0, r1)
        if verbose:
            print(f"\r  fields {r1}/{HS} rows", end="", flush=True)
    if verbose:
        print(f"  ({time.time() - t0:.0f}s)")

    # pass 2: colour each strip with shared paper, texture and normalisation
    pal = PALETTES[palette]
    paper = vellum(H, W, pal, 7)                        # at output resolution
    tex = fbm(H, W, max(H, W) / 80, 4, 16)
    inside = mu < 0
    trap_scale = float(np.percentile(tr[inside][::97], 95)) if inside.any() else 1.0
    out = np.empty((H, W, 3), np.float32)
    pad = 2 * ss                                        # for contour gradients
    for r0 in range(0, HS, strip_rows):
        r1 = min(HS, r0 + strip_rows)
        a0, a1 = max(0, r0 - pad), min(HS, r1 + pad)
        rows = slice(a0 // ss, (a1 + ss - 1) // ss)
        up = lambda arr: np.repeat(np.repeat(arr[rows], ss, 0), ss, 1)[: a1 - a0, :WS]
        img = plate((mu[a0:a1], dpx[a0:a1], st[a0:a1], tr[a0:a1], None), 1.0, palette,
                    line=1.1 * ss, paper=up(paper), trap_scale=trap_scale,
                    fill_tex=up(tex))
        img = img[r0 - a0: r0 - a0 + (r1 - r0)]
        out[r0 // ss: r1 // ss] = downsample(img, ss)
    if verbose:
        print(f"  wrapper {W}x{H} px ({Wmm:.0f}x{Hmm:.0f} mm incl. bleed, "
              f"{dpi} dpi) mode={mode} in {time.time() - t0:.0f}s")
    info = dict(W=W, H=H, dpi=dpi, spec=spec, fort_y=fort_y, lam=lam,
                fort_centres_mm=(xb_mm, xf_mm))
    return to_uint8(out), info


def guides(arr, info, color=(30, 110, 200)):
    """Proof marks: trim and bleed (solid), folds (dashed), board centres."""
    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)
    spec, dpi = info["spec"], info["dpi"]
    X = lambda mm: int(round((mm + spec.bleed) * MM * dpi))
    Y = lambda mm: int(round((mm + spec.bleed) * MM * dpi))
    tw, th = spec.trim_width, spec.height
    dr.rectangle((X(0), Y(0), X(tw), Y(th)), outline=color, width=1)
    for name, xm in spec.folds().items():
        for y0 in range(0, info["H"], 12):
            dr.line((X(xm), y0, X(xm), y0 + 6), fill=color, width=1)
    fy = info["fort_y"] * th
    for xm in info["fort_centres_mm"]:
        dr.line((X(xm) - 10, Y(fy), X(xm) + 10, Y(fy)), fill=color)
        dr.line((X(xm), Y(fy) - 10, X(xm), Y(fy) + 10), fill=color)
    return np.asarray(im)


# Front: the Heptabrot "snowflake" fort (period 10).  Followed to t = 1 it
# lands in the arms of a large period-7 Octabrot fort, which takes the back.
SNOWFLAKE = dict(c=0.948646674202614 + 0.589594547258562j, period=10)
OCTA_KEEP = dict(c=0.958143746042516 + 0.525108003889462j, period=7)


def load_fort():
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    fort = TrackedFort(SNOWFLAKE["c"], SNOWFLAKE["period"],
                       cache=os.path.join(HERE, "cache", "snowflake_p10.npz"))
    stab = os.path.join(HERE, "cache", "snowflake_p10_stab.npz")
    if not os.path.exists(stab):
        print("  measuring how the fort morphs (one-off, ~2 min)...")
        np.savez(stab, **dict(zip(("ts", "corr", "resid"), stabilise(fort, 161))))
    st = np.load(stab)
    fort.use_stabilisation(st["ts"], st["corr"], st["resid"])
    return fort


def main():
    d = WrapperSpec()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="pair", choices=["pair", "infinity", "branch"])
    for k in ("height", "panel", "spine", "flap", "turn", "bleed", "joint", "board"):
        ap.add_argument(f"--{k}", type=float, default=getattr(d, k), help=f"mm (default {getattr(d, k)})")
    ap.add_argument("--front-radius", type=float, default=21.0, help="front fort radius, mm")
    ap.add_argument("--back-radius", type=float, default=40.0, help="back fort radius, mm")
    ap.add_argument("--fort-y", type=float, default=0.5, help="fort height, fraction from top")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--palette", default="vellum", choices=sorted(PALETTES))
    ap.add_argument("--guides", action="store_true", help="draw trim/fold marks (proofs only)")
    ap.add_argument("--out", default="wrapper.png")
    a = ap.parse_args()
    spec = WrapperSpec(**{k: getattr(a, k) for k in asdict(d)})

    folds = spec.folds()
    print(f"  wrapper {spec.trim_width:.0f} x {spec.height:.0f} mm trimmed; "
          f"folds at " + ", ".join(f"{v:.0f}" for v in folds.values()) + " mm")
    fort = load_fort()
    B = find_nucleus(8, OCTA_KEEP["c"], OCTA_KEEP["period"])
    arr, info = wrapper(fort, spec, a.dpi, a.ss, a.front_radius, a.back_radius,
                        a.mode, a.fort_y, a.palette, B=B, pB=OCTA_KEEP["period"])
    if a.guides:
        arr = guides(arr, info)
    Image.fromarray(arr).save(a.out, dpi=(a.dpi, a.dpi))
    # a sidecar with the geometry, for the printer and for the OpTeX side
    side = dict(spec=asdict(spec), trim_width_mm=spec.trim_width,
                with_bleed_mm=[spec.trim_width + 2 * spec.bleed, spec.height + 2 * spec.bleed],
                folds_mm=folds, fort_centres_mm=info["fort_centres_mm"],
                fort_height_mm_from_top=a.fort_y * spec.height, dpi=a.dpi)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(side, f, indent=2)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
