"""
A seamless wrap-around jacket: Octabrot on the back board, Heptabrot on the
front, the same fort on both, joined across the spine.

Layout (jacket laid flat, outside up):   back board | spine | front board

Two things vary across the jacket.

1. The family parameter t (bridge.py): t = 1 (z^8 + c) on the back board,
   t = 0 (z^7 + c) on the front, changing smoothly across the spine.

2. Where each point of the jacket looks in the parameter plane.  The front
   and back show the same fort (its nucleus tracked through t), so both
   fort centres map to fort-relative position w = 0.  A smooth map that
   sends two points to the same place, keeping shapes undistorted at both,
   must have either a pole or a branch point between them (Riemann-Hurwitz).

   mode="infinity"  w = (D/pi) tan(pi (zeta - zeta_back) / D)
                    The spine is the point at infinity.  Out from each fort
                    the view widens until, at the spine, it takes in the
                    whole set and the calm field beyond it: a still eye where
                    the lettering and the publisher's device can sit.
   mode="branch"    w = (zeta - zeta_back)(zeta - zeta_front) / D
                    The spine is a branch point.  The spine centre is a
                    point near the fort, with angles there doubled; the back
                    is the front turned through 180 degrees and re-made in 8.

   Both of these magnify strongly at the spine, exactly where t changes,
   and the change of t then drags the picture into streaks.  The default
   avoids that:

   mode="pair"      One flat plate, no singular point.  The front shows fort
                    A (Heptabrot); the back shows fort B, a heptagonal
                    Octabrot fort in whose arms A sits once it has been
                    carried to t = 1.  The map
                        w = zoom (1 - exp(-lam (zeta - zeta_front))) / lam
                    is conformal everywhere (shapes undistorted), with a
                    gentle steady zoom across the jacket; complex lam lets
                    B land on the back board's centre with an arm upright.

Run  python -m visibilities.jacket --help.
"""
import argparse
import math
import os
import time

import numpy as np
from PIL import Image

from .bridge import TrackedFort, render_family, family_phase, dz_dc, stabilise
from .engine import find_nucleus, fort_geometry, fold_phase
from .grimoire import plate, downsample, to_uint8, vellum, PALETTES

HERE = os.path.dirname(__file__)


def smoothstep(e0, e1, x):
    s = np.clip((x - e0) / (e1 - e0), 0, 1)
    return s * s * (3 - 2 * s)


def solve_pair(fort, B, pB, Dl, zoom, rB_in, grid=241):
    """
    Choose lam (complex, per inch) and the t = 1 frame F1 so that fort B,
    of period pB, sits at the back board's centre with radius rB_in inches
    and one of its 8 arms pointing up.  Returns (lam, F1).
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


def jacket(fort, board=(6.25, 9.5), spine=1.25, wrap=0.75, dpi=100, ss=2,
           zoom=1.2, mode="pair", blend=1.2, fort_y=0.5, palette="vellum",
           max_iter=6000, B=None, pB=7, rB_in=1.6, verbose=True):
    """
    Render the whole case cover.  Sizes in inches.  'wrap' is the extra
    material on every side that turns in over the board edges.  'zoom' is
    fort sizes per inch at the front fort.  'blend' is how far either side
    of the spine the 7 -> 8 change extends.  For mode="pair", B is the back
    fort's nucleus (period pB) and rB_in its radius on the page.
    """
    bw, bh = board
    Wi, Hi = 2 * bw + spine + 2 * wrap, bh + 2 * wrap
    W, H = int(round(Wi * dpi)), int(round(Hi * dpi))
    WS, HS = W * ss, H * ss
    # jacket coordinates (inches), origin at the back fore-edge, y up
    x = (np.arange(WS) + 0.5) / (dpi * ss) - wrap
    y = (HS / 2 - np.arange(HS) - 0.5) / (dpi * ss) + (0.5 - fort_y) * bh
    xb, xf = bw / 2, bw + spine + bw / 2
    xm = bw + spine / 2
    Dl = xf - xb

    # t: 1 on the back, 0 on the front; spaced so the picture changes at an
    # even rate (the eighth arm is born quickly, near t ~ 0.55)
    s_col = smoothstep(xm - spine / 2 - blend, xm + spine / 2 + blend, x)
    t_col = fort.t_of(1.0 - s_col)

    lam = None
    if mode == "pair":
        lam, F1 = solve_pair(fort, B, pB, Dl, zoom, rB_in)
        fort.set_end_frame(F1)
        if verbose:
            print(f"  pair: lam={lam:.4f}/in  zoom ratio across jacket "
                  f"{abs(np.exp(lam * Dl)):.2f}  twist {math.degrees(lam.imag * Dl):.1f} deg")

    # per-column fort centre and frame
    tu, inv = np.unique(np.round(t_col, 6), return_inverse=True)
    cen = np.array([fort.centre(t) for t in tu])[inv]
    frm = np.array([fort.frame(t) for t in tu])[inv]

    zeta = x[None, :] + 1j * y[:, None]
    if mode == "pair":
        e = np.exp(-lam * (zeta - xf))
        w = zoom * (1 - e) / lam
        dw = zoom * e
    elif mode == "infinity":
        w = zoom * (Dl / math.pi) * np.tan(math.pi * (zeta - xb) / Dl)
        dw = zoom / np.cos(math.pi * (zeta - xb) / Dl) ** 2
    elif mode == "branch":
        w = zoom * (zeta - xb) * (zeta - xf) / Dl
        dw = zoom * (2 * zeta - xb - xf) / Dl
    else:
        raise ValueError(mode)
    C = cen[None, :] + frm[None, :] * w
    T = np.broadcast_to(t_col[None, :], C.shape)

    t0 = time.time()
    mu, de, s7, s8, tr = render_family(np.ascontiguousarray(C.real),
                                       np.ascontiguousarray(C.imag),
                                       np.ascontiguousarray(T, dtype=np.float64),
                                       max_iter, 1000.0)
    st = (1 - T) * s7 + T * s8
    # size of a pixel in c varies over the page: |dc/dzeta| / (dpi*ss)
    pix = np.abs(frm[None, :] * dw) / (dpi * ss)
    img = plate((mu, de / pix, st, tr, None), 1.0, palette, line=1.1 * ss)
    img = downsample(img, ss)
    if verbose:
        print(f"  jacket {W}x{H} px ({Wi:.2f}x{Hi:.2f} in) mode={mode} "
              f"in {time.time() - t0:.0f}s")
    return to_uint8(img), dict(W=W, H=H, dpi=dpi, wrap=wrap, board=board,
                               spine=spine)


def guides(arr, info, color=(40, 120, 200)):
    """Draw trim/spine/fold guides (for proofs only)."""
    from PIL import ImageDraw
    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)
    d, wr = info["dpi"], info["wrap"]
    bw, bh = info["board"]
    sp = info["spine"]
    xs = [wr, wr + bw, wr + bw + sp, wr + 2 * bw + sp]
    for xi in xs:
        X = int(xi * d)
        dr.line((X, 0, X, info["H"]), fill=color, width=1)
    for yi in (wr, wr + bh):
        Y = int(yi * d)
        dr.line((0, Y, info["W"], Y), fill=color, width=1)
    return np.asarray(im)


# Front: the Heptabrot "snowflake" fort (period 10).  Followed to t = 1 it
# lands in the arms of a large period-7 Octabrot fort, which takes the back.
SNOWFLAKE = dict(c=0.948646674202614 + 0.589594547258562j, period=10)
OCTA_KEEP = dict(c=0.958143746042516 + 0.525108003889462j, period=7)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="pair", choices=["pair", "infinity", "branch"])
    ap.add_argument("--back-radius", type=float, default=1.6,
                    help="radius of the back fort on the page, inches (pair mode)")
    ap.add_argument("--board", default="6.25x9.5")
    ap.add_argument("--spine", type=float, default=1.25)
    ap.add_argument("--wrap", type=float, default=0.75)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--zoom", type=float, default=1.2,
                    help="fort sizes per inch at the front fort")
    ap.add_argument("--blend", type=float, default=1.2)
    ap.add_argument("--fort-y", type=float, default=0.5)
    ap.add_argument("--palette", default="vellum", choices=sorted(PALETTES))
    ap.add_argument("--guides", action="store_true")
    ap.add_argument("--out", default="jacket.png")
    a = ap.parse_args()
    bw, bh = (float(v) for v in a.board.split("x"))
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    fort = TrackedFort(SNOWFLAKE["c"], SNOWFLAKE["period"],
                       cache=os.path.join(HERE, "cache", "snowflake_p10.npz"))
    stab = os.path.join(HERE, "cache", "snowflake_p10_stab.npz")
    if not os.path.exists(stab):
        print("  measuring how the fort morphs (one-off, ~2 min)...")
        np.savez(stab, **dict(zip(("ts", "corr", "resid"), stabilise(fort, 161))))
    st = np.load(stab)
    fort.use_stabilisation(st["ts"], st["corr"], st["resid"])
    B = find_nucleus(8, OCTA_KEEP["c"], OCTA_KEEP["period"])
    arr, info = jacket(fort, (bw, bh), a.spine, a.wrap, a.dpi, a.ss, a.zoom,
                       a.mode, a.blend, a.fort_y, a.palette, B=B,
                       pB=OCTA_KEEP["period"], rB_in=a.back_radius)
    if a.guides:
        arr = guides(arr, info)
    Image.fromarray(arr).save(a.out, dpi=(a.dpi, a.dpi))
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
