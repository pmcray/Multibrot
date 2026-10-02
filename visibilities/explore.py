"""
Hunt for mini-forts whose surroundings show strong d-fold structure.

Method ("Julia morphing"):
  1. take a parent fort (a low-period island);
  2. pick points A on the filaments of its arms;
  3. look for the lowest-period nucleus B within a small radius w of A --
     a much smaller fort hidden in that filament;
  4. render B at several magnifications.  Near B the critical point's
     local degree d shows: the features around A are repeated d times in a
     ring about the (d-1)-cusped fort, and again d-fold further out.

Usage
  python -m visibilities.explore parents 7
  python -m visibilities.explore hunt 7 --parent 3,0.942364,0.586911 \
        --seeds 12 --depth 3e-3 --sheet hunt7.png
"""
import argparse
import math
import numpy as np
from PIL import Image, ImageDraw

from .engine import (render_fields, search_forts, find_nucleus, find_period,
                     fort_geometry)
from .grimoire import plate, to_uint8


def preview(d, c, half_width, angle=0.0, size=320, max_iter=4000, palette="vellum"):
    f = render_fields(d, c.real, c.imag, half_width, size, size, angle,
                      max_iter, float(d), 1000.0)
    return to_uint8(plate(f, 2 * half_width / size, palette, line=0.9,
                          hatch=0.25, contour_alpha=0.4))


def parents(d, max_period=5, count=12):
    """Low-period islands in one symmetry sector of the set."""
    fs = search_forts(d, 0j, 1.5, grid=500, max_period=max_period)
    sec = 2 * math.pi / (d - 1)
    out = [f for f in fs if f["island"]
           and -1e-9 <= math.atan2(f["c"].imag, f["c"].real) < sec - 1e-9]
    return out[:count]


def hunt(d, parent_period, parent_c, seeds=8, depth=3e-3, ring=(1.2, 2.8),
         seed=0, grid=600):
    """Return deep forts found in the arms of a parent island."""
    c0 = find_nucleus(d, parent_c, parent_period)
    s, _ = fort_geometry(d, c0, parent_period)
    hw = 3 * s
    mu, de, *_ = render_fields(d, c0.real, c0.imag, hw, grid, grid, 0.0, 4000,
                               float(d), 1000.0)
    pix = 2 * hw / grid
    jj, ii = np.where((mu > 0) & (de < 0.5 * pix))
    u = (ii + 0.5 - grid / 2) * pix
    v = (grid / 2 - jj - 0.5) * pix
    r = np.hypot(u, v)
    sel = np.where((r > ring[0] * s) & (r < ring[1] * s))[0]
    rng = np.random.default_rng(seed)
    found = []
    for k in rng.choice(sel, min(len(sel), seeds * 4), replace=False):
        A = c0 + complex(u[k], v[k])
        w = s * depth
        p = find_period(d, A, w, max_period=5000)
        if not p:
            continue
        B = find_nucleus(d, A, p, steps=200)
        if not np.isfinite(B.real) or abs(B - A) > 5 * w:
            continue
        sb, ab = fort_geometry(d, B, p)
        found.append(dict(period=p, c=B, size=sb, angle=ab))
        if len(found) >= seeds:
            break
    return found


def contact_sheet(d, forts, mults, path, tile=320):
    im = Image.new("RGB", (tile * len(mults), tile * len(forts)), "white")
    dr = ImageDraw.Draw(im)
    for r, f in enumerate(forts):
        for k, m in enumerate(mults):
            arr = preview(d, f["c"], f["size"] * m, f["angle"], tile,
                          max_iter=max(4000, 60 * f["period"]))
            im.paste(Image.fromarray(arr), (k * tile, r * tile))
        dr.rectangle((0, r * tile, tile, r * tile + 14), fill="black")
        dr.text((3, r * tile + 1), f"#{r} p={f['period']} {f['c'].real:.12f}"
                f"{f['c'].imag:+.12f}i", fill="white")
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("parents")
    a.add_argument("d", type=int)
    a.add_argument("--sheet")
    b = sub.add_parser("hunt")
    b.add_argument("d", type=int)
    b.add_argument("--parent", required=True, help="period,re,im")
    b.add_argument("--seeds", type=int, default=8)
    b.add_argument("--depth", type=float, default=3e-3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--mults", default="2.5,8,30,120")
    b.add_argument("--sheet", default="hunt.png")
    args = ap.parse_args()

    if args.cmd == "parents":
        ps = parents(args.d)
        for f in ps:
            print(f"{f['period']},{f['c'].real:.15f},{f['c'].imag:.15f}  size={f['size']:.2e}")
        if args.sheet:
            contact_sheet(args.d, ps, [2.5, 6], args.sheet)
    else:
        p, re_, im_ = args.parent.split(",")
        forts = hunt(args.d, int(p), complex(float(re_), float(im_)), args.seeds,
                     args.depth, seed=args.seed)
        for i, f in enumerate(forts):
            print(f"#{i} {f['period']},{f['c'].real:.15f},{f['c'].imag:.15f}  "
                  f"size={f['size']:.2e} angle={math.degrees(f['angle']):.1f}")
        contact_sheet(args.d, forts, [float(x) for x in args.mults.split(",")], args.sheet)


if __name__ == "__main__":
    main()
