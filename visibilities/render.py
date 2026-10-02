"""Render a region of a Multibrot to a finished grimoire plate (PNG)."""
import math
import time
import numpy as np
from PIL import Image

from .engine import render_fields, find_nucleus, fort_geometry
from .grimoire import plate, downsample, to_uint8


def render_plate(d, center, half_width, width, height, angle=0.0, ss=2,
                 max_iter=6000, palette="vellum", stripe_k=None, line=1.2,
                 contour_every=1.0, contour_alpha=0.55, hatch=0.35, wash=0.35,
                 seed=7, verbose=True):
    """Return an RGB uint8 array. half_width is half the frame width in c."""
    t0 = time.time()
    W, H = width * ss, height * ss
    k = float(d if stripe_k is None else stripe_k)
    fields = render_fields(d, center.real, center.imag, half_width, H, W,
                           angle, max_iter, k, 1000.0)
    pix = 2 * half_width / W
    img = plate(fields, pix, palette, line=line * ss, contour_every=contour_every,
                contour_alpha=contour_alpha, hatch=hatch, wash=wash, seed=seed)
    img = downsample(img, ss)
    if verbose:
        print(f"  rendered {width}x{height} (ss={ss}) in {time.time() - t0:.1f}s")
    return to_uint8(img)


def fort_frame(d, nucleus_guess, period, zoom=3.0, turn=0.0):
    """
    Frame a mini-fort: returns (centre, half_width, angle).
    zoom   frame half-width in units of the fort's size (3 = fort plus its
           first ring of d-fold arms; 15-80 = the embedded Julia halos)
    turn   extra rotation in units of one bastion (2*pi/(d-1))
    """
    c0 = find_nucleus(d, nucleus_guess, period)
    size, ang = fort_geometry(d, c0, period)
    return c0, zoom * size, ang + turn * 2 * math.pi / (d - 1)


def save(arr, path, dpi=300):
    Image.fromarray(arr).save(path, dpi=(dpi, dpi))
