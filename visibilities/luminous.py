"""
Luminous colouring: smooth bands of colour by escape time, glowing where
they crowd against the set, the forts themselves black with an ember rim.

This is the look of a deep-zoom rendering rather than an engraving.  Two
palettes are tuned for the wrapper:

  amaranth   front board (Heptabrot): indigo night, violet, amaranth and
             rose-white glows
  trifle     back board (Octabrot): muted autumn / trifle -- sherry brown,
             sponge gold, custard, cream and a raspberry-jam accent

Colour index = log of the smooth escape count, so the bands keep an even
rhythm as one zooms toward the set; brightness rises near the boundary
(distance estimate), which makes the filigree glow.
"""
import math
import numpy as np


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


# Each palette: cyclic gradient stops (position 0..1, colour), plus the
# colours of the fort's interior and its rim.
PALETTES = {
    "amaranth": dict(
        stops=[(0.00, "#12061f"), (0.18, "#2a0b4a"), (0.36, "#5a1a8c"),
               (0.52, "#9b2fc0"), (0.64, "#d23a8f"), (0.72, "#e8a0d8"),
               (0.80, "#9b2fc0"), (0.90, "#3d1066"), (1.00, "#12061f")],
        inside="#050208", rim="#7a0f2e", glow="#f6c8ec"),
    "trifle": dict(
        stops=[(0.00, "#2a1810"), (0.16, "#5c3a1e"), (0.32, "#9a6a2e"),
               (0.46, "#d4a94a"), (0.58, "#f1deaa"), (0.66, "#fbf3dc"),
               (0.74, "#c9787a"), (0.84, "#8e2f3a"), (0.92, "#4a2a1c"),
               (1.00, "#2a1810")],
        inside="#120a06", rim="#8e2f3a", glow="#fff4d8"),
}


def _lut(stops, n=1024):
    pos = np.array([p for p, _ in stops])
    cols = np.array([_hex(c) for _, c in stops])
    x = np.linspace(0, 1, n, endpoint=False)
    return np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], -1).astype(np.float32)


def colour(mu, dpx, pal="amaranth", cycles=5.0, phase=0.0, glow=0.35,
           glow_px=3.0, ink=0.55, ink_px=1.2, inside=None, blend=None,
           pal2=None, ref=None):
    """
    mu      smooth escape count (<0 inside)
    dpx     exterior distance estimate in output pixels
    cycles  how many times the palette cycles between the 5th and 95th
            percentile of log(mu) in the view (deep zooms have large but
            narrow-ranged counts, so the bands are fitted to the view)
    glow    brightening of the halo round every filament
    ink     darkening of the filaments themselves
    blend   optional per-pixel 0..1 array mixing pal (0) into pal2 (1)
    ref     (lo, hi) of log(mu) to use instead of this view's percentiles,
            so pictures rendered in pieces share one set of bands
    Returns float32 RGB in 0..1.
    """
    if inside is None:
        inside = mu < 0
    lm = np.log(np.where(inside, 1.0, np.maximum(mu, 1.0)))
    if ref is None:
        ext = lm[~inside]
        lo, hi = (np.percentile(ext[:: max(1, ext.size // 200000)], [5, 95])
                  if ext.size else (0.0, 1.0))
    else:
        lo, hi = ref
    idx = ((lm - lo) / max(hi - lo, 1e-9) * cycles + phase) % 1.0
    dd = np.clip(dpx, 0, None)

    def shade(name):
        P = PALETTES[name]
        lut = _lut(P["stops"])
        c = lut[(idx * len(lut)).astype(np.int32) % len(lut)]
        g = (glow * (np.exp(-dd / glow_px) - np.exp(-dd / ink_px)))
        c = c * (1 - g[..., None]) + _hex(P["glow"]) * g[..., None]
        k = ink * np.exp(-dd / ink_px)
        c = c * (1 - k[..., None]) + _hex(P["rim"]) * 0.6 * k[..., None]
        return np.where(inside[..., None], _hex(P["inside"]), c)

    out = shade(pal)
    if blend is not None and pal2 is not None:
        b = np.clip(blend, 0, 1)[..., None]
        out = out * (1 - b) + shade(pal2) * b
    return np.clip(out, 0, 1).astype(np.float32)
