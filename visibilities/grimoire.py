"""
Grimoire / bestiary colouring for the raw Multibrot fields.

The look being aimed at is an engraved plate in an old book of marvels:
iron-gall ink linework on vellum, the beast itself (the fort) rubricated in
red lead or laid in gold, concentric outworks drawn as fine contour lines,
and the field between washed and hatched.

Each element of the picture is driven by one piece of the mathematics:

    ink line        exterior distance estimate (crisp filaments at any zoom)
    outworks        level lines of the smooth escape count; around a fort these
                    are nested copies of its outline, like the ravelins and
                    hornworks of a bastioned fortress
    hatching        stripe average with frequency d -- d-fold by construction,
                    so the 7 (or 8) insists itself into the wash
    fort fill       interior; textured by the orbit trap, edged with a rim
    paper           procedural vellum (independent of the maths)
"""
import math
import numpy as np


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


PALETTES = {
    # sepia ink on warm vellum, fort in red lead with a gold rim
    "vellum": dict(paper=("#efe2c2", "#d9bf8c"), stain="#a8783e",
                   ink="#2a1a0e", wash="#7a4b23", contour="#5a3518",
                   fill=("#7d1a12", "#b8331f"), rim="#c99a2e", vignette=0.55),
    # gold leaf on black vellum (a nocturnal grimoire)
    "nigredo": dict(paper=("#16110c", "#0b0806"), stain="#3a2410",
                    ink="#e2b956", wash="#8a6424", contour="#b48a3a",
                    fill=("#3b0d0a", "#6e1a12"), rim="#f0cf6e", vignette=0.35,
                    light_ink=True),
    # ultramarine and gold: an illuminated page
    "lapis": dict(paper=("#1d2f63", "#0f1a3d"), stain="#0a1130",
                  ink="#e8c463", wash="#5b74b8", contour="#c9a44c",
                  fill=("#7a1414", "#a52a1c"), rim="#f3d77a", vignette=0.4,
                  light_ink=True),
    # verdigris and bone: a herbal / bestiary plate
    "verdigris": dict(paper=("#e8e0c8", "#cbbd96"), stain="#8f7d4c",
                      ink="#1d2a24", wash="#3f6f62", contour="#2f5148",
                      fill=("#2f5d52", "#4f8a78"), rim="#b0892e", vignette=0.5),
}


# ---------------------------------------------------------------------------
# Procedural vellum
# ---------------------------------------------------------------------------

def _value_noise(h, w, scale, rng):
    gh, gw = int(h / scale) + 3, int(w / scale) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    y = np.arange(h, dtype=np.float32) / scale
    x = np.arange(w, dtype=np.float32) / scale
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = y - y0, x - x0
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fbm(h, w, base_scale, octaves=5, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot, sc = 1.0, 0.0, base_scale
    for _ in range(octaves):
        out += amp * _value_noise(h, w, max(sc, 1.0), rng)
        tot += amp
        amp *= 0.5
        sc /= 2.0
    return out / tot


def vellum(h, w, pal, seed=7):
    """A parchment/vellum ground with mottling, fibres, foxing and a vignette."""
    p0, p1 = _hex(pal["paper"][0]), _hex(pal["paper"][1])
    s = max(h, w)
    mottle = fbm(h, w, s / 6, 6, seed)
    fibres = fbm(h, w, s / 300, 3, seed + 1)
    t = np.clip(0.15 + 0.9 * (mottle - 0.5) * 2.2 + 0.15 * (fibres - 0.5), 0, 1)
    img = p0[None, None, :] * (1 - t[..., None]) + p1[None, None, :] * t[..., None]

    # foxing: a few soft stains
    rng = np.random.default_rng(seed + 2)
    stain = _hex(pal["stain"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(9):
        cy, cx = rng.random() * h, rng.random() * w
        r = s * (0.01 + 0.05 * rng.random())
        a = 0.06 + 0.10 * rng.random()
        m = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
        img = img * (1 - a * m[..., None]) + stain * (a * m[..., None])

    # vignette, darker toward the edges like a handled page
    ny, nx = (yy / h - 0.5) * 2, (xx / w - 0.5) * 2
    rr = np.sqrt(nx ** 2 * 0.9 + ny ** 2 * 0.9)
    edge = np.clip(rr - 0.55, 0, None) ** 1.6 * pal["vignette"]
    edge = edge * (0.8 + 0.4 * fbm(h, w, s / 10, 3, seed + 3))
    img = img * (1 - edge[..., None]) + stain * edge[..., None] * 0.6
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# Compositing the plate
# ---------------------------------------------------------------------------

def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def plate(fields, pix, pal_name="vellum", line=1.2, contour_every=1.0,
          contour_alpha=0.55, hatch=0.35, wash=0.35, seed=7, paper=None,
          trap_scale=None, fill_tex=None):
    """
    fields   (mu, de, stripe, trap, period) from engine.render_fields
    pix      size of one pixel in the complex plane
    line     ink line weight in pixels
    contour_every  spacing of the outworks in escape-count units
    """
    pal = PALETTES[pal_name]
    mu, de, st, tr, pe = fields
    h, w = mu.shape
    inside = mu < 0
    img = vellum(h, w, pal, seed) if paper is None else paper.copy()

    ink = _hex(pal["ink"])
    wash_c = _hex(pal["wash"])
    cont_c = _hex(pal["contour"])

    # distance in pixels from the set
    dpx = np.where(inside, 0.0, de / pix)

    # --- wash: a halo of tone that hugs the set and fades into the margin
    halo = np.exp(-np.clip(dpx, 0, None) / (40.0 * line))
    tone = wash * halo * (0.55 + 0.45 * st)
    img = img * (1 - tone[..., None]) + wash_c * tone[..., None]

    # --- hatching: d-fold stripe field turned into fine engraved strokes
    if hatch > 0:
        strokes = 0.5 + 0.5 * np.cos(2 * math.pi * (st * 6.0))
        strokes = _smoothstep(0.80, 0.97, strokes)
        hmask = hatch * strokes * _smoothstep(0.0, 0.25, halo) * (1 - _smoothstep(0.5, 1.0, halo) * 0.5)
        hmask = np.where(inside, 0, hmask)
        img = img * (1 - hmask[..., None]) + cont_c * hmask[..., None]

    # --- outworks: level lines of the escape count
    if contour_alpha > 0:
        m = np.where(inside, 0, mu) / contour_every
        f = np.abs(m - np.round(m))
        # line width in escape-count units from the local gradient
        gy, gx = np.gradient(m)
        g = np.sqrt(gx * gx + gy * gy) + 1e-6
        cl = 1 - _smoothstep(0.35 * line, 0.9 * line, f / g)
        fade = _smoothstep(0.0, 0.08, halo)
        ca = contour_alpha * cl * fade * (1 - _smoothstep(0.6, 1.0, halo))
        ca = np.where(inside, 0, ca)
        img = img * (1 - ca[..., None]) + cont_c * ca[..., None]

    # --- the fort itself: rubricated interior with a gilded rim
    f0, f1 = _hex(pal["fill"][0]), _hex(pal["fill"][1])
    # trap_scale / fill_tex let a picture rendered in strips share one
    # normalisation and one texture, so the strips meet without seams
    if trap_scale is None:
        trap_scale = np.percentile(tr[inside], 95) if inside.any() else 1.0
    tt = np.clip(tr / (trap_scale + 1e-12), 0, 1)
    tex = fbm(h, w, max(h, w) / 80, 4, seed + 9) if fill_tex is None else fill_tex
    t = np.clip(0.6 * tt + 0.4 * tex, 0, 1)
    fill = f0[None, None, :] * (1 - t[..., None]) + f1[None, None, :] * t[..., None]
    img = np.where(inside[..., None], fill, img)

    # rim: thin band just inside the boundary (approximated by dilation)
    rim = _rim_mask(inside, max(1, int(round(1.5 * line))))
    rim_c = _hex(pal["rim"])
    img = np.where(rim[..., None], rim_c * 0.85 + img * 0.15, img)

    # --- ink: the filaments and the outline of every fort
    a = 1 - _smoothstep(0.25 * line, line, dpx)
    a = np.where(inside, 0, a)
    a = np.where(rim, 0, a)
    img = img * (1 - a[..., None]) + ink * a[..., None]

    # a little ink grain so the black is not dead flat
    grain = fbm(h, w, 2.0, 2, seed + 11) - 0.5
    img = np.clip(img + 0.04 * grain[..., None], 0, 1)
    return img


def _rim_mask(inside, r):
    """Interior pixels within r pixels of the exterior."""
    out = np.zeros_like(inside)
    ext = ~inside
    near = np.zeros_like(inside)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            near |= np.roll(np.roll(ext, dy, 0), dx, 1)
    out = inside & near
    return out


def downsample(img, ss):
    if ss == 1:
        return img
    h, w, c = img.shape
    return img[: h - h % ss, : w - w % ss].reshape(h // ss, ss, w // ss, ss, c).mean(axis=(1, 3))


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
