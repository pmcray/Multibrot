"""
Core numerics for the Multibrot sets  z -> z**d + c  (d = 7 Heptabrot, d = 8 Octabrot).

Geometry that the cover images play on
--------------------------------------
* The main body ("fort") of the degree-d Multibrot is a (d-1)-cusped epicycloid:
  a hexagonal fort for d = 7, a heptagonal fort for d = 8.  Every mini-copy
  ("mini-fort") carries the same (d-1)-fold plan.
* The critical point z = 0 of z**d has local degree d, so wherever the
  parameter plane echoes the dynamical plane (embedded Julia sets, the halos
  that ring every mini-fort) branching is d-fold: 7 arms for the Heptabrot,
  8 for the Octabrot.

So around a mini-fort we see a (d-1)-gon pushing outward into a field that
insists on d-fold symmetry.  That mismatch is the "creative tension".

Everything here is float64 and runs on the CPU with numba (parallel).
"""
import math
import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _powers(zr, zi, d):
    """Return (z**(d-1), z**d) as (re, im, re, im) by repeated multiplication."""
    pr, pi = 1.0, 0.0
    for _ in range(d - 1):
        pr, pi = pr * zr - pi * zi, pr * zi + pi * zr
    return pr, pi, pr * zr - pi * zi, pr * zi + pi * zr


# ---------------------------------------------------------------------------
# The per-pixel kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=False, cache=True)
def render_fields(d, cx, cy, half_w, height, width, angle, max_iter,
                  stripe_k, bailout):
    """
    Iterate z -> z**d + c over a (possibly rotated) window and return the
    raw fields that colouring works from.

    The window is centred on (cx, cy); its horizontal half-width is half_w
    (in the complex plane); 'angle' rotates the frame (radians) so a fort can
    be stood upright on the page.

    Returns float32 arrays of shape (height, width):
        mu      smooth escape count (-1 for points that never escaped)
        de      exterior distance estimate in units of c (0 inside)
        stripe  stripe-average (Harkonen) with frequency stripe_k (0..1)
        trap    min |z| along the orbit (interior texture)
        period  detected period of the attracting cycle (0 outside)
    """
    mu = np.empty((height, width), np.float32)
    de = np.empty((height, width), np.float32)
    st = np.empty((height, width), np.float32)
    tr = np.empty((height, width), np.float32)
    pe = np.zeros((height, width), np.float32)

    pix = 2.0 * half_w / width
    ca, sa = math.cos(angle), math.sin(angle)
    log_b = math.log(bailout)
    log_d = math.log(d)
    b2 = bailout * bailout

    for j in prange(height):
        v = (height * 0.5 - j - 0.5) * pix
        for i in range(width):
            u = (i + 0.5 - width * 0.5) * pix
            cr = cx + u * ca - v * sa
            ci = cy + u * sa + v * ca

            zr, zi = 0.0, 0.0
            dr, di = 0.0, 0.0          # dz/dc
            trap = 1e30
            s_acc, s_prev = 0.0, 0.0
            n_st = 0
            # Brent-style periodicity check for the interior
            sr, si = 0.0, 0.0
            check, check_at = 0, 8
            escaped = False
            period = 0
            n = 0
            while n < max_iter:
                pr, pim, qr, qi = _powers(zr, zi, d)
                # dz' = d z^(d-1) dz' + 1
                dr, di = d * (pr * dr - pim * di) + 1.0, d * (pr * di + pim * dr)
                zr, zi = qr + cr, qi + ci
                n += 1
                m2 = zr * zr + zi * zi
                if m2 < trap:
                    trap = m2
                if n > 1:
                    s_prev = s_acc
                    s_acc += 0.5 + 0.5 * math.sin(stripe_k * math.atan2(zi, zr))
                    n_st += 1
                if m2 > b2:
                    escaped = True
                    break
                if abs(zr - sr) < 1e-13 and abs(zi - si) < 1e-13:
                    period = n - check
                    break
                if n == check_at:
                    sr, si = zr, zi
                    check = n
                    check_at *= 2

            if escaped:
                lz = 0.5 * math.log(m2)
                nu = math.log(lz / log_b) / log_d       # 0..1 fractional overshoot
                mu[j, i] = n + 1.0 - nu - 1.0
                dm = math.sqrt(dr * dr + di * di)
                de[j, i] = (math.sqrt(m2) * lz / dm) if dm > 0.0 else 0.0
                if n_st > 1:
                    # interpolate between last two averages using the smooth part
                    a1 = s_acc / n_st
                    a0 = s_prev / (n_st - 1)
                    f = 1.0 - nu
                    st[j, i] = a0 + (a1 - a0) * f
                else:
                    st[j, i] = 0.5
                tr[j, i] = math.sqrt(trap)
            else:
                mu[j, i] = -1.0
                de[j, i] = 0.0
                st[j, i] = 0.0
                tr[j, i] = math.sqrt(trap)
                pe[j, i] = period
    return mu, de, st, tr, pe


# ---------------------------------------------------------------------------
# Mini-fort location: nucleus, period, size and orientation
# ---------------------------------------------------------------------------

@njit(cache=True)
def _newton_nucleus(d, cr, ci, period, steps):
    for _ in range(steps):
        zr, zi, dr, di = 0.0, 0.0, 0.0, 0.0
        for _k in range(period):
            pr, pim, qr, qi = _powers(zr, zi, d)
            dr, di = d * (pr * dr - pim * di) + 1.0, d * (pr * di + pim * dr)
            zr, zi = qr + cr, qi + ci
        den = dr * dr + di * di
        if den == 0.0:
            break
        # c -= z / dz
        sr = (zr * dr + zi * di) / den
        si = (zi * dr - zr * di) / den
        cr -= sr
        ci -= si
        if sr * sr + si * si < 1e-30:
            break
    return cr, ci


def find_nucleus(d, c, period, steps=64):
    cr, ci = _newton_nucleus(d, c.real, c.imag, period, steps)
    return complex(cr, ci)


def find_period(d, c, radius, max_period=5000):
    """
    Lowest period p for which a mini-fort nucleus plausibly lies within
    'radius' of c: first n with |z_n| < |dz_n/dc| * radius.
    """
    z, dz = 0j, 0j
    for n in range(1, max_period + 1):
        dz = d * z ** (d - 1) * dz + 1
        z = z ** d + c
        if abs(z) > 1e6:
            return None
        if abs(z) < abs(dz) * radius:
            return n
    return None


def fort_geometry(d, c0, period):
    """
    Renormalisation of the period-p return map at a nucleus c0.

    Near the critical point f^p(z) ~ L z^d + D (c - c0), so the mini-fort is
    the whole Multibrot pulled back by  c = c0 + C / (D L^(1/(d-1))).

    Returns (size, angle) where 'size' is the scale factor (the main fort has
    radius ~1, so the mini-fort spans about 2*size) and 'angle' is its
    rotation relative to the main fort (defined modulo 2*pi/(d-1)).
    """
    z = 0j
    dz = 0j
    L = 1 + 0j
    for k in range(1, period + 1):
        dz = d * z ** (d - 1) * dz + 1
        z = z ** d + c0
        if k < period:
            L *= d * z ** (d - 1)
    scale = dz * L ** (1.0 / (d - 1))
    size = 1.0 / abs(scale)
    angle = -np.angle(scale)
    return size, angle


def canonical_angle(d, angle):
    """Reduce an orientation to (-pi/(d-1), pi/(d-1)]."""
    sector = 2 * math.pi / (d - 1)
    return (angle + sector / 2) % sector - sector / 2


# ---------------------------------------------------------------------------
# Searching a window for mini-forts
# ---------------------------------------------------------------------------

@njit(cache=True)
def _period_at(d, cr, ci, radius, max_period):
    zr, zi, dr, di = 0.0, 0.0, 0.0, 0.0
    for n in range(1, max_period + 1):
        pr, pim, qr, qi = _powers(zr, zi, d)
        dr, di = d * (pr * dr - pim * di) + 1.0, d * (pr * di + pim * dr)
        zr, zi = qr + cr, qi + ci
        m2 = zr * zr + zi * zi
        if m2 > 1e12:
            return 0
        if m2 < (dr * dr + di * di) * radius * radius:
            return n
    return 0


@njit(parallel=True, cache=True)
def _seed_grid(d, cx, cy, half_w, n, max_period):
    out_p = np.zeros(n * n, np.int64)
    out_r = np.zeros(n * n)
    out_i = np.zeros(n * n)
    h = 2.0 * half_w / n
    for k in prange(n * n):
        j, i = k // n, k % n
        cr = cx - half_w + (i + 0.5) * h
        ci = cy - half_w + (j + 0.5) * h
        p = _period_at(d, cr, ci, 0.7 * h, max_period)
        if p >= 2:
            nr, ni = _newton_nucleus(d, cr, ci, p, 60)
            if math.isfinite(nr) and math.isfinite(ni) and \
               (nr - cr) ** 2 + (ni - ci) ** 2 < (3 * h) ** 2:
                out_p[k], out_r[k], out_i[k] = p, nr, ni
    return out_p, out_r, out_i


def _exact_period(d, c0, p):
    """Smallest q dividing p with z_q ~ 0 (rejects nuclei found at a multiple)."""
    z = 0j
    for k in range(1, p + 1):
        z = z ** d + c0
        if abs(z) < 1e-9 and p % k == 0:
            return k
    return p


def search_forts(d, center, half_width, grid=256, max_period=64, min_size=0.0):
    """
    Seed a grid over the window, find the nucleus each seed points at, and
    return a de-duplicated list of dicts sorted by size (largest first):
        {period, c, size, angle, island}
    'island' is a heuristic: True for free-floating mini-forts, False for
    bulbs fused to a larger component.
    """
    ps, rs, is_ = _seed_grid(d, center.real, center.imag, half_width, grid, max_period)
    seen = {}
    tol = 2.0 * half_width / grid * 1e-3
    for p, r, i in zip(ps, rs, is_):
        if p == 0:
            continue
        c0 = complex(r, i)
        q = _exact_period(d, c0, int(p))
        key = (q, round(r / tol), round(i / tol))
        if key in seen:
            continue
        size, ang = fort_geometry(d, c0, q)
        if not math.isfinite(size) or size < min_size:
            continue
        seen[key] = dict(period=q, c=c0, size=size, angle=ang,
                         island=is_island(d, c0, size))
    return sorted(seen.values(), key=lambda f: -f["size"])


def is_island(d, c0, size, samples=48, ring=3.0, max_iter=2000):
    """A mini-fort is an island if a ring at ~3x its size is almost all exterior."""
    inside = 0
    for k in range(samples):
        c = c0 + ring * size * complex(math.cos(2 * math.pi * k / samples),
                                       math.sin(2 * math.pi * k / samples))
        z = 0j
        for _ in range(max_iter):
            z = z ** d + c
            if abs(z) > 2:
                break
        else:
            inside += 1
    return inside <= samples // 12


def fold_phase(d, c0, size, angle, mode, radius, interior_only=False,
               grid=400, max_iter=4000):
    """
    Direction (radians, in the image of a frame rotated by 'angle') of the
    strongest 'mode'-fold feature on an annulus around a nucleus.  With
    mode=d and ink on the annulus this finds the arms; with mode=d-1 and
    interior_only it finds the fort's bastions.
    """
    hw = radius[1] * size * 1.05
    mu, de, *_ = render_fields(d, c0.real, c0.imag, hw, grid, grid, angle,
                               max_iter, float(d), 1000.0)
    pix = 2 * hw / grid
    jj, ii = np.mgrid[0:grid, 0:grid]
    u = (ii + 0.5 - grid / 2) * pix
    v = (grid / 2 - jj - 0.5) * pix
    r = np.hypot(u, v)
    ring = (r > radius[0] * size) & (r < radius[1] * size)
    w = ring & ((mu < 0) if interior_only else ((mu < 0) | (de < 2 * pix)))
    th = np.arctan2(v[w], u[w])
    z = np.exp(1j * mode * th).sum()
    return float(np.angle(z) / mode)


def arm_phase(d, c0, size, angle, radius=(2.0, 3.5)):
    """Direction of one of the d arms that ring a mini-fort."""
    return fold_phase(d, c0, size, angle, d, radius)


def bastion_phase(d, c0, size, angle, radius=(0.6, 1.3)):
    """Direction of one of the fort's d-1 bastions (lobes)."""
    return fold_phase(d, c0, size, angle, d - 1, radius, interior_only=True)
