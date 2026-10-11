"""
Deep zoom for z^d + c by perturbation, so forts far below double precision
(1e-13) can be rendered and searched for.

* A fort's nucleus c0 is found to high precision (mpmath).  Its orbit
  Z_0 = 0, Z_1, ..., Z_{p-1} is periodic (Z_p = 0), and is stored in
  ordinary complex128 -- the values themselves are of order 1.
* Each pixel iterates only its offset from that orbit,
      delta' = (Z + delta)^d - Z^d + dc,
  in float64.  dc can be as small as 1e-300.  When the pixel's orbit
  passes closer to 0 than its offset ("rebasing", Zhuoran), the offset
  becomes the whole value and the reference restarts at Z_0 -- so one
  periodic reference serves every pixel without glitches.
* Interior (the fort itself and its bulbs) is detected when the pixel's
  returns near the critical point stop moving.

Also: iterated "Julia morphing" to build forts wrapped in nested rings
(morph_path), working at whatever depth that takes.
"""
import math
import numpy as np
from numba import njit, prange
import mpmath as mp

from .engine import render_fields

# ---------------------------------------------------------------------------
# High-precision nucleus, period and size
# ---------------------------------------------------------------------------


def _dps_for(scale):
    return max(30, int(-math.log10(max(scale, 1e-300))) + 25)


def nucleus_mp(d, c, p, scale=1e-12, steps=200):
    """Newton for f^p(0) = 0 at high precision.  Returns mpc."""
    mp.mp.dps = _dps_for(scale)
    c = mp.mpc(c)
    tol = mp.mpf(scale) * mp.mpf(10) ** -12
    for _ in range(steps):
        z = mp.mpc(0)
        dz = mp.mpc(0)
        for _k in range(p):
            dz = d * z ** (d - 1) * dz + 1
            z = z ** d + c
        if dz == 0:
            break
        step = z / dz
        c -= step
        if abs(step) < tol:
            break
    return c


def period_mp(d, A, radius, max_period=200000):
    """
    Lowest p for which a nucleus plausibly lies within 'radius' of A:
    first n with |z_n| < |dz_n/dc| * radius.  A may be an mpc.
    """
    mp.mp.dps = _dps_for(radius)
    z = mp.mpc(0)
    dz = 0j
    A = mp.mpc(A)
    for n in range(1, max_period + 1):
        zc = complex(z)
        dz = d * zc ** (d - 1) * dz + 1
        z = z ** d + A
        az = abs(complex(z))
        if az > 1e6:
            return None
        if az < abs(dz) * radius:
            return n
    return None


def geometry(d, c0, p):
    """Size and orientation of the fort at c0 (renormalisation, see engine)."""
    z, dz, L = 0j, 0j, 1 + 0j
    orbit = reference_orbit(d, c0, p)
    for k in range(p):
        zc = orbit[k]
        dz = d * zc ** (d - 1) * dz + 1
        if k >= 1:
            L *= d * zc ** (d - 1)
    scale = dz * L ** (1.0 / (d - 1))
    return 1.0 / abs(scale), -float(np.angle(scale))


def reference_orbit(d, c0, p):
    """Z_0..Z_{p-1} computed in high precision, stored as complex128."""
    z = mp.mpc(0)
    out = np.empty(p, np.complex128)
    for k in range(p):
        out[k] = complex(z)
        z = z ** d + c0
    return out


# ---------------------------------------------------------------------------
# The perturbation kernel
# ---------------------------------------------------------------------------

def ref_coefficients(d, ref):
    """coef[m, j] = C(d, j) * Z_m^(d-j): the expansion of (Z+dl)^d - Z^d."""
    j = np.arange(d + 1)
    B = np.array([math.comb(d, k) for k in j], float)
    return np.ascontiguousarray(B[None, :] * ref[:, None] ** (d - j)[None, :])


@njit(parallel=True, fastmath=False, cache=True)
def render_deep(d, ref, coef, dcr, dci, max_iter, bailout):
    """
    ref        reference orbit (complex128, periodic, ref[0] = 0)
    dcr, dci   per-pixel offsets of c from the nucleus (float64 arrays)
    Returns mu (smooth escape count, -1 inside), de (distance estimate in
    units of c), both float32.
    """
    H, W = dcr.shape
    p = ref.shape[0]
    mu = np.empty((H, W), np.float32)
    de = np.empty((H, W), np.float32)
    log_b = math.log(bailout)
    log_d = math.log(d)
    b2 = bailout * bailout
    for j in prange(H):
        for i in range(W):
            dc = complex(dcr[j, i], dci[j, i])
            dl = 0j
            der = 0j
            m = 0
            n = 0
            escaped = False
            z = 0j
            last = complex(1e300, 0.0)
            calm = 0
            while n < max_iter:
                Z = ref[m]
                z = Z + dl
                # derivative dz/dc of the full orbit
                zd1 = 1 + 0j
                for _ in range(d - 1):
                    zd1 *= z
                der = d * zd1 * der + 1.0
                # delta' = (Z+dl)^d - Z^d + dc, Horner in dl
                acc = coef[m, d]
                for jj in range(d - 1, 0, -1):
                    acc = acc * dl + coef[m, jj]
                dl = acc * dl + dc
                m += 1
                n += 1
                if m == p:
                    m = 0
                z = ref[m] + dl
                az2 = z.real * z.real + z.imag * z.imag
                if az2 > b2:
                    escaped = True
                    break
                adl2 = dl.real * dl.real + dl.imag * dl.imag
                if az2 < adl2:             # rebase onto the start of the reference
                    dl = z
                    m = 0
                if m == 0:                 # back near the critical point
                    dz_ = z - last
                    if abs(dz_) < 1e-10 * abs(z) + 1e-300:
                        calm += 1
                        if calm >= 3:
                            break
                    else:
                        calm = 0
                    last = z
            if escaped:
                r = math.sqrt(az2)
                lz = math.log(r)
                nu = math.log(lz / log_b) / log_d
                mu[j, i] = n - nu
                dm = abs(der)
                de[j, i] = r * lz / dm if dm > 0.0 else 0.0
            else:
                mu[j, i] = -1.0
                de[j, i] = 0.0
    return mu, de


def window_offsets(height, width, half_w, angle=0.0, centre_offset=0j):
    """Per-pixel c offsets (float64) for a rotated window round a nucleus."""
    pix = 2.0 * half_w / width
    jj, ii = np.mgrid[0:height, 0:width]
    u = (ii + 0.5 - width / 2) * pix
    v = (height / 2 - jj - 0.5) * pix
    w = (u + 1j * v) * np.exp(1j * angle) + centre_offset
    return np.ascontiguousarray(w.real), np.ascontiguousarray(w.imag)


class DeepFort:
    """A fort known to high precision, ready to render around."""

    def __init__(self, d, c, p, scale_hint=1e-12):
        self.d, self.p = d, p
        self.c = nucleus_mp(d, c, p, scale_hint)
        self.ref = reference_orbit(d, self.c, p)
        self.size, self.angle = geometry(d, self.c, p)
        if not (self.size > 0 and math.isfinite(self.size)):
            raise ValueError(f"no fort of period {p} here")
        # re-polish at the right precision now the size is known
        self.c = nucleus_mp(d, self.c, p, self.size)
        self.ref = reference_orbit(d, self.c, p)
        self.coef = ref_coefficients(d, self.ref)

    def render(self, half_w, height, width, angle=None, max_iter=None,
               bailout=1e3, centre_offset=0j):
        """Fields (mu, de) for a window of half-width half_w (in c)."""
        ang = self.angle if angle is None else angle
        dcr, dci = window_offsets(height, width, half_w, ang, centre_offset)
        it = max_iter or max(20000, 60 * self.p)
        return render_deep(self.d, self.ref, self.coef, dcr, dci, it, bailout)

    def __repr__(self):
        return (f"DeepFort(d={self.d}, p={self.p}, size={self.size:.3e}, "
                f"c={mp.nstr(self.c, 30)})")


# ---------------------------------------------------------------------------
# Building forts wrapped in nested rings
# ---------------------------------------------------------------------------

def deeper(fort, ring=(1.2, 2.0), theta=None, deltas=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
           grid=400, rng=None, tries=40):
    """
    One morph step: aim at a filament in the annulus 'ring' (fort sizes)
    round the fort, optionally near angle theta (radians, in the fort's own
    frame), and find the lowest-period fort within a small radius of it.
    """
    s = fort.size
    hw = ring[1] * s * 1.05
    mu, de = fort.render(hw, grid, grid, max_iter=max(8000, 30 * fort.p))
    pix = 2 * hw / grid
    jj, ii = np.mgrid[0:grid, 0:grid]
    u = (ii + 0.5 - grid / 2) * pix
    v = (grid / 2 - jj - 0.5) * pix
    r = np.hypot(u, v) / s
    ok = (mu > 0) & (de < 0.5 * pix) & (r > ring[0]) & (r < ring[1])
    if theta is not None:
        ok &= np.abs(np.angle(np.exp(1j * (np.arctan2(v, u) - theta)))) < 0.35
    idx = np.flatnonzero(ok.ravel())
    rng = rng or np.random.default_rng(0)
    rng.shuffle(idx)
    rot = complex(np.exp(1j * fort.angle))
    for k in idx[:tries]:
        off = complex(u.ravel()[k], v.ravel()[k]) * rot
        mp.mp.dps = _dps_for(s * 1e-4)
        A = fort.c + mp.mpc(off)
        for dl in deltas:
            q = period_mp(fort.d, A, dl * s)
            if q:
                N = nucleus_mp(fort.d, A, q, dl * s * 1e-3)
                if abs(complex(N - A)) < 5 * dl * s:
                    try:
                        g = DeepFort(fort.d, N, q, dl * s * 1e-3)
                    except (ValueError, ZeroDivisionError, OverflowError):
                        continue                  # Newton wandered off
                    if 0 < g.size < s:
                        return g
    return None


def morph_path(d, c, p, levels, ring=(1.2, 2.0), theta=None, seed=0, **kw):
    """Starting from the fort (c, p), morph 'levels' times.  Returns the forts."""
    rng = np.random.default_rng(seed)
    forts = [DeepFort(d, c, p, 1e-6)]
    for _ in range(levels):
        nxt = deeper(forts[-1], ring, theta, rng=rng, **kw)
        if nxt is None:
            break
        forts.append(nxt)
    return forts
