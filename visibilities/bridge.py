"""
A seamless bridge between the Heptabrot and the Octabrot.

Fractional powers z^7.5 need a branch cut and tear the picture.  This
module uses no fractional power.  Every pixel iterates an honest polynomial
from the one-parameter family

    f_t(z) = (1 - t) z^7 + t z^8 + c ,      0 <= t <= 1,

which is the Heptabrot at t = 0 and the Octabrot at t = 1.  Its critical
points are z = 0 (local degree 7 while t < 1) and

    z* = -7 (1 - t) / (8 t),

a second critical point that comes in from infinity as t grows and fuses
with 0 at t = 1.  That fusion is when the eighth fold is born: the
hexagonal fort with its 7-armed ring becomes a heptagonal fort with an
8-armed ring.  It happens continuously, so a picture in which t varies
from place to place has no seam.

A fort can be followed through the family: Newton's method on
f_t^p(0) = 0 tracks its nucleus c(t) from t = 0 to t = 1, so the front
board's hexagonal fort and the back board's heptagonal fort can be the same
creature.
"""
import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def render_family(cr, ci, tt, max_iter, bailout):
    """
    Iterate f_t at every pixel, with c = cr + i ci and t = tt given per pixel.
    Returns mu, de, stripe7, stripe8, trap (float32).  The two stripe
    averages use whole-number frequencies 7 and 8, so neither has a branch
    cut; blend them with t.
    """
    H, W = cr.shape
    mu = np.empty((H, W), np.float32)
    de = np.empty((H, W), np.float32)
    s7 = np.empty((H, W), np.float32)
    s8 = np.empty((H, W), np.float32)
    tr = np.empty((H, W), np.float32)
    log_b = math.log(bailout)
    b2 = bailout * bailout
    for j in prange(H):
        for i in range(W):
            c_r, c_i, t = cr[j, i], ci[j, i], tt[j, i]
            a = 1.0 - t
            zr, zi, dr, di = 0.0, 0.0, 0.0, 0.0
            trap = 1e30
            acc7, acc8, prev7, prev8 = 0.0, 0.0, 0.0, 0.0
            nst = 0
            sr, si = 0.0, 0.0
            check, check_at = 0, 8
            escaped = False
            n = 0
            m2 = 0.0
            while n < max_iter:
                # z^6 and z^7
                p6r, p6i = 1.0, 0.0
                for _ in range(6):
                    p6r, p6i = p6r * zr - p6i * zi, p6r * zi + p6i * zr
                p7r, p7i = p6r * zr - p6i * zi, p6r * zi + p6i * zr
                # f'(z) = z^6 (7a + 8 t z)
                gr, gi = 7.0 * a + 8.0 * t * zr, 8.0 * t * zi
                fr, fi = p6r * gr - p6i * gi, p6r * gi + p6i * gr
                dr, di = fr * dr - fi * di + 1.0, fr * di + fi * dr
                # f(z) = z^7 (a + t z) + c
                hr, hi = a + t * zr, t * zi
                zr, zi = p7r * hr - p7i * hi + c_r, p7r * hi + p7i * hr + c_i
                n += 1
                m2 = zr * zr + zi * zi
                if m2 < trap:
                    trap = m2
                if n > 1:
                    th = math.atan2(zi, zr)
                    prev7, prev8 = acc7, acc8
                    acc7 += 0.5 + 0.5 * math.sin(7.0 * th)
                    acc8 += 0.5 + 0.5 * math.sin(8.0 * th)
                    nst += 1
                if m2 > b2:
                    escaped = True
                    break
                if abs(zr - sr) < 1e-13 and abs(zi - si) < 1e-13:
                    break
                if n == check_at:
                    sr, si = zr, zi
                    check = n
                    check_at *= 2
            if escaped:
                r = math.sqrt(m2)
                lz = math.log(r)
                # effective degree of the last step: 7 -> 8 as t|z| outgrows (1-t)
                deff = 7.0 + t * r / (a + t * r)
                nu = math.log(lz / log_b) / math.log(deff)
                mu[j, i] = n - nu
                dm = math.sqrt(dr * dr + di * di)
                de[j, i] = r * lz / dm if dm > 0.0 else 0.0
                if nst > 1:
                    f = 1.0 - nu
                    s7[j, i] = prev7 / (nst - 1) + (acc7 / nst - prev7 / (nst - 1)) * f
                    s8[j, i] = prev8 / (nst - 1) + (acc8 / nst - prev8 / (nst - 1)) * f
                else:
                    s7[j, i] = 0.5
                    s8[j, i] = 0.5
            else:
                mu[j, i] = -1.0
                de[j, i] = 0.0
                s7[j, i] = 0.0
                s8[j, i] = 0.0
            tr[j, i] = math.sqrt(trap)
    return mu, de, s7, s8, tr


@njit(cache=True)
def _orbit_nb(c, t, p):
    z, dz = 0j, 0j
    for _ in range(p):
        z6 = z ** 3
        z6 = z6 * z6
        dz = (7 * (1 - t) * z6 + 8 * t * z6 * z) * dz + 1
        z = z6 * z * ((1 - t) + t * z) + c
        if abs(z) > 1e10:
            return complex(np.nan, np.nan), 0j
    return z, dz


def _orbit(c, t, p):
    z, dz = _orbit_nb(complex(c), float(t), int(p))
    if not math.isfinite(z.real):
        return None, None
    return z, dz


@njit(cache=True)
def _nucleus_nb(c, t, p, steps):
    for _ in range(steps):
        z, dz = _orbit_nb(c, t, p)
        if not math.isfinite(z.real) or dz == 0:
            return complex(np.nan, np.nan)
        step = z / dz
        c -= step
        if not (math.isfinite(c.real) and math.isfinite(c.imag)):
            return complex(np.nan, np.nan)
        if abs(step) < 1e-17:
            break
    return c


def nucleus(c, t, p, steps=80):
    """Newton's method for f_t^p(0) = 0."""
    r = _nucleus_nb(complex(c), float(t), int(p), steps)
    return None if not math.isfinite(r.real) else r


def track(c0, p, t0=0.0, t1=1.0, max_jump=None):
    """
    Follow a fort's nucleus from t0 to t1.  Returns arrays (t, c).
    max_jump bounds how far the nucleus may move in one step (defaults to
    1/20 of the distance to its nearest likely neighbour, estimated
    from |dz/dc|).
    """
    ts, cs = [t0], [c0]
    t, c, h = t0, c0, (t1 - t0) / 200
    while (t1 - t) * math.copysign(1, t1 - t0) > 1e-15:
        tn = t + h
        if (tn - t1) * math.copysign(1, t1 - t0) > 0:
            tn = t1
        cn = nucleus(c, tn, p)
        _, dz = _orbit(c, t, p)
        lim = max_jump or (0.05 / abs(dz) if dz else 1e-6)
        if cn is None or abs(cn - c) > lim:
            h /= 2
            if abs(h) < 1e-12:
                raise RuntimeError(f"tracking stalled at t={t}")
            continue
        t, c = tn, cn
        ts.append(t)
        cs.append(c)
        h = math.copysign(min(abs(h) * 1.6, 0.02), h)
    return np.array(ts), np.array(cs)


def fort_scale(c0, t, p):
    """
    Size and orientation of the fort at nucleus c0 of f_t.  Near z = 0,
    f(z) - c = z^7 (a + t z); the fort is governed by whichever of the two
    terms dominates on the fort's own scale.  Returns (size, angle, degree).
    """
    z, dz, L = 0j, 0j, 1 + 0j
    orbit = []
    for k in range(1, p + 1):
        dz = (7 * (1 - t) * z ** 6 + 8 * t * z ** 7) * dz + 1
        z = (1 - t) * z ** 7 + t * z ** 8 + c0
        orbit.append(z)
        if k < p:
            L *= 7 * (1 - t) * z ** 6 + 8 * t * z ** 7
    best = None
    for m, A in ((7, 1 - t), (8, t)):
        if A == 0:
            continue
        scale = dz * (L * A) ** (1.0 / (m - 1))
        alpha = abs(L * A) ** (-1.0 / (m - 1))  # fort's radius in z near 0
        cand = (1 / abs(scale), -np.angle(scale), m, alpha)
        # pick the degree whose term dominates at radius alpha
        other = t * alpha if m == 7 else (1 - t)
        mine = (1 - t) if m == 7 else t * alpha
        if best is None or mine > other:
            best = cand
    return best[0], best[1], best[2]


# ---------------------------------------------------------------------------
# Laying a tracked fort across a book jacket
# ---------------------------------------------------------------------------

def dz_dc(c, t, p):
    """D = d f_t^p(0)/dc at a nucleus: sets the fort's frame up to a constant."""
    return _orbit(c, t, p)[1]


def family_phase(c0, t, frame, mode, radius, interior_only=False, grid=360,
                 max_iter=6000):
    """
    Direction of the strongest mode-fold feature around a nucleus of f_t,
    in the image of the frame c = c0 + frame * w (w in fort sizes).
    """
    hw = radius[1] * 1.05
    jj, ii = np.mgrid[0:grid, 0:grid]
    pix = 2 * hw / grid
    u = (ii + 0.5 - grid / 2) * pix
    v = (grid / 2 - jj - 0.5) * pix
    C = c0 + frame * (u + 1j * v)
    mu, de, *_ = render_family(np.ascontiguousarray(C.real), np.ascontiguousarray(C.imag),
                               np.full((grid, grid), t), max_iter, 1000.0)
    r = np.hypot(u, v)
    ring = (r > radius[0]) & (r < radius[1])
    w = ring & ((mu < 0) if interior_only else ((mu < 0) | (de < 2 * pix * abs(frame))))
    z = np.exp(1j * mode * np.arctan2(v[w], u[w])).sum()
    return float(np.angle(z) / mode)


class TrackedFort:
    """A fort followed from t=0 (Heptabrot) to t=1 (Octabrot)."""

    def __init__(self, c0, period, cache=None):
        self.p = period
        if cache and _exists(cache):
            dat = np.load(cache)
            self.ts, self.cs = dat["ts"], dat["cs"]
        else:
            self.ts, self.cs = track(c0, period)
            if cache:
                np.savez(cache, ts=self.ts, cs=self.cs)
        # frame = K(t) / D(t): continuous in t.  Calibrate K at both ends so
        # the fort has unit size and one arm points straight up.
        self.K0 = self._calibrate(0.0, 7)
        self.K1 = self._calibrate(1.0, 8)
        # interpolate K along t without a jump in angle: choose the branch
        # of the end orientation (mod one arm) nearest the start
        a0, a1 = np.angle(self.K0), np.angle(self.K1)
        sector = 2 * math.pi / 8
        k = round((a0 - a1) / sector)
        self.a0, self.a1 = a0, a1 + k * sector
        self.l0, self.l1 = math.log(abs(self.K0)), math.log(abs(self.K1))

    def set_end_frame(self, F1):
        """Fix the frame at t = 1 (e.g. so a second fort lands where wanted)."""
        K1 = F1 * dz_dc(self.centre(1.0), 1.0, self.p)
        self.K1 = K1
        a0, a1 = np.angle(self.K0), np.angle(K1)
        a1 += 2 * math.pi * round((a0 - a1) / (2 * math.pi))
        self.a1 = a1
        self.l1 = math.log(abs(K1))

    def centre(self, t):
        return complex(np.interp(t, self.ts, self.cs.real),
                       np.interp(t, self.ts, self.cs.imag))

    def _calibrate(self, t, arms):
        c = self.centre(t)
        size, _, _ = fort_scale(c, t, self.p)
        D = dz_dc(c, t, self.p)
        K = size * abs(D) * D / abs(D)        # frame = size * conj-free unit
        frame = K / D
        phi = family_phase(c, t, frame, arms, (2.0, 3.5))
        # rotate so the arm at phi lands at +90 degrees
        return K * np.exp(1j * (phi - math.pi / 2))

    def frame(self, t):
        """Complex scale+rotation mapping fort-units to c at parameter t."""
        s = t * t * (3 - 2 * t)
        K = math.exp(self.l0 + (self.l1 - self.l0) * s) * \
            np.exp(1j * (self.a0 + (self.a1 - self.a0) * s))
        fr = K / dz_dc(self.centre(t), t, self.p)
        if getattr(self, "stab", None) is not None:
            ts, corr = self.stab
            # measured correction, less its linear part (the ends are pinned)
            c = complex(np.interp(t, ts, corr.real), np.interp(t, ts, corr.imag))
            fr *= np.exp(c - corr[-1] * s)
        return fr

    def use_stabilisation(self, ts, corr, resid):
        """Adopt a measured frame correction and an even-change t spacing."""
        self.stab = (ts, corr)
        cum = np.concatenate([[0], np.cumsum(0.5 * (resid[1:] + resid[:-1]) + 1e-3)])
        self.morph = (ts, cum / cum[-1])

    def t_of(self, s):
        """Map an even spatial fraction s (0..1) to t, slowing where the picture changes fast."""
        if getattr(self, "morph", None) is None:
            return s
        ts, cum = self.morph
        return np.interp(s, cum, ts)


def _exists(path):
    import os
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Stabilising the frame from the pictures themselves
# ---------------------------------------------------------------------------

def _logpolar(c0, t, frame, nr=96, nth=256, r0=0.6, r1=12.0, max_iter=4000):
    """Ink density around a nucleus sampled on a log-polar grid (r in fort units)."""
    lr = np.linspace(math.log(r0), math.log(r1), nr)
    th = np.linspace(0, 2 * math.pi, nth, endpoint=False)
    R, TH = np.meshgrid(np.exp(lr), th, indexing="ij")
    w = R * np.exp(1j * TH)
    C = c0 + frame * w
    mu, de, *_ = render_family(np.ascontiguousarray(C.real), np.ascontiguousarray(C.imag),
                               np.full(C.shape, float(t)), max_iter, 1000.0)
    # local pixel size in c on this grid ~ r * dlogr * |frame|
    pix = R * (lr[1] - lr[0]) * abs(frame)
    ink = np.where(mu < 0, 1.0, np.exp(-np.maximum(de, 0) / (2 * pix)))
    return ink - ink.mean(), lr[1] - lr[0], th[1] - th[0]


def _phase_shift(a, b):
    """Sub-pixel shift (dy, dx) such that b ~ a shifted by it (periodic in x)."""
    Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
    X = Fb * np.conj(Fa)
    X /= np.abs(X) + 1e-12
    r = np.fft.ifft2(X).real
    j, i = np.unravel_index(np.argmax(r), r.shape)
    def sub(v, k, n):
        vm, v0, vp = v[(k - 1) % n], v[k], v[(k + 1) % n]
        den = vm - 2 * v0 + vp
        return k + (0.5 * (vm - vp) / den if den != 0 else 0.0)
    ny, nx = r.shape
    dy, dx = sub(r[:, i], j, ny), sub(r[j, :], i, nx)
    if dy > ny / 2: dy -= ny
    if dx > nx / 2: dx -= nx
    return dy, dx


def stabilise(fort, n=241):
    """
    Measure how the picture around the fort turns and swells as t moves,
    and fold that into the frame so the fort's surroundings stay as still
    as possible in fort-relative coordinates.  Also returns the residual
    rate of change (for spacing t out evenly across the jacket).
    """
    ts = np.linspace(0, 1, n)
    corr = np.zeros(n, complex)      # accumulated log(scale) + i*angle
    resid = np.zeros(n)
    prev = None
    for k, t in enumerate(ts):
        fr = fort.frame(t) * np.exp(corr[k - 1] if k else 0)
        img, dlr, dth = _logpolar(fort.centre(t), t, fr)
        if prev is not None:
            dy, dx = _phase_shift(prev, img)
            # features moved outward by dy*dlr in log r and round by dx*dth;
            # follow them by scaling/rotating the frame the same way
            step = complex(dy * dlr, dx * dth)
            corr[k] = corr[k - 1] + step
            fr2 = fort.frame(t) * np.exp(corr[k])
            img2, *_ = _logpolar(fort.centre(t), t, fr2)
            resid[k] = float(np.sqrt(((img2 - prev) ** 2).mean()))
            img = img2
        prev = img
    return ts, corr, resid
