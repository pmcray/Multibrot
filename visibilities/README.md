# The Visibilities: cover plates from the Heptabrot and Octabrot

This package draws engraved, grimoire-style plates of the Multibrot sets
**z⁷ + c** (Heptabrot) and **z⁸ + c** (Octabrot), and lays them out as a book
cover. Each plate is centred on a mini-fort: a small copy of the set's main
body, which is the counterpart of the Mandelbrot "beetle".

## The tension the images show

| | Heptabrot (d = 7) | Octabrot (d = 8) |
|---|---|---|
| Fort (main body and every mini-copy) | hexagonal, 6 bastions | heptagonal, 7 bastions |
| Ring of arms around each mini-fort | 7-fold | 8-fold |

* The fort is a (d−1)-cusped epicycloid. That is why its symmetry is one less
  than the degree.
* The ring of arms comes from the critical point z = 0. Because z^d has local
  degree d there, every place where the parameter plane echoes the dynamical
  plane branches d-fold. This covers the embedded Julia sets and the halos
  around each mini-fort.

As a result, a 6-sided keep sits inside a 7-armed star, and a 7-sided keep
sits inside an 8-armed star. The cover's seal states this outright: d red
stars sit where the arms cross the ring, and d−1 gilt studs sit on the
bastion directions. Both are measured from the picture, not hard-coded
(`engine.fold_phase`).

How the drawing is built:

* **Ink.** The exterior distance estimate draws the filaments as crisp lines
  at any zoom.
* **Outworks.** Level lines of the smooth escape count form nested copies of
  the fort's outline, like the ravelins of a bastioned fortress.
* **Hatching.** A stripe average at frequency d puts the d-fold structure
  into the wash.
* **The fort.** The interior is filled in red lead with a gilt rim.

## Quick start

```bash
pip install -r visibilities/requirements.txt

# a finished 6x9in cover with 0.125in bleed at 300 dpi (takes about a minute on 4 cores)
python -m visibilities.cover hepta-snowflake --zoom 5 \
       --subtitle "A Bestiary of the Heptabrot" --plate plate.png --out cover.png

python -m visibilities.cover octa-palisade --palette nigredo --author "Your Name"
```

Options: `--palette vellum|nigredo|lapis|verdigris`, `--zoom` (frame
half-width in fort sizes: about 3 shows the fort and its palisade, 5–8 the
ring of arms, 20+ the outer halos), `--turn` (degrees), `--trim 6x9`,
`--dpi`, `--ss` (supersampling), `--no-seal`, `--fort-y` (vertical position),
`--plate` (also save the bare image without any type).

The fonts are IM Fell English, from Google Fonts under the SIL OFL. They are
downloaded once into `visibilities/fonts/`. Without a network connection the
code falls back to a system serif.

## Finding new beasts

```bash
# low-period island forts in one symmetry sector
python -m visibilities.explore parents 8

# hunt the arms of a parent for deep mini-forts and print a contact sheet
python -m visibilities.explore hunt 7 --parent 3,0.942364005881867,0.586910617077306 \
       --seeds 8 --depth 2e-3 --seed 4 --mults 2.5,8,30,120 --sheet hunt.png
```

`hunt` picks points on the parent's filaments. For each one it finds the
lowest-period nucleus nearby by Newton's method on f_c^p(0) = 0, then
estimates the size and orientation of that mini-fort from the renormalised
return map:

    f^p(z) ≈ L·z^d + D·(c − c₀)   ⇒   size = 1 / |D · L^(1/(d−1))|

Copy any good find into `specimens.py` and it becomes available to
`cover.py`. A smaller `--depth` gives deeper, higher-period forts, which have
more layers of d-fold halo. The engine uses float64, which is good to a frame
width of about 1e-12.

## Files

* `engine.py`: the numba kernels (escape count, distance estimate, stripe
  average, orbit trap), the nucleus/period/size finders, and the phase
  detection for arms and bastions.
* `grimoire.py`: the palettes, the procedural vellum, and how the plate is
  layered together.
* `render.py`: `render_plate()` for using the engine directly from a
  notebook.
* `explore.py`: the parent and hunt search tools and the contact sheets.
* `cover.py`: the cover layout (frame, corner roundels {d/3} with the
  (d−1)-gon inside, seal, title cartouche).
* `specimens.py`: the named mini-forts.

## The wrap-around case: joining the Heptabrot and the Octabrot

The Heptabrot and the Octabrot *can* be joined seamlessly. A fractional
power such as z^7.5 does tear the picture, but only because it needs a
branch cut. The join does not need fractional powers. `bridge.py` uses the
family

    f_t(z) = (1 − t) z⁷ + t z⁸ + c        (t = 0 Heptabrot, t = 1 Octabrot)

Every point iterates an ordinary polynomial, so there is no branch cut, and
the picture depends continuously on t. The family has a second critical
point, z* = −7(1 − t)/(8t). As t → 1 it comes in from infinity and merges
with z = 0, and that merger is when the eighth fold is born.

**The same creature on both boards.** Newton's method on f_t^p(0) = 0
follows a fort's nucleus through the family. The front fort,
`hepta-snowflake` (period 10, hexagonal, 7 arms), stays a period-10 fort all
the way to t = 1 and ends up heptagonal with 8 arms, at about the same size.
See `gallery/transfiguration_*.png`. On these strips the eighth arm appears
between t ≈ 0.55 and 0.7.

**The case** (`jacket.py`, default `--mode pair`) is one flat, conformal
plate. Fort A is on the front at t = 0. A carried to t = 1 sits in the arms
of a large period-7 heptagonal Octabrot fort, B, which takes the back. A
gentle steady zoom and twist across the jacket (exp(−λζ) with complex λ)
put B at the centre of the back board with an arm pointing up. There is no
singular point anywhere. t goes from 0 at the front fort's centre to 1 at
the back fort's centre. It is spaced by the measured rate of change of the
picture (`bridge.stabilise`), so the burst of change around t ≈ 0.55 is
spread out rather than crammed into one place.

```bash
python -m visibilities.jacket --blend 3.1 --dpi 150 --guides --out jacket.png
#   --board 6.25x9.5 --spine 1.25 --wrap 0.75   (inches; set to the real case)
#   --zoom (front fort sizes per inch)  --back-radius (back fort radius, in)
```

What happens at the spine, and why. The two forts drift relative to each
other as t changes: a tiny fort moves tens of thousands of its own sizes
between t = 0 and t = 1. Multibrot sets are connected, so filaments must
cross any line drawn between the two forts. Those filaments are therefore
combed out into a flowing, marbled current across the spine. It is
continuous: a sweep, not a tear. Two other layouts were tried and are kept
for reference:

* `--mode infinity`: the spine is the point at infinity.
* `--mode branch`: the spine is a branch point.

Mathematically, if both boards show the same fort, the spine *must* contain
one or the other (Riemann–Hurwitz). Both magnify the spine strongly, and
the change of t then streaks it (`gallery/rejected_infinity_spine.png`).

Large forts (the period-3 islands) drift only 5–14 of their own sizes. They
would give a nearly invisible join, but they have much less filigree than
the deep forts.

Still to do: extra horizontal supersampling in the transition zone, to clean
up ladder-like aliasing on the most swept filaments, and a tiled renderer so
that a 300 dpi run of the full case fits in memory.
