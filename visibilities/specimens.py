"""
Named mini-forts ("specimens") for the bestiary.

Each is a nucleus of z^d + c of the given period.  'zoom' is the cover's
frame half-width measured in fort sizes: ~3 shows the fort and its inner
palisade, ~6-10 the ring of d arms, larger values the d-fold embedded Julia
halos.  Find more with  python -m visibilities.explore hunt ...
"""

SPECIMENS = {
    # --- Heptabrot (z^7 + c): hexagonal forts, 7-fold rings ---------------
    "hepta-snowflake": dict(d=7, period=10, c=(0.948646674202614, 0.589594547258562),
                            zoom=7.0, note="clean seven-armed star round a hexagonal keep"),
    "hepta-citadel": dict(d=7, period=15, c=(0.937439826484936, 0.577494348019577),
                          zoom=4.0, note="large hexagon behind a dense palisade; seven bastioned arms"),
    "hepta-warren": dict(d=7, period=10, c=(0.944405965659876, 0.578050838530890),
                         zoom=4.0, note="hexagon with a thick glacis of thorns"),
    # --- Octabrot (z^8 + c): heptagonal forts, 8-fold rings ---------------
    "octa-palisade": dict(d=8, period=14, c=(0.950929087748376, 0.525665421294712),
                          zoom=6.0, note="heptagonal keep inside a ring of stakes; eight arms"),
    "octa-rose": dict(d=8, period=11, c=(0.951390145688302, 0.525605380745360),
                      zoom=5.0, note="heptagon with an eight-petalled rose of filaments"),
    "octa-sentinel": dict(d=8, period=11, c=(0.956584435340929, 0.515085102023858),
                          zoom=5.0, note="heptagonal fort, eight radiating causeways"),
}
