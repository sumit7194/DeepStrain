"""
Two diagrams:
  A) collapse.png      - Penrose diagram of a REAL black hole formed by collapse
                         (one universe, no white hole, no second universe)
  B) many_holes.png    - one universe containing many black holes (the real picture)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch, FancyBboxPatch

OUT = "/Users/sumit/Github/BlackHole"

# ---------------------------------------------------------------- A: collapse
def collapse():
    fig, ax = plt.subplots(figsize=(9.5, 9))
    # key points
    iL_bot = (0,-1.4)          # center r=0 bottom (i-)
    sing_L = (0, 0.8)          # singularity left end (r=0 meets singularity)
    sing_R = (1.0, 0.8)        # singularity right end (-> i+)
    H0     = (0,-0.2)          # horizon forms on r=0 axis
    iplus  = (1.0, 0.8)        # future timelike infinity
    i0     = (2.0,-0.2)        # spatial infinity
    iminus_null = (0.6,-1.4)

    # region II (black hole interior): above horizon, below singularity
    ax.add_patch(Polygon([H0, iplus, sing_L], closed=True, color="#fde2e2"))
    # region I (our universe): right of horizon, inside null infinities
    ax.add_patch(Polygon([H0, iplus, i0, iminus_null], closed=True, color="#e2ecfb"))
    # collapsing star (shaded): between center axis and star surface
    sx = np.array([0.85,0.78,0.62,0.42,0.22,0.0])
    sy = np.array([-1.4,-1.0,-0.5,0.0,0.45,0.8])
    star = list(zip(sx,sy)) + [(0,0.8),(0,-1.4)]
    ax.add_patch(Polygon(star, closed=True, color="#fff3cd"))
    ax.plot(sx,sy, color="#b8860b", lw=2)

    # singularity (jagged)
    xj = np.linspace(0,1.0,18)
    ax.plot(xj, 0.8+0.03*np.array([(-1)**i for i in range(18)]), color="black", lw=3)
    # event horizon (45 deg dashed)
    ax.plot([H0[0],iplus[0]],[H0[1],iplus[1]], color="#c0392b", ls="--", lw=2)
    # null infinities and axis
    ax.plot([iplus[0],i0[0]],[iplus[1],i0[1]], color="#34495e", lw=1.5)   # scri+
    ax.plot([i0[0],iminus_null[0]],[i0[1],iminus_null[1]], color="#34495e", lw=1.5)  # scri-
    ax.plot([0,0],[iL_bot[1],sing_L[1]], color="#7f8c8d", lw=1.5)          # r=0 axis

    # labels
    ax.text(1.15,0.45,"BLACK HOLE\nINTERIOR", ha="center", fontsize=10, fontweight="bold", color="#922b21")
    ax.text(1.25,-0.25,"OUR UNIVERSE\n(the only one)", ha="center", fontsize=11, fontweight="bold", color="#1f3a93")
    ax.text(0.33,-0.55,"collapsing\nstar", ha="center", fontsize=9, color="#7a5c00", rotation=58)
    ax.text(0.5,1.0,"SINGULARITY (forms in the future)", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.62,0.45,"event\nhorizon", color="#c0392b", fontsize=9, rotation=45)
    ax.text(2.02,-0.2,"  spatial ∞", fontsize=8, color="#34495e")
    ax.text(1.55,0.42,"light → ∞", fontsize=8, color="#34495e", rotation=-45)

    # the two big "NO" callouts
    ax.annotate("NO second universe here\n(left side is just the\nstar's own center, r=0)",
                xy=(0,-0.7), xytext=(-1.25,-0.5), fontsize=9.5, color="#7b241c", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#7b241c"))
    ax.annotate("NO white hole here\n(bottom is just the\nstar before it collapsed)",
                xy=(0.4,-1.25), xytext=(-1.25,-1.6), fontsize=9.5, color="#1e7d32", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#1e7d32"))

    ax.set_xlim(-1.6,2.4); ax.set_ylim(-2.0,1.4)
    ax.set_title("A REAL BLACK HOLE (formed by a collapsing star)", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{OUT}/collapse.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- B: many holes
def many():
    fig, ax = plt.subplots(figsize=(11, 7))
    # the one universe
    ax.add_patch(FancyBboxPatch((0.3,0.5),11.4,6.0, boxstyle="round,pad=0.1,rounding_size=0.4",
                                facecolor="#eef3fb", edgecolor="#1f3a93", lw=2.5))
    ax.text(6.0,6.05,"OUR ONE UNIVERSE  (a single spacetime)", ha="center", fontsize=13,
            fontweight="bold", color="#1f3a93")
    # scatter many black holes
    rng = [(2.0,4.6,0.32),(3.6,2.3,0.24),(5.1,4.9,0.20),(6.4,3.2,0.36),(7.9,5.0,0.22),
           (8.8,2.4,0.30),(10.2,4.3,0.26),(4.4,5.4,0.16),(9.6,5.7,0.14),(2.7,1.6,0.18),
           (6.9,1.5,0.20),(10.6,2.0,0.16)]
    for i,(x,y,r) in enumerate(rng):
        ax.add_patch(Circle((x,y), r*1.5, color="#f4c7c3", alpha=0.5))   # horizon glow
        ax.add_patch(Circle((x,y), r, color="black"))                    # the hole
    ax.text(2.0,4.6+0.55,"BH", ha="center", fontsize=8, color="#922b21")
    ax.text(6.4,3.2+0.62,"BH", ha="center", fontsize=8, color="#922b21")
    ax.text(11.0,1.0,"…and billions more", ha="right", fontsize=10, style="italic", color="#555")

    # "you"
    ax.add_patch(Circle((1.2,3.0),0.12, color="#8e44ad"))
    ax.text(1.2,2.6,"you", ha="center", fontsize=10, color="#6c3483", fontweight="bold")
    for (x,y,r) in [rng[0],rng[1],rng[3],rng[9]]:
        ax.add_patch(FancyArrowPatch((1.4,3.05),(x-r*1.4,y), arrowstyle="-|>",
                     mutation_scale=12, color="#8e44ad", alpha=0.6, ls=(0,(4,3))))
    ax.text(6.0,0.15,"Many black holes, ONE universe. Each is a local dent you can fall into —\n"
                     "NOT a doorway that creates a new universe.",
            ha="center", fontsize=11, style="italic")

    ax.set_xlim(0,12); ax.set_ylim(-0.4,6.6)
    ax.set_title("THE REAL PICTURE: one universe, many black holes", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{OUT}/many_holes.png", dpi=130); plt.close(fig)

collapse(); many()
print("done: collapse.png, many_holes.png")
