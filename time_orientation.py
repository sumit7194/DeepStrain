"""
Diagram 4: time orientation of the four regions of the eternal Schwarzschild
spacetime. Shows that future light cones open UP in ALL four regions, plus the
two mirror symmetries (time-reverse: BH<->WH; space: our<->other universe).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = "/Users/sumit/Github/BlackHole"
fig, ax = plt.subplots(figsize=(10, 9))

# region fills
ax.add_patch(Polygon([(0,0),(1,1),(2,0),(1,-1)], color="#e2ecfb"))    # I  our universe
ax.add_patch(Polygon([(0,0),(-1,1),(-2,0),(-1,-1)], color="#ede7f6")) # III other universe
ax.add_patch(Polygon([(0,0),(1,1),(-1,1)], color="#fde2e2"))          # II  black hole
ax.add_patch(Polygon([(0,0),(1,-1),(-1,-1)], color="#e8f8e8"))        # IV  white hole

# horizons
for (a,b) in [((-1,1),(1,-1)), ((-1,-1),(1,1))]:
    ax.plot([a[0],b[0]],[a[1],b[1]], color="#c0392b", ls=":", lw=1.5, alpha=0.7)
# outer null boundaries
for seg in [((2,0),(1,1)),((2,0),(1,-1)),((-2,0),(-1,1)),((-2,0),(-1,-1))]:
    ax.plot([seg[0][0],seg[1][0]],[seg[0][1],seg[1][1]], color="#34495e", lw=1.3)
# singularities
xj = np.linspace(-1,1,25)
ax.plot(xj, 1+0.035*np.array([(-1)**i for i in range(25)]), color="black", lw=3)
ax.plot(xj, -1+0.035*np.array([(-1)**i for i in range(25)]), color="#555", lw=2.5)

# upward future light cones in every region
def upcone(cx, cy, sz=0.17):
    ax.add_patch(Polygon([(cx,cy),(cx-sz,cy+1.3*sz),(cx+sz,cy+1.3*sz)],
                         closed=True, facecolor="#27ae60", alpha=0.75, edgecolor="#0b3d1f", lw=1.5))
    ax.annotate("", xy=(cx, cy+1.55*sz), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#0b3d1f", lw=1.6))
for (cx,cy) in [(1.15,-0.12),(-1.15,-0.12),(0,0.45),(0,-0.78)]:
    upcone(cx,cy)

# the two mirror axes
ax.plot([-2.05,2.05],[0,0], color="#e67e22", ls="--", lw=2)
ax.plot([0,0],[-1.05,1.05], color="#1565c0", ls="--", lw=2)

# region labels
ax.text(1.25,0.2,"OUR\nUNIVERSE", ha="center", fontsize=10, fontweight="bold", color="#1f3a93")
ax.text(-1.25,0.2,"OTHER\nUNIVERSE", ha="center", fontsize=10, fontweight="bold", color="#5e35b1")
ax.text(0,0.72,"BLACK HOLE", ha="center", fontsize=10, fontweight="bold", color="#922b21")
ax.text(0,-0.5,"WHITE HOLE", ha="center", fontsize=10, fontweight="bold", color="#1e7d32")
ax.text(0,1.17,"future singularity", ha="center", fontsize=9)
ax.text(0,-1.2,"past singularity", ha="center", fontsize=9, color="#555")

# "future = up" callouts on each cone
for (cx,cy) in [(1.15,-0.12),(-1.15,-0.12)]:
    ax.text(cx+0.33, cy+0.12, "future ↑", fontsize=8, color="#0b3d1f")
ax.text(0.27,0.55,"future ↑", fontsize=8, color="#0b3d1f")
ax.text(0.27,-0.68,"future ↑", fontsize=8, color="#0b3d1f")

# mirror labels
ax.text(2.1,0.06,"TIME-REVERSE mirror\nblack hole ↔ white hole", fontsize=9,
        color="#b8651b", va="bottom", fontweight="bold")
ax.text(0.05,1.32,"SPACE mirror\nour ↔ other universe", fontsize=9,
        color="#1565c0", ha="left", fontweight="bold")

# master-clock arrows (the grain of truth): up in I, down in III
ax.annotate("", xy=(1.7,0.32), xytext=(1.7,-0.32), arrowprops=dict(arrowstyle="-|>", color="#1f3a93", lw=2.2))
ax.text(1.74,0.0,"geometry's\nclock ↑", fontsize=8, color="#1f3a93")
ax.annotate("", xy=(-1.7,-0.32), xytext=(-1.7,0.32), arrowprops=dict(arrowstyle="-|>", color="#5e35b1", lw=2.2))
ax.text(-2.05,0.0,"geometry's\nclock ↓", fontsize=8, color="#5e35b1")

ax.text(0,-1.62,"Physical future (light cones, aging) points UP in ALL four regions.\n"
                "The only thing that 'reverses' in the other universe is the geometry's master clock.",
        ha="center", fontsize=10, style="italic")

ax.set_xlim(-2.6,2.7); ax.set_ylim(-1.8,1.6)
ax.set_title("WHICH WAY DOES TIME RUN IN EACH REGION?", fontsize=13, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(f"{OUT}/time_orientation.png", dpi=130); plt.close(fig)
print("done: time_orientation.png")
