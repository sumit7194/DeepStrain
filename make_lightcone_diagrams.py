"""
Generate three teaching diagrams about light cones and black holes:
  1) lightcone_basic.png   - what a light cone is
  2) tipping_cones.png     - light cones tipping toward a black hole
  3) penrose.png           - black hole / white hole causal map
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = "/Users/sumit/Github/BlackHole"

# ----------------------------------------------------------------------
# Figure 1: What a light cone is
# ----------------------------------------------------------------------
def fig_basic():
    fig, ax = plt.subplots(figsize=(8, 8))
    R = 5
    # forbidden (faster-than-light) left & right wedges
    ax.add_patch(Polygon([(0,0),( R, R),( R,-R)], closed=True, color="#f8d7da"))  # right
    ax.add_patch(Polygon([(0,0),(-R, R),(-R,-R)], closed=True, color="#f8d7da"))  # left
    # future cone (allowed)
    ax.add_patch(Polygon([(0,0),(-R,R),(R,R)], closed=True, color="#d4edda"))
    # past cone
    ax.add_patch(Polygon([(0,0),(-R,-R),(R,-R)], closed=True, color="#e2e3e5"))
    # light rays (45 deg)
    ax.plot([-R,R],[-R,R], color="#e0a800", ls="--", lw=2)
    ax.plot([R,-R],[-R,R], color="#e0a800", ls="--", lw=2)
    # a worldline (your possible path) inside the future cone
    yy = np.linspace(0,R,100); xx = 0.9*np.sin(yy*1.1)
    ax.plot(xx,yy, color="#155724", lw=2.5)
    ax.annotate("", xy=(xx[-1],yy[-1]), xytext=(xx[-3],yy[-3]),
                arrowprops=dict(arrowstyle="-|>", color="#155724", lw=2.5))
    # the event
    ax.plot(0,0,"o", color="black", ms=10)
    # labels
    ax.text(0,-0.55,"YOU, NOW", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0,4.3,"FUTURE\n(everywhere you're allowed to go)", ha="center",
            fontsize=12, color="#155724", fontweight="bold")
    ax.text(0,-4.3,"PAST\n(where you came from)", ha="center",
            fontsize=11, color="#383d41")
    ax.text(4.4,0,"FORBIDDEN\n(faster than\nlight)", ha="center", va="center",
            fontsize=10, color="#a71d2a", fontweight="bold")
    ax.text(-4.4,0,"FORBIDDEN\n(faster than\nlight)", ha="center", va="center",
            fontsize=10, color="#a71d2a", fontweight="bold")
    ax.text(2.55,4.2,"path of light", color="#b8860b", fontsize=10, rotation=45)
    ax.text(0.95,3.0,"your path", color="#155724", fontsize=10, rotation=70)
    ax.set_xlim(-R,R); ax.set_ylim(-R,R)
    ax.set_xlabel("space  →", fontsize=12); ax.set_ylabel("time  →  (future is up)", fontsize=12)
    ax.set_title("A LIGHT CONE: the set of places your future can reach", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{OUT}/lightcone_basic.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: Light cones tipping toward a black hole
# ----------------------------------------------------------------------
def fig_tipping():
    fig, ax = plt.subplots(figsize=(11, 7))
    rs = 4.0          # horizon radius
    L  = 0.95         # cone size
    base = 1.0        # baseline time for cones
    # background regions
    ax.axvspan(0, rs, color="#fde2e2")      # inside
    ax.axvspan(rs, 11, color="#e2ecfb")     # outside
    ax.axvline(rs, color="#c0392b", ls="--", lw=2)
    ax.axvline(0,  color="black", lw=3)
    # cones at several radii
    for r0 in [1.0, 2.5, 4.0, 5.5, 7.5, 9.5]:
        s = (r0 - rs) / (r0 + rs)          # outgoing-light horizontal step per unit time
        apex = (r0, base)
        left  = (r0 - L,        base + L)  # ingoing light: always toward smaller r
        right = (r0 + s*L,      base + L)  # outgoing light: escapes / frozen / falls in
        ax.add_patch(Polygon([apex,left,right], closed=True,
                             facecolor="#2ecc71" if r0>rs else "#27ae60",
                             alpha=0.55, edgecolor="#145a32", lw=1.5))
        # arrow showing general future direction (cone centerline)
        cx = (left[0]+right[0])/2; cy = base + L
        ax.annotate("", xy=(cx, cy+0.15), xytext=apex,
                    arrowprops=dict(arrowstyle="-|>", color="#0b3d1f", lw=1.8))
    # an infalling observer worldline
    rr = np.linspace(10.2, 0.05, 200)
    tt = base + 0.18*(10.2-rr) + 0.02*(10.2-rr)**1.6
    ax.plot(rr, tt, color="#8e44ad", lw=2.8)
    ax.annotate("", xy=(rr[-1],tt[-1]), xytext=(rr[-6],tt[-6]),
                arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=2.8))
    ax.text(10.6, base+0.35, "you,\nfalling in", color="#6c3483", fontsize=10)
    # labels
    ax.text(rs+0.15, 6.4, "EVENT HORIZON\n(r = r_s)", color="#c0392b", fontsize=11, fontweight="bold")
    ax.text(1.15, 6.5, "SINGULARITY\n(r = 0)", color="black", ha="center", fontsize=11, fontweight="bold")
    ax.text(8.5, 0.2, "OUTSIDE", color="#2c3e50", fontsize=12, fontweight="bold")
    ax.text(1.4, 0.2, "INSIDE", color="#922b21", fontsize=12, fontweight="bold")
    ax.text(9.5, 3.0, "cone upright:\nfuture can go\nin OR out", ha="center", fontsize=9, color="#145a32")
    ax.text(4.0, 5.0, "at horizon:\noutgoing edge\nstraight up\n(light frozen)", ha="center", fontsize=9, color="#145a32")
    ax.text(1.0, 3.4, "tipped over:\nEVERY future\nleads to r = 0", ha="center", fontsize=9, color="#7b241c", fontweight="bold")
    ax.set_xlim(11, -0.4)   # reversed: r decreases to the right toward singularity on the left
    ax.set_ylim(0, 7.2)
    ax.set_xlabel("radius  r   (distance from center)  —  singularity at left, far away at right", fontsize=11)
    ax.set_ylabel("time  →  (future is up)", fontsize=11)
    ax.set_title("LIGHT CONES TIP OVER AS YOU APPROACH A BLACK HOLE", fontsize=13, fontweight="bold")
    ax.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/tipping_cones.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: Penrose (causal) map  - black hole + white hole + 2 universes
# ----------------------------------------------------------------------
def fig_penrose():
    fig, ax = plt.subplots(figsize=(9, 9))
    # region fills
    ax.add_patch(Polygon([(0,0),(1,1),(2,0),(1,-1)], color="#e2ecfb"))   # I  right exterior
    ax.add_patch(Polygon([(0,0),(-1,1),(-2,0),(-1,-1)], color="#ede7f6")) # III left exterior
    ax.add_patch(Polygon([(0,0),(1,1),(-1,1)], color="#fde2e2"))          # II  black hole
    ax.add_patch(Polygon([(0,0),(1,-1),(-1,-1)], color="#e8f8e8"))        # IV  white hole
    # horizons (diagonals through centre)
    for (a,b) in [((-1,1),(1,-1)), ((-1,-1),(1,1))]:
        ax.plot([a[0],b[0]],[a[1],b[1]], color="#c0392b", ls="--", lw=2)
    # outer null boundaries (to infinity)
    for seg in [((2,0),(1,1)),((2,0),(1,-1)),((-2,0),(-1,1)),((-2,0),(-1,-1))]:
        ax.plot([seg[0][0],seg[1][0]],[seg[0][1],seg[1][1]], color="#34495e", lw=1.5)
    # singularities (jagged horizontal lines top & bottom)
    xj = np.linspace(-1,1,25); yj_top = 1 + 0.04*np.array([(-1)**i for i in range(25)])
    yj_bot = -1 + 0.04*np.array([(-1)**i for i in range(25)])
    ax.plot(xj, yj_top, color="black", lw=3)
    ax.plot(xj, yj_bot, color="#555", lw=2.5)
    # infalling worldline: region I -> cross horizon -> hit top singularity
    wx = np.array([1.55,1.2,0.85,0.55,0.32,0.18]); wy = np.array([-0.35,-0.05,0.32,0.62,0.85,0.97])
    ax.plot(wx,wy, color="#8e44ad", lw=2.8)
    ax.annotate("", xy=(wx[-1],wy[-1]), xytext=(wx[-2],wy[-2]),
                arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=2.8))
    ax.plot(wx[0],wy[0],"o",color="#8e44ad",ms=7)
    # little 45-degree future light cones (V shapes) along the path
    def cone(x,y,sz=0.16,color="#0b3d1f"):
        ax.plot([x-sz,x,x+sz],[y+sz,y,y+sz], color=color, lw=1.8)
    cone(1.55,-0.35)        # in region I
    cone(0.55,0.62,color="#7b241c")  # just inside region II
    cone(0.85,0.32,color="#7b241c")  # crossing
    # labels
    ax.text(1.25,0,"OUR\nUNIVERSE\n(outside)", ha="center", va="center", fontsize=11, fontweight="bold", color="#1f3a93")
    ax.text(-1.25,0,"OTHER\nUNIVERSE\n(unreachable)", ha="center", va="center", fontsize=10, color="#5e35b1")
    ax.text(0,0.62,"BLACK HOLE\nINTERIOR", ha="center", va="center", fontsize=11, fontweight="bold", color="#922b21")
    ax.text(0,-0.62,"WHITE HOLE\n(time-reverse)", ha="center", va="center", fontsize=10, color="#1e7d32")
    ax.text(0,1.18,"SINGULARITY  —  your FUTURE (a TIME, not a place)", ha="center", fontsize=11, fontweight="bold")
    ax.text(0,-1.2,"white-hole singularity  (the past)", ha="center", fontsize=9, color="#555")
    ax.text(0.62,0.95,"event\nhorizon", color="#c0392b", fontsize=9, rotation=-45)
    ax.text(1.62,-0.45,"you start here", color="#8e44ad", fontsize=9)
    ax.text(1.75,0.55,"every 45° path\nin region I can\nstill go out", fontsize=8, color="#0b3d1f", ha="center")
    ax.text(-0.55,0.55,"inside: every 45°\npath hits the\ntop singularity", fontsize=8, color="#7b241c", ha="center")
    ax.set_xlim(-2.2,2.4); ax.set_ylim(-1.5,1.5)
    ax.set_title("THE CAUSAL MAP  (light always travels at 45°)", fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{OUT}/penrose.png", dpi=130); plt.close(fig)

fig_basic(); fig_tipping(); fig_penrose()
print("done: lightcone_basic.png, tipping_cones.png, penrose.png")
