"""
Diagram: the Hashimoto et al. mapping — a neural network whose depth IS the
holographic radial dimension, boundary data at the input, black-hole horizon
at the output.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch

OUT = "/Users/sumit/Github/BlackHole"
fig, ax = plt.subplots(figsize=(12, 7))

n_layers = 6
xs = np.linspace(1.5, 9.5, n_layers)
n_nodes = 4
ys = np.linspace(1.8, 4.6, n_nodes)

# spacetime gradient background: boundary (left, blue) -> deep bulk (right, red)
for i in range(60):
    x0 = 0.8 + i*(9.6-0.8)/60
    ax.add_patch(Rectangle((x0, 1.2), (9.6-0.8)/60, 4.0,
                 color=plt.cm.coolwarm(i/60), alpha=0.18, lw=0))

# boundary wall (left)
ax.add_patch(Rectangle((0.55, 1.2), 0.25, 4.0, color="#1f3a93"))
ax.text(0.67, 5.45, "BOUNDARY", ha="center", fontsize=11, fontweight="bold", color="#1f3a93")
ax.text(0.67, 0.9, "quantum system,\nno gravity\n(your data lives here)", ha="center", fontsize=8.5, color="#1f3a93")

# horizon (right)
ax.add_patch(Rectangle((9.85, 1.2), 0.25, 4.0, color="black"))
ax.text(9.97, 5.45, "HORIZON", ha="center", fontsize=11, fontweight="bold")
ax.text(9.97, 0.9, "black hole,\ndeep in the bulk\n(end of the network)", ha="center", fontsize=8.5)

# network: nodes and connections
for li, x in enumerate(xs):
    for y in ys:
        ax.add_patch(Circle((x, y), 0.14, facecolor="white", edgecolor="#0b3d1f", lw=1.6, zorder=5))
    if li < n_layers-1:
        for y1 in ys:
            for y2 in ys:
                ax.plot([x+0.14, xs[li+1]-0.14], [y1, y2], color="#0b3d1f", lw=0.5, alpha=0.45, zorder=4)

# weight labels = metric h(eta)
for li in range(n_layers-1):
    xm = (xs[li]+xs[li+1])/2
    ax.text(xm, 4.95, f"W{li+1} = h(η{li+1})", ha="center", fontsize=9, color="#7b241c", fontweight="bold")
ax.text(5.5, 5.55, "the WEIGHTS are the geometry — one number per layer: the metric h(η)",
        ha="center", fontsize=10.5, color="#7b241c", fontweight="bold")

# input/output arrows
ax.annotate("", xy=(1.35, 3.2), xytext=(0.82, 3.2), arrowprops=dict(arrowstyle="-|>", color="#1f3a93", lw=2.2))
ax.text(0.45, 3.45, "input:\nfield value &\nits slope\n(φ, π)", ha="right", fontsize=9, color="#1f3a93")
ax.annotate("", xy=(9.82, 3.2), xytext=(9.65, 3.2), arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2))
ax.text(10.7, 3.35, "output: does it\nend cleanly at\na horizon?\n→ the LOSS", ha="left", fontsize=9)

# depth axis = radial direction
ax.annotate("", xy=(9.5, 0.35), xytext=(1.5, 0.35), arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=2.5))
ax.text(5.5, 0.05, "network DEPTH  =  the holographic radial direction η  (zoom level → direction of space)",
        ha="center", fontsize=11, fontweight="bold", color="#34495e")

ax.text(5.5, 6.15, "A NEURAL NETWORK THAT *IS* A SPACETIME  (Hashimoto et al. 2018)",
        ha="center", fontsize=13, fontweight="bold")

ax.set_xlim(-1.2, 12.4); ax.set_ylim(-0.4, 6.5)
ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/nn_spacetime.png", dpi=130); plt.close(fig)
print("done: nn_spacetime.png")
