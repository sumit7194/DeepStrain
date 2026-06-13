# Spacetime in 3+1 vs 2+1 Dimensions

*A concept-by-concept map of our world and its flattened twin.*

---

## 0. The premise

We live in **3+1 spacetime**: three space dimensions (call them *x, y, z*) plus one
time dimension *t*. The shorthand "3+1" keeps space and time separate on purpose,
because they are not interchangeable — you can stand still in space but you cannot
stand still in time.

Three facts frame everything below:

1. **Time stays 1-dimensional in both worlds.** Time is the "easy" axis — it is one
   number, it only goes forward, and *everything* we say about it (causality,
   before/after, the arrow of time) carries over to the 2D world **unchanged**. All
   the interesting differences live in *space*.

2. **We can reason rigorously about *lower* dimensions but not *higher* ones.** We can
   delete a spatial dimension and still picture and compute everything — that gives us
   **2+1 spacetime** (two space + one time), the world of *Flatland*. We cannot
   genuinely picture a *fourth* spatial dimension; we have the math but no intuition
   and barely any vocabulary. So this document only ever looks *downward*: 3+1 → 2+1
   (and occasionally → 1+1).

3. **"Fewer dimensions" is not the same as "just delete a coordinate."** This is the
   payoff of the whole exercise. Going from 3+1 to 2+1 does three different things to
   our concepts:
   - **Some shrink cleanly** (one fewer number to track). ✅ *parity*
   - **Some collapse or vanish** (the concept loses meaning). ⚠️ *degenerate*
   - **Some survive but behave qualitatively differently** (same idea, new physics).
     🔀 *different*

   The verdict column in the tables below uses these three tags.

---

## 1. Coordinates — the "address system"

Coordinates are just how you name a point. The number of coordinates **is** the
dimension; changing the *style* of coordinates never changes that count.

| | 3+1 world | 2+1 world |
|---|---|---|
| Straight-line (Cartesian) address of a point | (x, y, z) | (x, y) |
| Centered (polar/spherical) address | (r, θ, φ) — radius + 2 angles | (r, θ) — radius + 1 angle |
| An **event** (where *and* when) | (t, x, y, z) → 4 numbers | (t, x, y) → 3 numbers |

**Takeaway:** spherical/polar coordinates are the *same space* re-addressed. The honest
dimension count is the number of independent numbers: **3** for our space, **2** for
Flatland's. ✅ clean parity — just one fewer.

---

## 2. The master table

The big picture in one place. Details for the interesting rows follow in §3–§9.

| Concept | 3+1 (our world) | 2+1 (Flatland) | Verdict |
|---|---|---|---|
| Numbers to locate a point | 3 | 2 | ✅ fewer |
| Numbers to locate an event | 4 | 3 | ✅ fewer |
| Velocity / momentum / acceleration | 3 components | 2 components | ✅ fewer |
| Ways to *translate* (slide) | 3 | 2 | ✅ fewer |
| Ways to *rotate* | 3 (planes xy, yz, zx) | **1** (only the xy plane) | ✅ fewer |
| "Axis" of rotation | a line (the leftover 3rd dim) | **doesn't exist** | ⚠️ degenerate |
| Rigid-body degrees of freedom | 6 (3 slide + 3 turn) | 3 (2 slide + 1 turn) | ✅ fewer |
| Spacetime interval s² | −c²t² + x² + y² + z² | −c²t² + x² + y² | ✅ fewer terms |
| Light cone shape | hyper-cone (can't draw it) | an **ordinary cone** (drawable!) | 🔀 simpler to see |
| Time dilation / length contraction | yes | yes, identical formulas | ✅ same |
| A sharp pulse (flash, clap) | clean edge, heard once | leaves a **fading wake/tail** | 🔀 different |
| Force from a point source (gravity, charge) | ∝ 1/r² | ∝ 1/r | 🔀 different power |
| Potential of a point source | ∝ 1/r | ∝ ln(r) | 🔀 different shape |
| Gravity as a theory | rich: attraction, waves, black holes | **no local gravity, no waves** | ⚠️ degenerate |
| Can a closed boundary trap you? | need a 2D surface (wall) | a 1D loop (fence) suffices | 🔀 cheaper to trap |
| Knots in a string | exist (only in 3D!) | **impossible** | ⚠️ vanish |
| Left/right handedness (chirality) | real; needs a flip through 4D to undo | real; needs a flip through 3D to undo | 🔀 dim-dependent |
| Full angle "all around" | 4π steradians (a sphere) | 2π radians (a circle) | ✅ simpler |
| Causality, past/future, arrow of time | from the time axis | **identical** | ✅ same |

---

## 3. Directions, rotation, and degrees of freedom

This is the most surprising row, so it gets its own section.

**Translation (sliding)** is the intuitive one: 3 independent slide-directions in our
world, 2 in Flatland. Clean.

**Rotation** is where intuition misleads us. We think of a rotation as happening
*around an axis* (a line). That picture is a **coincidence of 3D**. A rotation really
happens *in a plane*; the "axis" is just the one leftover direction perpendicular to
that plane — and a perpendicular leftover direction only exists in 3D.

- The number of independent rotation planes in *n* spatial dimensions is **n(n−1)/2**.
  - n = 3 → **3** planes (xy, yz, zx). Three leftover perpendicular lines → we *can*
    name three rotation axes. (This is why a phone has 3-axis "roll/pitch/yaw.")
  - n = 2 → **1** plane (just xy). The "axis" would have to stick *out* of the plane,
    into a third dimension Flatlanders don't have. So a Flatlander can spin, but there
    is **no axis** — only a center point. ⚠️
  - n = 4 → 6 planes (a hint of why 4D breaks our intuition).

**Rigid-body degrees of freedom** (how robotics counts "DOF"):
- Our world: 3 slide + 3 turn = **6**.
- Flatland: 2 slide + 1 turn = **3**.

---

## 4. Relativity and the light cone

Special relativity is built on the **spacetime interval**, the one quantity all
observers agree on:

```
3+1:   s² = −c²t² + x² + y² + z²
2+1:   s² = −c²t² + x² + y²
```

Same structure, one fewer spatial term. Everything that depends only on time and the
direction of motion — **time dilation, length contraction, relativity of simultaneity,
E = mc²** — is **identical** in both worlds. ✅

The **light cone** (all the places a flash of light can reach) is the one piece that
actually gets *easier* to see in Flatland:

- **3+1:** the light cone is a 3-dimensional surface inside 4D spacetime. We literally
  cannot draw it; we draw cartoons.
- **2+1:** spacetime is (t, x, y) — just 3 axes — and the light cone is a genuine,
  honest **cone**. Every "light cone" diagram you've ever seen is secretly the 2+1
  case, because it's the highest dimension we can actually render. 🔀
- **1+1:** the cone degenerates further into two crossed lines — an "X."

---

## 5. Waves — and why a Flatlander shout would smear

Drop the same physics (a wave equation) into 2 space dimensions vs 3 and something
genuinely strange happens.

- **In 3 space dimensions, sharp signals stay sharp.** A clap or a flash has a clean
  leading edge *and* a clean trailing edge: you hear the clap once and then silence.
  (The technical name is that **Huygens' principle** holds in odd space dimensions ≥ 3.)
- **In 2 space dimensions, sharp signals develop a tail.** The leading edge still
  arrives on time, but behind it trails a **fading wake** that never quite cuts off —
  like the train of ripples after a stone hits a pond, rather than a single clean ring.

So a Flatlander trying to say "hey!" would be heard as a smeared, reverberating
"heyyyyy…". Same concept (waves, fields), **qualitatively different behavior** — not
just "one fewer component." 🔀

---

## 6. Forces and fields — the 1/r² → 1/r rule

Why is gravity (and the electric force) an **inverse-square** law in our world? Because
a point source sends its field-lines out through a **sphere**, and a sphere's area
grows as r². Spread a fixed number of field lines over an area ∝ r² and the strength
falls as 1/r².

Flatland has no spheres around a point — only **circles**, whose circumference grows as
r (not r²). So:

| | 3+1 (sphere, area ∝ r²) | 2+1 (circle, length ∝ r) |
|---|---|---|
| Force vs distance | **1/r²** | **1/r** |
| Potential vs distance | **1/r** | **ln(r)** (logarithm!) |

This is one of the cleanest "dimension dials" in physics: the exponent in the force law
is literally `−(dimensions − 1)`. The logarithm in the 2D potential is why 2D problems
often feel weird — a logarithm grows without bound, so a 2D charge's influence never
really levels off. 🔀

---

## 7. Gravity — where Flatland gets genuinely impoverished

Forces scaling as 1/r is one thing; **gravity as a theory** (general relativity) is
where the 2D world loses the most.

- **3+1 gravity** is rich: mass curves spacetime, objects attract, **gravitational
  waves** ripple outward, and **black holes** form with the structure we know.
- **2+1 gravity** has a famous, almost shocking property: **no local degrees of
  freedom.** Spacetime is *flat everywhere there is no matter*. A lone mass does not
  surround itself with an attractive field — it instead snips a wedge out of space and
  leaves a **cone** (a "conical deficit"). Two masses don't orbit and fall together the
  way Newton describes. **There are no gravitational waves in 2+1.** ⚠️

  *(Footnote for the curious: you can still get a 2+1 black hole — the "BTZ" black hole —
  but only by adding a negative cosmological constant. Plain 2+1 gravity has none.)*

The reason, in one line: a graviton has `d(d−3)/2` independent polarizations in *d*
spacetime dimensions. For d = 4 that's **2** (the two polarizations LIGO detects); for
d = 3 it's **0**. Gravity in Flatland has nothing to wave.

---

## 8. Shapes, traps, knots, and handedness

Pure geometry, no time needed — but it's where Flatland is most vivid.

**Trapping / enclosing.** A boundary of dimension one-less-than-space encloses a region.
- In our world you need a **2D surface** (a wall, a closed bag) to trap a 3D object.
- In Flatland a **1D loop** (a closed curve, a fence) already cuts the plane into
  inside and outside — a circle drawn around a Flatlander is an inescapable prison
  *unless* it can be lifted into the 3rd dimension. 🔀
- *The Flatland party trick:* a 3D being could reach "down," pluck a Flatlander out of
  its sealed circular cell, and drop it back outside — looking, to other Flatlanders,
  like teleportation. The same being could see the *inside* of any closed Flatland box
  from "above." This is exactly how a hypothetical 4D being would relate to *us*.

**Knots.** You can knot a 1D string **only in exactly 3 dimensions.**
- In 2D a loop can't knot — it just divides the plane. ⚠️ vanish
- In 4D+ there's so much room that any knot in a string falls open. (You *can* knot a
  2D sheet in 4D, though — knotting needs exactly "two spare dimensions.")
- We live in the one dimension where shoelaces, DNA tangles, and sailors' knots are
  possible. A nice reminder that 3 is special, not just "more than 2."

**Handedness (chirality).** A shape that isn't the same as its mirror image.
- In 2D, "b" and "d" are mirror twins: no in-plane *rotation* turns one into the other.
  But **flip the page through the 3rd dimension** and they match.
- In 3D, your left and right hands are mirror twins: no 3D rotation matches them, but a
  flip through a *4th* dimension would.
- So chirality is real in every dimension, and "undoing" it always needs **one
  dimension up**. 🔀 The pattern points at the higher dimension we can't picture.

---

## 9. The verdict — same catalog, fewer numbers, not a clean projection

**Does the 2D world need the same, more, or fewer "features" to describe everything?**

- **Fewer *numbers*.** Every quantity that is a position, velocity, or list of
  components loses exactly one entry. Locating an event drops from 4 numbers to 3;
  rigid-body freedom drops from 6 to 3. In this bookkeeping sense, Flatland is strictly
  cheaper. ✅

- **The *catalog of concepts* is the same.** Motion, inertia, relativity, light cones,
  waves, fields, forces, causality, chirality — every heading in our physics has a
  Flatland entry. Nothing brand-new is *required*, and nothing in the basic list is
  *missing as a heading*. ✅

- **But it is NOT a clean projection.** You cannot get Flatland physics by just crossing
  out *z* everywhere. Three things break that naive hope:
  - **Collapse / vanish** ⚠️ — the rotation *axis*, knots, and most of gravity simply
    stop existing.
  - **Qualitative change** 🔀 — waves grow tails, force laws change exponent, potentials
    turn logarithmic, light cones become drawable.
  - These changes are driven by *geometry that depends on the dimension count itself*
    (spheres vs circles, planes vs axes), not by the missing coordinate as a number.

- **And one thing is *perfectly* preserved:** everything anchored to the single time
  dimension — causal order, past/future, the arrow of time — is **identical**. ✅ The
  asymmetry between the two worlds is **entirely spatial**.

**One-sentence summary:** *Flatland keeps every chapter title of our physics and writes
each in fewer numbers, but several chapters have a completely different plot — and the
deeper lesson is that "3" is not just "one more than 2," it is the special dimension
where axes, knots, clean signals, and real gravity all happen to exist.*

---

## 10. Open threads (where we could take this next)

- **1+1 spacetime:** push the same table one rung lower. What survives with only *one*
  space dimension? (Spoiler: motion becomes purely back-and-forth, the light cone is an
  "X," and you can't even pass another object without colliding.)
- **The upward mirror:** every "needs one dimension up to undo" hint (chirality, lifting
  a trapped being out) is our clearest, most rigorous *finger pointing at* the 4th
  spatial dimension we can't visualize. Worth a section of its own.
- **Life in Flatland:** a straight-through digestive tract would cut a 2D animal in two
  — so biology, not just physics, constrains what a 2D creature could be.
- **A visual companion:** light-cone cones, the conical-deficit "snipped wedge," and the
  sphere-vs-circle field-line picture would each make a clean diagram.

*This is a living doc — pick a thread and we'll extend it.*
