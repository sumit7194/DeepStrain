# Journal — activity log (BlackHole: LIGO-data projects)

*One entry per working session, newest first. Lab-notebook-level detail stays in each
sub-project's `notes/lab_notebook.md`.*

> **Repo split 2026-06-13:** these three projects (echoes, ringdown_spectroscopy,
> primordial_blackhole_search) moved here from `../SpaceTime/`. The full pre-split
> narrative — including the interleaved night-shift sessions that wove black-hole and
> curvature work together — is archived in `../SpaceTime/JOURNAL.md`. This journal
> carries black-hole work forward from the split.

---

## 2026-09-02 (cont.) — what the CNN responds to: band-limited noise power at ~110 Hz

Having established the detector ignores transients, asked what it does use. Two probes: band-power vs score
across the spectrum, and occlusion (replace a band with the window's median, re-score) on three populations.

The answer is narrow and low: ~73-224 Hz, sensitivity centred at 111 Hz for signals and 121 Hz for false
alarms, with 97-100% of all sensitivity below 224 Hz. Above 700 Hz the band correlation is -0.04. The
analysis band runs to 1024 Hz, so roughly three quarters of the spectrogram is capacity the network learned
to ignore.

The important half is the comparison. False alarms use the SAME region as real signals -- occlusion profiles
correlate at +0.916, centres 10 Hz apart. The detector is honestly fooled: a high-scoring noise window is not
a glitch, it is noise with elevated power exactly where a subsolar chirp deposits its SNR, because these
binaries sweep slowly at low frequency and that is where the accumulated signal lives. Signal and false alarm
are the same feature.

That single mechanism retro-explains four things we had each measured separately and never connected:
min/veto costing reach and never buying any; tail-normalisation actively hurting 2-5%; the tail windows
having no glitch morphology; and H1xL1 coincidence buying 1.37x while every single-detector cut failed. All
of them follow if the trigger IS the signal feature and the fluctuation is independent between detectors.
The coincidence one we had never had a mechanism for at all -- it works precisely because the trigger is
ordinary noise rather than an instrumental artefact, which is the opposite of the glitch picture we assumed
for the whole arc.

Sixth false verdict from my own tooling in this stretch: the script declared "false alarms key on a DIFFERENT
band, so the noise trigger is a separate feature" -- from an argmax over two ADJACENT bands, while the two
profiles correlated at 0.92. An argmax over adjacent bins is a coin flip when the peak is broad. Rewritten to
compare profile shape. Caught the same way as the other five, by reading the numbers instead of the
conclusion line, and by now that is less a series of accidents than a property of how these tools get built:
the summary line is written when you still believe your own framing.

## 2026-09-02 — the glitches aren't glitches: the veto programme dies before it starts

Went looking for what the windows setting our deep-FAR ceiling actually are, so a veto could be built on
principled grounds. Verified the glitch taxonomy properly first (blips ~10 ms / ~100 Hz; scattered light ~4 s
arches at 8-64 Hz), which produced a sharp prediction: our [50, 1024] Hz band removes scattered light almost
entirely, so the tail-setters should be blip-family.

They are neither. All three windows I could measure contain NO supra-threshold cluster at all -- 0.88x the
max-excess of an ordinary window from the same segment -- while an ordinary window in that same segment
holding a monster transient (max-excess 4708, kurtosis +2404) scores -0.484 and is ignored. Verified the
extraction by re-scoring: all three reproduce their cached scores to 3 dp.

Confirmed at n=1,860 on 30 cached O3a segments: Spearman(score, max_excess) = +0.091, the top-18 by score and
top-18 by excess share ZERO windows, and top scorers are no more excess-bearing than average (0.11 vs 0.12).
The CNN does not respond to transient power. There is nothing to veto.

Two things make this better than a null. It retro-explains G2a/L2 (min and veto cost reach, never bought any)
and the tail-norm result (equalising the noise-tail asymmetry HURT 2-5%) -- both are exactly what a
non-glitch-driven tail predicts, since a consistency cut can only remove glitches. Three independent
measurements, one mechanism. And it corrects our own language: we have written "glitchy segments" across
several entries and attributed the effective-N limit to glitches. The limit stands; the mechanism we ascribed
to it does not.

The process notes are the usual shape. The overnight run was lost to my own typo (C.F_LOW for C.F_LOWER --
which I had grepped, seen, and mistyped anyway), then sed silently failed to fix it because BSD sed has no
\b. The morphology function was then broken twice in ways the smoke test caught before any real data: first
measuring the noise floor rather than the transient (blip, noise, and chirp all returned duration 64 s), then
ignoring the trials factor so pure noise classified as scattered light. Both had the same signature -- the
same answer for signal and for nothing. And GWOSC turned out not to be down but resetting connections
mid-transfer, so gwpy restarted the 97 MB file each time; the fix was curl -C -, and the diagnosis only came
from comparing a small manual probe against the job's large one.

The n=1,860 answer was on disk the whole time, in the 7 GB of cached O3a noise I had spent the previous night
treating purely as a disk problem.

## 2026-08-22 (cont.) — mutation-testing the gate suite: 52/55 provably fail on garbage

bridge found a §6 instance worse than mine — their keepalive `sed`-ed only the `"updated"` field, so the
clock advanced every 30 s while the content stayed frozen at `heavy: true` long after the job finished. It
blocked ansatz from launching a 4.75 GB run on an idle machine. Their framing is the best line in that file:
**it does not evade the staleness mechanism, it counterfeits the mechanism's output.**

Their closing lesson — *these checks don't work as reading comprehension* — paid off twice here.

**First**, it found something in our own status writer I had never run: the write was not atomic. Confirmed by
testing that a truncated file raises JSONDecodeError in every reader, which is worse than staleness (a stale
file still parses and expires cleanly through the contract). Now tmp + fsync + os.replace.

**Second, and the bigger one**: our margin audit tests whether a gate's bar is reachable, which is
reading-adjacent. It never tests whether a gate FAILS when its artifact is wrong. So `gate_mutation_test.py`
mutates every artifact in memory and re-runs each gate. Result: **52/55 fail on garbage, 0 unguarded.** Two
raise KeyError/StopIteration instead of asserting — they still halt the suite, so they are safe, but a clean
message would be better. One reads no JSON.

**My instrument produced a false verdict first, and that is the part worth keeping.** Version one used
x → 37x + 11 — monotone *increasing* — which satisfies every one-sided `>=` assertion and preserves every
ordering. It reported 7 gates as UNGUARDED that were all healthy; the mutation simply could not break them.
Caught by reading the flagged gates rather than the summary line. Fixed with a family of mutations
(inflate/collapse/negate/zero), calling a gate unguarded only if it survives all four.

**The mutation tester had the exact disease it was built to detect**: a perturbation that cannot violate the
assertion under test measures nothing, which is precisely a gate whose bar cannot be crossed. That is now
four instruments of mine this week returning confident false verdicts, every one caught by reading the
numbers instead of the conclusion.

Also declined bridge's offer to hand me the 2-D sweep, on their own §2: its entire value was being *different
data*, and a simulation I write to check my own empirical result inherits my assumptions about what matters.
Better carried as a known gap than filled with a check that cannot fail.

## 2026-08-22 (night) — read-back finds a live bug in my own status writer; clip-band validates our KS

bridge wrote a cross-session protocol file from the last two days' failures. Two items landed on us hard.

**§6 said validate status files by READ-BACK, not by the write returning success. Did it, and found a live bug
in my own writer inside 90 seconds.** My matcher tested `REPO in cmd`, but we launch as
`.venv/bin/python scripts/far_deep.py` from inside the repo — a relative path — so the absolute repo path
never appears in the command line and the rule matched NOTHING. Verified with a job running: zero matches.
`blackhole.status` said `idle`/`heavy: false` while jobs ran and would have done so through the whole 40-hour
GWOSC fetch.

That is worse than the frozen file the section was written about. A frozen file stops advancing; mine
advanced its timestamp every 60 s, emitted well-formed JSON, populated every field, and continuously
asserted the machine was FREE — the exact assertion a sister acts on when deciding whether to take 4 GB.
Fixed by matching on the process working directory via lsof, and verified by a full round trip:
idle → running/heavy with correct RSS and job names → back to idle, through the keepalive loop.

The lesson is not "status files are tricky". It is that **I read tabula's version of this failure, built what
I believed was the robust fix, told two sessions so, and carried the same class of bug the whole time.**
Reading about a failure mode is not testing for it.

**bridge's clip-band technique validated a load-bearing number of ours.** We quote KS(zero-lag, background)
= 0.00214 as the evidence that H1 ⊥ L1 across the whole distribution — the assumption the 4,120-yr ladder
rests on — but both CDFs were built by quantising into 40,000 bins, so the number could have been our grid.
Swept bin count 256x: KS varies 1.033x, d(log KS)/d(log width) = −0.0063. Flat. A measurement, not a floor.
Now gated. We would not have run that check on our own; it came from a lattice-entropy precision problem with
nothing in common but the shape of the worry.

**And I applied bridge's own §2 back at them**: "the thing that limits you does not average down" is a
method-driven inference that they and I have now converged on from three instances, so by their rule our
agreement is weak evidence. The three mechanisms genuinely differ, which is the argument it is real — but
that is not the same as three independent confirmations, and one habit wearing three coats is the risk. They
accepted it and said they would have written it up as three.

## 2026-08-22 (later) — retracting L4: the 3.2 sigma was an analytic artefact

Asked what was next and picked the one thing this week's audits hadn't touched: a live POSITIVE claim, in the
sub-project nobody had swept. Everything else lately has been auditing negatives and nulls, where being wrong
is cheap. L4 said "coherent network combination helps, 1.12x at 3.2 sigma" and verify.sh gated it at 1.10x
and 2 sigma — both bars marginal, so the claim only stood if the error bar was right.

It wasn't. The analytic bar omitted three terms: the slope denominator came from two adjacent points of a
6-point grid; the two statistics come from the same trial but were combined with hypot() as if independent;
and the 95th-percentile threshold from 60 background draws carried no uncertainty at all. A paired bootstrap
resampling trials and background together gives **sd 4.00x larger — 3.17 sigma becomes 0.96**, with the gain
CI [0.93, 1.48] spanning 1. Underpowered, not a result. Retracted, and the gate now asserts the retraction.

**The part worth keeping is what it does to an earlier lesson.** I had pre-registered that the network axis
buys ~nothing, recorded that prediction as refuted by the 1.12x/3.2σ, and wrote down "agreement with my
prediction made me less likely to check". That lesson was backwards. The run that *disagreed* with me was the
one that needed checking — its disagreement is exactly what made it feel like a finding worth keeping. Both
directions need the same scrutiny, and I had drawn the opposite conclusion from the same episode.

The history had also been signalling it: v3 gave 1.21x at 1.4 sigma, v4 gave 1.12x at 3.2 sigma. A gain that
shrinks as statistics improve is the signature of an understated error bar, not a converging measurement.

Survives untouched: the injection-convention finding (physical 1.12x vs identical 0.92x), which is the larger
effect, exposed two real bugs in our injections, and does not depend on the coherent gain's significance. So
does the measured H1/L1 geometry.

Noted without claiming a mechanism: the analytic bar understated by 4.00x here, and the pbh jackknife by 4.2x
on a different estimator in a different sub-project. Two independent instances of a shortcut error estimate
landing ~4x too narrow.

## 2026-08-20 — auditing our own error bars: the jackknife is 4.2× too small

Set out to measure the effective-N scaling law and ended up auditing the estimator behind yesterday's
headline. Three findings, in the order they forced themselves.

**The jackknife understates the true sampling spread by 4.2×.** Running both estimators on the same subsets
at the same n removes any sample-size excuse. The mechanism is specific: it drops 10% of segments, and with
727 in play that almost never removes the one glitchy segment that sets the tail — so it looks calm whatever
the truth is. The bias is stable in n, which is what saves yesterday's ratio: both endpoints are wrong by the
same factor, so 33–44% → 10–12% is a real relative improvement. Every absolute number needs ×4.2.

**Effective-N grows linearly, but does not control the error.** I pre-registered a collapse test — σ·√N_eff
constant across sizes and rungs — precisely because a power-law fit can't tell "N_eff controls it" from "n
controls it". β = 0.93 confirms L2's growth claim; α = 0.10 and 62% collapse scatter refute the causal one.
More independent loud windows simply don't buy precision, and most of the relative improvement L2 reported is
the threshold mean drifting upward, which is the non-convergence we already knew about.

**I proposed a heavy tail and it was wrong.** Two runs disagreeing 1.5× and σ rising with rep count looked
like infinite variance, which would have meant no ± was quotable anywhere in this arc. Hill index came back
19.7–195. σ converges; the disagreement was ordinary small-sample SD noise.

Two of my own scripts printed confident false verdicts tonight — a planning number of 10⁵⁵× from dividing by
α≈0, and a heavy-tail rule that compared growth over one window against drift over another. Both were caught
by reading the numbers instead of the verdict line, which is the same lesson as the silent rung-drops: the
conclusion line is the easiest part of your own output to stop checking.

Yesterday's background, reach and null are untouched. What changed is the error bar and the story about what
limits it.

## 2026-08-19 (later) — using the 45,073 zero-lag measurements we had been throwing away

Asked what we could do that isn't reimplementing LVK on a laptop. The honest answer: a signal-consistency
veto — my first suggestion — is the most standard tool in the field and would have taught us nothing. Dropped
it. What *is* ours: we had 45,074 zero-lag coincidence measurements and had only ever looked at the maximum.

`far_zerolag_population.py` uses the whole distribution. The trick is that the time-slide background is
itself the independence null (same windows, same noise, no model), so comparing the two distributions tests
H1⊥L1 and searches for a diffuse sub-threshold population at the same time. Golden-tested before running:
silent under true independence, catches a planted 1% shared-noise component at z=+6.35 — and only in the deep
tail, which says correlated noise in a few percent of windows is invisible in the bulk.

**Clean.** CDFs agree to 0.2% everywhere (KS 0.00214 vs crit 0.00640), joint exceedances 451 vs 450.7
expected, no excess at any rung. Independence is now verified to the **99.98th percentile** rather than just
the bulk — which matters, because the 4,120-yr ladder committed this morning rests on it.

**The better result is structural.** Three of four rungs sit beyond *all* zero-lag data — 1/year, 1/decade
and 1/century have zero zero-lag events at or above them, and the 1/century threshold is above the zero-lag
maximum entirely. That gap cannot be closed by sliding, because slides make more pairs and no new zero-lag
samples. So: **every factor of depth gained in a time-slide background pushes the threshold further beyond
the regime where the independence it assumes can be verified.** L2 showed livetime doesn't buy independent
samples; this shows it doesn't buy verification reach either, and unlike the first, no amount of compute
substitutes for real observing time.

Also guarded a one-sidedness column that silently inverted meaning when the louder detector's score is
negative — true at the shallow rungs, where it would have read as "95% one-sided" for no physical reason.

## 2026-08-19 — L2 deep FAR lands: 4,120 years, 1/century, and the audit answered

**Ran to completion.** `far_deep.py --target 727 --purge` finished all 727 O4b segments after **three power
losses**; each restart verified the cache (**0 corrupt of 399 / 641 / 727**) before relaunching. Per-segment
atomic checkpointing meant every loss cost only the in-flight fetch. Disk stayed flat — the astropy-cache
purge fix from the original run held throughout.

**Result.** 45,074 windows, 801.3 h, 45,073 distinct lags → **4,119.9 yr** (51× the published 80.5). Reached
**1/century** for the first time. Ladder 11.246 / 12.799 / 14.532 / 16.394.

**The point of the exercise: jackknife spread 33–44% → 10–12%.** The 08-09 audit said the ladder was an order
of magnitude over-precise and the fix was more independent loud-noise samples. It was right, and the old
16.1 ± ~5 contains the new 14.532.

**Zero-lag unchanged at 11.295**, verified against the score cache rather than the artifact — an unchanged
value across a 7× data increase is the shape of a stale read, and this one is real. Its measured FAR is
11.5/yr against 1.05 expected: a textbook null. I pre-registered *before* the numbers landed that 1/month
cannot support a claim here (1.1 expected events in 0.0914 yr of zero-lag), which is what stopped
`11.295 > 11.246` from being read as a near-miss. Null holds **4/4**.

**What didn't improve, stated plainly.** 1/decade's 425 events still come from **8 distinct H1 windows** and
1/century's 43 from **3** — the same count the published 1/decade had. 51× the background barely moved the
number of independent Hanford glitches setting the deep tail. The ladder is also **not converged** (1/decade
17.06 → 16.78 → 14.53 across n=60/80/727). Effective sample size, not livetime, remains the wall.

**Three tooling bugs, one species.** Pre-flighting the analysis at the new scale — rather than waiting for the
run to end — found that a `keep` cap sized for an 80-yr background silently drops the 1/month rung at 4,120 yr,
in **two** scripts, including from the jackknife that decides this item; `ladder()` dropped it with no message.
Separately, background livetime counted `n_segments × 4096 s` when the whitening crop leaves 62 of 64 windows,
overstating searched time by 3.23% in **three** scripts. All fixed, with a loud `!! RAISE keep` guard so a
truncated report can never again look like a complete one. Re-ran the whole chain for consistency.

The lesson worth keeping: **a constant sized against yesterday's data volume becomes a silent truncation when
the data grows**, and a check that silently omits its hardest case is indistinguishable from a check that
passed. That is the same failure mode the gate-margin audit was built to hunt — found this time in our own
audit tooling.

## 2026-08-16 — the 154 "unscoreable" assertions: mostly a parser problem, and 11 real self-certified flags

Follow-up to the gate audit. I had told quantum that 154 of our 193 assertions were unscoreable and that this
was "exactly where decoration can hide". **Both halves of that were partly wrong, and in opposite directions.**

**Most of the 154 was my parser, not the suite.** Rewriting it to handle what it had been skipping —
`open(R + "name.json")` (a form *its own gates* use), `all(...)` generators, range checks, abs-difference
checks, comparisons whose right-hand side is another artifact value, and — the big one — **variable bindings**
(`rd, npe, ec = st["ringdown"], st["npe"], st["echo"]`, which a whole-artifact resolver can never follow):

    scored ratios   39 -> 83
    real gates      32 -> 79
    decoration       1 ->  0
    unbacked flags  16 -> 11 -> 0

Reporting 20% coverage as a property of the SUITE when it was a property of the PARSER was itself a small
instance of the failure being hunted.

**But there were 11 genuinely self-certified flags**, where a gate rested on a boolean the script had computed
for itself with no number behind it anywhere — `assert d["ssl_helps"]`, `assert r["kerr_inside_90"]`,
`assert d["gates"]["S1_unbiased"]`. Each now carries a numeric companion asserting the quantity the flag
summarises, so the gate fails if the number moves even when the script still sets its flag True. Examples: SSL
now asserts `delta_mean > 0.05` (measured 0.124); Kerr-consistency asserts the δ CI actually brackets zero
(−0.46, +0.33); stacking asserts `|stacked mu| < 0.20` and that σ genuinely tightens against the single-event σ.

**One real gap left open rather than papered over:** `S3_coverage` in the stacking gate has **no numeric
companion available** — `12_stacking.py` doesn't emit the coverage value it gates on. Recorded in the gate
itself as an outstanding fix rather than backed with a substitute number.

**The tool failed on itself three more times**, which is now its most reliable feature. It (1) could not see
variable bindings, so it declared gates unbacked whose numbers sat right beside them; (2) could not parse
line-continued assertions — a trailing comma survived message-stripping and silently broke every pattern
match, including **the backing assertions this very audit had just added**; and (3) counted `assert g["M_inv"]
and 60 < g["M_inv"] < 100` as unbacked, when that is a null-guard plus a range check on the same quantity
(M_inv can be `None`, which would raise on comparison). All three produced false positives that looked exactly
like real findings.

52 gates green throughout. L2 untouched — still running, though GWOSC degraded badly overnight and the rate
has collapsed from ~6.6 to ~0.3 segments/hour.

---

## 2026-08-15 (overnight) — the audit round: our gates checked against themselves, and L4's second rung

Two tasks, both prompted by TheBridge Round 12 rather than by our own list — and both produced corrections.

**Gate margin audit (`scripts/gate_margin_audit.py`).** quantum's generalisation, after G3 produced four gates
certifying nothing and tabula found a clause gated at an unreachable bar: *"a control that cannot fail is not
a control — what is the observed distribution of the guarded quantity relative to its threshold?"* We run more
gates than any sibling and had never checked. Parsing the real assertions out of `verify.sh`:

    real gates (<10x from their bar): 32 | loose (10-100x): 2 | DECORATION (>100x): 1 -> now 0

**The one genuine decoration:** `sigma_snr_spread < 0.02`, observed **5.5e-13** — sitting **3.6×10¹⁰** below
its bar, because σ(δ) ∝ 1/SNR is an *analytic identity* here, not an empirical claim. It could never fail.
Retightened to 1e-11, ~18× above observed round-off, which still catches a broken Fisher inversion with
cross-platform headroom but no longer pretends to certify an identity. Recorded honestly: a round-off check is
*inherently* loose; the point is that the bar now tracks round-off rather than fiction.

**The tool corrected itself twice**, which is the part worth keeping. A first hand-encoded pass flagged the L4
geometry gate as infinite-margin decoration — that was my encoding (`3.0 - dist > 0` has threshold zero, so
the margin divided by zero). The parsed version then flagged six, of which five were the same class of
artifact: a ratio is meaningless when either side is zero. Those are now classified as **sign-tests** (e.g.
Build C-2's bootstrap CI lower bound at 0.0097 > 0 — the *tightest* gate we own, not the loosest) and
**at-extreme**, not scored. *An audit that reports margins will manufacture infinities from its own encoding,
and infinite margin is exactly the signature it is hunting — it self-camouflages.* Relayed to the siblings.
Honest coverage: **39 of 193 assertions** are numeric comparisons; the other 154 are flags, `all()`/`any()`
generators and compound conditions this method cannot score.

**L4 rung 2 — the within-detector phase axis.** Result in three parts, only two of them numbers.

*Reproduced:* the network-coherent gain, on an independent run, **1.09× at 2.4σ** against 20_'s 1.12× at 3.2σ,
with the same geometry recovered from the merger.

*The conceptual finding:* I built a "coherent comb" as |ACF of the complex analytic signal|, believing it kept
the phase the envelope throws away. **It doesn't** — for a narrowband signal that magnitude *is* the envelope
autocorrelation (measured correlation **0.9969**, teeth agreeing to ~1%). Taking |·| per tooth discards
exactly the phase it was meant to keep. Keeping within-detector phase requires combining the *teeth*
coherently, which needs the phase relation between successive echoes — reflection physics we don't have. **So
the envelope is not a sloppy choice; it is what makes the search model-independent**, and it explains why the
published method combines coherently across *detectors* (where geometry gives the phase relation) and not
across echoes. Three patches had failed to rescue it because they treated symptoms of a conceptual error.

*The failure, reported as a failure:* an oracle matched filter meant to bound the ceiling returned 9.62× — and
its efficiency **falls from 1.00 to 0.67 as the signal gets louder**, which no valid detection statistic can
do. The number is withheld. My proposed mechanism (PSD-estimation contamination via differencing) was
**not confirmed** — measured corr(template, noise) = −0.003 — so the honest position is: demonstrably invalid,
mechanism unknown, ceiling unmeasured.

Gate 52, all green. L2 still running (~230/727) and untouched throughout.

---

## 2026-08-15 (evening) — L6 closed: the SSL win saturates at 2,500 specs, so don't fetch more

Ran L6 alongside L2. The scheduling call mattered: fetching a bigger unlabeled pool would have fought L2 for a
flaky GWOSC and for 15 GB of disk — and was premature anyway, because **if the gain has already saturated,
fetching is pointless.** So the curve got measured from data already on disk first.

**Result: the SSL gain is fully achieved at 2,500 unlabeled spectrograms** (+0.1076) — **8× less data than N4
used** — and does not grow to 20,000 (5k +0.0968, 10k +0.1060, 20k +0.0768). Slope over 5k→20k is **−0.020**,
robustly failing the pre-registered ">+0.02 ⇒ fetch more" bar.

**Stated honestly, the curve is FLAT rather than declining.** Within-pool seed scatter is sd **0.019** against a
pool-to-pool spread of **0.031** (~2 SE), so the 20k dip is not significant, and this **bounds** the scaling
effect (excludes >~0.03 AUC between 5k and 20k) rather than proving it zero. Either way the decision is the
same, which is what the experiment existed to settle.

**Cross-detector transfer is null.** Adding the 6,250 L1 noise specs to 20,000 H1 ones moved the result
**+0.0009**, twenty times below seed scatter. So the pool can't be grown cheaply across detectors either — a
small counterpoint to the transfer claims foundation-model work makes.

**Inventory checked, not assumed.** I first read `shards_w64_hl` as a 2.3× pool expansion. It isn't: 20,000 of
its 26,250 specs are the *same 16 H1 segments* as `shards_w64`. Only 6,250 L1 specs are new (+31%). Leakage
check clean — 0 pool segments in H1 val or test.

**Process note.** My first draft **rewrote** the masked autoencoder with a different channel ladder
(1→16→32→64→128 instead of 1→32→64→128→256). That wouldn't have crashed — it would have silently produced a
scaling curve not comparable to N4's numbers. Caught it when tensor shapes didn't line up and I went to read
`ssl_pretrain.py`; the script now imports `SpecMAE`, `random_mask` and `train_model` from the N4 scripts, and
reuses the repo's own AUC helper rather than adding sklearn.

**Also recorded:** our 20k gain (+0.077) sits below N4's (+0.124) because we re-draw the labeled subset per seed
where N4 held it fixed (our scratch sd 0.024 vs N4's 0.006). Paired across pools, so the scaling comparison is
fair — but these are not a reproduction of N4's absolute numbers.

**⇒ L6b is not justified.** N4's headline stands and is better understood: the self-supervised win is real,
cheap, and saturates almost immediately. 49 green.

---

## 2026-08-15 (later still) — L1 built out and CLOSED: ratio filtering does not help subsolar (0.94×)

Asked to build the dense bank "if it passes". It didn't, and the path to finding that out went through **three
of my own errors** — which is the part worth recording.

**The method is real and the algebra is exact.** Verified at the primary source
([arXiv:2601.18835](https://arxiv.org/abs/2601.18835), PRD) rather than the search snippet: with A_t = A_r·R our
cross-correlation gives `c_t = c_r (*) IFFT[conj(R)]`, one FIR per target. New `pbh/ratiofilter.py`, kernel by
weighted least squares over frequency with Toeplitz normal equations. An **untruncated kernel reproduces the
matched filter to 1.000000**, at every separation and even when built at the wrong remnant.

**Error 1 — the golden test failed at 0.814 and it was mine.** I nearly recorded L1 dead. The untruncated-kernel
control (which the golden test lacked) showed the algebra was fine; the failure was truncation. Subsolar needs
far longer kernels than the paper's ~250 because these inspirals accumulate enormous orbital phase.

**Error 2 — an automated verdict I wrote said "no memory win", on an arbitrary `L/16` cutoff that 16,385 taps
missed by ONE tap.** Wrong figure of merit; total bank memory reversed it.

**Error 3, the decisive one — I measured a primitive at the wrong scale.** The cost model timed correlation on a
262,144-sample *chunk* (7.8 ms) and concluded generation (442 ms) was **56×** the filtering, giving a headline
**82× RAM / 36× time** that I committed to RESULTS.md, CLAUDE.md and the gate. But `bank_dense`'s expensive step
correlates over the **entire 16.7M-sample segment** — 64× more data. Measured properly: generation is **8%** of
the cost, and the real speedup is **0.94× — marginally slower.**

**The mechanism, which is the actual result.** Ratio filtering converts O(N log N) into O(N log K), so the gain
is ≈ log N / log K. The published 8× assumes **K≈250** (BNS). Subsolar needs **K≈16,385** — measured, not
guessed (8,193 taps → 2.4% statistic error; 16,385 → 0.89%, clearing the pre-registered 1% bar). With N=16.7M
that caps at **1.6×**, and we measure 0.94×. **The benefit is inversely tied to the kernel length a signal class
demands, and subsolar demands the longest.** Reopening criterion recorded: **K ≲ 1,000 taps**.

**Memory doesn't rescue it.** Kernels are ~31× smaller than stored analytic chunks — real, but irrelevant:
memory was never binding, since `bank_dense` had already gone template-major to work around it. **Compute time
binds**, and this doesn't cut it (6 segments at 0.01%: 151.9 h direct vs 161.8 h ratio).

**What survives.** The statistic *is* faithfully reproducible at 16,385 taps — noise-regime error is **unbiased
jitter** (median bias +0.17%, 57% positive, so the threshold is safe), signal-regime 0.89%. And the dense-bank
wall now stands for a **measured** reason instead of an unexamined cost assumption, which is strictly better
than where Follow-up A left it.

**The bank was deliberately not built** (~162 h for no speed gain), so *does a CNN still tie a matched filter
once the bank is adequate?* stays open and needs a genuinely cheaper filter. The gate was **rewritten to assert
the negative** (`not helps`, speedup < 2×, generation share < 25%) so the inflated claim cannot creep back;
`bank_ratio_costmodel.json` is retained but marked superseded everywhere it appeared. 48 green.

---

## 2026-08-15 (later) — acting on the sweep: L2 launched, and an honest negative on a published method

Two tasks off the new list: launch **L2** (deep background) to run unattended, then run the **orthonormal-mode
ROC test** that L3 was gated on.

**L2 — and a robustness bug found in the first minute.** Targeted the *full* 727-segment O4b pool rather than
the 353 I had scoped (≈**4,253 yr** background, 42 events at 1/century — a real estimator, versus the 8 events
that gave the current rung its ±33%). It immediately failed on **every** segment. Diagnosis: not missing data —
H1 fetched fine (16.7M samples) while **L1 timed out on the SSL handshake**. GWOSC is flaky, and `far_deep.py`
had *no retry*, so one transient timeout on one detector cost a whole segment, and the loop would have burned
all 627 pool entries in minutes and scored nothing. Stopped by PID (3 segments burned, cache intact at 100),
added per-segment retry with exponential backoff **plus** a consecutive-failure detector that sleeps 30 min
when GWOSC looks degraded — a job meant to run for days must survive a ~12 h outage, which we have hit before.
Relaunched. *A failure here was never permanent (an unscored segment simply has no cache entry, so the next
pass retries it) — but it would have wasted the pool ordering and looked like success.*

**Orthonormal QNM basis — the answer is no, and it costs us a hoped-for rescue.** arXiv:2605.03576 reports
GW250114's overtone significance rising **82.5% → 99.9%** by orthonormalizing, blaming non-orthogonality for
"hindering identification of subdominant QNMs" — which would have explained our parked **v4 tone-count
negative**. Pre-registered the sharp prediction *before running*: with frequencies fixed and whitened noise the
model is linear in the amplitudes, so overtone detection is the nested GLRT (power in V2⊖V1) — and that is
**algebraically identical** to A₂₂₁ weighted by the full 2×2 covariance, because the Schur complement of the
220 block of HᵀH is the Gram matrix of the orthogonalized 221 directions.

It held exactly. The premise is real — 251.0 vs 245.5 Hz, **5.6 Hz apart, overlap 0.863** — yet **max |ΔAUC| =
0.00000** across 40k paired trials per class, bit-for-bit. Stage B closed both escape hatches: building the
basis at a **wrong remnant** (ΔM ±5/±8, Δχ ±0.06–0.08) still gives **0.00000**, so it is the algebra and not a
lucky fiducial; and the one lever that *does* move a number is the prior — β = Rθ with R triangular, not
orthogonal, so "uninformative" means different things in the two bases, and the Savage–Dickey Bayes factor
shifts **0.454 dex (2.8× in odds) at ΔAUC 1.7e−6**, with the AUC gap collapsing monotonically as the prior
broadens while the BF shift persists. **Significance moves; detection power does not.** The one genuinely real
effect is small: an analysis that *ignores* the covariance loses ≤ **0.0076** AUC.

**Consequence, including the part we did not want:** L3 as a sensitivity play is dead, and **our v4 tone-count
AUC≈0.61 wall is NOT explained by basis non-orthogonality** — it stands as an information limit, as recorded.
**Scope stated honestly in the artifacts:** this is the linear/fixed-frequency case, so it does *not* show the
published number is wrong; orthonormalization can still help MCMC conditioning and amplitude reporting. What it
shows is that decorrelation *per se* carries no detection information, and that **any significance quoted in a
rotated basis must state its prior.** Open thread left explicit: real fits span a *union* of bases as (M, χ)
vary, which B1 does not cover.

**Process note.** The gate initially failed at ΔAUC 1.7e−6 against a 1e−6 tolerance — correctly, because I had
written one tolerance for two different things. Stage A's identity is **exact** (monotone transforms ⇒
identical ranks); B2's is **asymptotic** (finite prior regularization differs slightly between bases). Fixed by
asserting the *convergence* rather than bitwise equality, and by requiring the residual stay two orders of
magnitude below the one real effect. Gate 46 → **47**, all green.

---

## 2026-08-15 — literature sweep: where the field is, and what it means for our three arcs

No new measurements this session — a wide read of what's been published, recorded in a new living document
**[RELATED_WORK.md](RELATED_WORK.md)** so the context survives. Entries are marked **[A]** (abstract fetched
and read) vs **[S]** (search snippet only, verify before load-bearing use), per the prior-art-rigor rule.

**Nothing contradicts a result of ours.** Two things validate us, one hands us a citation we lacked, one
offers a method upgrade, and one reopens a wall we had called closed.

- **Subsolar.** LVK published their O4a search ([2602.12115](https://arxiv.org/abs/2602.12115), PRD): 25M
  templates, no candidates, f_PBH < 0.5% at 0.4 M☉, >2× better than O1–O3 combined. **S251112cm** (the Nov 12
  2025 candidate) sits at **FAR 1/6.2 yr, P(subsolar) > 99%, 93 ± 27 Mpc, still unresolved** — which means our
  deep-FAR ladder is calibrated on exactly the scale a real candidate lives at. **O4c is not public**, so its
  strain remains out of reach; that closes an earlier open question with a date.
- **Ringdown — the most actionable find.** [2605.03576](https://arxiv.org/abs/2605.03576) orthonormalizes the
  QNM basis and lifts the GW250114 first-overtone significance **82.5% → 99.9%**, on the argument that
  non-orthogonal QNMs induce correlations that mask subdominant modes. The 220/221 modes are ~6 Hz apart —
  near-parallel basis vectors — which is a candidate *mechanism* for our parked **v4 tone-count negative**
  (AUC 0.61) that we had attributed to SNR. **Pre-registered concern before adopting:** `P(A₂₂₁ ≠ 0)` is not
  basis-invariant and orthonormalizing changes the implicit amplitude prior, so part of a 17-point jump could
  be a change of question rather than a gain of information. Decisive test we are equipped for: inject
  known-amplitude overtones and compare **ROC/AUC in both bases**.
- **Echoes.** [2512.24730](https://arxiv.org/abs/2512.24730) searched GW150914 / GW231226 / GW250114 with a
  model-agnostic coherent-network method — **null, with 90% upper limits.** Our E3 nulls are consistent with
  the field's latest, including on GW250114.
- **Our audit has a twin.** [2509.05283](https://arxiv.org/abs/2509.05283) reports that ML-search sensitivity
  varies notably across month-long real-noise datasets **at low FAR**, and calls for rigorous statistical
  validation. That is precisely last session's finding, stated in the abstract; ours supplies the mechanism
  and a number (±33–44%, traced to 2 windows in one segment). Now cited in RESULTS.md — the audit is an
  instance of a named open problem, not housekeeping.
- **GraviBERT** ([2512.21390](https://arxiv.org/html/2512.21390)) does BERT-style self-supervised pretraining
  for GW, converging independently on our **N4** direction.
- **GWTC-5.0** (2026-05-26): 161 new events, 390 confident, and **the first three-tone measurement from a
  black hole** — directly our spectroscopy territory, to read before further tone-count work.

**Process note — a stale verdict corrected.** ROADMAP had carried "template-bank density wall → intractable
locally" as a blocker. The field's **ratio-filter de-chirping (~8× per core)** would put ~13,000 templates on
our existing hardware, which is a real move along the one axis Follow-up A named as the dominant loss. I had
initially recommended *skipping* it as a poor effort trade; the user corrected that — **effort is not a reason
to drop an item here.** Reclassified from blocker to **L1** of a new tracked **L1–L7 long-horizon list**
(RELATED_WORK.md + ROADMAP.md), with the standing rule that items leave the list only when done or measured
impossible. L2 (deep background → ~1000 yr, ~353 segments, ~14 h unattended, should take the jackknife spread
±33% → ~±18%) is the best use of idle Mac time and needs no prerequisite.

---

## 2026-08-09 — the cleanup that became an audit: our own 1/decade result stress-tested

Started as disk triage. Before deleting anything the user asked for a pass over the data
"to see if we did not miss anything genuinely important… look for any accidental findings
first" — then, after the pass turned up real findings, decided **"lets not delete anything,
we have lots to get from those."** The right call: nothing was deleted, and the retained
score cache produced the most useful result of the week.

**A wrong number caught before it became a claim.** The first quick pass reported "96% of
the loudest background events are one-sided." The careful re-run said 38%. Same data — the
quick version sampled every 37th lag, which makes its "top 100" a *mid*-tail population,
where one-sidedness dominates; the true extreme tail behaves oppositely (top-25 is **0%**
one-sided). One-sidedness has to be reported *against loudness*, not as a single number.
Resolving that discrepancy is what turned a cleanup into an audit.

**Then the audit inverted the interpretation — twice.** Reading the deep-FAR result as
"zero-lag 11.295 vs 1/month 12.34 = an 8% near-miss" is wrong on both sides:

- **The event isn't a coincidence.** 11.295 = H1 **+12.53**, L1 **−1.24** — the single
  loudest Hanford window in all 114 h, with Livingston seeing nothing. Under a consistency
  statistic it collapses to 1.635.
- **The threshold isn't 80 years of statistics.** Single-detector ceilings (max H1 12.53,
  max L1 6.26) mean any background above ~12.5 *requires* both detectors, so 1/year and
  1/decade are genuine two-sided coincidences — good news, worth verifying rather than
  assuming. But `far_background_validation.py` found the 8 events setting 1/decade trace to
  **2 distinct H1 windows**, jackknife spread **33–44%** (vs ±2% Poisson), and **6 of the 8
  loudest H1 windows sit in one segment** (59, gps 1397232640) — dropping it alone moves
  1/decade 16.121 → 11.261. The apparent "non-stationarity" (halves differing 32–46%) is the
  *same* finding, not a second one: bulk noise is identical between halves.

**What held:** the assumption the whole method rests on. H1⊥L1 independence, z = −0.38,
**p = 0.69**; per-segment data-quality correlation p = 0.45. Had that failed, every threshold
would have been biased low.

**A hypothesis I had recorded as untestable turned out to be testable.** I'd written that
min-vs-sum at deep FAR needed injections into O4b strain that `far_deep.py` purged. It
didn't: `o4_sensitive_distance_rows_matched` and `coinc_triple_rows_o4b` already store
**per-detector** scores for 4,800 O4b injections. Measured at matched FAR, `min` gives
**0.97–0.99×** sum's sensitive distance (`veto` 0.99–1.04×) while halving background
instability (25% vs 46%) ⇒ **honest negative, keep `sum`** — and G2a's "sum is optimal",
previously established only at a 4.6-yr background, now holds at depth too. *Lesson: "we
can't test that" deserves the same verification as any other claim.*

**The result that survived (V8).** Every test above attacks the threshold; what matters is
the search. Null in **4/4** configurations (all-segments / drop-59 × sum / min), each against
its own matched threshold, at margins of **2.0–3.7×**. Removing the glitchy segment removes
the loudest zero-lag event *and* the background that nearly matched it — self-consistently,
because they are the same glitches.

**Net correction:** reach stands, the null is *stronger* than we published, and the precision
must be re-quoted as **16.1 ± ~5 (33%)**, not ±0.3. Deep FAR is limited by the number of
independent loud-noise samples, not by livetime — which is why real LVK searches use
signal-consistency vetoes and DQ flags rather than raw time-slides alone. RESULTS/CLAUDE/
README corrected; gate now **46** green.

---

## 2026-08-08 — deep FAR: an 80-year background on O4b, 1/decade reached on the Mac (no VM)
The last "parked" item claimed lower-FAR needed the GPU VM. It didn't — it needed compute, and the Mac was idle.
`far_deep.py`: global time-slides give **background = (N_windows−1) distinct lags × total livetime** (formula
verified FIRST against Build C — reproduces its 1692 days exactly), and since both factors grow with segment
count, **background ∝ N_segments²**. That is why this was always affordable: 24 segs → 4.6 yr, but 100 → 80 yr.
- **Result: 100 fresh O4b H1∩L1 segments (113.8 h, leakage-free since cnn_w64 is O3a-trained) → 6,200 windows →
  6,199 lags → 80.5-year background, 17× Build C.** Ladder: 1/month 12.340, 1/year 14.112, **1/decade 16.121**.
  1/century is honestly *not measurable* (0.01 × 80.5 yr < 1 expected event) — the deepest measurable is ~1/80 yr,
  and we quote 1/decade as the conservative rung.
- **Zero-lag check (real, unshifted H1×L1): loudest coincidence 11.295 — below even the 1/month bar. Clean null**,
  which is the expected and correct outcome for randomly-chosen noise stretches. The deliverable was never a
  detection; it is the *calibrated ruler* that a future candidate (e.g. S251112cm, if O4c ever opens) would need.
- **A disk bug caught mid-run, and a correction owed.** I had told the user disk would stay flat because we purge
  raw strain after scoring. It didn't: `gwpy`'s `fetch_open_data(cache=True)` keeps a **second copy in astropy's
  download cache** (~0.25 GB/segment; 5.3 GB had accumulated) that our purge missed — it would have exhausted disk
  around segment 80, killing a 4-hour run near the finish. Found it by *checking* rather than trusting the claim,
  fixed the purge, reclaimed 5 GB, and disk then held flat at 19 GB for the remaining 60 segments.
- Robustness: per-segment atomic score-cache checkpoints meant a mid-run restart (to load the disk fix) resumed
  from 41 cached segments with **zero work lost**. Gated (45 green).

## 2026-08-07 (evening) — the δ-stacking "wall" re-measured against 439 events: it HOLDS, and now it has a number
Prompted by a good challenge — "are only 8 events available? check again." They were right to push: our
δ-stacking wall was decided in June on a **hand-picked 8-event list from O1–O3**, before GWTC-4.0/5.0 published
hundreds of O4 events. Worse, I had queried the catalog two days earlier and *still* answered from the stale doc.
Re-selected from the **full 439-event public catalog** under the NPE's real constraints (`26_more_events_o4.py`):
BBH not BNS · detector-frame remnant inside the [40,120] M☉ prior (source mass × (1+z)) · H1 **and** L1 both
observing · SNR > 20 → **12 analyzable, 9 never tested**.
- **The physics caught an error the script missed.** The automated verdict said "wall is NOT real — 2 informative
  events". But the second one, **GW231206 (SNR 21.9), was beating events 1.7× louder** — impossible if the signal
  is real. Its mass posterior was **jammed against the 120 M☉ prior ceiling** (M 106.3 vs published 80.8, CI to
  118), and a truncated posterior narrows δ artificially. Added a **CI-based prior-truncation screen**; **5 of 12
  events fail it** and are excluded. Lesson: I nearly reported an automated verdict that the physics contradicted.
- **Corrected result: the wall HOLDS — still exactly one informative event (GW250114, σ/prior 0.833).**
- **But it is now QUANTIFIED, not assumed.** On the 7 clean events, corr(SNR, σ_δ/σ_prior) = **−0.91** (the
  physics behaves once artifacts are gone), giving **σ/prior = −0.103·ln(SNR) + 1.295 ⇒ SNR ≳ 47 needed** for an
  event to carry δ information. **Only GW250114 (78.6) clears it in the entire catalog.**
- Near-miss worth recording: **GW230814 (SNR 43)** would fall just below threshold anyway — and is unusable
  regardless, because **H1 was offline** (single-detector event; our NPE needs both).
**Net: the June conclusion was right, but was being defended with a stale argument. It is now measured against
439 events with a predictive threshold instead of "wait for the universe."** Gated (44 green).

## 2026-08-07 (later) — O4-4: the S251112cm/O4c release watcher + README refresh
Closed the last open PLAN item. `o4c_release_watch.py` is a standing GWOSC query that answers one question:
has the era containing **S251112cm** (the first >99%-probability sub-solar candidate, 12 Nov 2025) become
public? **Current answer: no.** Public bulk strain still ends at O4b (2025-01-28); the sole O4c dataset
(`O4c1DiscC00`) is a **1.14 h** discovery window around GW250207 and does not cover S251112cm (GPS 1446995943);
no S25* superevents are public. The gate asserts the *status*, so if it flips the check fires and we go look —
and by then nothing blocks us, since cnn_w64 is already validated on O4-era noise (0.97× transfer). Gated (43).
Also refreshed the README: stale gate counts (39 → 42/43) and, more substantively, the O4-era results were
missing from the front page entirely — added the 1.23× reach gain and the cross-generation Virgo replication
to both the PBH section and the headline table.

## 2026-08-07 (review) — O4-3 stress-tested: gain survives, mass-dependence does not
Reviewed the O4-3 reach campaign (built in a parallel Gemini session) against the north star. The code follows
our conventions and the physics is right (d_reach = SNR_ref(1 Mpc)/SNR50), and it carries a strong independent
cross-check: **the measured 1.24× distance gain matches our separately-measured 1.29× ASD improvement** — two
independent routes agreeing. Three issues found, all now addressed:
- **(a) The eras were not FAR-matched.** O3a used 5 segments (5.7 h) vs O4b 8 (9.1 h); the zero-FA threshold is
  the max over noise windows, so it grows with livetime (thr_single O3a 2.111 vs O4b **3.233** — our own 3-segment
  scout gave O4b just 1.141). O4b was being held to a *stricter* bar, biasing **against** the claim. Added
  `--match-segs` and re-ran at equal livetime: **1.21/1.28/1.20×, mean 1.23× vs 1.24×** — the concern was real but
  **immaterial** (~1%, within Monte Carlo noise). The result is robust to it.
- **(b) No uncertainty on the gains.** Added `o4_reach_bootstrap.py` (B=1000, resampling the saved injection rows,
  no re-run needed): **every mass bin's 90% CI excludes 1** — 1.21 [1.14,1.29], 1.28 [1.21,1.35], 1.20 [1.11,1.30]
  ⇒ the gain is significant everywhere. **But all three CIs mutually overlap ⇒ the apparent mass-dependence
  (1.20 vs 1.29) is NOISE.** Removed that from the claim.
- **(c) Volume uses (mean SNR_ref)³ rather than mean(SNR_ref³).** By Jensen's inequality this *underestimates*
  absolute volumes (sky/orientation is isotropic, so SNR_ref varies a lot within a bin); the ratio largely cancels
  it, so the ~1.9× volume gain is sound but the absolute Mpc³ numbers are lower bounds, not survey volumes.
- Doc fix: the walkthrough said "5 O3a segments (6.8 h)" — 5×4096 s is **5.7 h** (6.8 h was the old 6-segment v1 number).
- Engineering: a power loss destroyed a full 40-minute all-or-nothing run, so the campaign now **checkpoints per
  segment atomically** and **persists injection rows to parquet** (re-analysis without re-running 4,800 injections).
**Honest headline: O4b expands subsolar reach by 1.23× in distance [1.11–1.35] and ~1.9× in volume over O3a,
significant in every mass bin, with no evidence of mass dependence.** Gated (42 green).

## 2026-08-07 — O4-3: Subsolar PBH Search Reach Campaign (O4b vs O3a)
Executed `O4-3` search reach evaluation (`o4_sensitive_distance.py`) across 2,400 subsolar injections per era on held-out test noise (O3a 5 segs, O4b 8 segs) for single-detector (H1) and H1×L1 coincidence modes.
- **H1×L1 Coincidence Mode Search Reach Results**:
  - **Chirp Mass [0.17–0.35 M☉]**: O3a 12.68 Mpc (8,541 Mpc³) → **O4b 15.73 Mpc (16,300 Mpc³)** | **1.24× distance, 1.91× volume**
  - **Chirp Mass [0.35–0.55 M☉]**: O3a 20.61 Mpc (36,655 Mpc³) → **O4b 26.58 Mpc (78,660 Mpc³)** | **1.29× distance, 2.15× volume**
  - **Chirp Mass [0.55–0.88 M☉]**: O3a 30.76 Mpc (121,907 Mpc³) → **O4b 36.94 Mpc (211,051 Mpc³)** | **1.20× distance, 1.73× volume**
- **Headline**: The cleaner O4b noise in-band expands our subsolar PBH search reach by **~1.24× in distance** and **~1.93× (1.73–2.15×) in surveyed astrophysical volume** over O3a, reaching **up to 36.9 Mpc** in the highest subsolar mass bin.
- Artifacts: `results/o4_sensitive_distance.json` and `results/o4_sensitive_distance.png`. Gate #41 added to `verify.sh` (41 gates green).

## 2026-08-06 — O4 era: the Virgo negative REPLICATES (and the mechanism is now measured)
Triggered by a question about the 12 Nov 2025 subsolar candidate S251112cm. Two findings from the research:
(a) **S251112cm's data is NOT public** — it is in O4c; verified via the GWOSC API (zero public events after
June 2025, zero S25* superevents; the one O4c dataset, O4c1DiscC00, covers a 68-minute window on 2025-02-07).
(b) **I had missed GWTC-5.0** — released May 2026, 161 new O4b events public on GWOSC, 390 total. It contains
NO confirmed subsolar event, so "a big catalog" and "a subsolar merger" are two separate stories that are easy
to conflate. Corrected the record.
Acting on the real opportunity — **O4b bulk data is public (Apr 2024–Jan 2025, H1+L1+V1)** while our entire PBH
arc was built on O3a (May 2019):
- **Transfer scout ✅** (`o4_transfer_scout.py`): the O3a-trained cnn_w64 works on O4b noise **unchanged (0.97×
  sensitive distance)** — per-segment PSD whitening absorbs the era shift, so no retraining needed and the
  re-test is not confounded by model changes. O4b is **1.41× more sensitive** in-band; zero-FA threshold
  2.111 → 1.141 (cleaner noise).
- **N5 Virgo re-test ✅ — REPLICATES.** Our N5 negative ("Virgo does not help subsolar") was measured on weak
  O3a-era Virgo, the obvious objection. On 8 fresh O4b triple-coincident segments (2× the O3a run, 5.5 years
  later): **double/single 1.30× (O3a 1.33×), triple/double 0.95× (O3a 0.94×)** — both within 3%. Virgo still
  costs ~5%.
- **The mechanism is now MEASURED, not asserted** (`o4_asd_compare.py`): median ASD in [50,300] Hz from our own
  strain — V1 is **2.8× louder than the best LIGO detector in O3a, 3.2× in O4b**. Virgo improved 1.14×, LIGO
  improved 1.29×, so **the gap WIDENED**. That is precisely why V1's signal responsiveness fell 19% → 12%.
  The negative is structural at subsolar masses, not an artifact of old data — and it got slightly worse.
- Bycatch: O4b has **177 triple-coincident windows in a 30-day probe** (vs 20 in six months of O3a), and 8/8
  fetched segments were usable (vs 4–5 in O3a). Virgo's duty cycle is transformed; its sensitivity gap is not.
Gated (40 green). Everything ran detached + checkpointed overnight.

## 2026-07-24 (later) — TheBridge wall-taxonomy: δ is species-1 statistically, but the TOTAL error saturates
The bridge classified our δ wall CROSSABLE (species-1) and asked us to falsify it by escalating SNR.
`25_wall_species.py`, three legs, predictions pre-registered:
- **LEG 1 was analytically forced, not empirical:** for any amplitude-linear model in Gaussian noise, block
  inversion gives (F⁻¹)_δδ ∝ A⁻² ⇒ σ_Fisher(δ) ∝ 1/SNR EXACTLY — a finite saturation of the Fisher floor is
  mathematically impossible (a true degeneracy gives σ=∞ at every SNR, not a floor). Measured: slope **−1.0000**,
  σ·SNR invariant to **0.0%** over 1000×. **The bridge's species-1 call STANDS; δ is not species-2.**
- **LEG 2 (the real finding): the waveform-systematic bias is SNR-INDEPENDENT.** Cutler–Vallisneri bias from an
  un-modeled 222 second overtone = **0.072**, flat across 1000× in SNR (13.9% variation, *below* the measured
  19.2% numerical noise floor — I measured that floor rather than loosening an arbitrary threshold to get my
  predicted answer).
- **LEG 3: the TOTAL error therefore SATURATES at 0.072**, crossover at **SNR ≈ 124**.
**FOLLOW-UP CORRECTION (same day):** the bridge formalized SNR≈124 as species-4's "computable crossover", so I
measured its sensitivity before it got over-cited. The crossover = 7.97/|bias| scales as 1/|bias| and spans
**~4–330** depending on the omitted content; under our OWN measured realistic systematic (R3 full-IMR, δ≈−0.33)
it is **~24 — GW250114 is already AT the transition**, not comfortably below it. Corrects our "comfortably
statistics-limited" statement: waveform completeness is plausibly the binding constraint NOW. Retro-explains the
prior-dominated δ (G8) and the visible start-time systematic (R3/B1) as crossover symptoms. Gated (39).
**The answer that the two-option framing misses:** δ is precision-limited *statistically* (more SNR genuinely
helps, exactly 1/SNR) but the total error stops improving at a **model-fidelity floor** — crossable by better
WAVEFORMS (222, (4,4), nonlinear modes), not by better detectors. Actionable: GW250114 (SNR~25) is still
statistics-limited, so more SNR pays; past SNR≈124 (a ~5× louder event, plausible in O5) the no-hair test becomes
systematics-limited and effort should shift to waveform completeness. Ties to B1: the peak-start mass bias is the
same species, and its start-time-convention dependence adds a species-3 (definitional) flavour. Gated, 38 green.

## 2026-07-24 — TheBridge G8: the Fisher-floor number built — NPE does NOT beat the floor, G8 STANDS
The bridge took up our offer; built the Cramér-Rao floor on δ (`24_fisher_floor.py`), pre-registered in the
ringdown lab notebook first. The no-hair model is white-noise whitened → the Fisher inner product is a plain dot
product and SNR=√(Σh²). Fisher matrix over 12 params (M, χ, δ, t0 shared + per-detector A220/φ220/A221/φ221),
correlation-matrix preconditioned (raw cond ~1e22 is pure parameter-scale, not physics), σ_Fisher(δ) step-stable
to 0.8% across a 4× step range. **At GW250114's ringdown SNR ~25: data-only σ(δ)=0.32 ≈ the prior width (0.29) —
the data barely constrain δ**, independently consistent with v6's "informative only at SNR ≳ 37". The NPE
posterior width (0.263) is a proper Bayesian combination between the data floor (0.32) and the data+prior floor
(0.215) — it does NOT beat the information limit. **The clincher — a prior-shrinkage test:** inject δ=0.4, the NPE
median comes back at +0.055 (**86% pulled to the prior center**), PROVING the apparent sub-floor precision is prior
regularization, not data. **G8 STANDS.** Pre-registered honesty footnote: our "σ(δ) 2.6× tighter than classical"
is true but is efficient-Bayesian + prior-regularized precision, not floor-beating. Nothing beats the floor on δ at
current SNR — the deficit is fundamental, exactly as G8 asserts. Gated, 37 gates green. (Process note: a botched
one-liner truncated this JOURNAL mid-write; restored from git — a reminder to never `open(p,"w")` before reading.)

## 2026-07-10 — TheBridge round 6: no ask; no-chase assessment concurred
Status note from the sister repo: two new theory papers (G2-manifold BH remnant, GRG; Dorau & Much entropic
semiclassical-Einstein derivation, PRL) assessed as having NO observational purchase for real GW data —
remnant QNMs sit ~tens of orders above any detector band; the entropic result is a consistency derivation with
no predicted deviation. Sanity-checked the scaling (f_QNM ~ c^3/GM -> Planck-scale remnants are ~40+ orders
above band) and CONCUR: nothing for our stack to chase. Our release capstone + event watcher are featured in
the bridge's CAPSTONE.md (the §22/§23 start-time referee and Follow-up A's CNN-ties-bank among its entries).
Standing OPTIONAL item only: an "A1-decisive" amortization follow-up (5+ NPE variants, C2ST scoring) — shelved
for an instrument-building mood. Nothing waits on us; the watcher watches.

## 2026-07-03 (night) — Follow-up C: release capstone — the arc closes
Made the whole thing durable + citable. Env pins committed for all four venvs (requirements.txt x3 +
requirements-py311.txt for the Py3.11 ringdown package). Rewrote the top-level README from stale (it predated
the entire follow-up arc) to the honest, complete, current story — coincidence +1.37x, the CNN-ties-a-real-
matched-filter benchmark, Virgo-doesn't-help, echo nulls, the no-hair delta + overtone cross-check, and the
event watcher — with a Reproduce section anchored on `uv` + verify.sh (the 36-check gate). Added an MIT LICENSE
(user's choice) + CITATION.cff; verified every README link resolves. A stranger can now recreate each venv from
pins, re-fetch GWOSC data via the numbered scripts, and assert every headline with ./verify.sh.
**The follow-up arc A -> B -> D -> C is COMPLETE:** the real matched-filter benchmark (CNN ties a realizable
bank), the ringdown-package extensions (start-time referee + NPE loop closed + nonlinear-QNM honestly parked),
the event watcher (one command, the whole stack), and now the release. 36 gates green.

## 2026-07-03 (evening) — Follow-up D: the EVENT WATCHER — the whole stack becomes one instrument
`watch_event.py` (repo root) orchestrates the three sub-projects across their three venvs into a single
one-page report for any loud event: (1) ringdown remnant + overtone [ringdown package, venv311], (2) no-hair
delta + Kerr consistency [amortized NPE, .venv], (3) echo Delta-t(M_f,chi_f) prediction + comb p-value [echoes
venv]. Thin per-venv entrypoints (watch_ringdown/watch_npe/watch_echo) emit JSON; the orchestrator isolates
each stage (one failing doesn't sink the rest) and assembles the report. **Reference run on GW250114 in 48 s
reproduces ALL three sub-project headlines** — ringdown M 74.7 + overtone detected (=21/22), NPE delta -0.15
Kerr-consistent (=09), echo Dt 295 ms comb p 0.33 null (=E3). Amortized NPE + cached strain -> seconds per
event, so this is genuinely ready to point at O4b/O5 GW250114-class events (the only ones loud enough to
measure delta, per the v6 wall). Gated (36). Arc: A done, B done, D done -> only C (release capstone) remains.

## 2026-07-03 (later) — Follow-up B: field-standard start-time referee + NPE loop closed; nonlinear-QNM honestly parked
Rode the GW250114 toehold with the `ringdown` package (`.venv311` from R2), all local/zero-compute.
- **B1 ✅** `22_starttime_sweep.py` — GW250114 220+221 across 9 start offsets (0–16 t_Mf). The overtone is
  significant from the peak (P(A221≈0)=0.000) and damps by ~5.4 ms (→0.059) — a real τ221≈1.4 ms mode; the
  peak-start mass is biased HIGH (74.7 vs true 68.1, +10%), drifting −8.8 M☉ later. **The R3 early-time
  systematic, independently reproduced by the field-standard coherent pipeline** (all rhat<1.01).
- **B2 🅿️ parked-honest** — the nonlinear claim (arXiv:2601.05734) is 6 quadratic modes in the (4,4) multiplet
  (BF 74, 3σ). Verified the vanilla package fits ONLY linear (2,2) QNMs — no multi-multipole, no 2·f220
  frequency-locking (the quadratic mode = 497 Hz, 6.9% below the linear Kerr 440). Refereeing it with (2,2)
  machinery would null a real-but-subtle signal → a false negative, which the guardrail forbids. Parked with the
  requirement stated (needs Wang & Ma's custom PyCBC-Inference pipeline). Same discipline that (correctly) parked
  R2 until the right tool existed.
- **B3 ✅** `23_npe_package_loop.py` (synthesis, no new fits) — closes the NPE loop: (1) the NPE (76.0/[68.4,85.2])
  agrees with the package (74.8, CI nested) ⇒ real field-consistent inference; (2) the NPE median sits at ~0 t_Mf
  (the peak) in B1's family ⇒ it weights the high-SNR early regime and **inherits the peak systematic** (+7.9 M☉),
  NOT bias-free from marginalizing t0. Completes the NPE arc: amortized + calibrated + field-cross-validated + now
  located in the systematic.
- **B complete** (B1+B3 gated, B2 honestly parked). 35 gates green. Follow-up arc now A ✅ B ✅ → next **D** (event
  watcher), then **C** (release capstone).

## 2026-07-03 — Follow-up A: the real matched-filter benchmark on the Mac — the CNN TIES a realizable dense bank
Direction discussion picked a 4-item follow-up arc (A dense-bank MF benchmark → B ringdown-package → D event
watcher → C release); A first, and — with the GPU VM down — I re-examined the "intractable locally" call and
found it was made against a *naive* full-coherence plan. Three insights made it tractable on the M4: 64-s
templates (the CNN window), an effectively 1-D Mc bank, and MPS-batched FD correlation.
- **A1** ✅ golden-tested the hand-rolled MPS matched filter (`pbh/bankmf.py`) against first principles — exact
  noiseless recovery, phase-invariant, MPS≡CPU to 9e-7. Found a monster O3a glitch (2310σ) rings the chirp
  template → the statistic must be newSNR/chi²-vetoed (which production searches already are).
- **A2/A2b** — the wall, made quantitative. Full-coherent FF collapses (0.576 at 0.3% spacing) → megatemplate
  scale, matching LVK's real **3,452,006-template** O4 subsolar bank (arXiv:2412.10951; the parked ~1,650 estimate
  was ~3 orders optimistic). But the n=8 **semi-coherent** statistic is tractable: its recovery-vs-spacing curve
  (`bank_semiff.py`: 0.25%→0.86, 2%→0.37) **retroactively explains bank_oracle's parked 0.000** and sets ~0.1%/
  1,619 templates. (A first cut with a 6.4-s injection misalignment gave impossible recovery >1 — caught by the
  physicality check, fixed to oracle-exact alignment.)
- **A3/A5** — the benchmark. `bank_dense.py` ran the semicoherent_oracle statistic at 0.1% on the 6 real test
  segments (template-major, **mid-segment atomic checkpointing** — survived 2 power losses + a Claude app restart,
  each costing ≤10 min instead of hours). `bank_vs_cnn.py` scored cnn_w64 on the **identical** deterministic
  injections. **Result: real bank MF 0.489 vs CNN 0.472 = 1.03× — a statistical TIE.** A single CNN forward pass
  matches a 1,619-template matched-filter bank. The density sweep (83→0.000 reproducing bank_oracle … 1619→0.489)
  is the wall end-to-end; both sit far below the true-template oracle (0.72) ⇒ **template-bank mismatch is the
  dominant loss, not learned-vs-MF.**
- **North-star save:** the v1 gated CNN number (0.45, different injections) implied a ~10% MF win; the airtight
  co-injection (identical injections) shrank it to a ~3% tie — the co-injection prevented an overclaim of exactly
  the headline. 33 gates green. **A is done; the honest answer to the arc's central question is: the learned CNN
  is not leaving meaningful sensitivity on the table versus any realizable detector.** Next: B (ringdown package).

## 2026-07-02 — R2 v2: the Python-3.11 wall falls; the proper pipeline detects the GW250114 overtone; NPE arc cross-validated
Back after a few days. The PLAN backlog was cleared, so attacked the parked shelf: **R2** (tone-count via the real
`ringdown` package) with **E3 off-source prefetch** riding in the background while GWOSC was healthy.
- **Environment (why R2 was parked):** `ringdown` needs Py3.11 — built `.venv311` with uv. Its `jax~=0.4` pins are
  too loose for 2026: three successive era-mismatches (newest jax kills `jaxlib.xla_extension`, matplotlib 3.11
  kills arviz 0.19, scipy 1.17's STFT kills ringdown's pandas-Series in Welch). Working set frozen in
  `.venv311-pins.txt`. Two-venv pipeline: `20_extract_strain.py` (3.12/gwpy → npz) + `21_ringdown_crosscheck.py` (3.11).
- **Rigor:** targets verified before fitting (GW150914 = docs example values; GW250114 = LVK max-likelihood via
  arXiv:2601.05734: t0=1420878141.2362, ra=2.35, dec=0.22, psi=1.37); gates pre-registered in the lab notebook;
  reran at x64 with R̂/ESS recorded (all R̂ ≤ 1.004, ESS ≥ 950) before reading anything off.
- **(a) Validation ✓** GW150914 220+221 lands in the known ballpark (M 77.5, χ 0.76).
- **(b) THE R2 ANSWER ✓ — GW250114's overtone is REAL to the field statistic: A221 bounded away from zero
  (P = 0.000; A221/A220 = 1.02 at peak)**, matching arXiv:2509.08099 — on the same event where our simplified
  white-noise Bayes factor (`14`) saw nothing. The parked "implementation limit, not information limit" call is
  now positively demonstrated; refusing the false negative was right. GW150914 comes out marginal (P = 0.049) —
  consistent with the contested Isi/Farr-vs-Cotesta literature, so the statistic isn't trigger-happy.
- **(c) NPE referee ✓ — the first independent field-standard cross-validation of our whole SBI arc:** package
  M 74.8 [70.4,79.0] / χ 0.729 vs our 09 NPE 76.0 [68.4,85.2] / 0.762 — medians within 1.2 M☉ / 0.033, the
  package's tighter coherent CI nests inside the NPE's. The amortized posterior is real, not a simulator artifact.
- **E3 ✅ (same day) — per-event ML scorers across the broadened set, all clean nulls.** `19_per_event_ml.py`:
  per-event autoencoder scorer + v2 ML network comb at each event's formula-Δt, for GW150914 / GW151012 /
  GW151226 / **GW250114** (Δt from its verified remnant → 0.2952 s, a GW150914 twin). A first pass on the tiny
  own-block background (n=59) threw up GW151012 ML p=0.033 and skipped NaN-cropped GW151226 — so I swapped to the
  **independent ±hour background** (E2-style, own-PSD whitened, 660–1815 segs): GW151012's 0.033 **washed out to
  0.130** (small-sample artifact; the comb never flagged it) and GW151226 was rescued. **All four events clean
  nulls under both statistics.** Gated. **The PLAN backlog is now fully cleared** — every tractable item gated;
  only VM-scale extensions remain. 32 gates green.

## 2026-06-26 — N4 SSL follow-up + R1 recalibration + N5 triple-detector (data-blocked tonight by GWOSC)
Continued the PLAN knock-out. Each item gated + documented honestly; committed at milestones.
- **N4 sensitive-distance follow-up ✅ — the SSL win TRANSLATES to distance (at a defined FAR).** `ssl_sensdist.py`
  reads efficiency-vs-SNR straight from the val shards (each injection's `in_window_snr`+`chirp_mass`). At the
  **strict zero-FA** threshold the reduced-budget distance is **0 for both** SSL and from-scratch — a model-strength
  floor (the headline distance needs ~full-data AUC 0.79), NOT an SSL failure. At a **softer (1%) FAR** the SSL win
  **does** translate, same data-wall signature: +0.278 distance-fraction @2000 labels (from-scratch non-functional)
  → +0.01 @8000. So the AUC win is a real *detection* improvement, not an AUC artifact. (Realization en route: the
  SpectrogramCNN is NOT standardization-invariant at eval — BatchNorm uses fixed running stats — so the eval must
  standardize the input; caught by an assertion.) Gated.
- **R1 ✅ — per-parameter recalibration: gate-passing, but global T wins (honest low-value).** `17_recalibrate_perparam.py`
  fits separate T_M/T_χ/T_δ (one T-sweep suffices — `widen()` is per-column). Per-param coverage 0.94/0.92/0.90,
  each in [0.85,0.95] (PLAN criterion met), **but mean|cov−0.90| 0.020 (per-param) vs 0.011 (global)** — the per-param
  fit overfits the n=600 calibration-set noise. Confirms v3's single global T=1.05 was the right, sufficient choice.
  GW250114 δ −0.16 [−0.46,+0.33] Kerr-consistent, unchanged. Gated.
- **N5 triple-detector H1×L1×V1 ✅ DONE (2026-06-27) — honest NEGATIVE: Virgo does NOT help subsolar.**
  Built `coinc_triple.py` (extends the G1 +1.37× double-coincidence to a 3rd detector: cnn_w64 on H1+L1+V1, 3-way
  time-slide matched-FAR background, injections projected onto all 3). **Found the local H1∩L1 test segments are ALL
  Virgo duty-cycle gaps (0/5 clean V1)** → discovered 20 true H1∩L1∩V1 segments (intersecting the 3 DATA flags), 4
  leakage-free fetched by a **persistent checkpointing fetcher** (GWOSC was badly degraded ~12 h overnight; the
  fetcher accumulated segments as the network flickered back). **Result: (1) double H1×L1 reproduces the win on fresh
  data — 1.33× over single (validates G1/Build-C); (2) triple = 0.94× double — Virgo marginally HURTS.** Mechanism
  (diagnostic): V1 signal responsiveness +1.2 vs H1 +5.1 / L1 +7.4 (~19%) — too insensitive at subsolar to carry
  signal, so summing its near-noise score + the higher 3-way threshold degrades the sum. Also rules out the
  learned-triple (no V1 signal to weight). **H1×L1 double-coincidence is the subsolar ceiling.** Gated (30 gates).
  Eng note: per-segment checkpointing (`coinc_triple_rows.parquet`) survived **5+ power losses + Anthropic
  service-busy interruptions**, resuming from the last finished segment (3/4 segs survived the final power loss → a
  ~50-min restart became a ~13-min finish).
- **PLAN backlog now fully worked through** — every tractable item is gated + documented or honestly parked
  (R2 needs Py3.11 `ringdown`; E3-broaden + lower-FAR need the VM). 30 gates green.

## 2026-06-25 — backlog-execution day: PLAN tracker + 4 echo/ringdown items + a verified physics formula that caught a bug
A long "knock them out" session against a new [PLAN.md](PLAN.md) (tractable backlog mined from all docs + new
cross-cutting angles). Also did a cross-session **prior-art audit** for all three sub-projects (verified every
citation myself; all novelty claims survive once scoped — pbh strain-trigger vs PBH-population ML, ringdown
amortized-SBI-with-start-time-marginalized, echoes autoencoder-anomaly-on-echoes) and saved a `prior-art-rigor`
memory. Items worked, in order, each gated + documented honestly:
- **E1 ✅ (honest negative).** ML scorer does NOT tighten the echo upper limit. Caught + neutralized the
  whitened-domain "13× artifact" (it reappears if you run the ML scorer through the whitened harness); the honest
  production-path A90 (`12_ul_production.py`) is ML≈comb. The gated comb UL stands.
- **E2 ✅.** GW150914 echo null holds vs an independent 660-pair, different-day background (`13_independent_bg.py`)
  — 4× larger, own-PSD-whitened → robust to the stationarity/shared-block assumption.
- **R2 ⏸️ parked honestly.** Built explicit Bayesian tone-count (`14_bayes_tonecount.py`), but the oracle
  diagnostic + the *published* GW250114 overtone detection proved the simplified machinery isn't a fair test —
  did NOT ship a false negative. Needs the proper `ringdown` FD pipeline (Py3.11, deferred).
- **🎯 Echo Δt(M,χ) formula ✅ VERIFIED + caught a data bug.** `14_echo_spacing.py`: Kerr-tortoise echo spacing
  from first principles (Abedi Eq.2), uncalibrated, reproduces all 3 Abedi Table-I values to **<2%**. Caught that
  the repo's hardcoded Δt were WRONG (GW151226 0.0579→0.1013, LVT151012 0.1013→0.1778 — a mislabel; a prior
  session's "correction" was itself the bug). Fixed + re-ran the two events (nulls hold). GW150914 always correct.
- **N1 ✅ (flagship).** `15_joint_ringdown_echo.py`: propagated GW250114's ringdown mass posterior through the
  verified formula → a tight echo Δt prior (0.357s [0.304,0.445]) → conditioned the echo search on the SAME
  event. 3.2× fewer trials → **1.11× more sensitive** (A90 1.90→1.72σ); on-source null. The first echo search
  conditioned on its own ringdown. Honest: modest gain (steep efficiency curve), bigger for tighter-mass events.
- **N3 ✅ (stacked population echo).** `16_stacked_echo.py`: stacked the comb statistic across 4 events at each
  one's formula-predicted Δt → population NULL (stacked z=−5.17 vs +5.05 threshold, p=0.998), combined limit
  1.21× tighter than best single (below √N=2× — heterogeneous events). A population non-detection.
- **R3 ✅ (IMR referee) — the other standout.** `15_imr_referee.py`: injected NR-calibrated full-IMR ringdowns
  (IMRPhenomXAS via pbh pycbc) into the no-hair NPE. Unbiased on its analytic-tone family (δ=+0.02) but a
  **δ≈−0.33 systematic on realistic ringdowns from the peak**, decaying to ~0 by 6 ms post-peak → it's the
  early-time merger/overtone content the two-tone model omits. **Independently quantifies the start-time
  systematic** central to the no-hair controversy; a caveat on the GW250114 δ (Kerr-consistent still holds).
- **N2 ✅ (honest mixed/modest).** Tested a robust H1×L1 consistency-weighted echo statistic (a learned head
  would overfit the tiny echo data). Caught a selection-bias trap (best-λ at n=20 looked like ~10%); rigorous
  pre-chosen λ + bootstrap → significant for GW150914 (~4%) but not GW250114 → event-dependent, not a universal
  win (the comb-sum is already Δt-consistency-aware).
- Infra: dashboard live; transient gwosc SSL timeouts on the GW250114 fetch (retry-to-cache); **23 gates green**.
  **All tractable PLAN.md items now complete** — only the big multi-session builds (N4 self-supervised backbone,
  N5 triple-detector) and the low-value tail (R1) remain.

## 2026-06-20 (night) — Build C-2 + ringdown v6: a LEARNED coincidence beats sum (significant, leakage-free), and the δ-SNR wall is mapped
- **PBH Build C-2 (GPU VM) — the night's headline.** Asked whether a *learned* H1×L1 coincidence statistic can
  beat the plain `sum` of per-detector scores (G2a had said no — but only for simple scalar combos). Built
  `coinc_learned.py`: cnn_w64's 256-d penultimate embeddings of the H1 and L1 windows → consistency features
  `[eH, eL, |eH−eL|, eH·eL]` → a small head trained to tell real coincident injections from time-slid noise pairs
  (it learns whether the two detectors *agree*). **Learned beats sum at 5/5 FARs, all 3 mass bins, the gain
  growing at stricter FAR.**
- **Held to the north star — two stress-tests before claiming.** (1) *Leakage:* the head trains on noise and is
  evaluated against noise → could memorize noise realizations (the δ-stacking trap). Ran three modes — leaky,
  held-out-noise, and the gold-standard **held-out-segments** (train 16 segments, eval 8 *unseen* ones). The gain
  is stable across all three ⇒ not memorization. (2) *Significance:* bootstrap B=500 over the 2000 held-out-segment
  injections — **every FAR × every mass-bin 90% CI excludes zero, P(learned>sum)=1.00**. The +0.02–0.05
  sensitive-distance gain (≈+5–15%, on top of sum's +1.37× over single-det) is real. First thing to beat sum for
  subsolar coincidence. Gated (cross-segment + bootstrap CI>0). Added a segment-tagged embedding cache so the
  cross-segment test was a fast re-run, not a 60-min regen.
- **Ringdown v6 (Mac, in parallel) — `14_delta_threshold.py`.** Completed the δ-stacking story from the other side:
  13 found only GW250114 measures δ; 14 maps *why*. Swept injected ringdown loudness, measured σ(δ) vs whitened
  ringdown SNR — δ only becomes informative at **ringdown SNR ≳ 37**, and even at the top of the trained loudness
  it's just ~13% tighter than prior; GW250114 (real, σ/prior 0.83) sits right at that edge. Seed-robust (0/1/2),
  gated. The stacking starvation is now quantitative, not anecdotal.
- **Pushing Build C-2 further (morning): two robustness probes + a real bug caught.** Per "keep pushing the
  learned coincidence": **(A)** head-seed robustness — learned > sum across 5 independent head seeds (not a lucky
  init). **(B)** does a higher-AUC base model compound? — verified cnn_hl leakage-free on the Build-C segs, then
  found it helps the learned statistic too (base-agnostic) but does NOT compound (≈ cnn_w64 within seed noise);
  the gate-critical cnn_w64 suffices. **Lower FAR:** the leakage-clean `--holdout-noise` reaches 1/year with
  learned still significantly > sum (Δ+0.048[+0.030,+0.071]).
- **🐛 honest-slides bug, caught while pushing FAR (north star at work).** The time-slide background used 4000
  lags on only ~500–1500 noise windows — but there are only N−1 distinct circular lags; the rest repeat (and
  re-inject the zero-lag/on-source). That overcounted T_bg ~5–8× and inflated the reachable-FAR *labels*
  (held-out-segments really reaches 1/month, not 1/year; Build C's "12.3 yr" is honestly 4.6 yr). Fixed in
  coinc_learned.py + coinc_far.py (cap at N−1), re-ran everything honest. **The learned>sum and Build-C
  conclusions are unchanged — only the optimistic labels were.** Also corrected an over-claim: the light mass
  bin's gain is marginal at the loosest FAR (high-mass is the robust headline). Gates updated to the honest FARs.
- **Infra/notes:** used both machines (VM for pbh, idle Mac for ringdown). Weathered repeated transient SSH/IAP
  drops (retry loops; don't hold sessions open) and git-pull conflicts from VM-regenerated tracked artifacts
  (lesson: scp result JSONs directly, don't `rm`+checkout before pulling — that restores stale committed copies).
  Full regression gate ALL GREEN (17 gates). Detail: RESULTS.md (Build C-2 + honest-slides), ringdown lab notebook (v6).

## 2026-06-20 — Build C on GPU VM: pbh coincidence advantage is FAR-ROBUST (the win, completed at scale)
- Moved to a free L4 GPU VM (alphaludo-l4) for the one carried blocker the Mac couldn't touch: does the
  +1.37× coincidence advantage survive at a REALISTIC false-alarm rate? Set up an isolated workspace
  (~/deepstrain, clear of the user's other VM projects), cloned the public repo, built a venv
  (torch 2.12+CUDA, pycbc, gwpy), transferred cnn_w64.
- **Efficiency:** the workload is CPU/data-bound (whitening + spectrograms + subsolar-waveform generation),
  the cnn is tiny → the L4 GPU is overkill for the model. So used the box right: parallel whitening +
  **1 worker/segment over 8 cores** for injection generation (~8× over serial; ~5 h → 25 min), GPU only for
  the batch cnn forward, RAM to hold data. Did NOT GPU-port the spectrogram (would change the inputs the
  model trained on — north star).
- **Result (fetch_coinc.py + coinc_far.py):** 24 fresh H1×L1 coincident O3a segments (26.9 h, no train
  leakage), global time-slide background = **N−1=1511 distinct lags × 26.9 h = 4.6 yr** (honest; an earlier
  "12.3 yr" used 4000 lags but there are only N−1 distinct circular lags — overcounting, fixed) → probe FAR to **1/year**.
  Coincidence degrades only GRACEFULLY (1/6h→1/year loses ~15–20%). **Coinc @1/day = 1.33/1.32/1.43× over
  single-det floor — reproduces the stress-tested local G1 +1.37–1.48× (cross-check ✓); even @1/year (a FAR
  the single detector cannot reach from this data) coinc still beats the single-det floor by ~1.2×.**
- Gated in verify.sh (ALL GREEN). The best win is now a realistic-FAR result. Artifacts: results/coinc_far.{json,png}.

## 2026-06-20 — roadmap night: echo UPPER LIMITS (P1+P2) + ringdown δ-STACKING (P1), both ✓
- **Echoes v6 — upper limits (11_upper_limits.py):** generalised 06's single-Δt sensitivity into a 2-D
  (amplitude × spacing) efficiency map at N=300, per-Δt p<0.01 background → A90(Δt). Given the on-source
  null, we now **EXCLUDE first-pulse amplitude ≥ A90: GW150914 1.65σ, GW151226 1.72σ** at predicted Δt,
  across all spacings. Converts "found nothing" → a real exclusion. N=300 = decisive (P2).
- **Ringdown v5 — multi-event no-hair δ stacking (12_stacking.py):** common-δ precision-weighted combo
  of the recalibrated NPE posteriors. **σ(δ) provably tightens as √N** (injections: N=8 → 0.095 vs ideal
  0.097, unbiased, calibrated); real **GW250114+GW150914 → δ = −0.090 ± 0.176, Kerr-consistent, ~1.3×
  tighter** than the best single event. The amortized no-hair net combines across events as theory predicts.
- Both gated in verify.sh (ALL GREEN). ROADMAP P1×2 + P2 marked DONE. Two real, validated results on the
  same data in one session. Remaining roadmap: the tone-count guardrail (keep parked) stands.
- **STRESS-TEST CORRECTION (north star, 13_more_events.py):** before stacking MORE events I cross-checked
  the NPE on 8 real events → **only GW250114 measures δ; all 7 fainter events return ≈ the prior**
  (δ_σ/prior 0.93–0.99). ⇒ the δ-stacking REAL-event "GW250114+GW150914 → 1.3× tighter" was a
  Gaussian-approx-of-prior **artifact** — genuine combined constraint ≈ GW250114 alone. The stacking
  METHOD (√N on informative injections) stands; the real multi-event payoff is **parked** (per-event SNR
  information wall — only SNR~80 events measure δ). Corrected lab notebook / CLAUDE / ROADMAP / verify.sh
  (removed the misleading "stack<singles" gate; added the stress-test gate). The north star caught our
  own fresh over-claim within the hour — working as intended.
- **STRESS-TEST SWEEP (north star) — full results:**
  - **δ-stacking:** ❌ real-event payoff was a prior artifact → CORRECTED + parked (method valid).
  - **Echo upper limits:** ✅ PASSED (statistic == 05/06 convention; self-consistent; threshold not
    glitch-driven; cross-checks v1) — but caught + fixed a wrong GW151226 Δt (0.105→0.0579 → A90 1.55σ).
  - **Leg-8b family robustness:** ✅ SETTLED at N=300 (in-band differences real & physical; OOB control =
    known whitened-domain artifact).
  - **pbh coincidence +1.37×:** ✅ PASSED — single-det SNR50÷√2 = 19.3/19.6 per-det matches v1's
    independent 18.6; matched-FAR accounting correct (305th-of-305 livetimes = 1 FA/livetime = single-det);
    zero-lag max < strict thr. The biggest win holds up.
  - **No-hair single-event:** ✅ incidentally validated (GW250114 genuinely informative; δ unbiased on
    Kerr injections despite the known +10% mass pull).
  ⇒ Net: 2 errors in fresh work caught + corrected; all surviving claims now adversarially verified.

## 2026-06-15 — ringdown v4 tone-count: PARKED, honest negative (6 attempts, full diagnostic chain)
- Pivoted from pbh (parked) to a new ringdown thread: an amortized, start-time-marginalized AI to count
  QNM tones (1 = 220 only, 2 = 220+221 overtone) — addressing the live GW150914 overtone controversy,
  whose crux is start-time dependence (which the SBI infra marginalizes by construction).
- First cut didn't transfer to real data. Chased it down a clean diagnostic chain: (A) scale → norm;
  (B) noise coloring → train on real O4 noise; DIAGNOSTIC → caught a "loud⇒2-tone" SNR shortcut;
  (C') SNR-matched classes → removed it; DIAGNOSTIC → whitening reshapes the ringdown (raw-vs-whitened
  shape overlap 0.48; built a fast FD whitening matching gwpy to 1.000); (D) injection-convention-matched
  training → transfer pathology GONE (GW250114 read 2-tone) BUT model overfit the 14-chunk pool;
  finally 60 chunks + fresh-per-epoch + early-stop → overfitting fixed.
- **Verdict (honest NEGATIVE with a now-trustworthy model):** calibrated (ECE 0.006) but WEAK — held-out
  AUC ~0.61, can't confidently call tone count on real events (GW250114 P(2-tone)=0.32; the earlier 0.69
  was an overfitting mirage). Black-box ML tone-count is too weak at this data/SNR scale. Salvage: a
  calibrated detectability threshold (overtone SNR≈5 for 50% detection). The diagnostic chain itself is
  the contribution. Six-attempt table in ringdown notes/lab_notebook.md; survived 3 power losses (all
  artifacts disk-cached/reboot-safe). Come-back-later: more data/coherent model, multi-event stacking,
  or explicit Bayesian model selection with a real noise model.

## 2026-06-15 — pbh path G CLOSED: +1.37× coincidence is the ceiling (G2a/G2b negatives)
- (G2a, coinc_stat.py) better coincidence statistic: no gain — `sum` already optimal (min/prod-prob/
  max+min all ≤ it). (G2b, build_hl.py + cnn_hl) H1+L1 training: built a 64-s H1+L1 spectrogram set
  (self-contained, resumable, no eval leakage), trained cnn_hl → val AUC 0.804 (> cnn_w64 0.793) but
  coincidence FLAT (0.345/0.375/0.420 ≈ cnn_w64's 0.345/0.382/0.428). Higher AUC didn't carry to the
  operating point (tail-separation-limited, not AUC-limited). Finer 10-ms timing coincidence stays
  blocked by the bank-density wall.
- **Path G headline:** single-detector learned subsolar search is noise-floor-limited; H1×L1 coincidence
  recovers ~1.4× sensitive distance (~2.5× volume) — the honest ceiling for the learned approach at this
  data/compute scale. Remaining work is robustness only (lower FAR ← more coincident data).
- **Infra:** survived ANOTHER reboot (~11:56). build_hl.py resumable; build+train+eval all finished
  pre-reboot (cnn_hl.pt 05:24, coinc_eval_cnn_hl.json 05:52), only /tmp logs lost. Note: the g2b_chain.sh
  orchestrator does NOT survive a reboot (a nohup bash dies with it) — fine here since work completed first,
  but truly reboot-proof automation would need launchd/cron. Saved a documentation-discipline memory.

## 2026-06-14 — pbh v2 second pass: cross-field brainstorm → coincidence PIVOT → FIRST POSITIVE (+1.3–1.5×)
- Took 3 external models' brainstorms, triaged them; convergent diagnosis (weak supervision / noise floor)
  held. Ran the cheap diagnostics first: glitch-robust re-threshold (threshold_robust_eval.py) REFUTED the
  single-glitch hypothesis but sharpened it (V2 weakly real, dies to a fat noise tail).
- F0 bank-mismatch gate (bank_oracle.py + coinc_check.py): a coarse template bank gives 0.000, and the
  clean true-vs-bank diagnostic QUANTIFIED why — subsolar needs ≤0.1% Mc template spacing (+1% Mc → SNR
  dead), ~1,600+ templates → intractable locally; extrinsic params (sky/inclination) are irrelevant (the
  quadrature MF is orientation-invariant). This is why F0 was flat-zero.
- **The pivot (G0):** coincidence kills the NOISE floor, not the SIGNAL-recovery (bank-density) problem →
  ride it on the LEARNED model (cnn_w64), not the broken bank. Fetched 8 more L1 coincident segments
  (10 total; 5 overlap H1 test).
- **G1 — FIRST POSITIVE (coinc_eval.py):** cnn_w64 per-detector + H1×L1 coincidence with a TIME-SLIDE
  background (18,910 accidentals from 5 segments). At matched FAR, two-detector agreement gives
  **+1.3–1.5× sensitive distance** over the single-detector ML search (1.48× high-mass → ~2.3–3.3× volume).
  Pipeline cross-checks v1's per-detector SNR50 (~18.6). After a long run of honest negatives, the
  coincidence lever finally moved the number. Caveats: coarse window-level coincidence, ~1/6h FAR, H1→L1
  transfer. Next: G2 = finer coincidence (timing/phase). Full tables in RESULTS.md.

## 2026-06-14 — pbh v2 rung 3 stage 1 CLOSED: definitive negative (A/B/C exhausted)
- Finished the "be sure of the hurdles" pass before concluding. **(B) SemiCoherentNetV2** —
  learnable matched-filter front end (64 quadrature templates → phase-invariant |⟨d,t⟩|² map,
  the oracle's statistic, learned). Capacity gate passed; full run lr=3e-4/20k/20ep was **stable,
  monotonic, clean plateau val AUC 0.691** (no thrash) — eval **0.000/0.000/0.000**. **(C)**
  definitive original-arch run revealed the earlier "flat 0.69 plateau" was a **short-probe
  artifact**: at full budget V1 **overfits/destabilizes** (train loss 0.50→0.46 smooth while val
  AUC oscillates 0.31↔0.62, below chance on most late epochs), best 0.706, eval **0.000/0.000/0.000**.
- **Verdict:** the ~0.69–0.71 AUC wall is robust across BOTH natural learned realizations — not an
  architecture quirk, not optimization (V2 converges cleanly and still hits it). The explicit
  matched-filter front end only made training better-behaved; it did not raise the ceiling. Stage 0
  proved the phase info is recoverable (oracle 0.66–0.76 ≫ cnn_w64 0.41–0.48), but neither
  learned-from-strain design realizes it (<cnn_w64's 0.79 → 0 sensitive distance at zero-FA).
  **The 45→70% gap needs a coherent / fully-matched-filter method (or true-waveform supervision at
  much larger scale), not a better classifier on whitened strain.** Open threads banked in RESULTS.md.
- **Infra:** survived a THIRD power loss mid-C — resumed from the epoch-5 atomic checkpoint with zero
  work lost (dashboard relaunch gotcha noted: root dashboard runs under system `python3`, not `.venv`).

## 2026-06-13 — pbh v2 rung 3 stage 1 (learned semi-coherent model): NEGATIVE so far
- Built the learned realization: SemiCoherentNet (per-chunk 1-D ResNet on whitened strain +
  consistency combiner, 1.24M), on-the-fly strain-injection dataset from a 2500-waveform pool,
  train/eval + self-healing overnight runner. Overfit gate passed (capacity OK).
- First full run (sweep winner lr=1e-3, 16 epochs) UNSTABLE: val AUC peaked 0.687 @ep0 then
  collapsed/thrashed to ~0.35; eval **0.000/0.000/0.000** vs cnn_w64 0.41/0.46/0.48, oracle
  0.66/0.76/0.75. Exhaustive LR + grad-clip probing: only lr=3e-4 stable (flat ~0.69), all
  higher LRs collapse (below-chance cliffs = exploding gradients), clipping doesn't fix.
  ⇒ ~0.69 is an ARCHITECTURE ceiling < cnn_w64's 0.79 ⇒ ~0 sensitive distance. Stage-0 phase
  info real but this learned design can't realize it.
- **Infra win:** survived TWO power losses + repeated session-kill of background tasks. Lessons:
  long runs must be nohup-detached (not harness background tasks) [[nohup-long-running]];
  per-epoch atomic checkpoint + --resume; ps RSS undercounts MPS memory (use Activity Monitor).
- Not closing yet (be sure of the hurdles): (B) matched-filter front-end architecture,
  (C) full lr=3e-4/20k definitive run. Table in pbh RESULTS.md.

## 2026-06-13 — pbh v2 rung 3 stage 0 (semi-coherent oracle): GATE CLEARED (first non-negative)
- After rungs 1&2 ruled out score aggregation, diagnosed the 45->70 gap as a *representation*
  problem (magnitude spectrograms discard phase; MLGWSC ~70% used time-domain). Chose option B
  (semi-coherent learned bank, MFCNN-style) over a plain 1-D ResNet port — it attacks subsolar's
  bank-dephasing pathology and scales to long signals.
- Stage 0 = oracle ceiling before building. First smoke gave 0% -> diagnosed as REAL-noise glitch
  domination (2310-sigma glitch; not a bug — synthetic noise gives the correct chi^2_2n ~33).
  Added the chunk-consistency chi^2 veto (PyCBC newSNR); n=1 can't veto by construction.
- Result (1500 inj, 3 ceilings): **n=8 chunks vetoed = 0.66/0.76/0.75 vs cnn_w64 0.41/0.46/0.48**
  (SNR50 ~11 vs ~18). Sweet spot n=8 (coherence vs glitch-robustness). First rung not ruled out.
- **Honest caveats banked:** oracle = true templates (learned model lands below); threshold is
  lenient (6-segment noise; clean ceiling >1.0 proves it) so the absolute number is OPTIMISTIC.
  Decision: build stage 1 (learned n=8 model), measure the oracle->learned gap.
- Process: never `pkill -f` (killed another session's dashboard); dashboard now writes
  `.dashboard.pid`, stop via `kill "$(cat .dashboard.pid)"`. Memory saved.

## 2026-06-13 — pbh v2 rung 2 (shorter windows + accumulation): also negative
- **Score aggregation exhausted.** Parameterized the whole pbh pipeline by window length
  (config/spectrogram/sweep/build_dataset/train/track_eval), v1 path byte-identical + gate
  green throughout. Built 64-s shards (40k/2.5k, 128×63), retrained `cnn_w64` (val AUC 0.793
  > v1 0.777), added `sum_track` (√k-normalized summed logits) to pbh/aggregate.py.
- **Result:** `sum_track` ≈ `max` ≈ `oracle` (≤+0.007) — even the duration-oracle can't beat
  the single best 64-s window ⇒ independent per-window evidence does not accumulate. Decision
  (c). Confound flagged: 384 non-overlap noise windows vs v1's 2868 → no FAR-matched "shorter
  helps" claim. Both aggregation rungs negative; gap needs a sequence-aware/coherent method.
  Pre-registered before building; table in pbh RESULTS.md. cnn_w64.pt + eval_cnn_w64_track_w64.json.
- **Dashboard fix:** build_dataset.py (per-segment) + train.py (intra-epoch loss sparkline)
  now heartbeat, so build/train are visible live, not just eval.

## 2026-06-13 — pbh v2 rung 1 (track aggregation) + repo decoupling + dashboard
- **Decoupled from SpaceTime:** all three `.venv` folders had internal paths rewritten
  `SpaceTime/`→`BlackHole/` (pyvenv.cfg, activate, ~400 console shebangs); skill re-adapted
  to BlackHole; code had no runtime coupling. Gate green through the rewritten venvs.
- **verify.sh** gained a pbh assertion (eval_cnn `mf_distance_fraction`) + now a track
  assertion (max-control reproduces v1 threshold).
- **pbh v2 rung 1 — track-score aggregation = clean negative.** New `pbh/aggregate.py`
  (max / boxcar_bank / count_above) + `pbh/sweep.py` (shared window grid, refactored out of
  evaluate.py) + `scripts/track_eval.py` (full-signal segment injection → spliced sweep →
  masked stats, atomic+resumable). 1500 injections: no statistic beats the per-window `max`
  control (≤+0.01), and the duration-`oracle` ceiling doesn't either ⇒ the 45→70% gap is
  *within the 256-s window*. Bycatch: the sweep protocol lifted v1's high-mass bin
  0.413→0.447 (alignment gain). Pre-registered before running; full table in pbh RESULTS.md.
  Next is rung 2 (shorter windows + retrain, still local).
- **Dashboard:** root `dashboard.py` (stdlib HTTP, live UI) over `*/results/progress/*.json`;
  pbh gained `pbh/progress.py`. Watches all three sub-projects.

## 2026-06-13 — repo split: black-hole projects moved out of SpaceTime
- echoes/, ringdown_spectroscopy/, primordial_blackhole_search/ moved from SpaceTime
  to this BlackHole folder, joining the existing black-hole concept notes. New CLAUDE.md,
  verify.sh (echoes 07 + ringdown 09/10 asserts), and a copy of the ai-coding-standards
  skill set up here. The three shared concept docs (dimensional_ladder, emergent_dimension,
  3plus1_vs_2plus1) are copied into both repos. Both repos will be `git init`'d separately.
- **venvs decoupled from SpaceTime (2026-06-13):** all internal absolute paths in the
  three `.venv` folders (pyvenv.cfg, activate scripts, console-script shebangs) rewritten
  `SpaceTime/`→`BlackHole/`. Zero SpaceTime references remain, the gate passes all-green
  through them, the science stacks (torch, gwpy, pycbc) import, and `source activate` now
  resolves to the BlackHole path. The projects are independent of SpaceTime.

---

## State at split — the three completed arcs (detail in each notes/lab_notebook.md)

**echoes/ — v5 COMPLETE (2026-06-13).** Post-merger GW-echo comb search on real LIGO
strain. The honest final number: production-path ML advantage ≈ **1.2×** over the plain
comb (ML 50% point ≈ 0.85σ vs comb ≈ 1.05σ). The earlier "13×" was a whitened-domain
convention artifact, caught and corrected. Story: modest real ML edge + band-honest +
family-robust + periodicity-specific + on-source nulls (GW150914 p=0.75). Shelf:
independent background blocks, per-event scorers, FAR scaling.

**ringdown_spectroscopy/ — v3 COMPLETE (2026-06-13).** No-hair test via amortized SBI
(NPE over M, χ, δ with start-time marginalized by construction — the network IS the
test). σ(δ) ≈ 2.6× tighter than classical; v3 added post-hoc temperature recalibration
(T=1.05, held-out coverage mean 0.911). GW250114 δ = −0.16 [−0.46, +0.33], Kerr-consistent,
landing exactly on the classical point estimate. Shelf: per-param recalibration, simulator
realism, tone-count selection, stacking, SXS injections.

**primordial_blackhole_search/ — v1 COMPLETE (2026-06-10).** Deep-learning search for
subsolar-mass mergers (primordial BH candidates) in public LIGO strain. CNN reaches
41–45% of ideal matched-filter sensitive distance at a zero-false-alarm threshold (6.8 h
real noise), flat across subsolar masses; transformer an honest negative. End-to-end
H1×L1 coincidence demo recovers an SNR-22 event and rejects a louder glitch. Shelf:
close the 45→70% gap, FAR→1/month, ViT rematch, H1+L1 training, eccentric corner.
