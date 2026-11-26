# INVALID: 2-box MPS pilot of 2026-11-26 21:07–21:11

These files are kept, but their scores are **invalid**. Do not use them for anything except the timing they record.

**What went wrong.** On MPS, `coh2t_drive.score` divided the engine's complex z by a Python float, σ·κ ≈ 4·10⁻²⁶.
That returns **NaN** on MPS (on the CPU it is correct). As a result:
- `tpeak_max` is NaN everywhere;
- `plain_net` kept its −1 placeholder, because `max(−1, nan)` silently returns −1;
- the χ² band pieces are finite, but they were computed at arbitrary templates, since argsort of NaN is meaningless.

**Scope.** Only `coh2t_drive.py` from 282cf15 onwards, run on MPS. That was only this pilot. Every Fable / verification
result divides on the CPU after `.cpu().numpy()`, and all of them are finite (files checked).

**Fixed in:** the commit that adds this README. The fix multiplies by the reciprocal; uses a −inf placeholder;
asserts finite values at every stage as a correctness gate; computes the band sums with fixed shapes; and empties
the MPS cache per box.

**What is still valid:** timing. Per candidate, about 2.5 s on box 1410 (Mc 0.35, 8,277 templates) and about 1.7 s on
box 2123 (Mc 0.50, 2,291 templates). The `seconds` field is cumulative per box across kinds. Footprint 3.2–4.3 GB
(engine).
