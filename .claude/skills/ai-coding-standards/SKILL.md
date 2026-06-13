---
name: ai-coding-standards
description: General engineering standards and anti-slop guardrails for AI-assisted coding in ANY language or framework. Use this skill whenever you write, review, refactor, or plan code in this project — features, bug fixes, dependency changes, tests, refactors — even for "quick fixes" and one-line changes. It encodes the working loop (search-before-write, smallest diff, verify-before-done), dependency restraint rules, and countermeasures for documented AI-generated-code failure modes (duplication, hallucinated packages, stale APIs, weakened tests, premature "done" claims). Grounded in 2024–2026 research (GitClear, DORA, METR, USENIX, Veracode).
---

# AI Coding Standards (framework-agnostic)

> Portable skill. To install: copy this folder to `<project>/.claude/skills/ai-coding-standards/` (project-level) or `~/.claude/skills/ai-coding-standards/` (all projects). Then fill in the "Project adaptation" section at the bottom for the specific stack.

## Why this exists — the evidence

Research on AI-generated code shows its failures are **additive**:

- Duplicated code blocks rose **~8× in 2024**; copy/paste exceeded refactoring for the first time on record (GitClear 2025, 211M changed lines).
- Each +25% AI adoption correlated with **−7.2% delivery stability**, driven by larger change batches; gains only materialize with small batches + strong tests (DORA 2024/2025).
- Experienced devs were **19% slower** with AI on real tasks — while believing they were ~20% faster (METR RCT 2025). Perceived speed is not real speed.
- **19.7%** of LLM-recommended packages don't exist; attackers pre-register hallucinated names with malware — "slopsquatting" (USENIX Security 2025).
- **45%** of AI-generated code introduced OWASP-class vulnerabilities; newer models were NOT more secure (Veracode 2025).
- Top developer frustration: "AI solutions that are *almost right, but not quite*" (66%, Stack Overflow 2025).

**The pattern:** AI failure is additive — more code, more duplication, more defensive bloat, bigger diffs, premature "done". Quality engineering is **subtractive and verificatory** — reuse, delete, scope down, prove. When in doubt, make the subtractive move.

## The loop — every task, no exceptions

**Before writing code:**
1. Read the 2–3 nearest existing files. Match their style, naming, and patterns exactly — consistency beats personal preference.
2. Search the codebase for existing helpers/components/modules that already do the job. A near-duplicate of existing code is a defect, not a style issue.
3. If the repo already solves a class of problem (error handling, HTTP client, DI, validation, date formatting), extend that solution. Introducing a parallel mechanism for a solved problem requires explicit human approval first.
4. Verify every API against the **installed** version (lockfile + that package's docs/changelog), not memory — training data is stale.
5. Plan briefly. For multi-file work, state the plan before editing.

**While writing:**
- Smallest diff that completes the task. No drive-by reformatting, renames, or refactors mixed in — propose those separately.
- Comments explain *why* only — never narrate what the next line does. No emoji, no "Step 1/2/3" comments, no banner comments.
- No speculative abstraction: no interface/abstract class with one implementation, no manager/wrapper/config layers "for later". Rule of three before extracting.
- Trust the type system: no redundant null/undefined checks on values the types already guarantee; every try/catch, retry, or timeout must name the failure scenario it handles.

**Before claiming done — the gate:**
1. Run the project's full local gate: formatter + linter/static analysis (zero findings) + test suite. If the project has a verify script, use it.
2. Show the real output. Never state that tests/analysis pass without fresh output from this session. "It should work" is not a status.
3. Self-review the diff hunk by hunk against the checklist below: dead code deleted, old paths removed, unused imports gone, docs updated if behavior or commands changed.
4. If a gate fails: fix it or report the failure honestly. Never weaken, skip, or delete a test to get green — if a test looks wrong, stop and say so.

**When stuck:** after 2 failed fix attempts, stop adding code. Re-read the code, reproduce the failure deliberately, write a one-line root-cause hypothesis, then edit. Layering workarounds (band-aid retries, swallowing errors, widening types, casting away) is forbidden.

## Dependency restraint (a top documented complaint)

Order of preference, strictly: **standard library → framework built-in → an existing dependency in the project → a new dependency**. A new dependency is a last resort with a stated justification, not a convenience.

Before adding ANY package:
1. Does the stdlib/framework or an existing dep already cover this? If yes, stop.
2. Confirm it **exists** on the official registry (npm/PyPI/pub.dev/crates.io/Maven) — query the registry, don't trust memory. ~20% of LLM package suggestions are hallucinated, and those names get weaponized.
3. Vet it: maintained (release within ~12 months), known/verified publisher, healthy adoption (downloads/stars), active issue tracker, compatible license, sane transitive dependency count.
4. Add with a standard version constraint; **commit the lockfile** (for applications). Version pinning lives in the lockfile, not the manifest.
5. One major-version upgrade per PR, with the changelog read and the full test suite run.
6. Dependency overrides/resolutions are temporary: each carries a TODO with an upstream issue link.

Never add a dependency to avoid writing 20 lines of code. Never add two packages that solve the same problem class.

## Failure-mode catalog → rules

1. **"Almost right, but not quite"** → Never trust plausibility. Prove every change with executed checks plus a deliberate trace of edge cases: empty, null/missing, error path, concurrent/duplicate call.
2. **Duplication instead of reuse** → Search before writing (by concept, not just exact name). Extending the existing helper is the default; near-duplicates block review.
3. **Ignoring conventions / second pattern for a solved problem** → Copy the style of neighboring files; one pattern per problem class (one error model, one HTTP client, one DI style, one state approach).
4. **False "done" claims** → No completion claim without fresh verification output shown in the same message.
5. **Weak or gamed tests** → Tests are a contract: never delete/skip/weaken an assertion to pass; never special-case test inputs in production code; assert behavior (outputs, state transitions), not internals.
6. **Swallowed exceptions / silent fallbacks** → No empty catch; no catch-log-return-null; no broad catch without a typed clause outside top-level handlers. Handle with a real recovery strategy or rethrow. Errors must surface.
7. **Doom loops — fixing by adding** → The 2-strikes rule above. Root cause before the third edit.
8. **Over-engineering** → Solve today's problem in the fewest concepts. No config layers, plugin systems, or generic frameworks for one caller.
9. **Hallucinated packages/APIs** → Registry check before adding; signature check against installed version before calling.
10. **Stale-training-data APIs** → Treat memorized API knowledge as suspect; check current docs for anything deprecated-prone; keep the linter/compiler strict so it catches what review misses.
11. **Huge unfocused diffs** → Touch only task-required files; keep changes reviewable (aim under ~300 changed lines per PR). DORA links big batches directly to instability.
12. **Narration comments** → Comments state constraints and *why*. Delete any comment restating the line below it.
13. **Dead code and leftovers** → Replacing logic means deleting the old path, its imports, and helpers in the same change. Commented-out code is never committed. Deleting code is progress.
14. **Defensive bloat** → See "trust the type system" above. Cargo-cult retries/timeouts without a named failure scenario get removed.
15. **Stale docs** → Any change altering behavior, commands, or setup updates the corresponding doc (README/CHANGELOG/comments) in the same diff — or states explicitly that none exists.

## Self-review checklist (run before every "done")

- [ ] Full gate run (format + lint/analyze + tests); real output shown; green
- [ ] No near-duplicate of an existing helper introduced
- [ ] Old code paths, unused imports, orphaned helpers deleted; no commented-out code
- [ ] Diff contains only task-related changes
- [ ] Every catch handles specifically or rethrows; nothing silently swallowed
- [ ] New dependencies registry-verified and vetted; APIs checked against installed versions
- [ ] Comments are why-only; zero narration
- [ ] Tests assert behavior; none weakened, skipped, or deleted
- [ ] Edge cases traced: empty, null, error path, concurrent/duplicate
- [ ] Docs/CHANGELOG updated if behavior, commands, or setup changed

## Project bookkeeping (generic)

- **Small scoped tasks**: well-defined tasks with clear acceptance criteria produce dramatically less slop than "build the whole thing" prompts. Decompose before generating.
- **Lockfiles committed** for applications; dependency audit (outdated check) monthly; routine upgrades in their own PR.
- **CHANGELOG.md** (Keep-a-Changelog): user-visible changes add a line under `[Unreleased]` in the same PR.
- **ADRs** (`docs/adr/NNNN-title.md`, Status/Context/Decision/Consequences): significant decisions are written down; accepted ADRs are superseded, never silently contradicted. Check `docs/adr/` before proposing architectural changes.
- **CI as the backstop**: the same gate (format check, lint with zero findings, tests, coverage floor that ratchets up) runs on every PR. A bug fix without a regression test isn't done.
- **README honesty**: setup commands in the README must actually work; they're part of the same-diff doc rule.

## Project adaptation — BlackHole repo

This repo is a research-hobby monorepo: prose docs at the root, independent
science sub-projects in subfolders. Each sub-project is self-contained.

- **Stack**: Python, one venv PER sub-project (`<sub>/.venv`), created from a brew
  Python. Torch-based projects use **python3.12** (known-good with torch on this
  Mac); pure gwpy/sympy projects may use 3.14. Never share venvs across
  sub-projects; never install into the brew Python.
- **One pattern per problem** (copy the neighbors, don't invent):
  - Layout: `<sub>/scripts/` (numbered steps + one shared `<x>lib.py`),
    `<sub>/data/` (cached downloads, regenerable), `<sub>/results/` (plots +
    saved arrays), `<sub>/notes/lab_notebook.md` (decisions, results, gotchas).
  - Scripts: standalone, runnable as `.venv/bin/python scripts/NN_name.py`,
    module docstring stating purpose, `matplotlib.use("Agg")`, save plots to
    `results/` and print the path. Shared logic lives in the sub-project's lib,
    not duplicated across scripts.
  - Reproducibility: fixed seeds for anything stochastic; physics claims
    validated against a known closed-form answer before use on unknowns.
- **The verify gate** (no test suite here; the gate is): run the touched script(s)
  fresh and show real output; sanity-check plots by opening them; record results +
  surprises in the sub-project lab notebook in the same change.
- **ADR equivalent**: design decisions go in the sub-project `README.md`
  ("Decisions" section) or `notes/lab_notebook.md` with date + why. CHANGELOG
  role is served by the lab notebook. Root `CLAUDE.md` carries a compact status
  block per sub-project — update it when a milestone lands.
- **Deprecation traps seen in this repo**: gwpy whitening needs PSD grid matched
  to data length (interpolate to fs/len); GWOSC blocks may contain NaN gaps —
  crop to longest finite run; macOS `multiprocessing.Pool` deadlocks → use
  `xargs -P`; torch MPS gains nothing on tiny MLPs — default to CPU.
- **Dependency restraint here**: numpy/scipy/matplotlib + the sub-project's
  domain stack (gwpy/gwosc, torch, sympy, scikit-learn) are the approved set.
  Anything beyond that needs a stated justification in the lab notebook.

### ML experiment methodology

Our science hygiene (pre-register → control → gate → report nulls honestly) is
strong; the ML-craft is where discipline slips into folklore (defaults + round
numbers). The throughline of every result this repo got right: **measure the
achievable floor first, then gate relative to it** — echoes' sensitivity-via-
injections, ringdown's calibration at matched event loudness, pbh's matched-filter
σ as the ground-truth oracle. Make that the rule, not the exception:

1. **Oracle floor before any quantitative gate.** Run the *true* system through the
   same pipeline and measure ITS performance; gate as "within k× of oracle," never an
   absolute number pulled from intuition. pbh does this right — `mf_distance_fraction`
   is sensitivity expressed as a fraction of the ideal matched filter, not a raw number.
2. **Diagnostic trio on a failed gate, BEFORE any fix round** (~1 h): (a) overfit a
   single batch → tests expressivity/capacity; (b) 2× model, same data → capacity-
   bound?; (c) same model, 2× data → data-bound? This says *which* knob failed
   instead of guessing architecture.
3. **LR is not a default.** 3-point sweep (3e-4 / 1e-3 / 3e-3, ~500 steps each) +
   cosine decay. Removes the "was it the optimizer?" ambiguity before you blame the model.
4. **Convergence-based length, not round numbers.** Stop on val-loss plateau; add a
   feasibility check at ~25% of budget to abort doomed runs early. When changing batch
   size, hold *samples-seen* roughly fixed — fewer steps *and* smaller batches means far
   fewer samples, the opposite of "more updates."
5. **≥3 seeds for any headline number.** Single-seed = anecdote.
6. **Receptive-field / capacity arithmetic before choosing an architecture.** Compute
   what the model can actually see versus the structure in the signal. pbh's live
   example: a per-window CNN cannot see a minutes-long inspiral track — the mismatch
   that motivates track-score aggregation rather than a bigger single-window model.

The one legitimately tuned knob in this repo to date is ringdown's post-hoc temperature
T=1.05 (fitted on held-out sims, coverage-checked) — the model to copy when a knob
genuinely needs setting.
