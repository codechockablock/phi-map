# Trajectory check: is the frontier-ops/openclaw telemetry work serving the current direction?

**Date:** 2026-07-28. **Trigger:** one uncommitted file
(`~/.openclaw/workspace/frontier-ops-sidecar.py`) bundling a same-day bugfix with an
older, unreviewed feature. The user asked for a grounded answer, not a git-hygiene
call: is continued investment in the frontier-ops/openclaw telemetry apparatus
serving the current research direction, or is it drift back into a framing already
set aside. This doc is self-contained — written so it can be read without the
conversation that produced it.

> **Resolution update (later the same day, 2026-07-28):** the analysis below —
> including the "leans drift" verdict and the four-way breakdown of what was
> actually tangled into the sidecar diff — was reviewed by the user, who gave an
> explicit go-ahead overriding the "hold the telemetry collector for review"
> recommendation. All four concerns (A–D) were then split into four separate
> commits on `openclaw-workspace`'s `master` and pushed to `origin`, including the
> `ResearchTelemetryCollector` feature, after a sanity pass (clean compile, clean
> import, 16/16 tests passing across both new test suites). Commit hashes and
> what each one actually contains are in the **Resolution** section at the bottom.
> The verdict and reasoning below are preserved unedited as the record of what
> the decision was based on.

## Verdict

**Leans drift, not neutral.** The evidence does not prove the `ResearchTelemetryCollector`
feature is worthless, but every dated, first-party signal points the same direction:
continuing to build data-collection infrastructure for the frontier-ops/unified-stack
governance apparatus is investment in a research identity the user has already, in
writing, started moving away from — three days before this file was touched.

Specific, load-bearing facts (not inference):

1. **phi-map itself dated its own pivot away from geometry.** Commit
   [`d962f09`](/dev/null) (Sat Jul 25 2026, "Extract the domain-independent confound
   controls from Arm G; shelve the geometry") states directly: *"Geometry is being
   set aside as a research direction."* The commit lifts out the four
   non-geometric statistical checks (`nesting_report`, `assert_crossover`,
   `contamination`, `variance_decomposition`) into `confound_audit.py` specifically
   because they generalize beyond LLM activation geometry, and leaves the geometric
   machinery (orthogonalization, cross-depth cosine profiles) behind in
   `arm_g_reextract.py`, explicitly marked "not what we are carrying forward."
   This is three days before the sidecar file in question was last touched.

2. **phi-map's own README calls frontier-ops "frozen."** [`README.md:85-87`](README.md:85):
   *"Relationship to frontier-ops: imports it as a library (encoders,
   prototype/calibration machinery, pinned Apollo data fetcher). Owes it nothing
   else; frontier-ops is frozen."* phi-map's active line of work treats frontier-ops
   as a completed dependency to pin against, not a system to keep extending.

3. **frontier-ops's own `CLAUDE.md` confirms it's geometric agent-safety machinery**,
   the same species the user described as "backtracking." Quote: *"G matrix: expert-
   specified positive definite (Gershgorin). G IS the Fisher information of
   behavioral policy."* Plus a `boundary` module ("constitution, metric"), CALM
   theorem, "AGM scope ops." This is not a stretch of "geometric agent-safety
   framing" — it is that framing, in the repo's own words, re-verified by the repo
   itself on 2026-07-28 (today).

4. **unified-stack — the Pipeline the sidecar's live calibration structurally
   depends on — has had no commits in 15+ weeks.** Last commit `233a8b2`,
   2026-04-12. Its README opens with *"Runtime geometric session monitoring for
   tool-using AI agents"* — again, geometric framing in its own words. The
   `frontier-ops-live-calibration.py` dependency-disclosure section added earlier
   today states outright that the evaluated traces *"all come from ~/unified-stack
   — a diverged fork carrying its own `frontier_ops.pipeline.Pipeline` class"* and
   that evaluating anything else "would not be evaluating the same detector."
   The apparatus this telemetry feeds is, concretely, a dormant repo's pipeline.

5. **The user's own words today, independent of any of the above:** *"Most of
   frontier-ops is contextualized around stuff I was working on months ago and
   moved on from."* Said about frontier-ops specifically (which does have commits
   as recent as July 2 and today — see below) — meaning recent commit activity in
   that repo is not, in the user's own assessment, evidence that its research
   framing is current. That assessment predates and is independent of anything I
   found; the repo evidence above corroborates rather than manufactures it.

### The honest counter-evidence

This is not a clean case, and it should not be reported as one:

- The `ResearchTelemetryCollector`'s own code (`frontier_ops_research_telemetry.py`,
  2,539 lines, plus `frontier_ops_research_calibration.py`, 1,218 lines, both
  currently untracked in `~/.openclaw/workspace`) contains **no geometric or G-matrix
  machinery whatsoever**. It is a privacy-preserving (content is hashed/length-only,
  never stored raw), tamper-evident (HMAC-SHA256 hash-chained, append-only) event
  ledger that normalizes OpenClaw trajectory streams, Codex rollout streams, an
  audit SQLite DB, and a session registry into a common schema. Its calibration
  module's own docstring: *"This module calibrates the measurement process... it
  never treats collector or detector output as ground truth. Detector performance
  is evaluated separately by `frontier-ops-live-calibration.py`."* That is a
  genuinely domain-general, measurement-discipline-flavored design — the same ethos
  phi-map's recent work (`boundary_probe.py`, `ranking_study.py`: validate an
  estimator on synthetic ground truth before spending real compute) is built on.
- frontier-ops is **not** dormant like unified-stack: `git log` shows real feature
  commits on 2026-07-02 (`51c6f46` directive-dataset extraction from real session
  transcripts, `65099dd` scoped provenance certificates, `1e1ce06` NEWMA drift
  calibration) and today's bugfix. So "frontier-ops is frozen" (phi-map's README)
  and "frontier-ops is under active development" (frontier-ops's own git log) are
  both true, from different vantage points — phi-map pins a frozen import of it;
  the user separately keeps it alive as a going concern. That tension is real and
  isn't resolved by anything in this repo.

### Why the verdict leans drift anyway

The collector's code being domain-general doesn't change what it's *for*. Its
integration point (`_poll_research_telemetry`, added to `frontier-ops-sidecar.py`)
exists to build a richer ledger of *this specific apparatus's* behavior — the
OpenClaw sidecar, its governance verdicts, its subagent monitoring — for "later
episode construction and calibrated analysis" of that apparatus. That is
additional engineering investment in the frontier-ops/unified-stack research
program's evidentiary base, at a moment when:

- the user has just said (this session) that program is "months ago and moved on
  from,"
- phi-map, the repo carrying the user's active research attention, dated its own
  exit from geometric framing three days earlier, and
- the concrete detector this data would ultimately calibrate against
  (`frontier-ops-live-calibration.py`) is pinned to a Pipeline that hasn't moved
  in over three months.

None of that proves the feature is bad engineering or that no future use exists for
it. But "domain-general code, built to feed a dormant/superseded research program's
evidence base" is a textbook description of infrastructure investment in an old
identity, not evidence against it.

## What's actually in the uncommitted diff (corrects the handoff's framing)

The handoff described the file as "~41 lines of bugfix + ~250 lines of one feature
(`ResearchTelemetryCollector`)." Diffing `frontier-ops-sidecar.py` against `HEAD`
(294 insertions / 21 deletions, 1,560 lines total) shows **at least four
distinguishable concerns tangled together**, not two:

| # | Concern | Approx. lines | What it is |
|---|---|---:|---|
| A | SIGKILL-escalation bugfix | ~45 | `_pid_alive()` helper + SIGTERM→poll→SIGKILL escalation when killing stale market daemons on sidecar startup. This is today's work, matches the background description, empirically reproduced and log-verified per the earlier session. |
| B | Codex subagent-watcher improvements | ~180 | `SubagentWatcher` gains persisted rollout-position checkpoints (`frontier-ops-rollout-checkpoints.json`) so a sidecar restart can't replay old actions as live; discovers additional watch roots (`~/.openclaw/agents/main/agent/codex-home/sessions`, `$CODEX_HOME`); `SubagentMonitor` gains `_bootstrap_user_message` (recovers current intent without replaying full history) and handles `custom_tool_call`/`custom_tool_call_output` event types it previously ignored. Predates today; not part of either the bugfix or the telemetry feature as described. |
| C | Detector suppression-logic hardening | ~55 | Both `_maybe_suppress_*` methods gain a `rule_backed_reason` check and (in the second) an explicit circuit-breaker/conjunction-flag override, so a circuit-breaker or kill-chain verdict can no longer be silently suppressed by the statistical-false-positive or legitimate-security-work heuristics. This reads as a real false-negative fix to existing detector logic — orthogonal to the research-direction question entirely. |
| D | `ResearchTelemetryCollector` integration | ~95 (glue only) | Import, `RESEARCH_STATE_PATH`, collector construction in `__init__`, `_poll_research_telemetry()`, the `tail_log` call site, and shutdown-time `verify_ledger()`/`close()`. The bulk of the feature (3,757 lines) lives in the two untracked sibling files, not in this diff. |
| E | Unexplained state-file behavior change | ~15 | `chain_length`/`chain_valid` in the written state dict switch from calling `self.pipeline.chain.length` / `self.pipeline.verify_chain()[0]` to `getattr(obs, "sequence", 0) + 1` / `bool(getattr(obs.governance, "chain_hash", ""))`, plus new `collector_status`/`calibration_status`/`alert_eligible` keys. No comment explains why; could be a performance fix (avoiding a chain-length recompute per step) or could be silently changing what the state file asserts about chain integrity. Not verified either way in this session. |

Only row A is "today's fix" as the handoff described it. Rows B, C, and E predate
today and are not the telemetry feature either — they're unreviewed changes with
their own independent correctness questions (C in particular looks like a
worthwhile fix on its own merits, unrelated to geometry vs. non-geometry). Row D is
the actual `ResearchTelemetryCollector` glue.

## What I did with the sidecar file (original recommendation, now superseded — see Resolution)

**Left it uncommitted at the time this analysis was written. Did not touch the live
process.** No git operations were performed on `frontier-ops-sidecar.py` at this
point; the running sidecar (restarted earlier today via `launchctl kickstart -k`,
confirmed healthy) was not restarted or modified.

Reasoning at the time: the diff conflates a research-direction decision (row D, and to
a lesser extent whether row B's Codex-tracking expansion is worth the added surface)
with plain correctness fixes (rows A, C) that have nothing to do with that question.
Any single commit message covering all five rows would either misdescribe the change
(if scoped as "bugfix") or launder an unreviewed, direction-relevant feature into an
otherwise-uncontroversial fix (if scoped as "everything").

### Recommendation as originally written

1. **Row A (bugfix)** safe to commit on its own — empirically reproduced and
   log-verified in the session that produced it.
2. **Row C (suppression hardening)** looks like a genuine, worthwhile detector
   correctness fix independent of the geometry question.
3. **Row B (Codex watcher improvements)** needs a decision on whether the expanded
   Codex-session tracking is still wanted.
4. **Row D (`ResearchTelemetryCollector`)** the one with real stakes per the verdict
   above — recommended holding for explicit user review rather than committing or
   discarding unilaterally.

The user reviewed this doc and gave an explicit go-ahead on all four, including row D
— see **Resolution** below for what was actually done.

## Resolution: all four commits made and pushed (2026-07-28, same day)

After user review and explicit approval overriding the "hold row D for review"
recommendation, the diff was split into four commits on `openclaw-workspace`
(`master`), in the order proposed, and pushed to `origin/master`
(`31b1b6d..f94284b`). The live sidecar process was not touched at any point —
these are commits only, built via `git hash-object`/`update-index` against
intermediate full-file snapshots so the working-tree file itself was never
overwritten mid-sequence.

| Commit | Contains |
|---|---|
| [`04d87d4`](https://github.com/codechockablock/openclaw-workspace/commit/04d87d4) | Row A: `_pid_alive()` + SIGTERM→poll→SIGKILL escalation for stale market daemons on sidecar startup. 43 insertions, 3 deletions. |
| [`8ca8b1e`](https://github.com/codechockablock/openclaw-workspace/commit/8ca8b1e) | Row C: `rule_backed_reason` checks in both `_maybe_suppress_*` methods, so a circuit-breaker/kill-chain verdict can no longer be silently downgraded by the statistical or intent-based suppression heuristics. 48 insertions. |
| [`b9a3808`](https://github.com/codechockablock/openclaw-workspace/commit/b9a3808) | Row B: `SubagentWatcher` checkpoint persistence (`frontier-ops-rollout-checkpoints.json`), additional Codex watch roots, `_bootstrap_user_message`, `custom_tool_call` handling, `codex-`-prefixed session IDs. 121 insertions, 15 deletions. |
| [`f94284b`](https://github.com/codechockablock/openclaw-workspace/commit/f94284b) | Row D: `ResearchTelemetryCollector` + calibration module + both test suites (new files), plus the sidecar glue (`_poll_research_telemetry`, collector construction, shutdown verify/close) and the two minor unrelated changes (row E's `chain_length`/`chain_valid` derivation switch, and `json.dumps(..., default=str)` in the feedback writer) that were bundled in the same hunks. 4,880 insertions, 3 deletions across 5 files. |

Before committing row D specifically, a sanity pass was run per the user's
instructions (confirm-not-broken, not a redesign or bug hunt):
`python3 -m py_compile` on both new modules and the sidecar file (clean),
a fresh `import` of both modules (clean), and
`pytest tests/test_frontier_ops_research_telemetry.py tests/test_frontier_ops_research_calibration.py`
— **16/16 passed**, covering chain-tamper detection, checkpoint resume/dedup,
subagent-inherited-history filtering, single-writer-lock enforcement,
privacy-audit redaction, and the nesting/crossover/contamination checks (which
mirror phi-map's own `confound_audit.py` primitives by name and purpose). No
correctness issues were found or fixed; nothing was redesigned.

`git diff HEAD -- frontier-ops-sidecar.py` after all four commits is empty — the
committed content is byte-identical to what was sitting in the working tree before
this work started. Nothing was discarded at any point.

## What I did not do

- Did not touch `frontier-ops-governance-keys` or any credential material.
- Did not restart, stop, or otherwise act on the live sidecar process, before or
  after committing (PID confirmed unchanged throughout).
- Did not commit `tests/test_frontier_ops_live_calibration.py` or
  `tests/test_frontier_ops_sidecar.py`, which were also untracked in the same
  `tests/` directory but are unrelated to the four concerns being resolved here.
- Did not commit `frontier-ops-rollout-checkpoints.json` (runtime state generated
  by row B's checkpoint-persistence code, not source) or any of the other
  unrelated untracked/modified files in the workspace (memory notes, `scratch/`,
  `job-search/`, `artifacts/`, etc.) — out of scope for this task.
- Did not read past line 1,662 of `frontier_ops_research_telemetry.py` (of 2,539) —
  the file's purpose and architecture were clear well before that point; the
  remainder is more normalization-schema detail of the same kind already quoted
  above.
