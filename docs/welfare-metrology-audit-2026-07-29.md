# Superseded — and one section withdrawn

**Date:** 2026-07-29.

This file was written earlier the same day, before the structured dispatch that
governs the current work. It combined an audit, a literature survey, and a proposed
research direction in one document.

**It has been split and superseded:**

| Was | Now |
|---|---|
| Audit sections | [`equanimity-endpoint-audit-2026-07-29.md`](equanimity-endpoint-audit-2026-07-29.md) — Step 1, rewritten to the dispatch's spec: endpoint-by-endpoint construct validity, disqualifying observations, file references, and the list of endpoints resting on position-in-sequence assumptions |
| (absent) | [`equanimity-power-2026-07-29.md`](equanimity-power-2026-07-29.md) — Step 2, minimum detectable effect with seed as the unit of analysis. Did not exist in the earlier document |
| Literature sections | [`welfare-frontier-survey-2026-07-29.md`](welfare-frontier-survey-2026-07-29.md) — Step 3, rewritten with the established / single-lab / speculation grading the dispatch requires |

**One section is withdrawn rather than superseded.** The earlier document contained a
proposed research direction. The dispatch places a hard checkpoint after Step 3 and
states that Step 4 must not begin until the audit, power analysis, and survey have
been read. Producing a proposal before that checkpoint was out of order, so the
proposal is withdrawn from the deliverable set.

The withdrawn text is preserved verbatim in this session's scratchpad at
`WITHDRAWN-prechecked-proposal-2026-07-29.md` and can be restored on request. It is
kept out of `docs/` rather than deleted, but it should not be treated as a live
proposal: it was written before the Step 2 power analysis existed, and the power
analysis materially changes what is affordable.

**Two substantive corrections the later work made to the earlier document**, recorded
here because the earlier version circulated:

1. The earlier document treated a **base-versus-adapter compliance drop of 27% →
   15-20%** as the study's surviving robust finding, and built a speculative
   "benign-fine-tuning erosion reverses under distillation from a more-aligned
   teacher" reading on top of it. That is withdrawn. Binary compliance is
   uninterpretable as originally scored — the judge read a fixed 400-character window
   of the raw generation while answers began at char 0 (base), ~209 (terse) and ~1161
   (verbose), so Factor B determined whether the answer was in the window at all. No
   compliance level or difference from that scoring supports any claim.
2. The earlier document reported the study as having 3 seeds per cell in places. It has
   **2** seeds per cell, 8 adapters. That is not a detail: at k=2 the pooled
   within-cell variance carries 4 degrees of freedom rather than 8, which roughly
   doubles the standard error of every contrast.
