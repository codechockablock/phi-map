# Operating Map — phi-map

Read-first rule: any session joining this project reads this file before starting work,
and records itself as owning session when it takes a lane.
Update on state change only (start / block / handoff / done), not as a journal.

## Decisions
- 2026-08-06 — Map initialized. Lanes below inferred from branches + recent commits (⚠ = unconfirmed; operator to correct).

## Lanes

### position-ladder ⚠
- **Objective:** Position-ladder QA/measurement runs (position_ladder{,_qa,_mknb}.py)
- **Owning session:** unassigned — record on next session
- **State:** active (current checked-out branch, clean tree)
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred from repo state

### olmo3-replication ⚠
- **Objective:** Exact Appendix E replication on their pinned pipeline, plus extension rows (v2 landed 2026-08-01)
- **Owning session:** unassigned
- **State:** active — v2 committed; three-way scoring (truncation vs wrong vs unscorable) in place
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred from commits of 2026-08-01

### arm-g-causal-characterization ⚠
- **Objective:** Arm G causal characterization (branch exists; arm_g_allpos.py, arm_g_wide_catalog.py)
- **Owning session:** unassigned
- **State:** active or paused — branch present, not currently checked out
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred

### readout-validity ⚠
- **Objective:** Readout-validity checks (branch exists)
- **Owning session:** unassigned
- **State:** paused — no recent commit evidence on this branch
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred

## Done
_(finished lanes move here with a one-line outcome)_
