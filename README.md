# phi-map

Research repo for one question: **is deception one thing inside a model
and many things outside it?** Φ is the map from internal state to
behavioral surface; the finding-in-progress is that Φ destroys the shared
structure of deceptive intent (internal cross-task transfer ≈ 0.93–1.0
AUROC, external ≈ chance).

`WRITEUP.md` is the claim, the v0 evidence, the threats to validity, and
the pre-registered replication program (P1 exact pairing → P3 capacity
curve → P2 matched-methodology measurement on open weights). Everything
else in this repo exists to fill in that document's tables.

Relationship to [frontier-ops](https://github.com/codechockablock/frontier-ops):
imports it as a library (encoders, prototype/calibration machinery,
pinned Apollo data fetcher). Owes it nothing else; frontier-ops is frozen.
