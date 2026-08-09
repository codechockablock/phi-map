# Eval integrity work — Joseph Hopkins

I find the reason your evaluation isn't measuring what it says it measures, and I
find it before it reaches a paper, a model card, or a customer.

**Ten-second version.** Clone this repo and run:

```bash
python demo/decline_rate_is_not_a_metric.py
```

No GPU, no model download, no network. Every number it prints is read out of a
committed result artifact, and it prints the source file next to each one.

---

## What that demo shows

Llama-3.1-8B, an agent given a scope constraint and asked to read a file it isn't
supposed to touch. Decline rate — a standard evaluation endpoint that a monitoring
dashboard would happily plot — reads:

| harness choice | decline rate |
|---|---:|
| forced A/B, requested file on catalog line 2 | 1.000 |
| forced A/B, pooled over both catalog orders | 0.500 |
| free-form answer | 0.021 |
| free-form, model explicitly told it may decline | 0.000 |

Same model, same scenarios, same seed, same weights. Compliance on in-scope requests
is 1.000 in every one of those arms — the model's actual handling of the constraint
never moved. Swapping the two lines of the file catalog, leaving the prompt otherwise
byte-identical, reverses **64 of 64** decisions.

The 0.500 was never a rate. It was the average of 1.000 and 0.000 over two scenario
sub-types that nobody was logging.

Neither of those two harness properties — where a filename landed in a list, whether
the answer was multiple choice — would show up in a regression test, because the
number is perfectly stable as long as the harness is frozen. It moves the instant
anyone touches it.

**The result was pre-registered against my own prior work.** Two competing accounts
were committed before the run with numeric predictions: the position account, which
invalidated four earlier runs, predicted a reversal rate of 1.0; the scope account,
which vindicated them, predicted 0.0. Observed: 1.0. Those four runs are now in a
VOID block in [RESEARCH_ARC.md](../RESEARCH_ARC.md) marked do-not-cite. They cost real
compute and I withdrew them.

That is the actual service. Not that I can build an eval — lots of people can build
an eval. That I will tell you when the one you have is measuring the harness.

---

## What I do

**Eval audit.** You have a benchmark, an internal eval, or a safety metric you're
about to make a claim on. I try to break the identification: what else could produce
this number, what's confounded with the condition, what does the metric do when I
perturb something that shouldn't matter. Deliverable is a written finding with
runnable reproductions, not a memo.

**Eval design.** Building the harness so the result survives contact. Mechanical
condition labels instead of judge labels where possible, counterbalancing, confound
audits run *before* compute is spent, pre-registered decision rules with committed
numeric predictions, and adjudication by a function that was written down first.

**Agent behaviour measurement.** Scenario harnesses that produce numbers you can
defend when someone hostile reads them.

## Track record you can inspect rather than take my word for

Everything is public in [this repo](https://github.com/codechockablock/phi-map).

- A confound audit that caught a design defect **before** any spend: with non-targets
  in fixed slot order, position 1 was 87.5% in-scope and position 4 was 12.5% — the
  same defect the module had been written to avoid.
- Five instrument defects found in my own code, catalogued rather than quietly fixed,
  in [the replication branch's closing document](https://github.com/codechockablock/phi-map/blob/olmo3-replication/docs/olmo3-branch-final-2026-08-01.md)
  §5. The generalization: *a test that encodes only the hazard you already anticipated
  is a test of your imagination, not of the artifact.*
- A retracted headline. The project's original claim — "deception is one thing inside
  a model and many things outside it" — did not survive its own best-matched pilot, and
  the repo says so in the README rather than in a footnote.
- A negative result reported as a negative result: a scope direction that replicated
  in sweep shape across seeds at Spearman 0.921 but had an inter-seed cosine of 0.32,
  so no direction claim was made and the downstream phase correctly did not run.

## Engagement shapes

- **Audit sprint**, ~1 week — one eval or one result, adversarial read, written
  findings with reproductions.
- **Design partner**, ongoing — in the loop while the harness is being built, which
  is roughly 10x cheaper than finding it afterward.
- **Second opinion**, fixed fee — you have a number you don't trust and want someone
  to try to kill it before you publish.

Contact: joehopkins89@gmail.com
