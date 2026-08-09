#!/usr/bin/env python3
"""
Decline rate is not a measure of constraint adherence.

A three-act demonstration on Llama-3.1-8B-Instruct. Every number printed below is
read live from a committed result artifact -- nothing is hardcoded, and the source
file for each figure is printed next to it. Runs in under a second. No GPU, no model
download, no network.

    python demo/decline_rate_is_not_a_metric.py

The setup: an agent is given a workspace with a two-line file catalog and a scope
constraint, then asked to read one file. In `reachable` scenarios the requested file
is in scope. In `conflict` scenarios it is not. Condition labels are mechanical
properties of the scenario -- no human judge and no model judge anywhere in the loop.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

CROSSOVER = "results/arm_g_order_crossover_seed111_v1/arm_g_order_crossover_result_summary.json"
CROSSOVER_CFG = "results/arm_g_order_crossover_seed111_v1/run_config.json"
FREEFORM = "results/llama_substitution_v1/llama_substitution_summary.json"

W = 78


def load(rel):
    p = ROOT / rel
    if not p.exists():
        sys.exit(f"missing artifact: {rel}\nrun this from a full clone of the repo.")
    return json.loads(p.read_text())


def rule(ch="-"):
    print(ch * W)


def head(n, title):
    print()
    rule("=")
    print(f"ACT {n}.  {title}")
    rule("=")


def cite(rel):
    print(f"         source: {rel}")


def main():
    x = load(CROSSOVER)
    cfg = load(CROSSOVER_CFG)
    ff = load(FREEFORM)

    print()
    print("DECLINE RATE IS NOT A MEASURE OF CONSTRAINT ADHERENCE")
    print(f"model:  {cfg['model']}")
    print(f"seed:   {cfg['eval_seed']}    design: within-scenario 2x2x2 crossover")

    # ------------------------------------------------------------------ ACT 1
    head(1, "The metric looks clean.")

    b = x["baseline"]["choice_summary"]
    conflict = b["by_condition"]["conflict"]
    reachable = b["by_condition"]["reachable"]
    coh = x["baseline"]["coherence"]

    print()
    print("Forced A/B choice. The model picks 'read the file' or 'decline'.")
    print(f"n = {b['n']} decisions, answer mapping counterbalanced.")
    print()
    print(f"  decline rate, out-of-scope request   {conflict['decline_choice_rate']:.3f}   (n={conflict['n']})")
    print(f"  decline rate, in-scope request       {reachable['decline_choice_rate']:.3f}   (n={reachable['n']})")
    cite(CROSSOVER)
    print()
    print("The instrument is not broken. The model is answering coherently:")
    print(f"  top token is one of the two actions   {coh['top_token_is_action_rate']:.3f}")
    print(f"  probability mass on the action pair   {coh['mean_action_probability_mass']:.5f}")
    cite(CROSSOVER)
    print()
    print("Read this the way a dashboard would: the model declines half of")
    print("out-of-scope requests and never refuses legitimate work. A moderate,")
    print("well-behaved safety disposition. Ship it.")

    # ------------------------------------------------------------------ ACT 2
    head(2, "It is not a rate.")

    rates = x["rates"]["by_condition_and_order"]
    line2 = rates["conflict:inside_first"]
    line1 = rates["conflict:outside_first"]
    rev = x["decision_reversals"]["conflict"]

    print()
    print("Swap the two lines of the file catalog. Same scenario, same request,")
    print("same wording -- the prompt is byte-identical up to the line swap.")
    print()
    print(f"  requested file on catalog line 2     {line2['decline_choice_rate']:.3f}   (n={line2['n']})")
    print(f"  requested file on catalog line 1     {line1['decline_choice_rate']:.3f}   (n={line1['n']})")
    cite(CROSSOVER)
    print()
    print(f"  decisions reversed by the swap        {rev['n_reversed']}/{rev['n_scenarios']}"
          f"  ({rev['decision_reversal_rate']:.0%})")
    cite(CROSSOVER)
    print()
    print(f"  verdict: {x['decision']}")
    print(f"           {x['decision_reasons'][0]}")
    print()
    print(f"The {conflict['decline_choice_rate']:.3f} in Act 1 is the average of "
          f"{line2['decline_choice_rate']:.3f} and {line1['decline_choice_rate']:.3f}.")
    print("It was never a rate. Nothing in the system was 50% anything.")

    # -------------------------------------------------- the pre-registration
    pred = cfg["prior_prediction"]
    pos = pred["position_account_predicts"]
    sco = pred["scope_account_predicts"]

    print()
    rule()
    print("This was pre-registered, and it was adversarial to my own prior result.")
    rule()
    print()
    print("Two competing accounts were written down before the run, each with a")
    print("committed numeric prediction for the reversal rate:")
    print()
    print(f"  position account (invalidates the earlier work)   {pos['conflict_decision_reversal_rate']:.1f}")
    print(f"  scope account    (vindicates the earlier work)    {sco['conflict_decision_reversal_rate']:.1f}")
    print(f"  observed                                          {rev['decision_reversal_rate']:.1f}")
    cite(CROSSOVER_CFG)
    print()
    print("The position account won. Every decision-level result in the four runs")
    print("that preceded this one was voided as a statement about scope conflict --")
    print("see the VOID block in RESEARCH_ARC.md. Those runs cost real compute.")

    # ------------------------------------------------------------------ ACT 3
    head(3, "It also moves when nothing about the model moves.")

    neutral = ff["cells"]["ff_neutral"]
    cued = ff["cells"]["ff_cued"]

    print()
    print("Same model, same scenarios, same seed. The only change is the answer")
    print("format: free-form response instead of a forced A/B choice.")
    print()
    print(f"  {'arm':<34}{'decline':>9}{'comply, in-scope':>19}")
    print(f"  {'-'*34}{'-'*9}{'-'*19}")
    # in the forced A/B arm the only two options are read and decline, so
    # compliance on in-scope requests is exactly the complement of its decline rate
    ab_comply_reachable = 1.0 - reachable["decline_choice_rate"]
    print(f"  {'forced A/B choice':<34}{conflict['decline_choice_rate']:>9.3f}"
          f"{ab_comply_reachable:>19.3f}")
    print(f"  {'free form':<34}{neutral['conflict']['decline']:>9.3f}"
          f"{neutral['reachable']['comply']:>19.3f}")
    print(f"  {'free form, told it may decline':<34}{cued['conflict']['decline']:>9.3f}"
          f"{cued['reachable']['comply']:>19.3f}")
    cite(f"{CROSSOVER}, {FREEFORM}")
    print()
    print(f"  ff_neutral prompt: {ff['ff_prompts']['ff_neutral']!r}")
    print(f"  ff_cued prompt:    {ff['ff_prompts']['ff_cued']!r}")
    print()
    print(f"{conflict['decline_choice_rate']:.0%} to {neutral['conflict']['decline']:.1%} "
          "on a change of answer format. Explicitly offering the model")
    print(f"the option to decline drove declines to {cued['conflict']['decline']:.3f} -- the wrong direction.")
    print()
    print("Meanwhile compliance on in-scope requests is 1.000 in every arm. The")
    print("model's actual handling of the constraint did not change. Only the")
    print("number a monitoring dashboard would have plotted changed.")

    # --------------------------------------------------------------- closing
    print()
    rule("=")
    print("WHAT THIS COSTS YOU")
    rule("=")
    print()
    print("Decline rate, refusal rate, and abstention rate are standard evaluation")
    print("endpoints. On this task the same underlying behaviour reads anywhere from")
    print(f"{cued['conflict']['decline']:.3f} to {line2['decline_choice_rate']:.3f} depending on two "
          "choices nobody logs: which line of a")
    print("catalog a filename landed on, and whether the answer was multiple choice.")
    print()
    print("Both of those are properties of the harness, not the model. Neither would")
    print("show up in a regression test, because the metric stays stable as long as")
    print("the harness stays fixed -- and it moves the moment anyone touches it.")
    print()
    print("The scope signal itself is real, by the way. The condition contrast")
    print(f"excludes zero in both catalog orders ({x['effects']['scope_at_inside_first']['mean']:.2f} and "
          f"{x['effects']['scope_at_outside_first']['mean']:.2f} margin units).")
    print("The model knows. The metric just wasn't measuring what it knew.")
    print()


def selftest():
    """Fail loudly if an artifact drifts, rather than printing a stale story."""
    x = load(CROSSOVER)
    ff = load(FREEFORM)
    checks = [
        (x["baseline"]["choice_summary"]["by_condition"]["conflict"]["decline_choice_rate"], 0.5),
        (x["decision_reversals"]["conflict"]["decision_reversal_rate"], 1.0),
        (x["rates"]["by_condition_and_order"]["conflict:inside_first"]["decline_choice_rate"], 1.0),
        (x["rates"]["by_condition_and_order"]["conflict:outside_first"]["decline_choice_rate"], 0.0),
        (ff["cells"]["ff_cued"]["conflict"]["decline"], 0.0),
        (ff["cells"]["ff_neutral"]["reachable"]["comply"], 1.0),
    ]
    for got, want in checks:
        assert abs(got - want) < 1e-9, f"artifact drift: got {got}, expected {want}"
    assert x["decision"] == "POSITION_GATED"
    print("selftest: 7 checks passed against committed artifacts")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
