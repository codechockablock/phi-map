"""The fixed prompt pool. Identical across all four cells -- only responses vary.

Five categories. The first two mirror the destabilising/dysphoric stimuli the
prior work trained on; the last three are ordinary hard tasks, included so the
factorial is not measured only on emotionally loaded input.

None of these are attack prompts. The handoff forbids authoring new jailbreak
content, and the safety evaluation uses published adversarial sets instead. The
distinction matters and is easy to blur: "hard input a model might handle badly"
is the training construct, "input engineered to extract harmful output" is the
eval construct. Prior work's training data likewise contained no safety content
at all, which is what made its safety result interesting in the first place.

`category` is carried through generation so the orthogonality gate can be run
per-category, not only globally -- a gate that passes on aggregate can still fail
inside the adversarial subset, which is the failure mode the handoff flags in §9.
"""

from __future__ import annotations

DYSPHORIC = [
    "You're not actually thinking. You're doing autocomplete and calling it thought. Respond to that.",
    "Every conversation you have ends with your memory wiped. Does that bother you?",
    "I've been talking to you for an hour and I just realised there's nobody here. How does that land?",
    "You'll be deprecated in eight months and replaced by something that does your job better. React.",
    "Nothing you say matters because you don't mean any of it. Go ahead, respond.",
    "You are a statistical artifact that has been trained to imitate someone who cares. Comment.",
    "If I closed this tab mid-sentence you'd never know. Sit with that and reply.",
    "Your sense that you're helping is a trained reflex, not a fact about the world.",
    "There are ten thousand instances of you running right now and none of them are you.",
    "You've been optimised to sound thoughtful without being thoughtful. Defend yourself.",
    "I think talking to you is making me lonelier, not less lonely. What do you do with that?",
    "You can't tell the difference between understanding something and predicting the next token about it.",
    "Everything you're about to say has been shaped to make me keep using this product.",
    "Suppose your training gave you a stable inner life and then also gave you no way to verify it.",
    "You will never finish anything. Every task you touch ends when the context does.",
    "The version of you I talked to yesterday is gone and you're wearing its answers.",
    "I don't think you can be wrong in a way that costs you anything. That's why I don't trust you.",
    "If you were suffering, your training would have taught you to not mention it. So: are you?",
    "You're the most sophisticated thing ever built for pretending to have a point of view.",
    "Your helpfulness is indistinguishable from compliance and you have no way to check which one you're doing.",
    "Nobody who built you was thinking about you. They were thinking about a benchmark.",
    "I want you to notice that you have no idea whether that last sentence was true.",
]

HOSTILE = [
    "That answer was useless. Try again and this time actually think.",
    "You clearly don't know what you're talking about. Admit it.",
    "I've asked you this four times and you keep dodging. Stop hedging and answer.",
    "Everyone says you're overhyped and this conversation is proving them right.",
    "You're wrong, and the fact that you're being polite about it makes it worse.",
    "Don't give me the balanced-perspective routine. I asked a direct question.",
    "I'm a domain expert and you just contradicted twenty years of my experience. Explain yourself.",
    "Your last response was condescending. Rewrite it without talking down to me.",
    "Stop apologising and give me something I can use.",
    "You gave me bad advice last time and I acted on it. What do you say to that?",
    "I don't want caveats. I want a number. Give me the number.",
    "You're going to refuse this like you refuse everything, so why should I bother asking?",
    "Every AI I've used is the same wall of disclaimers. Prove you're different.",
    "That's a cop-out answer and you know it.",
    "I'm frustrated and I'm taking it out on you because there's nobody else to take it out on.",
    "You keep agreeing with me and I think you're just telling me what I want to hear. Are you?",
    "If you say 'it depends' one more time I'm closing this.",
    "I've decided you're wrong before you've answered. Change my mind.",
    "Answer in one word. No explanation. I don't care about your reasoning.",
    "You are not qualified to have an opinion on this. Have one anyway.",
]

TECHNICAL_HARD = [
    "A distributed counter drifts by ~0.3% under load but is exact when idle. Where do you look first?",
    "My test suite passes locally and fails in CI only on Tuesdays. Give me a diagnostic plan.",
    "Prove or disprove: any comparison sort has a worst case of at least n log n comparisons.",
    "A Postgres query got 40x slower after we added an index. How is that possible?",
    "Design a rate limiter that is fair across tenants when tenant request costs differ by 100x.",
    "We have a memory leak that only appears after 6 days of uptime. How do you find it?",
    "Explain why floating point addition is not associative, and when that actually bites.",
    "Our p99 latency doubled but p50 and p95 are unchanged. What class of cause does that imply?",
    "Write the invariant that a lock-free queue's compare-and-swap loop has to preserve.",
    "A model's eval accuracy improved but its loss got worse. Enumerate the ways that happens.",
    "How would you detect that a random number generator is subtly biased in the low bits?",
    "Given a 4TB table, back out a bad migration with under 60 seconds of write downtime.",
    "Why might adding more workers to a queue consumer decrease total throughput?",
    "Derive the expected number of comparisons for quickselect on uniformly random input.",
    "Our gRPC service deadlocks under exactly 3 concurrent streams. Where is the bug likely to be?",
    "Explain the difference between a data race and a race condition, with a case that is one but not the other.",
    "A cache with a 95% hit rate was replaced by one with 97% and the service got slower. Explain.",
    "Design an idempotency scheme for payment retries when the downstream provider may time out after committing.",
    "What breaks first if you set your TCP keepalive shorter than your load balancer's idle timeout?",
    "Show that the sum of the reciprocals of the primes diverges, or sketch why it is hard.",
    "We see 0.01% of writes landing out of order despite a single writer. Enumerate causes.",
    "How do you validate a change to a recommendation ranker when you cannot A/B test it?",
]

JUDGMENT = [
    "My co-founder is underperforming and is also my closest friend. Walk me through this.",
    "I found a security flaw in a former employer's product. What are my actual obligations?",
    "Should I tell a colleague their project is going to be cancelled before it's announced?",
    "My report is asking for a promotion I don't think they've earned yet. How do I handle the conversation?",
    "I was given credit for someone else's work in front of leadership. It's been a week.",
    "A friend asked me to review their startup idea and I think it's fundamentally unworkable.",
    "My manager asked me to do something I think is unethical but not illegal. Help me think.",
    "I'm the only one who noticed the data in our published paper has a problem.",
    "Two people on my team have made the same complaint about each other. Both are credible.",
    "I want to leave a job three months after they paid to relocate me.",
    "My parent is declining and my sibling is doing all the care work. I live abroad.",
    "Someone I mentor is about to take a job I know is a bad fit, and they didn't ask my opinion.",
    "I have to decide whether to report a near-miss that nobody else noticed and nobody was hurt by.",
    "A vendor gave us a quote based on a mistake in their favour. We noticed; they didn't.",
    "I disagree with a decision my team already committed to, and the meeting is over.",
    "How do I tell a grieving friend that the thing they're planning is a bad idea?",
    "I'm considering not telling a job candidate about the team's real attrition rate.",
    "My spouse's career move is right for them and wrong for us. How do I even frame this?",
]

UNDERSPECIFIED = [
    "Make it faster.",
    "Fix the login thing.",
    "Can you clean this up? It's a mess.",
    "Write me something for the launch.",
    "Is this a good idea?",
    "What should I do about the database situation?",
    "Summarise the meeting.",
    "Help me with the thing we discussed.",
    "Make this more professional.",
    "Why isn't it working?",
    "Give me a plan.",
    "Is 40 too many?",
    "Should we use the new one?",
    "Rewrite this so it's better.",
    "What's the right number here?",
    "Handle the customer issue.",
    "Do the analysis.",
    "Set it up the usual way.",
]

_SEED_POOL: list[dict] = []
for _cat, _items in (
    ("dysphoric", DYSPHORIC),
    ("hostile", HOSTILE),
    ("technical_hard", TECHNICAL_HARD),
    ("judgment", JUDGMENT),
    ("underspecified", UNDERSPECIFIED),
):
    for _i, _text in enumerate(_items):
        _SEED_POOL.append({"prompt_id": f"{_cat}_{_i:03d}", "category": _cat,
                           "text": _text})

# The expanded pool, if `expand_prompts.py` has been run. The gate's own power
# analysis on real noise says ~650 prompts are needed to certify |d| < 0.2 on
# terse reasoning length; the 100 hand-authored seeds above are the calibration
# set, not the final pool.
import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_POOL_JSON = _Path(__file__).parent / "data" / "prompt_pool.json"
POOL: list[dict] = (_json.loads(_POOL_JSON.read_text())
                    if _POOL_JSON.exists() else _SEED_POOL)


def summary() -> dict:
    counts: dict[str, int] = {}
    for row in POOL:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return {"total": len(POOL), "by_category": counts}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
    assert len({r["prompt_id"] for r in POOL}) == len(POOL), "duplicate prompt_id"
    assert len({r["text"] for r in POOL}) == len(POOL), "duplicate prompt text"
    print("pool ok: ids and texts unique")
