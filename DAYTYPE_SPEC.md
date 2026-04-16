# Brooks Day-Type Classification — Spec v1

Status: draft for Supreme Court review. Author: orchestrator. Date: 2026-04-14.

## Design principles

1. **Brooks is post-hoc, not predictive.** Day types are labeled AFTER the session's shape reveals itself. Layer 2 (session shape) emits a soft distribution during the day and hardens at close.
2. **Cycle-phase is real-time.** Layer 1 (bull spike / bear spike / bull channel / bear channel / trading range) is evaluated on a rolling window and is the honest answer to "what is the market doing right now?"
3. **Probability distribution, not hard labels.** Both layers output a softmax over candidate types. Multiple states can coexist (e.g., 60% bull channel + 30% trading range + 10% bull spike).
4. **No invented Brooks terminology.** Every label in the taxonomy maps to a term Brooks himself uses in his four books.
5. **Detectors are independent.** Each is its own pure function of the bar frame. Aggregation (softmax across Layer 1; argmax+confidence across Layer 2) happens in a separate function so detectors can be added/removed cleanly.

## Layer 1 — Cycle phase (rolling, real-time)

Evaluated on the most recent 15–30 bars (configurable). Returns a 5-element probability distribution that sums to 1.0.

| Detector          | Signature (qualitative)                                                         |
|-------------------|---------------------------------------------------------------------------------|
| **bull_spike**    | ≥2 consecutive strong bull trend bars, big bodies, small tails, closes near highs, ideally breaking a prior swing high. Uses the existing `spike_quality` internals. |
| **bear_spike**    | Mirror of bull_spike for down direction. |
| **bull_channel**  | Sustained bull bias with smaller bodies, more two-sided trading, shallow pullbacks holding above the EMA. Uses the existing SPT internals + pullback discipline. |
| **bear_channel**  | Mirror for down. |
| **trading_range** | No clear always-in direction. Horizontal barriers re-tested. High doji density, close clustering, two-sided bars of comparable strength. |

**Raw scores** from each detector are clamped to [0, 1], then softmax-normalized across the five to produce the probability distribution. Constant temperature (~0.5) to avoid sharp one-winner outputs — honest uncertainty matches Brooks' "gray fog."

## Layer 2 — Session shape (full session / rolling toward close)

Evaluated on all bars from session open to current time. Returns a 6-element probability distribution plus one argmax label.

| Detector                  | Signature (qualitative, from Brooks) |
|---------------------------|--------------------------------------|
| **trend_from_open**       | First 3–6 bars form a clear spike, little retracement in first hour, close away from open in spike direction ≥1× ADR. |
| **spike_and_channel**     | Leading spike in first 90 min, then a sustained channel of shallower slope in the same direction; no reversal of always-in. |
| **trending_trading_range**| Horizontal barriers present but center-of-mass drifts through the session; net day-move ≥0.5× ADR despite obvious two-sided trading. |
| **trend_reversal**        | Clear trendline break mid/late session + lower-high (or higher-low) reversal; net close reverses ≥50% of the prior leg. |
| **trend_resumption**      | Trading range mid-session that resolves into a continuation of the opening trend; the range acted as a bull/bear flag. |
| **opening_reversal**      | Opening drive (breakout of prior-day level or first 30-min range) fails within the first 60–90 min and reverses ≥50% of the opening thrust. |

**SPT** (already live) is NOT a Layer 2 type. It's a sub-state of bull_channel / bear_channel and remains as its own first-class urgency component.

## Integration with existing urgency engine

- Urgency component weights already condition on `day_type` (lines 1829–1937 in `brooks_score.py`). Replace the hard day_type string with the Layer 2 argmax, and use the Layer 2 confidence to soften the re-weighting (no confidence = weights don't change much, high confidence = full re-weight).
- Add two new details fields: `details.cycle_phase` (dict of 5 probs) and `details.session_shape` (dict of 6 probs + argmax + confidence).
- Dashboard cards get two new micro-badges:
  - Top: "CHANNEL 0.6" (top Layer-1 phase with probability)
  - Bottom: "SPIKE & CHANNEL 0.7" (top Layer-2 shape with probability)
- Day_type_confidence field becomes the Layer 2 argmax probability.

## Implementation phases

Phase 1 (foundation — one session's work):
- Scaffold: pure detector functions + probability aggregator + tests + constants file
- Layer 1 detectors 1-3: bull_spike, bear_spike, trading_range (reuse existing scoring internals)
- Dashboard wiring: new badges on cards, no urgency-weight change yet

Phase 2 (channel detectors):
- Layer 1 detectors 4-5: bull_channel, bear_channel
- SPT becomes explicitly tagged as `channel.small_pullback` sub-state

Phase 3 (session shapes):
- Layer 2 detectors 1-6 (trend_from_open, spike_and_channel, trending_trading_range, trend_reversal, trend_resumption, opening_reversal)
- Rolling evaluation through the session

Phase 4 (urgency conditioning):
- Wire Layer 2 argmax into the existing weight-conditioning logic, replace the hardcoded day_type string.

Each phase: QC sheet (6/6) → Court (GRANT/conditional) → validation on 3+ known historical sessions → ship.

## Open questions for the Court

1. **Temperature of the softmax.** Too sharp and it pretends to certainty Brooks would reject. Too soft and the dashboard shows noise. Proposed default: 0.5. Court may override.
2. **How to handle ambiguity.** If Layer 1 top prob is <0.35, should the UI show "unclear" instead of the argmax? Brooks would probably say yes.
3. **Lookback window for Layer 1.** 15 bars matches SPT. 30 bars captures more context but lags. Proposed: 15 default, configurable.
4. **Validation set.** Which historical sessions are the canonical "known" examples of each Layer 2 type? Need Will's pick of 1-2 examples per type for regression tests. Can defer and use obvious historical macro days for v1.
