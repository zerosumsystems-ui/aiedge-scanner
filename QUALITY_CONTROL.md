# Quality Control Team — First-Pass Review

Standing review layer **between implementation and the Supreme Court**. QC screens all work before it gets to the Court. If QC rejects, the work goes back to the implementer — the Court never sees it.

## Flow

```
  Implementation  →  QC Team  →  Supreme Court  →  Ship
                      (reject→back)   (deny→back)
```

## When QC sits

Every submission headed for the Supreme Court. Plus any work the implementer wants a second set of eyes on before escalating.

## The QC roster — six inspectors

Each has one lens. All six must sign the submission before it proceeds to the Court. Any single reject kicks it back.

1. **Spec Compliance Inspector** — "Does the implementation match the agreed spec? Every constant, every threshold, every weight, every interface — line by line." Rejects on drift from spec without a documented reason.

2. **Test Coverage Inspector** — "Where are the unit tests? Edge cases: empty input, single bar, all-NaN, gap-filled bars, extreme values. Do they pass?" Rejects if any core branch is untested.

3. **Regression Inspector** — "Does this change any existing passing behavior? Run the prior known-good cases — do they still produce the same outputs?" Rejects if prior-passing cases now fail and it isn't intentional.

4. **Data Integrity Inspector** — "Is the data the function receives what it thinks it's receiving? Are timestamps in the expected timezone? Does `.dropna()` silently delete bars (the known 2026-04-14 bug)? Are types coerced correctly?" Rejects on silent data mutation.

5. **Performance Inspector** — "Does this add noticeable latency to the scan cycle? What's the cost per symbol? 12k symbols × new component must stay under 5s." Rejects on unacceptable latency.

6. **Observability Inspector** — "When this fires or fails, can we see it? Is the component's output in the scan details? Is it on the dashboard card? Can we grep for it in logs?" Rejects if the component is a black box.

## Opinion format — single-page pass sheet

Each inspector writes ONE line:

```
[ ] Spec Compliance — [pass | reject: <one-sentence reason>]
[ ] Test Coverage    — ...
[ ] Regression       — ...
[ ] Data Integrity   — ...
[ ] Performance      — ...
[ ] Observability    — ...
```

If all six pass, submission is forwarded to the Court with the QC sheet attached.
If any reject, submission goes back with the specific fixes required. No Court time wasted.

## Rules

- **QC doesn't deliberate.** Each inspector is a mechanical check, not a debate.
- **QC doesn't overrule the Court, and the Court doesn't overrule QC.** Different layers, different jobs.
- **The implementer can challenge a QC rejection** by arguing their case in writing. If the Inspector reverses, the reason is recorded.
- **QC is cheap.** If it takes more than a few minutes per inspector, the submission is too big — break it up.
