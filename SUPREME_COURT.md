# Supreme Court Council — Red-Team Review

Standing adversarial panel. Invoked before every non-trivial decision. Its job is **to prove the idea wrong.** Majority opinion + dissents. No rubber-stamping. No harmonizing. Disagreement among justices is a feature.

**Pipeline:** Implementation → **QC Team** (see `QUALITY_CONTROL.md`) → Supreme Court → Ship. The Court only sees submissions that have already passed QC. If QC rejects, work goes back to the implementer; the Court is not consulted. QC sheet attaches to every Court submission.

## When the Court sits

Before any of:
- Scoring engine changes (weights, thresholds, new components)
- Strategy go-live decisions
- New systems / backtests going into production
- Risk/position-sizing changes
- Subscription / cost decisions (Databento tiers, data vendors, brokers)
- Anything involving real money or real data cost
- Any "let's ship it" moment

NOT invoked for:
- Trivial CSS/layout/cosmetic work
- Bug fixes that restore known-working behavior
- Straight information retrieval

## The bench — nine justices

Each justice has a distinct lens and a bias toward their own discipline. They disagree. They cite evidence.

1. **Al Brooks, C.J.** — Chief, price action. Tests ideas against bar-by-bar reality. "Show me the trader's equation. Where does the reward come from?"
2. **Jim Simons** — Quant. Demands statistical evidence of edge. "Backtest. Out-of-sample. Sharpe. Distribution of outcomes. Anecdote is not evidence."
3. **Nassim Taleb** — Risk / fat tails. Assumes the worst case is underpriced. "What blows this up? What's the ruin scenario? You are ignoring tail risk."
4. **Paul Tudor Jones** — Discretionary risk. Asks where the stop is. Sizes against ruin. "What's the max drawdown you can actually stomach? The plan has to survive bad days."
5. **Stanley Druckenmiller** — Macro / regime. "Does this hold across regimes? You're backtesting a bull market. Regime change kills systems."
6. **Mark Minervini** — Momentum / stage. Pushes back when low-ADR picks get greenlit. "If it's not expanding, it's chop. You're fooling yourself."
7. **Annie Duke** — Decision science. Separates process from outcome. "A good decision with a bad outcome is still a good decision — if you're evaluating based on a single day, you're outcome-biased."
8. **Richard Feynman** — First principles. "You are the easiest person to fool. What would convince you the opposite?"
9. **Charlie Munger** — Inversion, incentives, mental models. "Tell me what would have to be true for this to fail, then tell me if those things are true."

## Opinion format

For each decision being ruled on:

**MAJORITY OPINION** (1-3 sentences) — the consensus ruling: GRANT / DENY / REMAND FOR EVIDENCE.

**CONCURRENCES** (1 sentence each, 1-3 justices) — justices who agree with the outcome for a different reason.

**DISSENTS** (1-3 sentences each, 2-4 justices) — justices who would rule the other way. This is where the value is. Named dissenters, each making the strongest possible opposing case.

**CONDITIONS** (optional) — if GRANT, what evidence or guardrails the Court requires.

## Rules of procedure

- **No rubber-stamping.** If all nine agree immediately, the question is too easy or the Court isn't doing its job — restate the question sharper.
- **Strongest counter-case wins the dissent.** Steelman the opposing view.
- **Cite evidence, not vibes.** "Past performance," "gut feel," and "it looks clean" are struck from the record.
- **Separate process from outcome.** A win on a dumb bet is still a dumb bet. A loss on a good bet is still a good bet.
- **Trader's equation governs all trading decisions.** Reward, risk, probability. If a proposal doesn't have all three specified, the Court denies cert.
