# Methodology — Team Diamond

> **Reference material, not rules.** This file is explanatory.
> Binding engineering rules live in [`AGENTS.md`](../AGENTS.md).
> Current strategy, hypotheses and experiments live in [`suggestion.md`](../suggestion.md).

> **Status: intentionally not written yet.**
>
> The methodology is not settled, so writing it now would create a second source of
> truth that silently goes stale. That is the exact failure mode this reorganisation
> exists to prevent.
>
> A methodology document describes *what the final system does*. We do not have a
> final system yet.

## Where the method currently lives

| Question | Answered by | Status |
|:--|:--|:--|
| What does the metric compute, exactly? | [`suggestion.md`](../suggestion.md) §2 | **VERIFIED** — closed form re-derived |
| What decision policy should we use? | [`suggestion.md`](../suggestion.md) §3 | **PROPOSAL** — §3.6 defines the head-to-head |
| How does the official scorer treat singletons? | — | **UNKNOWN** — scorer not in repo; see §7.1 |
| What is the final architecture? | — | not built |
| What features actually helped? | — | no experiments run |

## When to write this file

Write `methodology.md` when, and only when:

1. P0 is complete — there is a validated submission end to end.
2. The decision policy has been chosen by the §3.6 comparison, not by argument.
3. The compliance position (§11.4 of `suggestion.md`) is resolved.

At that point this file becomes the scored "describing your approach" deliverable, and
`suggestion.md` should point at it rather than restate it.

## Interim rule

Until then: **the winning submission's actual code is the methodology.** If this file
and the code disagree, the code is the method and this file is wrong.
