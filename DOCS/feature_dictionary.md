# Feature Dictionary — Team Diamond

> **Reference material, not rules.** This file is explanatory.
> Binding engineering rules live in [`AGENTS.md`](../AGENTS.md).
> Current strategy, hypotheses and experiments live in [`suggestion.md`](../suggestion.md).

**Status: PROPOSAL.** Nothing here is implemented. Features are listed by family so
that a retrieval or matching change can be checked against a written catalogue rather
than from memory. Treat individual features as candidates until an ablation shows they
help (`AGENTS.md` §Ablation Testing).

**Not yet decided:** how many features exist, which are cheap vs expensive, and the
stage boundary between them. The two-stage cascade and its feature split are PROPOSAL
in [`suggestion.md`](../suggestion.md) §4.2 — this file does not fix them.

---

## 22. Pairwise Matching

After retrieval:

```text
S1 entity
   |
   +-- candidate A
   +-- candidate B
   +-- candidate C
```

Generate features for each pair.

Potential features include:

#### Name

- exact normalized equality
- edit similarity
- Jaro-Winkler
- token overlap
- token-set similarity
- character n-gram similarity
- length difference
- token count difference

#### Address

- exact normalized equality
- character similarity
- token similarity
- component overlap
- number overlap
- postal-code evidence when available
- locality/city evidence
- street evidence

#### Cross-field

- name + address agreement
- name agreement when address missing
- address agreement when name noisy
- country agreement
- source combination

#### Retrieval evidence

- retrieved by exact name
- retrieved by character index
- retrieved by address
- number of retrieval methods agreeing
- retrieval rank
- best retrieval score
- candidate frequency / rarity

---
