---
type: paper
node_id: paper:katzsamuels2019_true_sample_complexity_good_arms
title: "The True Sample Complexity of Identifying Good Arms"
authors: ["Julian Katz-Samuels", "Kevin Jamieson"]
year: 2019
venue: "arXiv/AISTATS 2020"
external_ids:
  arxiv: "1906.06594"
  doi: null
  s2: null
tags: ["bandits", "verifiability", "epsilon-good", "sample-complexity"]
added: 2026-07-31T00:00:00Z
---

# The True Sample Complexity of Identifying Good Arms

## One-line thesis
Splits identification into verifiable vs unverifiable regimes: sets defined against a fixed absolute threshold are finite-sample verifiable; sets defined within epsilon of the UNKNOWN best mean are not (the reference point never stops being an estimate).

## Key Results
- Our exact-tie observation (Delta-U identically 0 across all probes) lives in the verifiable regime; the epsilon-band around an estimated max does not.

## Relevance to This Project
Round-3 sweep: the theoretical basis for shipping the exact-tie trigger first and treating the 0<|dU|<epsilon band as a separate, later, lower-confidence extension.
