"""RETIRED: does keeping attribute content verbatim help an LLM attacker recover the
IDENTITY spans? (Finding written up in docs/research/inference-risk-enforcement.md.)

Needs the ranker-v1 arms world — `train_ranker.assemble` and
`reward_gate.{IDENTITY_TYPES,identity_only_choice}` — retired in commit e960fee
("refactor: retire legacy ranker-v1 stack"). Recover the sources with
`git show e960fee^:scripts/train_ranker.py`; the v2 renderer is
`cloak.ranker.environment.assemble_action_vector` over a frozen environment, which needs
a different arm construction than this spike's `choice` dicts.
"""
raise SystemExit(
    "retired: needs train_ranker.assemble + reward_gate (git show e960fee^:scripts/)"
)
