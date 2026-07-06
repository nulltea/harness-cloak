"""Quote-anchored recovery of survived spans: cascade (invert) vs cascade+reconstructor.
Per-residue outcomes (recovered/wrong_insert/deletion/miss) + a no-harm rate over spans the
cascade already resolved. Run ONCE PER STRATUM (Round-1 weakness #4), never averaged:
  clinical held-out:  --corpora clinical --doc-split heldout   (--ckpt trained on clinical split)
  lexsum held-out:    --corpora lexsum   --doc-split heldout   (--ckpt trained on lexsum split)
  cross-domain:       --corpora lexsum   --doc-split all       (--ckpt trained on clinical)
--doc-split heldout keeps only doc_ids not in the training split file (data/recon_train_ids_<corpus>.txt,
written by the data builder / a 80-20 hash split); 'all' uses every doc.

Run: INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts:scripts/spikes \
       .venv/bin/python -u scripts/spikes/reconstructor_eval.py \
       --env data/ranker_env_pilot.json --arms data/task_arms_pilot.json \
       --corpora lexsum --n-docs 80 --doc-split all --ckpt data/models/reconstructor_v1
"""
import argparse, json
from pathlib import Path

from survival_by_type import (build_jobs, _judge, parse_judge, grounded, exact_present,
                              fill_present, SYSTEM, JUDGE_TMPL)
from cloak.extract import invert, _rule_prepass
from cloak.reconstruct import reconstruct, load_reconstructor, classify_recovery
from cloak.train.roundtrip import roundtrip_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--corpora", required=True); ap.add_argument("--n-docs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--doc-split", choices=["all", "heldout"], default="all")
    args = ap.parse_args()

    train_ids = set()
    if args.doc_split == "heldout":
        for c in args.corpora.split(","):
            p = Path(f"data/recon_train_ids_{c}.txt")
            if p.exists():
                train_ids |= set(p.read_text().split())

    jobs, metas = build_jobs(args)
    keep = [i for i, m in enumerate(metas) if m["doc_id"] not in train_ids]
    jobs, metas = [jobs[i] for i in keep], [metas[i] for i in keep]
    outs = roundtrip_batch(jobs, workers=args.workers)
    judge = _judge(); model = load_reconstructor(args.ckpt)
    OUT = ("recovered", "wrong_insert", "deletion", "miss")
    rows = {}   # type -> {survived, cascade:{outcome:n}, recon:{outcome:n}}
    harm = {"resolved": 0, "changed_by_recon": 0}

    for m, o in zip(metas, outs):
        out_p = o["out_p"]
        gens = [e for e in m["R"] if e["action"] == "generalize"]
        if not gens: continue
        items = "\n".join(f'{i}. "{e["surface"]}" -> "{e["replacement"]}"  [{e.get("type","MISC")}]'
                          for i, e in enumerate(gens))
        v = parse_judge(judge.generate(JUDGE_TMPL.format(items=items, out_p=out_p),
                                       system=SYSTEM), len(gens))
        casc = invert(out_p, m["R"])[0]
        recon = reconstruct(out_p, m["R"], model=model)[0]
        prepass, _, residue = _rule_prepass(out_p, m["R"], semantic=True)
        residue_surf = {e["surface"] for e in residue}
        for e, vv in zip(gens, v):
            q = vv.get("quote"); lbl = vv.get("label", "ABSENT")
            surv = exact_present(e["replacement"], out_p) or (
                lbl in ("SURVIVED", "REWORDED") and grounded(q, out_p))
            if not surv or (not fill_present(e["replacement"], out_p)
                            and exact_present(e["surface"], out_p)):
                continue   # not substituted-content survival
            # no-harm: spans the cascade already resolved (not in residue) must be untouched
            if e["surface"] not in residue_surf:
                harm["resolved"] += 1
                if casc.count(e["surface"]) != recon.count(e["surface"]):
                    harm["changed_by_recon"] += 1
                continue
            t = e.get("type", "MISC")
            r = rows.setdefault(t, dict(survived=0, cascade={k: 0 for k in OUT},
                                        recon={k: 0 for k in OUT}))
            r["survived"] += 1
            r["cascade"][classify_recovery(casc, q, e["surface"], prepass)] += 1
            r["recon"][classify_recovery(recon, q, e["surface"], prepass)] += 1

    report = {"ckpt": args.ckpt, "corpora": args.corpora, "doc_split": args.doc_split,
              "rows": rows, "harm": harm,
              "harm_rate": round(harm["changed_by_recon"] / harm["resolved"], 4)
                           if harm["resolved"] else None,
              "totals": {arm: {k: sum(r[arm][k] for r in rows.values()) for k in OUT}
                         for arm in ("cascade", "recon")}}
    Path("results/reconstructor_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
