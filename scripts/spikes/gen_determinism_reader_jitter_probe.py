"""Probe round-trip gen determinism and reader jitter under parallelism.

This is gate 1 for docs/specs/RL/roundtrip-ranker-infiller.md §Gates:

  1. Run the same rollout jobs through roundtrip_batch at workers=1 and workers=W.
     Each arm gets a fresh scratch CLOAK_LLM_CACHE so remote generation recomputes.
     The gate statistic is per-job out_p exact match.
  2. Freeze out_final texts from the workers=1 arm, then re-read those fixed texts at
     workers=1 and workers=W with refresh=True. The gate statistic is reader answer
     flip rate plus recall-delta distribution.

Actual execution hits the GPU/proxy and must be explicitly confirmed. Do not run this
on the shared box without checking occupancy first.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fresh-cache probe for round-trip generation determinism and reader jitter. "
            "Requires the local GPU/proxy and explicit --confirm-run."
        )
    )
    ap.add_argument("--env", default="data/ranker_env_full.json")
    ap.add_argument("--arms", default="data/task_arms_full.json")
    ap.add_argument("--probes", default="data/probes_validated.json")
    ap.add_argument("--corpus", default="clinical")
    ap.add_argument("--n-docs", type=int, default=3)
    ap.add_argument("--g", type=int, default=6, help="rollouts per doc")
    ap.add_argument("--workers", type=int, default=6, help="parallel workers to compare vs 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON report path; defaults under results/")
    ap.add_argument(
        "--confirm-run",
        action="store_true",
        help="required: confirms GPU/proxy occupancy was checked on the shared box",
    )
    return ap.parse_args(argv)


def _reset_clients() -> None:
    """Force fresh clients so each arm observes the arm-specific cache environment."""
    import cloak.train.reward as rw
    import cloak.train.roundtrip as rt

    rt._client = None
    rw._qa = None


@contextmanager
def _cache_env(cache_dir: Path):
    prev = os.environ.get("CLOAK_LLM_CACHE")
    os.environ["CLOAK_LLM_CACHE"] = str(cache_dir)
    try:
        _reset_clients()
        yield
    finally:
        _reset_clients()
        if prev is None:
            os.environ.pop("CLOAK_LLM_CACHE", None)
        else:
            os.environ["CLOAK_LLM_CACHE"] = prev


def _fresh_cache_dir(root: Path, name: str) -> Path:
    path = root / name
    if path.exists():
        raise RuntimeError(f"scratch cache arm already exists, refusing non-fresh cache: {path}")
    path.mkdir(parents=True)
    if any(path.iterdir()):
        raise RuntimeError(f"scratch cache arm is not empty: {path}")
    return path


def _roundtrip_arm(jobs: list[dict[str, Any]], *, workers: int, cache_dir: Path) -> dict[str, Any]:
    from cloak.train.roundtrip import roundtrip_batch

    with _cache_env(cache_dir):
        t0 = time.time()
        rows = roundtrip_batch(jobs, workers=workers)
        wall = time.time() - t0
    return {"workers": workers, "cache_dir": str(cache_dir), "wall_s": wall, "rows": rows}


def _recall_from_f1s(probes: list[dict[str, Any]], f1s: list[float]) -> float | None:
    from cloak.train.reward import _max_by_fact

    by_fact = _max_by_fact(probes, f1s)
    return (sum(by_fact.values()) / len(by_fact)) if by_fact else None


def _reader_arm(
    jobs: list[dict[str, Any]],
    out_finals: list[str],
    *,
    workers: int,
    cache_dir: Path,
) -> dict[str, Any]:
    from cloak.concurrent import pmap
    from cloak.train.reward import _read_batch, fact_score

    def one(item: tuple[int, dict[str, Any], str]) -> dict[str, Any]:
        i, job, out_final = item
        answers = _read_batch([p["question"] for p in job["probes"]], out_final, refresh=True)
        f1s = [fact_score(answer, probe["surface"]) for answer, probe in zip(answers, job["probes"])]
        return {
            "job_index": i,
            "doc_id": job["doc_id"],
            "answers": answers,
            "f1s": f1s,
            "recall": _recall_from_f1s(job["probes"], f1s),
        }

    with _cache_env(cache_dir):
        t0 = time.time()
        rows = pmap(one, list(zip(range(len(jobs)), jobs, out_finals)), workers=workers)
        wall = time.time() - t0
    return {"workers": workers, "cache_dir": str(cache_dir), "wall_s": wall, "rows": rows}


def _summary(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"n": 0, "mean": None, "min": None, "p50": None, "max": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "p50": statistics.median(vals),
        "max": max(vals),
    }


def _build_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from build_arms_artifact import load_artifact
    from train_ranker import derive_spans, sample_rollout

    from cloak.corpora import load_task_docs
    from cloak.train.ranker import RankerPolicy

    import torch

    torch.manual_seed(args.seed)

    art = load_artifact(args.arms)
    env = json.loads(Path(args.env).read_text())
    probes_all = json.loads(Path(args.probes).read_text())["docs"]
    floors = dict(env["k_floors"])
    per_doc = env["corpora"][args.corpus]
    texts = {d["id"]: d["text"] for d in load_task_docs(args.corpus, 300)}

    docs = []
    for doc_id, d in per_doc.items():
        probes = probes_all.get(doc_id, {}).get("train", [])
        if doc_id not in texts or not d.get("trainable") or not d["spans"] or len(probes) < 3:
            continue
        spans, feats = derive_spans(d["spans"], floors, args.corpus, "cpu")
        docs.append({
            "id": doc_id,
            "corpus": args.corpus,
            "text": texts[doc_id],
            "R_walk": art[args.corpus][doc_id]["tau_walk"][1],
            "spans": spans,
            "raw_spans": d["spans"],
            "feats": feats,
            "probes_train": probes,
        })
        if len(docs) >= args.n_docs:
            break

    if not docs:
        raise RuntimeError(
            f"no trainable docs with >=3 train probes for corpus={args.corpus!r}; "
            "check --env/--arms/--probes"
        )

    policy = RankerPolicy()
    jobs: list[dict[str, Any]] = []
    for doc in docs:
        for rollout_idx in range(args.g):
            _choice, _logps, _ph_rate, doc_p, R, _legals = sample_rollout(
                doc, doc["spans"], doc["feats"], policy
            )
            jobs.append({
                "job_id": f"{doc['id']}:{rollout_idx}",
                "doc_id": doc["id"],
                "corpus": doc["corpus"],
                "doc_p": doc_p,
                "R": R,
                "probes": doc["probes_train"],
            })

    meta = {
        "n_docs": len(docs),
        "n_jobs": len(jobs),
        "n_probe_reads": sum(len(j["probes"]) for j in jobs),
        "doc_ids": [d["id"] for d in docs],
    }
    return jobs, meta


def _gen_determinism_rows(
    jobs: list[dict[str, Any]],
    serial_rows: list[dict[str, Any]],
    parallel_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for i, (job, serial, parallel) in enumerate(zip(jobs, serial_rows, parallel_rows)):
        rows.append({
            "job_index": i,
            "job_id": job["job_id"],
            "doc_id": job["doc_id"],
            "out_p_exact": serial["out_p"] == parallel["out_p"],
            "out_final_exact": serial["out_final"] == parallel["out_final"],
            "recall_workers_1": serial["recall"],
            "recall_workers_parallel": parallel["recall"],
        })
    n_exact = sum(1 for r in rows if r["out_p_exact"])
    aggregate = {
        "n_jobs": len(rows),
        "out_p_exact_matches": n_exact,
        "out_p_exact_match_rate": n_exact / len(rows) if rows else None,
        "out_final_exact_matches": sum(1 for r in rows if r["out_final_exact"]),
    }
    return rows, aggregate


def _reader_jitter_rows(
    jobs: list[dict[str, Any]],
    serial_rows: list[dict[str, Any]],
    parallel_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    flips = 0
    total = 0
    deltas = []
    by_doc: dict[str, list[float]] = {}

    for i, (job, serial, parallel) in enumerate(zip(jobs, serial_rows, parallel_rows)):
        answer_rows = []
        for k, (probe, answer_1, answer_w) in enumerate(
            zip(job["probes"], serial["answers"], parallel["answers"])
        ):
            flipped = answer_1 != answer_w
            flips += int(flipped)
            total += 1
            answer_rows.append({
                "probe_index": k,
                "surface": probe["surface"],
                "question": probe["question"],
                "answer_workers_1": answer_1,
                "answer_workers_parallel": answer_w,
                "flipped": flipped,
            })
        r1 = serial["recall"]
        rw = parallel["recall"]
        delta = None if r1 is None or rw is None else rw - r1
        if delta is not None:
            deltas.append(delta)
            by_doc.setdefault(job["doc_id"], []).append(delta)
        rows.append({
            "job_index": i,
            "job_id": job["job_id"],
            "doc_id": job["doc_id"],
            "recall_workers_1": r1,
            "recall_workers_parallel": rw,
            "recall_delta_parallel_minus_1": delta,
            "answers": answer_rows,
        })

    aggregate = {
        "n_answers": total,
        "answer_flips": flips,
        "answer_flip_rate": flips / total if total else None,
        "recall_delta_distribution": _summary(deltas),
        "per_doc_recall_delta_distribution": {
            doc_id: _summary(vals) for doc_id, vals in sorted(by_doc.items())
        },
    }
    return rows, aggregate


def _default_report_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "results" / f"gen_determinism_reader_jitter_probe_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(
        "LOUD REMINDER: this probe hits the shared GPU/proxy. Check that no other GPU "
        "process is running and get user confirmation before launching.",
        flush=True,
    )
    if not args.confirm_run:
        print("Refusing to run without --confirm-run.", flush=True)
        return 2

    jobs, meta = _build_jobs(args)
    avg_reads = meta["n_probe_reads"] / max(meta["n_jobs"], 1)
    print(
        f"{meta['n_docs']} docs, {meta['n_jobs']} jobs, {meta['n_probe_reads']} "
        f"probe-reads ({avg_reads:.1f}/job)",
        flush=True,
    )

    previous_cache = os.environ.get("CLOAK_LLM_CACHE")
    if previous_cache:
        print(f"Existing CLOAK_LLM_CACHE is ignored for probe arms: {previous_cache}", flush=True)

    out_path = Path(args.out) if args.out else _default_report_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cloak-gen-det-reader-jitter-") as scratch:
        scratch_root = Path(scratch)
        print(f"scratch cache root={scratch_root}", flush=True)

        serial_cache = _fresh_cache_dir(scratch_root, "gen_workers_1")
        parallel_cache = _fresh_cache_dir(scratch_root, f"gen_workers_{args.workers}")
        reader_serial_cache = _fresh_cache_dir(scratch_root, "reader_workers_1")
        reader_parallel_cache = _fresh_cache_dir(scratch_root, f"reader_workers_{args.workers}")

        gen_serial = _roundtrip_arm(jobs, workers=1, cache_dir=serial_cache)
        print(
            f"[gen workers=1] wall={gen_serial['wall_s']:6.1f}s "
            f"{len(jobs) / max(gen_serial['wall_s'], 1e-9):.2f} rt/s",
            flush=True,
        )
        gen_parallel = _roundtrip_arm(jobs, workers=args.workers, cache_dir=parallel_cache)
        print(
            f"[gen workers={args.workers}] wall={gen_parallel['wall_s']:6.1f}s "
            f"{len(jobs) / max(gen_parallel['wall_s'], 1e-9):.2f} rt/s",
            flush=True,
        )

        gen_rows, gen_agg = _gen_determinism_rows(
            jobs, gen_serial["rows"], gen_parallel["rows"]
        )
        print(
            "SUMMARY gen_out_p_exact "
            f"{gen_agg['out_p_exact_matches']}/{gen_agg['n_jobs']} "
            f"rate={gen_agg['out_p_exact_match_rate']:.4f}",
            flush=True,
        )

        fixed_out_finals = [row["out_final"] for row in gen_serial["rows"]]
        reader_serial = _reader_arm(
            jobs, fixed_out_finals, workers=1, cache_dir=reader_serial_cache
        )
        print(
            f"[reader workers=1 refresh] wall={reader_serial['wall_s']:6.1f}s",
            flush=True,
        )
        reader_parallel = _reader_arm(
            jobs, fixed_out_finals, workers=args.workers, cache_dir=reader_parallel_cache
        )
        print(
            f"[reader workers={args.workers} refresh] wall={reader_parallel['wall_s']:6.1f}s",
            flush=True,
        )

        reader_rows, reader_agg = _reader_jitter_rows(
            jobs, reader_serial["rows"], reader_parallel["rows"]
        )
        delta_stats = reader_agg["recall_delta_distribution"]
        print(
            "SUMMARY reader_answer_flips "
            f"{reader_agg['answer_flips']}/{reader_agg['n_answers']} "
            f"rate={reader_agg['answer_flip_rate']:.4f}",
            flush=True,
        )
        print(
            "SUMMARY reader_recall_delta "
            f"n={delta_stats['n']} mean={delta_stats['mean']} "
            f"min={delta_stats['min']} p50={delta_stats['p50']} max={delta_stats['max']}",
            flush=True,
        )

        report = {
            "meta": {
                "created": datetime.now().isoformat(timespec="seconds"),
                "args": vars(args),
                "scratch_root": str(scratch_root),
                "external_cloak_llm_cache_ignored": previous_cache,
                **meta,
            },
            "gen_determinism": {
                "workers_1": {
                    "wall_s": gen_serial["wall_s"],
                    "cache_dir": gen_serial["cache_dir"],
                },
                f"workers_{args.workers}": {
                    "wall_s": gen_parallel["wall_s"],
                    "cache_dir": gen_parallel["cache_dir"],
                },
                "aggregate": gen_agg,
                "jobs": gen_rows,
            },
            "reader_jitter": {
                "workers_1": {
                    "wall_s": reader_serial["wall_s"],
                    "cache_dir": reader_serial["cache_dir"],
                },
                f"workers_{args.workers}": {
                    "wall_s": reader_parallel["wall_s"],
                    "cache_dir": reader_parallel["cache_dir"],
                },
                "aggregate": reader_agg,
                "jobs": reader_rows,
            },
        }
        out_path.write_text(json.dumps(report, indent=2))

    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
