#!/usr/bin/env python3
"""Build the embedding index sidecar for a lattice profile artifact."""
import argparse

import numpy as np

from cloak.lattice.profile_match import DEFAULT_MODEL_ID, build_embindex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profiles_path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    args = ap.parse_args()

    out = build_embindex(args.profiles_path, out_path=args.out, model_id=args.model)
    rows = len(np.load(out, allow_pickle=False)["vectors"])
    print(f"wrote {out} ({rows} rows)")


if __name__ == "__main__":
    main()
