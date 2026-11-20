#!/usr/bin/env python3
"""Selection des evenements (coupures pt/eta/masse) + cutflow.

    python scripts/select_events.py --input data/raw/sig.root \
        --output work/selected/sig.parquet --cutflow work/selected/sig.cutflow.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

BRANCHES = ["pt", "eta", "mass", "weight"]
OUTPUT_COLUMNS = BRANCHES + ["sample"]


@dataclass(frozen=True)
class Cuts:
    # None = pas de coupure
    pt_min: float | None = None
    abs_eta_max: float | None = None
    mass_min: float | None = None
    mass_max: float | None = None


def read_events(path, tree_name, branches=BRANCHES):
    with uproot.open(path) as fin:
        if tree_name not in fin:
            raise KeyError(f"TTree {tree_name!r} absent de {path} (clés : {fin.keys()})")
        tree = fin[tree_name]
        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            raise KeyError(f"Branches manquantes dans {path}: {missing}")
        return tree.arrays(branches, library="ak")


def build_masks(events, cuts):
    # masques cumulatifs -> le i-eme masque = les i premieres coupures, donc
    # c'est directement le cutflow. Marche aussi avec un dict de np.array ou un df.
    n = len(events["pt"])
    mask = np.ones(n, dtype=bool)
    steps = [("all", mask.copy())]

    def add(name, cond):
        nonlocal mask
        mask = mask & np.asarray(cond, dtype=bool)
        steps.append((name, mask.copy()))

    pt, eta, mass = (np.asarray(events[k]) for k in ("pt", "eta", "mass"))
    if cuts.pt_min is not None:
        add(f"pt > {cuts.pt_min:g}", pt > cuts.pt_min)
    if cuts.abs_eta_max is not None:
        add(f"|eta| < {cuts.abs_eta_max:g}", np.abs(eta) < cuts.abs_eta_max)
    if cuts.mass_min is not None:
        add(f"mass > {cuts.mass_min:g}", mass > cuts.mass_min)
    if cuts.mass_max is not None:
        add(f"mass < {cuts.mass_max:g}", mass < cuts.mass_max)
    return steps


def compute_cutflow(steps, weights):
    weights = np.asarray(weights, dtype=np.float64)
    return [{"cut": name, "n_raw": int(m.sum()), "sum_weights": float(weights[m].sum())}
            for name, m in steps]


def select(events, cuts, scale=1.0, sample="unknown"):
    """Retourne (df, cutflow). poids final = weight * scale"""
    steps = build_masks(events, cuts)
    sel = steps[-1][1]
    w = np.asarray(events["weight"], dtype=np.float64) * scale

    df = pd.DataFrame({
        "pt": np.asarray(events["pt"])[sel].astype(np.float32),
        "eta": np.asarray(events["eta"])[sel].astype(np.float32),
        "mass": np.asarray(events["mass"])[sel].astype(np.float32),
        "weight": w[sel],
    })
    df["sample"] = pd.Series([sample] * len(df), dtype="string")
    df = df[OUTPUT_COLUMNS]
    # sinon on perd le nom de l'echantillon quand le df est vide
    df.attrs["sample"] = sample
    return df, compute_cutflow(steps, w)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cutflow", type=Path, required=True)
    p.add_argument("--tree", default="events")
    p.add_argument("--sample", default=None, help="defaut: nom du fichier")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--pt-min", type=float, default=None)
    p.add_argument("--abs-eta-max", type=float, default=None)
    p.add_argument("--mass-min", type=float, default=None)
    p.add_argument("--mass-max", type=float, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sample = args.sample or args.input.stem
    cuts = Cuts(args.pt_min, args.abs_eta_max, args.mass_min, args.mass_max)

    events = read_events(args.input, args.tree)
    df, cutflow = select(events, cuts, scale=args.scale, sample=sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cutflow.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    args.cutflow.write_text(json.dumps({"sample": sample, "scale": args.scale, "cutflow": cutflow}, indent=2))

    n_in, n_out = cutflow[0]["n_raw"], cutflow[-1]["n_raw"]
    print(f"[select] {sample}: {n_out}/{n_in} evts -> {args.output}")
    if n_out == 0:
        print(f"[select] WARNING rien ne passe la selection pour {sample}")


if __name__ == "__main__":
    main()
