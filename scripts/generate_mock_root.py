#!/usr/bin/env python3
"""Genere un .root bidon (TTree avec pt, eta, mass, weight) pour tester le pipeline.

shape=peak -> gaussienne (signal), shape=falling -> exponentielle (bkg)

    python scripts/generate_mock_root.py --output data/raw/sig.root \
        --shape peak --mean 125 --width 5 --n-events 10000 --seed 1
"""
import argparse
from pathlib import Path

import numpy as np
import uproot

MASS_FLOOR = 20.0  # GeV, debut de l'expo


def generate_events(n_events, shape, rng, mean=125.0, width=5.0, slope=50.0):
    if n_events < 0:
        raise ValueError(f"n_events doit être >= 0 (reçu {n_events})")

    if shape == "peak":
        mass = rng.normal(mean, width, n_events)
    elif shape == "falling":
        mass = MASS_FLOOR + rng.exponential(slope, n_events)
    else:
        raise ValueError(f"forme inconnue : {shape!r} (attendu 'peak' ou 'falling')")

    pt = rng.exponential(30.0, n_events) + 5.0
    eta = rng.uniform(-4.0, 4.0, n_events)
    # ~2% de poids negatifs, comme en NLO, pour tester la ponderation
    weight = rng.normal(1.0, 0.1, n_events)
    weight[rng.random(n_events) < 0.02] *= -1.0

    return {
        "pt": pt.astype(np.float32),
        "eta": eta.astype(np.float32),
        "mass": np.clip(mass, 0.0, None).astype(np.float32),
        "weight": weight.astype(np.float64),
    }


def write_root(path, tree_name, branches):
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as fout:
        fout[tree_name] = branches


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tree", default="events")
    p.add_argument("--n-events", type=int, default=10000)
    p.add_argument("--shape", choices=["peak", "falling"], default="falling")
    p.add_argument("--mean", type=float, default=125.0)
    p.add_argument("--width", type=float, default=5.0)
    p.add_argument("--slope", type=float, default=50.0, help="pente de l'expo (falling)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    branches = generate_events(args.n_events, args.shape, rng, mean=args.mean, width=args.width, slope=args.slope)
    write_root(args.output, args.tree, branches)
    print(f"[generate_mock] {args.n_events} events ({args.shape}) -> {args.output}")


if __name__ == "__main__":
    main()
