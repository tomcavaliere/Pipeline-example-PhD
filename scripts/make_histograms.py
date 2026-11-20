#!/usr/bin/env python3
"""Histos ponderes (counts + sumw2) et quelques metriques par echantillon.

    python scripts/make_histograms.py --inputs work/selected/*.parquet \
        --output-hists work/histograms.npz --output-metrics work/metrics.json

npz : edges, variable, {sample}__counts, {sample}__sumw2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def weighted_histogram(values, weights, bins, range_):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    counts, edges = np.histogram(values, bins=bins, range=range_, weights=weights)
    sumw2, _ = np.histogram(values, bins=bins, range=range_, weights=weights**2)
    return counts, sumw2, edges


def weighted_stats(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    sumw = float(weights.sum())
    if values.size == 0 or sumw == 0.0:
        return {"mean": None, "std": None}
    mean = float(np.sum(weights * values) / sumw)
    var = float(np.sum(weights * (values - mean) ** 2) / sumw)
    # var < 0 possible avec les poids negatifs (NLO)
    return {"mean": mean, "std": float(np.sqrt(var)) if var >= 0 else None}


def summarize_sample(df, variable, bins, range_):
    values, weights = df[variable].to_numpy(), df["weight"].to_numpy()
    counts, sumw2, edges = weighted_histogram(values, weights, bins, range_)
    lo, hi = range_
    metrics = {
        "n_events": int(len(df)),
        "sum_weights": float(weights.sum()) if len(df) else 0.0,
        "sum_weights_in_range": float(counts.sum()),
        "n_underflow": int(np.sum(values < lo)),
        "n_overflow": int(np.sum(values >= hi)),
    }
    for k, v in weighted_stats(values, weights).items():
        metrics[f"{variable}_{k}"] = v
    return counts, sumw2, edges, metrics


def sample_name(path, df):
    # attrs parquet > colonne sample > nom de fichier
    if df.attrs.get("sample"):
        return str(df.attrs["sample"])
    if "sample" in df.columns and len(df):
        return str(df["sample"].iloc[0])
    return path.stem


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output-hists", type=Path, required=True)
    p.add_argument("--output-metrics", type=Path, required=True)
    p.add_argument("--variable", default="mass")
    p.add_argument("--bins", type=int, default=35)
    p.add_argument("--range", type=float, nargs=2, default=(60.0, 200.0), metavar=("LO", "HI"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.bins <= 0 or args.range[0] >= args.range[1]:
        raise SystemExit(f"binning invalide : bins={args.bins}, range={args.range}")

    arrays = {}
    all_metrics = {}
    edges = None
    for path in args.inputs:
        df = pd.read_parquet(path)
        name = sample_name(path, df)
        counts, sumw2, edges, m = summarize_sample(df, args.variable, args.bins, tuple(args.range))
        arrays[name + "__counts"] = counts
        arrays[name + "__sumw2"] = sumw2
        all_metrics[name] = m
        print(f"[hist] {name}: {m['n_events']} evts, sumw = {m['sum_weights']:.2f}")

    args.output_hists.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_hists, edges=edges, variable=np.array(args.variable), **arrays)
    out = {"variable": args.variable, "bins": args.bins, "range": list(args.range), "samples": all_metrics}
    args.output_metrics.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
