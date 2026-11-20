#!/usr/bin/env python3
"""Plot de synthese (bkg empiles + signal) et tableau csv par echantillon.

    python scripts/plot_summary.py --hists work/histograms.npz \
        --metrics work/metrics.json --cutflows work/selected/*.cutflow.json \
        --output-png results/summary_mass.png --output-csv results/summary.csv \
        --kinds sig=signal bkg=background
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de display sur le cluster
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_mapping(items):
    # ['a=x', 'b=y'] -> {'a': 'x', 'b': 'y'}
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Format attendu clé=valeur, reçu {item!r}")
        out[key] = value
    return out


def load_histograms(path):
    """-> edges, variable, {sample: (counts, sumw2)}"""
    with np.load(path) as data:
        edges = data["edges"]
        variable = str(data["variable"]) if "variable" in data else "x"
        samples = sorted({k.split("__")[0] for k in data.files if "__" in k})
        hists = {s: (data[f"{s}__counts"], data[f"{s}__sumw2"]) for s in samples}
    return edges, variable, hists


def build_summary_table(metrics, cutflows, kinds):
    cf_by_sample = {cf["sample"]: cf["cutflow"] for cf in cutflows}
    var = metrics["variable"]
    rows = []
    for sample, m in metrics["samples"].items():
        cf = cf_by_sample.get(sample, [])
        n_total = cf[0]["n_raw"] if cf else None
        n_sel = m["n_events"]
        rows.append({
            "sample": sample,
            "kind": kinds.get(sample, "unknown"),
            "n_generated": n_total,
            "n_selected": n_sel,
            "efficiency": (n_sel / n_total) if n_total else None,
            "sum_weights": m["sum_weights"],
            "sum_weights_in_range": m["sum_weights_in_range"],
            f"{var}_mean": m.get(f"{var}_mean"),
            f"{var}_std": m.get(f"{var}_std"),
        })
    return pd.DataFrame(rows)


def plot_stack(edges, variable, hists, kinds, labels, output):
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(8, 5.5))

    bkgs = [s for s in hists if kinds.get(s) == "background"]
    sigs = [s for s in hists if kinds.get(s) == "signal"]
    data = [s for s in hists if kinds.get(s) == "data"]

    bottom = np.zeros(len(centers))
    tot_w2 = np.zeros(len(centers))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(bkgs), 1)))
    for color, s in zip(colors, bkgs):
        counts, sumw2 = hists[s]
        ax.bar(edges[:-1], counts, width=np.diff(edges), bottom=bottom, align="edge",
               color=color, edgecolor="none", label=labels.get(s, s))
        bottom = bottom + counts
        tot_w2 = tot_w2 + sumw2

    # bande stat MC sur la somme des bkg
    # (le append du dernier point c'est pour que step="post" aille jusqu'au bord)
    if bkgs:
        err = np.sqrt(tot_w2)
        ax.fill_between(edges, np.append(bottom - err, (bottom - err)[-1]),
                        np.append(bottom + err, (bottom + err)[-1]), step="post",
                        facecolor="none", edgecolor="gray", hatch="////", linewidth=0,
                        label="Stat. MC")

    for s in sigs:
        ax.stairs(hists[s][0], edges, color="crimson", linewidth=2, label=labels.get(s, s))
    for s in data:
        counts, sumw2 = hists[s]
        ax.errorbar(centers, counts, yerr=np.sqrt(sumw2), fmt="ko", markersize=4, label=labels.get(s, s))

    if sum(h[0].sum() for h in hists.values()) == 0:
        ax.text(0.5, 0.5, "Aucun événement sélectionné", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"{variable} [GeV]")
    ax.set_ylabel(f"Événements pondérés / {np.diff(edges)[0]:g} GeV")
    ax.set_title("Distribution après sélection")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hists", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--cutflows", type=Path, nargs="*", default=[])
    p.add_argument("--output-png", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--kinds", nargs="*", default=[], help="sample=signal|background|data")
    p.add_argument("--labels", nargs="*", default=[])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kinds, labels = parse_mapping(args.kinds), parse_mapping(args.labels)

    edges, variable, hists = load_histograms(args.hists)
    metrics = json.loads(args.metrics.read_text())
    cutflows = [json.loads(p.read_text()) for p in args.cutflows]

    plot_stack(edges, variable, hists, kinds, labels, args.output_png)

    table = build_summary_table(metrics, cutflows, kinds)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False, float_format="%.4g")
    print(table.to_string(index=False))
    print(f"[plot] -> {args.output_png}, {args.output_csv}")


if __name__ == "__main__":
    main()
