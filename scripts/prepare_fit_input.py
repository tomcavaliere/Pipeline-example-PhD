#!/usr/bin/env python3
"""npz -> TH1D (via uproot) pour fit_mass.py.

Tourne dans l'env du pipeline, pas besoin de ROOT ici (le fit lui a besoin de PyROOT).
Ecrit h_{sample}, mc_signal, mc_background et fit_data. Le contenu de fit_data
depend de --dataset : mc_sum (sig+bkg), pseudo_data (toy poisson de mc_sum), data.

    python scripts/prepare_fit_input.py --hists work/histograms.npz \
        --output work/fit_input.root --dataset mc_sum --kinds sig=signal bkg=background
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import uproot
from uproot.writing.identify import to_TAxis, to_TH1x

from plot_summary import load_histograms, parse_mapping

DATASETS = ("mc_sum", "pseudo_data", "data")


def make_th1d(name, title, counts, sumw2, edges, axis_title=""):
    counts = np.asarray(counts, dtype=np.float64)
    sumw2 = np.asarray(sumw2, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # bins 0 et n+1 = under/overflow, on les laisse a 0
    pad = lambda a: np.concatenate([[0.0], a, [0.0]])  # noqa: E731
    xaxis = to_TAxis(fName="xaxis", fTitle=axis_title, fNbins=len(counts),
                     fXmin=float(edges[0]), fXmax=float(edges[-1]))
    return to_TH1x(
        fName=name, fTitle=title, data=pad(counts),
        fEntries=float(counts.sum()), fTsumw=float(counts.sum()), fTsumw2=float(sumw2.sum()),
        fTsumwx=float((counts * centers).sum()), fTsumwx2=float((counts * centers**2).sum()),
        fSumw2=pad(sumw2), fXaxis=xaxis,
    )


def sum_by_kind(hists, kinds, kind, nbins):
    counts, sumw2 = np.zeros(nbins), np.zeros(nbins)
    for sample, (c, w2) in hists.items():
        if kinds.get(sample) == kind:
            counts, sumw2 = counts + c, sumw2 + w2
    return counts, sumw2


def build_fit_data(hists, kinds, dataset, nbins, seed=0):
    if dataset not in DATASETS:
        raise ValueError(f"dataset inconnu : {dataset!r} (attendu {DATASETS})")
    if dataset == "data":
        if "data" not in kinds.values():
            raise SystemExit("dataset=data mais aucun échantillon de kind 'data' n'est déclaré")
        return sum_by_kind(hists, kinds, "data", nbins)

    sig, sig_w2 = sum_by_kind(hists, kinds, "signal", nbins)
    bkg, bkg_w2 = sum_by_kind(hists, kinds, "background", nbins)
    expected, expected_w2 = sig + bkg, sig_w2 + bkg_w2
    if dataset == "mc_sum":
        return expected, expected_w2
    # toy : n_i ~ Poisson(nu_i), erreurs poissoniennes donc sumw2 = n
    toy = np.random.default_rng(seed).poisson(np.clip(expected, 0.0, None)).astype(np.float64)
    return toy, toy.copy()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hists", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=DATASETS, default="mc_sum")
    p.add_argument("--kinds", nargs="*", default=[], help="sample=signal|background|data")
    p.add_argument("--seed", type=int, default=0, help="seed du toy (pseudo_data)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kinds = parse_mapping(args.kinds)
    edges, variable, hists = load_histograms(args.hists)
    nbins = len(edges) - 1

    objects = {f"h_{s}": make_th1d(f"h_{s}", s, c, w2, edges, variable) for s, (c, w2) in hists.items()}
    for kind, name in (("signal", "mc_signal"), ("background", "mc_background")):
        objects[name] = make_th1d(name, name, *sum_by_kind(hists, kinds, kind, nbins), edges, variable)
    counts, sumw2 = build_fit_data(hists, kinds, args.dataset, nbins, args.seed)
    objects["fit_data"] = make_th1d("fit_data", args.dataset, counts, sumw2, edges, variable)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.output) as fout:
        for name, obj in objects.items():
            fout[name] = obj
    print(f"prepare_fit: dataset={args.dataset}, sum={counts.sum():.1f} -> {args.output}")


if __name__ == "__main__":
    main()
