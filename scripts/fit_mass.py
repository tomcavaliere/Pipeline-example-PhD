#!/usr/bin/env python3
"""Fit RooFit de la masse : gaussienne + bkg (expo ou bernstein), binned, etendu.

A lancer avec un python qui a PyROOT (cle fit.python de la config) -> que ROOT
+ stdlib ici, pas de numpy.

    python3 scripts/fit_mass.py --input work/fit_input.root \
        --output-json results/fit_mass.json --output-png results/fit_mass.png \
        --output-workspace work/fit_workspace.root
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import ROOT
except ImportError:
    sys.exit(
        f"PyROOT introuvable pour l'interpréteur {sys.executable}.\n"
        "Installez ROOT (ex. `brew install root`) et réglez `fit.python` dans config/config.yaml."
    )

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

BACKGROUNDS = ("exponential", "bernstein")


def read_hist(path, name):
    f = ROOT.TFile.Open(str(path))
    if not f or f.IsZombie():
        raise OSError(f"Impossible d'ouvrir {path}")
    h = f.Get(name)
    if h:
        h.SetDirectory(0)
    f.Close()
    return h or None


def build_model(x, total, args):
    # ATTENTION : PyROOT ne garde pas les dependances des pdf en vie -> on
    # renvoie tout dans `keep`, sinon segfault au moment du fit
    mean = ROOT.RooRealVar("mean", "#mu", args.mean_init, *args.mean_range)
    sigma = ROOT.RooRealVar("sigma", "#sigma", args.sigma_init, *args.sigma_range)
    sig = ROOT.RooGaussian("sig", "signal", x, mean, sigma)

    keep = [mean, sigma]
    if args.background == "exponential":
        tau = ROOT.RooRealVar("tau", "#tau", -0.02, -1.0, 0.0)
        bkg = ROOT.RooExponential("bkg", "bruit de fond", x, tau)
        keep.append(tau)
    else:
        coefs = ROOT.RooArgList()
        for i in range(args.bkg_order + 1):
            c = ROOT.RooRealVar(f"b{i}", f"b_{{{i}}}", 1.0, 0.0, 100.0)
            if i == 0:
                c.setConstant(True)  # sinon degenere avec nbkg
            coefs.add(c)
            keep.append(c)
        bkg = ROOT.RooBernstein("bkg", "bruit de fond", x, coefs)
        keep.append(coefs)

    # nsig peut aller a 0, on ne veut pas forcer un pic
    nsig = ROOT.RooRealVar("nsig", "N_{sig}", 0.1 * total, 0.0, 2.0 * total)
    nbkg = ROOT.RooRealVar("nbkg", "N_{bkg}", 0.9 * total, 0.0, 2.0 * total)
    model = ROOT.RooAddPdf("model", "signal + bruit de fond",
                           ROOT.RooArgList(sig, bkg), ROOT.RooArgList(nsig, nbkg))
    keep += [sig, bkg, nsig, nbkg]
    return model, keep


def param_dict(var):
    d = {"value": var.getVal(), "error": var.getError()}
    if var.hasAsymError():
        d["error_lo"], d["error_hi"] = var.getAsymErrorLo(), var.getAsymErrorHi()
    return d


def draw(x, data, model, frame_title, weighted, n_float, summary_lines, output):
    """plot + pulls, renvoie (chi2/ndf, ndf)"""
    frame = x.frame(ROOT.RooFit.Title(frame_title))
    err = ROOT.RooAbsData.SumW2 if weighted else ROOT.RooAbsData.Poisson
    data.plotOn(frame, ROOT.RooFit.DataError(err), ROOT.RooFit.Name("data"))
    model.plotOn(frame, ROOT.RooFit.Components("bkg"), ROOT.RooFit.LineStyle(ROOT.kDashed),
                 ROOT.RooFit.LineColor(ROOT.kGray + 2), ROOT.RooFit.Name("bkg_curve"))
    model.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed + 1), ROOT.RooFit.Name("model_curve"))
    chi2_ndf = frame.chiSquare("model_curve", "data", n_float)
    ndf = data.numEntries() - n_float
    pulls = frame.pullHist("data", "model_curve")

    c = ROOT.TCanvas("c", "", 800, 700)
    top = ROOT.TPad("top", "", 0, 0.3, 1, 1)
    bot = ROOT.TPad("bot", "", 0, 0, 1, 0.3)
    top.SetBottomMargin(0.02)
    bot.SetTopMargin(0.03)
    bot.SetBottomMargin(0.3)
    for p in (top, bot):
        p.SetLeftMargin(0.12)
        p.Draw()

    top.cd()
    frame.GetXaxis().SetLabelSize(0)
    frame.GetYaxis().SetTitleOffset(1.3)
    frame.Draw()
    leg = ROOT.TLegend(0.62, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(frame.findObject("data"), "Donn#acute{e}es", "pe")  # pas d'utf8 dans TLatex...
    leg.AddEntry(frame.findObject("model_curve"), "Signal + bruit de fond", "l")
    leg.AddEntry(frame.findObject("bkg_curve"), "Bruit de fond", "l")
    leg.Draw()
    box = ROOT.TPaveText(0.62, 0.42, 0.88, 0.68, "NDC")
    box.SetBorderSize(0)
    box.SetFillStyle(0)
    box.SetTextAlign(12)
    box.SetTextSize(0.035)
    for line in summary_lines + [f"#chi^{{2}}/ndf = {chi2_ndf:.2f}"]:
        box.AddText(line)
    box.Draw()

    bot.cd()
    pframe = x.frame(ROOT.RooFit.Title(""))
    pframe.SetTitle("")
    pframe.addPlotable(pulls, "P")
    pframe.SetMinimum(-4.5)
    pframe.SetMaximum(4.5)
    pframe.GetYaxis().SetTitle("Pull")
    pframe.GetYaxis().SetNdivisions(505)
    for axis in (pframe.GetXaxis(), pframe.GetYaxis()):
        axis.SetLabelSize(0.11)
        axis.SetTitleSize(0.11)
    pframe.GetYaxis().SetTitleOffset(0.45)
    pframe.Draw()
    line = ROOT.TLine(x.getMin(), 0, x.getMax(), 0)
    line.SetLineStyle(ROOT.kDashed)
    line.Draw()

    output.parent.mkdir(parents=True, exist_ok=True)
    c.SaveAs(str(output))
    return chi2_ndf, ndf


def draw_empty(output, message):
    c = ROOT.TCanvas("c", "", 800, 700)
    t = ROOT.TLatex(0.5, 0.5, message)
    t.SetNDC()
    t.SetTextAlign(22)
    t.SetTextColor(ROOT.kGray + 2)
    t.Draw()
    output.parent.mkdir(parents=True, exist_ok=True)
    c.SaveAs(str(output))


def truth_yield(path, name):
    h = read_hist(path, name)
    return h.Integral() if h else None


def fit(args):
    h = read_hist(args.input, "fit_data")
    if h is None:
        raise KeyError(f"Histogramme 'fit_data' absent de {args.input}")
    dataset = h.GetTitle()
    variable = h.GetXaxis().GetTitle() or "x"
    lo, hi = h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax()
    total = h.Integral()
    weighted = dataset != "pseudo_data"

    model_info = {"signal": "gaussian", "background": args.background}
    if args.background == "bernstein":
        model_info["bkg_order"] = args.bkg_order
    result = {
        "dataset": dataset, "variable": variable, "range": [lo, hi], "bins": h.GetNbinsX(),
        "model": model_info,
        "sum_weights": total,
        "truth": {"nsig": truth_yield(args.input, "mc_signal"),
                  "nbkg": truth_yield(args.input, "mc_background")},
    }

    if total <= 0:
        draw_empty(args.output_png, "Aucun événement à ajuster")
        ROOT.RooWorkspace("w").writeToFile(str(args.output_workspace))
        return {**result, "status": "no_data"}

    x = ROOT.RooRealVar(variable, f"{variable} [GeV]", lo, hi)
    # NB: IntegrateBins integre sur le binning de x (100 par defaut) -> doit
    # etre le meme que le TH1 sinon la masse fittee est biaisee
    x.setBins(h.GetNbinsX())
    data = ROOT.RooDataHist("data", dataset, ROOT.RooArgList(x), h)
    model, _keep = build_model(x, total, args)

    # SumW2Error seulement si poids != 1 (mc_sum / data ponderees)
    res = model.fitTo(
        data, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True),
        ROOT.RooFit.SumW2Error(weighted), ROOT.RooFit.IntegrateBins(1e-3),
        ROOT.RooFit.PrintLevel(-1),
    )
    params = {p.GetName(): param_dict(p) for p in res.floatParsFinal()}

    nsig, mean, sigma = params["nsig"], params["mean"], params["sigma"]
    lines = [
        f"#mu = {mean['value']:.2f} #pm {mean['error']:.2f} GeV",
        f"#sigma = {sigma['value']:.2f} #pm {sigma['error']:.2f} GeV",
        f"N_{{sig}} = {nsig['value']:.0f} #pm {nsig['error']:.0f}",
        f"N_{{bkg}} = {params['nbkg']['value']:.0f} #pm {params['nbkg']['error']:.0f}",
    ]
    chi2_ndf, ndf = draw(x, data, model, f"Fit {dataset}", weighted,
                         res.floatParsFinal().getSize(), lines, args.output_png)

    ws = ROOT.RooWorkspace("w", "fit de masse")
    ws.Import(model)
    ws.Import(data)
    ws.Import(res, "fit_result")
    ws.writeToFile(str(args.output_workspace))

    converged = res.status() == 0 and res.covQual() == 3
    return {
        **result,
        "status": "ok" if converged else "not_converged",
        "minuit_status": res.status(),
        "cov_quality": res.covQual(),
        "edm": res.edm(),
        "min_nll": res.minNll(),
        "chi2_ndf": chi2_ndf,
        "ndf": ndf,
        "parameters": params,
        "signal_over_error": nsig["value"] / nsig["error"] if nsig["error"] > 0 else None,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-png", type=Path, required=True)
    p.add_argument("--output-workspace", type=Path, required=True)
    p.add_argument("--background", choices=BACKGROUNDS, default="exponential")
    p.add_argument("--bkg-order", type=int, default=4, help="ordre bernstein")
    p.add_argument("--mean-init", type=float, default=125.0)
    p.add_argument("--mean-range", type=float, nargs=2, default=(100.0, 150.0), metavar=("LO", "HI"))
    p.add_argument("--sigma-init", type=float, default=5.0)
    p.add_argument("--sigma-range", type=float, nargs=2, default=(1.0, 20.0), metavar=("LO", "HI"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.bkg_order < 1:
        raise SystemExit(f"--bkg-order doit être >= 1 (reçu {args.bkg_order})")
    args.output_workspace.parent.mkdir(parents=True, exist_ok=True)
    out = fit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))

    print(f"fit: status={out['status']}, dataset={out['dataset']}")
    for name, p in out.get("parameters", {}).items():
        print(f"  {name:6s} = {p['value']:.4g} +- {p['error']:.2g}")
    if "chi2_ndf" in out and math.isfinite(out["chi2_ndf"]):
        print(f"  chi2/ndf = {out['chi2_ndf']:.2f} (ndf = {out['ndf']})")


if __name__ == "__main__":
    main()
