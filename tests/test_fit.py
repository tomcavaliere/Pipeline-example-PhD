"""tests export TH1D + fit RooFit.
Les tests de fit tournent avec fit.python (ou $FIT_PYTHON), skip si pas de ROOT.
"""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import uproot
import yaml

import prepare_fit_input as pfi

ROOT_DIR = Path(__file__).resolve().parent.parent
FIT_SCRIPT = ROOT_DIR / "scripts" / "fit_mass.py"
KINDS = ["sig=signal", "bkg=background"]


def _root_python():
    cfg = yaml.safe_load((ROOT_DIR / "config" / "config.yaml").read_text())
    exe = os.environ.get("FIT_PYTHON") or cfg.get("fit", {}).get("python", "python3")
    exe = shutil.which(exe)
    if exe is None:
        return None
    ok = subprocess.run([exe, "-c", "import ROOT"], capture_output=True).returncode == 0
    return exe if ok else None


ROOT_PYTHON = _root_python()
needs_root = pytest.mark.skipif(ROOT_PYTHON is None, reason="PyROOT indisponible (fit.python)")


def write_npz(path, sig_counts, bkg_counts, edges):
    np.savez(path, edges=edges, variable=np.array("mass"),
             sig__counts=sig_counts, sig__sumw2=sig_counts.copy(),
             bkg__counts=bkg_counts, bkg__sumw2=bkg_counts.copy())


def expected_spectrum(n_sig=3000.0, n_bkg=20000.0):
    # attendu exact par bin (pas de fluctuation) : gauss(125, 5) + expo(50) a partir de 60
    edges = np.linspace(60.0, 200.0, 36)
    lo, hi = edges[:-1], edges[1:]
    gauss_cdf = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)((x - 125.0) / (5.0 * math.sqrt(2.0))))  # noqa: E731
    expo_cdf = lambda x: 1.0 - np.exp(-(x - 60.0) / 50.0)  # noqa: E731
    sig = n_sig * (gauss_cdf(hi) - gauss_cdf(lo))
    bkg = n_bkg * (expo_cdf(hi) - expo_cdf(lo))
    return sig, bkg, edges


# export, pas besoin de ROOT

def test_th1d_roundtrip(tmp_path):
    counts, sumw2, edges = np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 0.5]), np.array([0.0, 1.0, 2.0, 3.0])
    path = tmp_path / "h.root"
    with uproot.recreate(path) as f:
        f["h"] = pfi.make_th1d("h", "titre", counts, sumw2, edges, "mass")
    h = uproot.open(path)["h"]
    np.testing.assert_allclose(h.values(), counts)
    np.testing.assert_allclose(h.variances(), sumw2)
    np.testing.assert_allclose(h.axis().edges(), edges)
    assert h.title == "titre" and h.axis().member("fTitle") == "mass"


def test_build_fit_data_modes():
    hists = {"sig": (np.array([1.0, 2.0]), np.array([0.5, 1.0])),
             "bkg": (np.array([10.0, 5.0]), np.array([2.0, 2.0])),
             "obs": (np.array([12.0, 6.0]), np.array([12.0, 6.0]))}
    kinds = {"sig": "signal", "bkg": "background", "obs": "data"}

    counts, sumw2 = pfi.build_fit_data(hists, kinds, "mc_sum", 2)
    np.testing.assert_allclose(counts, [11.0, 7.0])
    np.testing.assert_allclose(sumw2, [2.5, 3.0])

    counts, sumw2 = pfi.build_fit_data(hists, kinds, "data", 2)
    np.testing.assert_allclose(counts, [12.0, 6.0])

    toy, toy_w2 = pfi.build_fit_data(hists, kinds, "pseudo_data", 2, seed=1)
    assert np.all(toy == np.round(toy)) and np.array_equal(toy, toy_w2)
    again, _ = pfi.build_fit_data(hists, kinds, "pseudo_data", 2, seed=1)
    np.testing.assert_array_equal(toy, again)


def test_dataset_data_requires_data_sample():
    hists = {"sig": (np.ones(2), np.ones(2))}
    with pytest.raises(SystemExit, match="kind 'data'"):
        pfi.build_fit_data(hists, {"sig": "signal"}, "data", 2)


def test_prepare_writes_expected_histograms(tmp_path):
    sig, bkg, edges = expected_spectrum(n_sig=100, n_bkg=500)
    npz, out = tmp_path / "h.npz", tmp_path / "fit_input.root"
    write_npz(npz, sig, bkg, edges)
    pfi.main(["--hists", str(npz), "--output", str(out), "--kinds", *KINDS])
    f = uproot.open(out)
    assert {k.split(";")[0] for k in f.keys()} == {"h_sig", "h_bkg", "mc_signal", "mc_background", "fit_data"}
    assert f["fit_data"].values().sum() == pytest.approx(sig.sum() + bkg.sum())
    assert f["mc_signal"].values().sum() == pytest.approx(sig.sum())


# fit (PyROOT)

def run_fit(tmp_path, sig, bkg, edges, *extra):
    npz, inp = tmp_path / "h.npz", tmp_path / "fit_input.root"
    write_npz(npz, sig, bkg, edges)
    pfi.main(["--hists", str(npz), "--output", str(inp), "--kinds", *KINDS, *extra])
    out = {k: tmp_path / f"fit.{k}" for k in ("json", "png", "root")}
    proc = subprocess.run(
        [ROOT_PYTHON, str(FIT_SCRIPT), "--input", str(inp), "--output-json", str(out["json"]),
         "--output-png", str(out["png"]), "--output-workspace", str(out["root"])],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out["json"].read_text()), out


@needs_root
def test_fit_is_unbiased_on_expected_spectrum(tmp_path):
    # non-regression : sans x.setBins(nbins) la masse sortait biaisee de ~0.9 GeV
    sig, bkg, edges = expected_spectrum()
    res, _ = run_fit(tmp_path, sig, bkg, edges, "--dataset", "mc_sum")
    assert res["status"] == "ok"
    p = res["parameters"]
    assert p["mean"]["value"] == pytest.approx(125.0, abs=0.02)
    assert p["sigma"]["value"] == pytest.approx(5.0, abs=0.02)
    assert p["nsig"]["value"] == pytest.approx(sig.sum(), rel=0.005)
    assert p["tau"]["value"] == pytest.approx(-1.0 / 50.0, rel=0.005)


@needs_root
def test_fit_recovers_injected_signal(tmp_path):
    sig, bkg, edges = expected_spectrum()
    res, out = run_fit(tmp_path, sig, bkg, edges, "--dataset", "pseudo_data", "--seed", "3")
    assert res["status"] == "ok"
    p = res["parameters"]
    assert abs(p["mean"]["value"] - 125.0) < 3 * p["mean"]["error"]
    assert abs(p["sigma"]["value"] - 5.0) < 3 * p["sigma"]["error"]
    assert abs(p["nsig"]["value"] - sig.sum()) < 3 * p["nsig"]["error"]
    assert res["truth"]["nsig"] == pytest.approx(sig.sum())
    assert 0.3 < res["chi2_ndf"] < 2.0
    assert out["png"].stat().st_size > 0 and out["root"].stat().st_size > 0


@needs_root
def test_fit_empty_histogram(tmp_path):
    edges = np.linspace(60.0, 200.0, 36)
    zeros = np.zeros(35)
    res, out = run_fit(tmp_path, zeros, zeros, edges)
    assert res["status"] == "no_data" and "parameters" not in res
    assert out["png"].stat().st_size > 0
