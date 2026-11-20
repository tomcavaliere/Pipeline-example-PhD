"""tests selection / histos / tableau"""
import json

import numpy as np
import pandas as pd
import pytest

import generate_mock_root as gen
import make_histograms as mh
import plot_summary as ps
import select_events as se


@pytest.fixture
def toy_events():
    # 5 evts faits a la main
    return {
        "pt": np.array([10.0, 30.0, 50.0, 40.0, 100.0], dtype=np.float32),
        "eta": np.array([0.0, 1.0, 3.0, -2.0, 0.5], dtype=np.float32),
        "mass": np.array([125.0, 90.0, 120.0, 300.0, 130.0], dtype=np.float32),
        "weight": np.array([1.0, 2.0, 1.0, 1.0, -0.5]),
    }


CUTS = se.Cuts(pt_min=25.0, abs_eta_max=2.5, mass_min=60.0, mass_max=200.0)


# selection

def test_selection_keeps_expected_events(toy_events):
    df, _ = se.select(toy_events, CUTS, sample="toy")
    # 0: pt, 2: eta, 3: masse
    assert df["mass"].tolist() == [90.0, 130.0]
    assert (df["sample"] == "toy").all()
    assert list(df.columns) == se.OUTPUT_COLUMNS


def test_cutflow_is_monotonic_and_weighted(toy_events):
    _, cutflow = se.select(toy_events, CUTS)
    n_raw = [step["n_raw"] for step in cutflow]
    assert n_raw == [5, 4, 3, 3, 2]
    assert all(a >= b for a, b in zip(n_raw, n_raw[1:]))
    assert cutflow[-1]["sum_weights"] == pytest.approx(2.0 - 0.5)


def test_scale_is_applied_to_weights(toy_events):
    df, cutflow = se.select(toy_events, CUTS, scale=2.0)
    assert df["weight"].tolist() == pytest.approx([4.0, -1.0])
    assert cutflow[0]["sum_weights"] == pytest.approx(2.0 * 4.5)


def test_disabled_cuts_keep_everything(toy_events):
    df, cutflow = se.select(toy_events, se.Cuts())
    assert len(df) == 5
    assert len(cutflow) == 1


def test_zero_events_pass_selection(toy_events):
    df, cutflow = se.select(toy_events, se.Cuts(pt_min=1e6), sample="toy")
    assert len(df) == 0
    assert list(df.columns) == se.OUTPUT_COLUMNS
    assert cutflow[-1] == {"cut": "pt > 1e+06", "n_raw": 0, "sum_weights": 0.0}


def test_empty_parquet_keeps_sample_name(toy_events, tmp_path):
    # le nom doit survivre a un parquet vide (attrs)
    df, _ = se.select(toy_events, se.Cuts(pt_min=1e6), sample="toy")
    path = tmp_path / "other_name.parquet"
    df.to_parquet(path, index=False)
    assert mh.sample_name(path, pd.read_parquet(path)) == "toy"


def test_empty_input_file(tmp_path):
    root_path = tmp_path / "empty.root"
    gen.write_root(root_path, "events", gen.generate_events(0, "peak", np.random.default_rng(0)))
    events = se.read_events(root_path, "events")
    df, cutflow = se.select(events, CUTS, sample="empty")
    assert len(df) == 0 and cutflow[0]["n_raw"] == 0

    counts, sumw2, edges, metrics = mh.summarize_sample(df, "mass", 10, (60.0, 200.0))
    assert counts.sum() == 0 and sumw2.sum() == 0 and len(edges) == 11
    assert metrics["n_events"] == 0
    assert metrics["mass_mean"] is None and metrics["mass_std"] is None


# lecture ROOT

def test_root_roundtrip(tmp_path):
    branches = gen.generate_events(1000, "peak", np.random.default_rng(1), mean=125, width=5)
    path = tmp_path / "sig.root"
    gen.write_root(path, "events", branches)
    events = se.read_events(path, "events")
    assert len(events) == 1000
    np.testing.assert_allclose(np.asarray(events["mass"]), branches["mass"])
    assert abs(float(np.mean(events["mass"])) - 125) < 1.0


def test_missing_branch_raises(tmp_path):
    path = tmp_path / "bad.root"
    gen.write_root(path, "events", {"pt": np.ones(3)})
    with pytest.raises(KeyError, match="Branches manquantes"):
        se.read_events(path, "events")


def test_generator_is_reproducible():
    a = gen.generate_events(100, "falling", np.random.default_rng(7))
    b = gen.generate_events(100, "falling", np.random.default_rng(7))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


# histos

def test_weighted_histogram_counts_and_errors():
    counts, sumw2, edges = mh.weighted_histogram([1.5, 1.5, 3.5, 10.0], [1.0, 2.0, 0.5, 9.0], 4, (0.0, 4.0))
    np.testing.assert_allclose(counts, [0.0, 3.0, 0.0, 0.5])  # 10 -> overflow
    np.testing.assert_allclose(sumw2, [0.0, 5.0, 0.0, 0.25])
    np.testing.assert_allclose(edges, [0, 1, 2, 3, 4])


def test_weighted_stats():
    stats = mh.weighted_stats([1.0, 3.0], [1.0, 3.0])
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["std"] == pytest.approx(np.sqrt(0.75))
    assert mh.weighted_stats([], []) == {"mean": None, "std": None}
    # sumw = 0
    assert mh.weighted_stats([1.0, 2.0], [1.0, -1.0]) == {"mean": None, "std": None}


def test_summarize_sample_under_overflow():
    df = pd.DataFrame({"mass": [50.0, 100.0, 250.0], "weight": [1.0, 1.0, 1.0]})
    _, _, _, m = mh.summarize_sample(df, "mass", 10, (60.0, 200.0))
    assert (m["n_underflow"], m["n_overflow"]) == (1, 1)
    assert m["sum_weights"] == 3.0 and m["sum_weights_in_range"] == 1.0


# summary

def test_summary_table_efficiency_handles_zero_input():
    metrics = {
        "variable": "mass",
        "samples": {
            "a": {"n_events": 5, "sum_weights": 5.0, "sum_weights_in_range": 5.0, "mass_mean": 100.0, "mass_std": 1.0},
            "b": {"n_events": 0, "sum_weights": 0.0, "sum_weights_in_range": 0.0, "mass_mean": None, "mass_std": None},
        },
    }
    cutflows = [
        {"sample": "a", "cutflow": [{"cut": "all", "n_raw": 10, "sum_weights": 10.0}]},
        {"sample": "b", "cutflow": [{"cut": "all", "n_raw": 0, "sum_weights": 0.0}]},
    ]
    table = ps.build_summary_table(metrics, cutflows, {"a": "signal"}).set_index("sample")
    assert table.loc["a", "efficiency"] == pytest.approx(0.5)
    assert pd.isna(table.loc["b", "efficiency"])
    assert table.loc["b", "kind"] == "unknown"


def test_parse_mapping():
    assert ps.parse_mapping(["a=x", "b=Z' (m = 1 TeV)"]) == {"a": "x", "b": "Z' (m = 1 TeV)"}
    with pytest.raises(ValueError):
        ps.parse_mapping(["no_equal_sign"])


def test_end_to_end_scripts(tmp_path):
    # gen -> select -> hist -> plot, comme snakemake
    raw, sel = tmp_path / "raw.root", tmp_path / "sel.parquet"
    cf, hists, met = tmp_path / "cf.json", tmp_path / "h.npz", tmp_path / "m.json"
    png, csv = tmp_path / "out.png", tmp_path / "out.csv"

    gen.main(["--output", str(raw), "--n-events", "500", "--shape", "peak", "--seed", "3"])
    se.main(["--input", str(raw), "--output", str(sel), "--cutflow", str(cf), "--sample", "sig",
             "--pt-min", "25", "--mass-min", "60", "--mass-max", "200"])
    mh.main(["--inputs", str(sel), "--output-hists", str(hists), "--output-metrics", str(met)])
    ps.main(["--hists", str(hists), "--metrics", str(met), "--cutflows", str(cf),
             "--output-png", str(png), "--output-csv", str(csv), "--kinds", "sig=signal"])

    assert png.stat().st_size > 0
    table = pd.read_csv(csv)
    assert table.loc[0, "sample"] == "sig" and table.loc[0, "n_generated"] == 500
    assert json.loads(cf.read_text())["cutflow"][-1]["n_raw"] == table.loc[0, "n_selected"]
