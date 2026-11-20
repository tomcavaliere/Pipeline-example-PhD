# Pipeline : raw .root -> selection -> histos -> plot + fit RooFit
#   snakemake -n -p         dry-run
#   snakemake --cores 2

import shlex

configfile: "config/config.yaml"

SAMPLES = list(config["samples"])
SEL = config["selection"]
HIST = config["histogram"]
FIT = config["fit"]

wildcard_constraints:
    sample="|".join(SAMPLES),


rule all:
    input:
        "results/summary_mass.png",
        "results/summary.csv",
        "results/fit_mass.json",
        "results/fit_mass.png",


## --- donnees mock ---
if config.get("use_mock_data", False):

    def _mock(wc, key, default):
        return config["samples"][wc.sample]["mock"].get(key, default)

    rule generate_mock:
        output:
            "data/raw/{sample}.root",
        params:
            tree=config["tree_name"],
            n_events=lambda wc: _mock(wc, "n_events", 10000),
            shape=lambda wc: _mock(wc, "shape", "falling"),
            mean=lambda wc: _mock(wc, "mean", 125.0),
            width=lambda wc: _mock(wc, "width", 5.0),
            slope=lambda wc: _mock(wc, "slope", 50.0),
            seed=lambda wc: config["seed"] + SAMPLES.index(wc.sample),
        log:
            "logs/generate_mock/{sample}.log",
        shell:
            "python scripts/generate_mock_root.py"
            " --output {output}"
            " --tree {params.tree}"
            " --n-events {params.n_events}"
            " --shape {params.shape}"
            " --mean {params.mean}"
            " --width {params.width}"
            " --slope {params.slope}"
            " --seed {params.seed}"
            " > {log} 2>&1"


## --- analyse ---
rule select_events:
    input:
        "data/raw/{sample}.root",
    output:
        events="work/selected/{sample}.parquet",
        cutflow="work/selected/{sample}.cutflow.json",
    params:
        tree=config["tree_name"],
        scale=lambda wc: config["samples"][wc.sample].get("scale", 1.0),
    log:
        "logs/select_events/{sample}.log",
    shell:
        "python scripts/select_events.py"
        " --input {input}"
        " --output {output.events}"
        " --cutflow {output.cutflow}"
        " --tree {params.tree}"
        " --sample {wildcards.sample}"
        " --scale {params.scale}"
        f" --pt-min {SEL['pt_min']}"
        f" --abs-eta-max {SEL['abs_eta_max']}"
        f" --mass-min {SEL['mass_min']}"
        f" --mass-max {SEL['mass_max']}"
        " > {log} 2>&1"


rule make_histograms:
    input:
        events=expand("work/selected/{sample}.parquet", sample=SAMPLES),
    output:
        hists="work/histograms.npz",
        metrics="work/metrics.json",
    log:
        "logs/make_histograms.log",
    shell:
        "python scripts/make_histograms.py"
        " --inputs {input.events}"
        " --output-hists {output.hists}"
        " --output-metrics {output.metrics}"
        f" --variable {HIST['variable']}"
        f" --bins {HIST['bins']}"
        f" --range {HIST['range'][0]} {HIST['range'][1]}"
        " > {log} 2>&1"


def _kinds_args():
    kinds = " ".join(f"{s}={c['kind']}" for s, c in config["samples"].items())
    # quote a cause des labels genre "Z' (m = 125 GeV)"
    labels = " ".join(
        shlex.quote(f"{s}={c.get('label', s)}") for s, c in config["samples"].items()
    )
    return f"--kinds {kinds} --labels {labels}"


rule plot_summary:
    input:
        hists="work/histograms.npz",
        metrics="work/metrics.json",
        cutflows=expand("work/selected/{sample}.cutflow.json", sample=SAMPLES),
    output:
        png="results/summary_mass.png",
        csv="results/summary.csv",
    params:
        extra=_kinds_args(),
    log:
        "logs/plot_summary.log",
    shell:
        "python scripts/plot_summary.py"
        " --hists {input.hists}"
        " --metrics {input.metrics}"
        " --cutflows {input.cutflows}"
        " --output-png {output.png}"
        " --output-csv {output.csv}"
        " {params.extra}"
        " > {log} 2>&1"


## --- fit ---
# fit_mass tourne avec le python de config["fit"] (PyROOT), pas celui du pipeline
def _kinds_only():
    return " ".join(f"{s}={c['kind']}" for s, c in config["samples"].items())


rule prepare_fit_input:
    input:
        hists="work/histograms.npz",
    output:
        "work/fit_input.root",
    params:
        dataset=FIT["dataset"],
        kinds=_kinds_only(),
        seed=config["seed"],
    log:
        "logs/prepare_fit_input.log",
    shell:
        "python scripts/prepare_fit_input.py"
        " --hists {input.hists}"
        " --output {output}"
        " --dataset {params.dataset}"
        " --kinds {params.kinds}"
        " --seed {params.seed}"
        " > {log} 2>&1"


rule fit_mass:
    input:
        "work/fit_input.root",
    output:
        json="results/fit_mass.json",
        png="results/fit_mass.png",
        workspace="work/fit_workspace.root",
    params:
        python=FIT["python"],
        background=FIT["background"],
        bkg_order=FIT.get("bkg_order", 3),
        sig=FIT["signal"],
    log:
        "logs/fit_mass.log",
    shell:
        "{params.python} scripts/fit_mass.py"
        " --input {input}"
        " --output-json {output.json}"
        " --output-png {output.png}"
        " --output-workspace {output.workspace}"
        " --background {params.background}"
        " --bkg-order {params.bkg_order}"
        " --mean-init {params.sig[mean]}"
        " --mean-range {params.sig[mean_range][0]} {params.sig[mean_range][1]}"
        " --sigma-init {params.sig[sigma]}"
        " --sigma-range {params.sig[sigma_range][0]} {params.sig[sigma_range][1]}"
        " > {log} 2>&1"
