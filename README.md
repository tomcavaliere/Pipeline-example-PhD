# pipeline-analyse-ATLAS

Pipeline Snakemake : selection d'evenements, histos et fit RooFit d'un pic de masse.
Tourne sur des donnees generees (mock).

## Install

```bash
git clone <url> && cd pipeline-analyse-ATLAS
mamba env create -f environment.yml && mamba activate pipeline-analyse
# ou : python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Le fit a besoin de ROOT (mac : `brew install root`). Mettre dans `config/config.yaml`
le python qui a PyROOT :

```yaml
fit:
  python: python3.15   # verifier avec : python3.15 -c "import ROOT"
```

## Lancer

```bash
snakemake -n -p        # dry-run
snakemake --cores 2    # donnees mock
pytest                 # tests
```

## Sorties

- `results/summary_mass.png`, `results/summary.csv` : distributions et tableau par echantillon
- `results/fit_mass.png`, `results/fit_mass.json` : fit et parametres
- `logs/` en cas de probleme

Coupures, binning, echantillons et options du fit : voir les commentaires dans `config/config.yaml`.
Apres un changement de config : `snakemake --cores 2 --forceall`.
