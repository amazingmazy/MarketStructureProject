# FINM 33100 - Market Structure Project

## What it is

This project studies whether common portfolio-construction methods diversify across **independent sources of risk**, rather than simply counting the number of stocks held.

The final analysis uses daily U.S. stock returns, rolling principal-component analysis (PCA), covariance estimation, and portfolio-level variance decompositions. It compares:

- Equal weight
- Market-cap weight
- Minimum variance
- Equal-risk contribution
- Maximum diversification

The main notebook is `project_notebook.ipynb`. The other notebooks are experimental drafts.

## Setup and run

The project requires Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/).

### 1. Open the project

Clone the repository and change into the project directory:

```bash
git clone https://github.com/amazingmazy/MarketStructureProject.git
cd MarketStructureProject
```

If the repository is already open in VS Code, make sure the commands below are run from the `MarketStructureProject/` directory.

### 2. Install dependencies

Run:

```bash
uv sync
```

This creates the project-local `.venv` and installs the dependencies recorded in `uv.lock`.

### 3. Download the data files

Download the returns-data file from [UChicago Box](https://uchicago.box.com/s/5akprzhsyyt3oj2v39al7rsqz939ff3h) and save it as:

```text
data/olhcv_merged.parquet
```

Download the portfolio-weight cache from [UChicago Box](https://uchicago.box.com/s/uy1o43fln8k7o32s2skxnwhjolji6mnw) and save it as:

```text
data/portfolios.pkl
```

The repository already includes `data/ccm_linktable.csv`.

### 4. Run the notebook

Start Jupyter from the project root:

```bash
uv run jupyter lab
```

Open `project_notebook.ipynb` and run all cells.

In VS Code, select the project interpreter at:

```text
MarketStructureProject/.venv/bin/python
```

Alternatively, register and select a Jupyter kernel with:

```bash
uv run python -m ipykernel install --user \
  --name uchicago-summer-course \
  --display-name "FINM 33100 — Market Structure (uv)"
```

The notebook must run with `MarketStructureProject/` as its working directory so that it can find `data/` and `functions/`.

## Caching

`data/portfolios.pkl` contains precomputed portfolio weights. When it is present, the notebook skips the approximately 18-minute portfolio-construction step and proceeds directly to the analysis.

To rebuild the portfolio weights, temporarily move `data/portfolios.pkl` out of the data directory and run the notebook again:

```bash
mv data/portfolios.pkl /tmp/portfolios.pkl.backup
```

Restore the cache afterward if desired:

```bash
mv /tmp/portfolios.pkl.backup data/portfolios.pkl
```

## Project structure

- `project_notebook.ipynb` — final analysis
- `functions/` — portfolio, PCA, diversification, and period-analysis helpers
- `data/` — input data and local caches
- `refs/` — reference notebooks and materials
