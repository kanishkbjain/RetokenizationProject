# Retokenization Invariance in Language Models

This repository contains code and figure notebooks for studying how language model
behavior changes when the same text is presented with different tokenizations or with
small character-level perturbations. This can be reproduced to generate figures from [this preprint](https://arxiv.org/abs/2606.15521). 

## Repository layout

- `eval/`: result loading, pass-curve computation, benchmark utilities, and shared helpers
- `experiments/passat/`: experiment scripts for retokenization and typo variants
- `figure_notebooks/`: notebooks for regenerating figures
- `data/`: place for raw experiment outputs
- `outputs/figures/`: generated figures

## Data

The raw result files are not tracked in Git. Put them under:

- `data/raw/Tokenizer_passK/`
- `data/raw/TokenizationProject/`

The notebooks and loaders use those paths by default. Generated figures go in
`outputs/figures/`.

## Setup

Install the Python dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Running the experiment scripts may require additional model/runtime setup depending
on the model and hardware.

## Reproducing figures

1. Add the raw result trees under `data/raw/`.
2. Open the notebooks in `figure_notebooks/`.
3. Run the notebook for the figure you want.

See [figure_notebooks/README.md](figure_notebooks/README.md) for the notebook-to-figure
mapping.

## Notes

The repository tracks source code, notebooks, small benchmark fixtures, and lightweight
figure data. Raw model outputs, generated figures, paper drafts, and local working
files are kept out of Git.
