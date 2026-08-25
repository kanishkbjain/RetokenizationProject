# Figure Notebooks

The notebooks in this directory regenerate the main figures from the saved experiment
outputs.

Notebook map:

- `figure_single_model_core_figures.ipynb`: `2_Figure1`-style pass@ figures, failure-distribution figures, `compute_l_k_over_k`, `olmo2_failure_tail_mass`
- `figure_humaneval_allmodels.ipynb`: `4_humaneval_allmodels`
- `figure_passat_retok_per_p_humaneval.ipynb`: `passat_retok_per_p_humaneval`
- `figure_humaneval_olmo2_alpha_retok.ipynb`: HumanEval OLMo-2 retokenized prompt length scale `alpha`
- `figure_passk_mmlu_random.ipynb`: `passk_mmlu_random`
- `figure_olmo2_humaneval_scaling_and_prepost.ipynb`: `olmo2_combined_passat`, OLMo-2 scaling figures, `pre_post_humaneval_pass_curves_and_mean_failure_probabilities`
- `figure_olmo2_figure4_combined.ipynb`: `olmo2_figure4_combined`

Raw result files go under `../data/raw/`. The notebooks write figures to
`../outputs/figures/` when run from this directory.
