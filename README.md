# IRAC Legal-Reasoning Evaluation of LLMs

This repository contains the code, prompts, data, and results for a study that
evaluates how well large language models (LLMs) perform **IRAC-style legal
reasoning** (Issue → Rule → Application → Conclusion) over international treaty
texts (BBNJ, CBD, Nagoya Protocol). Given a fact pattern and a set of treaty
provisions, each model is asked to construct **two opposing but defensible IRAC
analyses** (conclusions `C1` and `C2`), and its selected rules and supporting
argument edges are scored against a human-authored ground truth.

## Approach

The pipeline (in [IRAC_notebook2.ipynb](IRAC_notebook2.ipynb)) runs as follows:

1. **Input** — for each question `Q1`–`Q5`: a fact pattern, two contradictory
   conclusions (`C1`/`C2`), and a "superset" of candidate treaty rows (each with
   its exact treaty text and Hohfeldian mapping: actor-holder / action /
   actor-affected).
2. **Generation** — a strict system prompt instructs the model to produce two
   IRAC analyses using *only* the supplied facts and treaty text (no outside
   knowledge), returned as JSON. Calls are made through
   [OpenRouter](https://openrouter.ai/) (OpenAI-compatible client).
3. **Scoring** — the model's `rules_selected` and `edges_support` are compared
   to the ground truth using **precision / recall / F1** (via `nltk`).
4. **Comparison** — metrics are aggregated across models and visualized to
   compare performance, including a "with vs without reasoning" view.

Models evaluated: `anthropic/claude-opus-4.6`, `openai/gpt-5.4`,
`google/gemini-3.1-pro-preview`, `google/gemini-2.5-pro`,
`moonshotai/kimi-k2.5`.

## Repository layout

```
irac_paper/
├── IRAC_notebook2.ipynb        # Main pipeline: generate → evaluate → visualize
├── data/                       # Inputs
│   ├── Questions_superset_rows - Q1.csv … Q5.csv   # Candidate treaty rows per question
│   └── ground_truth_irac.json  # Gold rules/edges (git-ignored — see note below)
├── prompts/
│   ├── system_prompt_cot_updated.txt   # Active system prompt
│   └── archive/                # Earlier prompt versions (not used by the notebook)
├── irac_outputs/               # Model predictions & charts
│   ├── pred_Q*.json
│   ├── model_comparison_chart.html
│   ├── with_reasoning/  without_reasoning/  updated_cot/
├── results/                    # Aggregated metrics
│   ├── metrics_all_LLMs.csv    # All models, all questions
│   ├── metrics.csv
│   └── moonshot_final_results.csv
├── archive/notebooks/          # Legacy / scratch notebooks (paths not maintained)
├── docs/                       # Paper artifacts
├── requirements.txt
├── .env.example
└── LICENSE
```

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/rnanda17/irac_paper.git
cd irac_paper

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Provide your OpenRouter API key
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=...
```

The first time you use `nltk`, its metrics functions work out of the box; no
extra corpora download is required for this pipeline.

## Usage

Open [IRAC_notebook2.ipynb](IRAC_notebook2.ipynb) and run the cells top to
bottom:

- **Cells 0–6** load the system prompt, ground truth, question CSVs, and define
  the prompt builder, LLM call, and evaluation functions.
- **Cell 7** runs every model over `Q1`–`Q5`, writing predictions to
  `irac_outputs/updated_cot/pred_<model>_<Q>.json` and metrics to
  `irac_outputs/updated_cot/metrics.csv`.
- **Cell 8** flattens the JSON predictions into a single CSV.
- **Cell 9** renders the F1-score comparison chart
  (`irac_outputs/model_comparison_chart.html`).
- **Final cells** load `results/metrics_all_LLMs.csv` to report macro-F1 per
  model and the best-performing model.

## Data note

`data/ground_truth_irac.json` is intentionally **git-ignored** and held
privately, so it is not distributed with this repository. The pipeline requires
it to compute metrics; place your own copy at that path to run the evaluation
end to end.

## License

Released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**
license. See [LICENSE](LICENSE).
