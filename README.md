# Nature-Inspired Prompt Evolution Framework

Evolutionary prompt optimisation framework for reducing reasoning blind spots in LLM-generated advice. Accompanies the IEEE conference paper *"Nature-Inspired Prompt Evolution for Improving Reasoning Completeness in Large Language Models"*.

---

## Project Structure

```
project/
├── data/
│   ├── scenarios.json              # 20 advice-generation benchmark scenarios
│   └── prompts/
│       └── seed_prompts.json       # 8 manually designed seed chromosomes
├── src/
│   ├── evolution.py                # PromptChromosome + population utilities
│   ├── fitness.py                  # Blind-Spot Awareness Score (BAS) evaluator
│   ├── llm_interface.py            # OpenRouter API wrapper
│   ├── mutation.py                 # 7 mutation operators
│   ├── crossover.py                # Single-point and uniform crossover
│   ├── selection.py                # Tournament selection + elitism
│   ├── evaluation.py               # Result tracking and CSV export
│   └── visualisation.py           # Publication-quality figures (PNG + PDF)
├── results/                        # Auto-created; one sub-folder per run
├── run_experiment.py               # Main entry point
└── requirements.txt
```

---

## Installation

**Python 3.10+** is required.

```bash
# 1. Clone or copy the project folder
cd project

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## API Key Setup

This framework uses the [OpenRouter](https://openrouter.ai) API to access LLMs.

```bash
# Option A — environment variable (recommended)
export OPENROUTER_API_KEY="sk-or-..."

# Option B — pass directly on the command line
python run_experiment.py --api-key "sk-or-..."
```

Get a free key at <https://openrouter.ai/keys>.

---

## Running an Experiment

### Minimal run (debug / smoke test)

```bash
python run_experiment.py \
    --population-size 6 \
    --generations 3 \
    --scenarios-subset 5 \
    --seed 42
```

Completes in ~2–5 minutes depending on API latency. Good for verifying the pipeline end-to-end.

### Standard preliminary experiment (as reported in the paper)

```bash
python run_experiment.py \
    --population-size 10 \
    --generations 10 \
    --mutation-rate 0.3 \
    --crossover-rate 0.7 \
    --tournament-size 3 \
    --n-elite 1 \
    --seed 42 \
    --advice-model gpt4o-mini \
    --judge-model gpt4o-mini
```

### Larger exploratory run

```bash
python run_experiment.py \
    --population-size 20 \
    --generations 20 \
    --mutation-rate 0.35 \
    --crossover-rate 0.7 \
    --seed 0 \
    --advice-model claude-haiku \
    --judge-model gpt4o-mini
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--population-size` | 10 | Number of individuals per generation |
| `--generations` | 10 | Number of evolutionary generations |
| `--mutation-rate` | 0.3 | Per-individual mutation probability |
| `--crossover-rate` | 0.7 | Crossover probability per pair |
| `--tournament-size` | 3 | Competitors per tournament selection round |
| `--n-elite` | 1 | Elite individuals carried over unchanged |
| `--seed` | 42 | Global random seed |
| `--advice-model` | `gpt4o-mini` | Model alias for advice generation |
| `--judge-model` | `gpt4o-mini` | Model alias for fitness evaluation |
| `--temperature` | 0.7 | Sampling temperature for advice LLM |
| `--scenarios-subset` | None | Use only first N scenarios (None = all 20) |
| `--results-dir` | auto | Output directory (default: `results/run_<timestamp>_seed<N>`) |
| `--api-key` | env var | OpenRouter API key |

### Supported model aliases

| Alias | OpenRouter model string |
|---|---|
| `gpt4o-mini` | `openai/gpt-4o-mini` |
| `gpt4o` | `openai/gpt-4o` |
| `claude-haiku` | `anthropic/claude-haiku-4-5` |
| `claude-sonnet` | `anthropic/claude-sonnet-4-5` |
| `llama3-8b` | `meta-llama/llama-3-8b-instruct` |
| `llama3-70b` | `meta-llama/llama-3-70b-instruct` |
| `mistral-7b` | `mistralai/mistral-7b-instruct` |
| `gemini-flash` | `google/gemini-flash-1.5` |

---

## Output Files

Each run creates a timestamped folder under `results/`:

```
results/run_20250101_120000_seed42/
├── config.json              # Full experiment configuration
├── generation_stats.csv     # Per-generation: best/avg/worst fitness + diversity
├── score_records.csv        # Per-(chromosome, scenario) BAS dimension scores
├── best_prompt.json         # Best evolved chromosome + assembled prompt string
└── figures/
    ├── fitness_best.png / .pdf     # Best BAS vs Generation
    ├── fitness_avg.png  / .pdf     # Avg BAS vs Generation
    ├── diversity.png    / .pdf     # Population Diversity vs Generation
    └── fitness_combined.png / .pdf # Combined figure (IEEE two-column ready)
```

All PNG figures are saved at **300 dpi**. PDF figures are vector and can be included directly in LaTeX via `\includegraphics`.

---

## Prompt Chromosome Structure

Each individual in the population is a chromosome with five semantic genes:

| Gene | Description | Example |
|---|---|---|
| `role` | Advisor persona | `"You are a cautious and analytical advisor."` |
| `reasoning` | Systematic analysis instruction | `"Break the problem into its root causes."` |
| `risk_awareness` | Risk identification instruction | `"List potential negative consequences first."` |
| `reflection` | Deliberate reasoning instruction | `"Reconsider your initial answer before finalising."` |
| `output_format` | Response structure instruction | `"Structure your reply as: Analysis, Risks, Recommendation."` |

The assembled prompt is used verbatim as the LLM system message.

---

## Fitness Function: Blind-Spot Awareness Score (BAS)

An LLM judge evaluates each piece of generated advice on four dimensions (1–5 scale):

| Dimension | What it measures |
|---|---|
| Risk Identification (RI) | Coverage of risks and warning signs |
| Alternative Perspectives (AP) | Exploration of multiple interpretations or options |
| Context Awareness (CA) | Integration of situational and contextual factors |
| Reasoning Completeness (RC) | Thoroughness and nuance of the reasoning process |

$$\text{BAS} = \frac{\text{RI} + \text{AP} + \text{CA} + \text{RC}}{4}$$

The overall fitness of a prompt is the mean BAS across all evaluation scenarios.

---

## Reproducing Reported Results

All experiments reported in the paper use `--seed 42`. The exact configuration is saved to `config.json` in every results folder, enabling precise reproduction.

```bash
# Reproduce paper results
python run_experiment.py \
    --population-size 10 \
    --generations 10 \
    --mutation-rate 0.3 \
    --crossover-rate 0.7 \
    --tournament-size 3 \
    --n-elite 1 \
    --seed 42 \
    --advice-model gpt4o-mini \
    --judge-model gpt4o-mini
```

---

## Extending the Framework

### Add a new mutation operator

1. Open `src/mutation.py`.
2. Define a function with signature `(chromosome: PromptChromosome, rng: random.Random) -> PromptChromosome`.
3. Append it to `MUTATION_OPERATORS`.

### Add new scenarios

Add entries to `data/scenarios.json` following the existing schema (`id`, `category`, `title`, `scenario`).

### Swap the crossover strategy

Replace `single_point_crossover` in `run_experiment.py` with `uniform_crossover` from `src/crossover.py`.

---

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{torres2025promptevolution,
  title     = {Nature-Inspired Prompt Evolution for Improving Reasoning Completeness
               in Large Language Models},
  author    = {Torres, Nicolas},
  booktitle = {Proceedings of the IEEE ...},
  year      = {2025},
}
```

---

## License

MIT License. See `LICENSE` for details.
