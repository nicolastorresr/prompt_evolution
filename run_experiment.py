"""
run_experiment.py
-----------------
Entry point for the evolutionary prompt optimisation experiment.

Usage
-----
    python run_experiment.py [options]

Examples
--------
    # Run with default settings
    python run_experiment.py

    # Custom configuration
    python run_experiment.py \\
        --population-size 20 \\
        --generations 15 \\
        --mutation-rate 0.4 \\
        --crossover-rate 0.7 \\
        --seed 42 \\
        --advice-model gpt4o \\
        --judge-model gpt4o \\
        --scenarios-subset 10

    # Use a specific subset of scenarios (faster for debugging)
    python run_experiment.py --scenarios-subset 5 --generations 3
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on the Python path when run from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from evolution import (
    PromptChromosome,
    load_seed_prompts,
    initialise_population,
    population_diversity,
)
from fitness import FitnessEvaluator, load_scenarios
from evaluation import ExperimentTracker
from llm_interface import LLMInterface, OPENROUTER_MODELS
from mutation import apply_mutation
from crossover import single_point_crossover
from selection import select_parents, elitism
import visualisation  # generates publication-quality figures

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nature-Inspired Prompt Evolution for LLM Blind-Spot Reduction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--population-size",  type=int,   default=10,
                        help="Number of individuals in the population.")
    parser.add_argument("--generations",      type=int,   default=10,
                        help="Number of evolutionary generations.")
    parser.add_argument("--mutation-rate",    type=float, default=0.3,
                        help="Probability of mutation per individual.")
    parser.add_argument("--crossover-rate",   type=float, default=0.7,
                        help="Probability of crossover between two parents.")
    parser.add_argument("--tournament-size",  type=int,   default=3,
                        help="Tournament size for parent selection.")
    parser.add_argument("--n-elite",          type=int,   default=1,
                        help="Number of elite individuals preserved each generation.")
    parser.add_argument("--seed",             type=int,   default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--advice-model",     type=str,   default="gpt4o-mini",
                        choices=list(OPENROUTER_MODELS.keys()),
                        help="OpenRouter model alias for advice generation.")
    parser.add_argument("--judge-model",      type=str,   default="gpt4o-mini",
                        choices=list(OPENROUTER_MODELS.keys()),
                        help="OpenRouter model alias for fitness evaluation.")
    parser.add_argument("--temperature",      type=float, default=0.7,
                        help="Sampling temperature for advice generation.")
    parser.add_argument("--scenarios-subset", type=int,   default=None,
                        help="Use only the first N scenarios (None = all).")
    parser.add_argument("--results-dir",      type=str,   default=None,
                        help="Directory for output files (auto-named by default).")
    parser.add_argument("--api-key",          type=str,   default=None,
                        help="OpenRouter API key (falls back to OPENROUTER_API_KEY env var).")
    return parser


# ---------------------------------------------------------------------------
# Main evolutionary loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or str(
        PROJECT_ROOT / "results" / f"run_{timestamp}_seed{args.seed}"
    )
    os.makedirs(results_dir, exist_ok=True)

    config = vars(args)
    config["timestamp"] = timestamp

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    scenarios_path  = PROJECT_ROOT / "data" / "scenarios.json"
    seed_prompts_path = PROJECT_ROOT / "data" / "prompts" / "seed_prompts.json"

    scenarios = load_scenarios(str(scenarios_path))
    if args.scenarios_subset:
        scenarios = scenarios[: args.scenarios_subset]
    logger.info("Loaded %d advice scenarios.", len(scenarios))

    seed_prompts = load_seed_prompts(str(seed_prompts_path))
    logger.info("Loaded %d seed prompts.", len(seed_prompts))

    # ------------------------------------------------------------------
    # Initialise LLMs
    # ------------------------------------------------------------------
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error(
            "OpenRouter API key not found.  "
            "Set OPENROUTER_API_KEY or pass --api-key."
        )
        sys.exit(1)

    advice_llm = LLMInterface(
        model=args.advice_model,
        api_key=api_key,
        temperature=args.temperature,
    )
    judge_llm = LLMInterface(
        model=args.judge_model,
        api_key=api_key,
        temperature=0.0,  # deterministic judgments
    )
    evaluator = FitnessEvaluator(advice_llm, judge_llm, scenarios)
    tracker   = ExperimentTracker(results_dir, config)

    # ------------------------------------------------------------------
    # Initialise population
    # ------------------------------------------------------------------
    population = initialise_population(seed_prompts, args.population_size, rng)
    logger.info(
        "Initialised population of %d individuals.", len(population)
    )

    # ------------------------------------------------------------------
    # Evaluate generation 0
    # ------------------------------------------------------------------
    logger.info("Evaluating initial population (generation 0)…")
    all_score_records = []
    population, gen0_records = _evaluate_population(population, evaluator, generation=0)
    all_score_records.extend(gen0_records)
    tracker.record_generation(0, population, gen0_records)

    # ------------------------------------------------------------------
    # Evolutionary loop
    # ------------------------------------------------------------------
    for gen in range(1, args.generations + 1):
        logger.info("─── Generation %d / %d ───", gen, args.generations)
        t0 = time.time()

        # Elitism: carry over the best individual(s)
        elite_individuals = elitism(population, args.n_elite)
        for e in elite_individuals:
            e.generation = gen

        # Build next generation
        next_population: list[PromptChromosome] = list(elite_individuals)

        while len(next_population) < args.population_size:
            # Select parents
            parents = select_parents(
                population,
                n_parents=2,
                tournament_size=args.tournament_size,
                rng=rng,
            )
            # Crossover
            child_a, child_b = single_point_crossover(
                parents[0], parents[1], args.crossover_rate, rng
            )
            # Mutation
            child_a = apply_mutation(child_a, args.mutation_rate, rng)
            child_b = apply_mutation(child_b, args.mutation_rate, rng)

            child_a.generation = gen
            child_b.generation = gen

            next_population.append(child_a)
            if len(next_population) < args.population_size:
                next_population.append(child_b)

        # Evaluate new (non-elite) individuals only
        new_individuals = next_population[args.n_elite:]
        evaluated_new, gen_records = _evaluate_population(
            new_individuals, evaluator, generation=gen
        )
        all_score_records.extend(gen_records)

        # Restore evaluated fitness for elite (already evaluated in prior gen)
        population = elite_individuals + evaluated_new

        tracker.record_generation(gen, population, gen_records)
        logger.info("Generation %d completed in %.1f s.", gen, time.time() - t0)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    tracker.save_results()
    tracker.print_summary()

    # Generate publication-quality figures
    visualisation.plot_all(tracker.generation_stats, results_dir)

    logger.info("Experiment complete. Results in: %s", results_dir)


# ---------------------------------------------------------------------------
# Helper: evaluate a list of individuals
# ---------------------------------------------------------------------------

def _evaluate_population(
    individuals: list[PromptChromosome],
    evaluator: FitnessEvaluator,
    generation: int,
) -> tuple[list[PromptChromosome], list]:
    """Evaluate each individual and attach its fitness score."""
    all_records = []
    for individual in individuals:
        fitness, records = evaluator.evaluate(individual)
        individual.fitness = fitness
        individual.generation = generation
        all_records.extend(records)
        logger.debug(
            "  %s fitness=%.4f", individual.chromosome_id, fitness
        )
    return individuals, all_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
