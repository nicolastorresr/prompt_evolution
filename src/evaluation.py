"""
evaluation.py
-------------
Logging, result aggregation, and CSV export utilities.

Tracks per-generation statistics and per-chromosome scoring records,
and serialises everything needed to reproduce and report the experiment.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional

from evolution import PromptChromosome, population_diversity
from fitness import ScoreRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation statistics record
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    diversity: float
    best_chromosome_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Result tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    Accumulates statistics across generations and writes result files.

    Parameters
    ----------
    results_dir : str
        Directory where CSV and JSON output files are saved.
    config : dict
        Experiment configuration dictionary (logged alongside results).
    """

    def __init__(self, results_dir: str, config: dict):
        self.results_dir = results_dir
        self.config = config
        os.makedirs(results_dir, exist_ok=True)

        self.generation_stats: list[GenerationStats] = []
        self.score_records: list[ScoreRecord] = []
        self.best_chromosome: Optional[PromptChromosome] = None

        # Save config immediately
        self._save_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_generation(
        self,
        generation: int,
        population: list[PromptChromosome],
        new_score_records: list[ScoreRecord],
    ) -> GenerationStats:
        """
        Compute and store statistics for the current generation.

        Parameters
        ----------
        generation : int
        population : list of PromptChromosome
            Evaluated population (all must have fitness values).
        new_score_records : list of ScoreRecord
            Detailed scores produced during this generation's evaluation.

        Returns
        -------
        GenerationStats
        """
        self.score_records.extend(new_score_records)

        fitnesses = [c.fitness for c in population if c.fitness is not None]
        if not fitnesses:
            logger.warning("No evaluated individuals in generation %d", generation)
            fitnesses = [0.0]

        best_individual = max(population, key=lambda c: c.fitness or 0.0)
        stats = GenerationStats(
            generation=generation,
            best_fitness=max(fitnesses),
            avg_fitness=sum(fitnesses) / len(fitnesses),
            worst_fitness=min(fitnesses),
            diversity=population_diversity(population),
            best_chromosome_id=best_individual.chromosome_id,
        )
        self.generation_stats.append(stats)

        # Update overall best
        if (
            self.best_chromosome is None
            or (best_individual.fitness or 0.0) > (self.best_chromosome.fitness or 0.0)
        ):
            self.best_chromosome = best_individual

        logger.info(
            "Gen %03d | best=%.4f avg=%.4f diversity=%.3f",
            generation,
            stats.best_fitness,
            stats.avg_fitness,
            stats.diversity,
        )
        return stats

    def save_results(self):
        """Write all accumulated results to disk."""
        self._save_generation_csv()
        self._save_scores_csv()
        self._save_best_prompt()
        logger.info("Results saved to %s", self.results_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_config(self):
        path = os.path.join(self.results_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def _save_generation_csv(self):
        path = os.path.join(self.results_dir, "generation_stats.csv")
        if not self.generation_stats:
            return
        fieldnames = list(asdict(self.generation_stats[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for stat in self.generation_stats:
                writer.writerow(asdict(stat))
        logger.info("Generation stats saved to %s", path)

    def _save_scores_csv(self):
        path = os.path.join(self.results_dir, "score_records.csv")
        if not self.score_records:
            return
        fieldnames = list(asdict(self.score_records[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.score_records:
                writer.writerow(asdict(rec))
        logger.info("Score records saved to %s", path)

    def _save_best_prompt(self):
        if self.best_chromosome is None:
            return
        path = os.path.join(self.results_dir, "best_prompt.json")
        data = self.best_chromosome.to_dict()
        data["assembled_prompt"] = self.best_chromosome.assemble()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Best prompt saved to %s", path)

    def print_summary(self):
        """Print a human-readable experiment summary to stdout."""
        if not self.generation_stats:
            print("No generation statistics recorded.")
            return
        first = self.generation_stats[0]
        last = self.generation_stats[-1]
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Generations run  : {len(self.generation_stats)}")
        print(f"Initial best BAS : {first.best_fitness:.4f}")
        print(f"Final best BAS   : {last.best_fitness:.4f}")
        improvement = last.best_fitness - first.best_fitness
        pct = (improvement / first.best_fitness * 100) if first.best_fitness > 0 else 0
        print(f"Improvement      : {improvement:+.4f}  ({pct:+.1f}%)")
        print(f"Final diversity  : {last.diversity:.3f}")
        if self.best_chromosome:
            print(f"\nBest chromosome  : {self.best_chromosome.chromosome_id}")
            print(f"Assembled prompt :\n{self.best_chromosome.assemble()}")
        print("=" * 60)
