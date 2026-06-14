"""
evolution.py
------------
Core data structures and utilities for the evolutionary prompt optimization framework.

A ``PromptChromosome`` is a structured prompt composed of five semantic genes:
  1. role             – persona adopted by the model
  2. reasoning        – instructions for systematic analysis
  3. risk_awareness   – instructions for identifying risks
  4. reflection       – instructions for deliberate reasoning
  5. output_format    – instructions for structuring the response

The assembled prompt is passed verbatim as the system message when querying
the advice-generation LLM.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Chromosome
# ---------------------------------------------------------------------------

@dataclass
class PromptChromosome:
    """
    Represents a single prompt individual in the evolutionary population.

    Attributes
    ----------
    role : str
        Persona / role definition gene.
    reasoning : str
        Reasoning instruction gene.
    risk_awareness : str
        Risk-awareness instruction gene.
    reflection : str
        Reflection instruction gene.
    output_format : str
        Output-format instruction gene.
    fitness : float or None
        Blind-Spot Awareness Score assigned during evaluation.
    generation : int
        Generation in which this individual was created.
    parent_ids : list of str
        IDs of parent chromosomes (for lineage tracking).
    chromosome_id : str
        Unique identifier assigned at creation.
    """

    role: str
    reasoning: str
    risk_awareness: str
    reflection: str
    output_format: str
    fitness: Optional[float] = field(default=None, compare=False)
    generation: int = field(default=0, compare=False)
    parent_ids: list = field(default_factory=list, compare=False)
    chromosome_id: str = field(default="", compare=False)

    def __post_init__(self):
        if not self.chromosome_id:
            self.chromosome_id = _generate_id()

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def assemble(self) -> str:
        """
        Concatenate all genes into a single system-message string.

        Returns
        -------
        str
            The full prompt string ready to be used as a system message.
        """
        parts = [
            self.role,
            self.reasoning,
            self.risk_awareness,
            self.reflection,
            self.output_format,
        ]
        return " ".join(p.strip() for p in parts if p.strip())

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PromptChromosome":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PromptChromosome":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clone(self) -> "PromptChromosome":
        """Return a deep copy with a fresh ID."""
        c = copy.deepcopy(self)
        c.chromosome_id = _generate_id()
        c.fitness = None
        return c

    def __repr__(self) -> str:
        fit = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return (
            f"PromptChromosome(id={self.chromosome_id}, "
            f"gen={self.generation}, fitness={fit})"
        )


# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------

def load_seed_prompts(path: str) -> list[PromptChromosome]:
    """
    Load seed prompts from a JSON file and convert them to chromosomes.

    Parameters
    ----------
    path : str
        Path to ``seed_prompts.json``.

    Returns
    -------
    list of PromptChromosome
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chromosomes = []
    for entry in data["seed_prompts"]:
        c = PromptChromosome(
            role=entry["role"],
            reasoning=entry["reasoning"],
            risk_awareness=entry["risk_awareness"],
            reflection=entry["reflection"],
            output_format=entry["output_format"],
            chromosome_id=entry.get("id", _generate_id()),
        )
        chromosomes.append(c)
    return chromosomes


def initialise_population(
    seed_prompts: list[PromptChromosome],
    population_size: int,
    rng: random.Random,
) -> list[PromptChromosome]:
    """
    Build an initial population of ``population_size`` chromosomes.

    The seeds are included as-is; additional individuals are generated by
    randomly sampling genes from the seed pool.

    Parameters
    ----------
    seed_prompts : list of PromptChromosome
        Base seed chromosomes loaded from ``seed_prompts.json``.
    population_size : int
        Desired population size.
    rng : random.Random
        Seeded random number generator for reproducibility.

    Returns
    -------
    list of PromptChromosome
    """
    population = [s.clone() for s in seed_prompts[:population_size]]

    while len(population) < population_size:
        # Create a new individual by sampling each gene from a random seed
        new_chr = PromptChromosome(
            role=rng.choice(seed_prompts).role,
            reasoning=rng.choice(seed_prompts).reasoning,
            risk_awareness=rng.choice(seed_prompts).risk_awareness,
            reflection=rng.choice(seed_prompts).reflection,
            output_format=rng.choice(seed_prompts).output_format,
            generation=0,
        )
        population.append(new_chr)

    return population


def population_diversity(population: list[PromptChromosome]) -> float:
    """
    Compute a simple diversity metric as the ratio of unique assembled prompts
    to total population size.

    Returns a value in [0, 1] where 1.0 means all individuals are distinct.
    """
    unique = len({c.assemble() for c in population})
    return unique / len(population) if population else 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_id_counter = 0


def _generate_id() -> str:
    global _id_counter
    _id_counter += 1
    return f"chr_{_id_counter:05d}"
