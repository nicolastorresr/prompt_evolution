"""
selection.py
------------
Selection operators for the evolutionary algorithm.

Tournament selection is the primary method used in the paper.
Roulette-wheel (fitness-proportionate) selection is provided as an alternative.
"""

from __future__ import annotations

import random

from evolution import PromptChromosome


def tournament_selection(
    population: list[PromptChromosome],
    tournament_size: int,
    rng: random.Random,
) -> PromptChromosome:
    """
    Select one individual via tournament selection.

    Draws ``tournament_size`` individuals uniformly at random (with
    replacement) from ``population`` and returns the one with the highest
    fitness score.

    Parameters
    ----------
    population : list of PromptChromosome
        Current population (all individuals must have a fitness value).
    tournament_size : int
        Number of competitors in each tournament (typically 2–5).
    rng : random.Random
        Seeded RNG.

    Returns
    -------
    PromptChromosome
        The winner of the tournament (original object, not a copy).
    """
    competitors = rng.choices(population, k=tournament_size)
    return max(competitors, key=lambda c: c.fitness or 0.0)


def roulette_wheel_selection(
    population: list[PromptChromosome],
    rng: random.Random,
) -> PromptChromosome:
    """
    Fitness-proportionate (roulette-wheel) selection.

    Individuals with higher fitness scores have a proportionally greater
    probability of being selected.  If all fitness values are zero, uniform
    random selection is used as a fallback.

    Parameters
    ----------
    population : list of PromptChromosome
    rng : random.Random

    Returns
    -------
    PromptChromosome
    """
    fitnesses = [max(c.fitness or 0.0, 0.0) for c in population]
    total = sum(fitnesses)
    if total == 0:
        return rng.choice(population)
    weights = [f / total for f in fitnesses]
    return rng.choices(population, weights=weights, k=1)[0]


def select_parents(
    population: list[PromptChromosome],
    n_parents: int,
    tournament_size: int,
    rng: random.Random,
) -> list[PromptChromosome]:
    """
    Select ``n_parents`` individuals via repeated tournament selection.

    Parameters
    ----------
    population : list of PromptChromosome
    n_parents : int
        Number of parents to select.
    tournament_size : int
    rng : random.Random

    Returns
    -------
    list of PromptChromosome
    """
    return [
        tournament_selection(population, tournament_size, rng)
        for _ in range(n_parents)
    ]


def elitism(
    population: list[PromptChromosome],
    n_elite: int = 1,
) -> list[PromptChromosome]:
    """
    Return the top ``n_elite`` individuals sorted by descending fitness.

    These elite individuals are carried over unchanged to the next generation.

    Parameters
    ----------
    population : list of PromptChromosome
    n_elite : int
        Number of elite individuals to preserve (default 1).

    Returns
    -------
    list of PromptChromosome
        Clones of the elite individuals (with preserved fitness).
    """
    sorted_pop = sorted(population, key=lambda c: c.fitness or 0.0, reverse=True)
    elites = []
    for individual in sorted_pop[:n_elite]:
        clone = individual.clone()
        clone.fitness = individual.fitness  # preserve evaluated fitness
        elites.append(clone)
    return elites
