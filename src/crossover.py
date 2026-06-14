"""
crossover.py
------------
Crossover operators for recombining two parent PromptChromosomes.

Single-point crossover splits the ordered list of genes at a randomly chosen
cut-point.  The offspring inherits all genes before the cut from parent A and
all genes from the cut onward from parent B (and vice-versa for the second
offspring).
"""

from __future__ import annotations

import random

from evolution import PromptChromosome


# Gene order must match the chromosome fields
GENE_NAMES = ["role", "reasoning", "risk_awareness", "reflection", "output_format"]


def single_point_crossover(
    parent_a: PromptChromosome,
    parent_b: PromptChromosome,
    crossover_rate: float,
    rng: random.Random,
) -> tuple[PromptChromosome, PromptChromosome]:
    """
    Perform single-point crossover between two parent chromosomes.

    With probability ``crossover_rate``, a cut-point is chosen uniformly
    from positions 1..len(GENE_NAMES)-1.  Genes before the cut come from
    parent A (B) and genes from the cut onward come from parent B (A).

    If crossover does not fire, clones of the two parents are returned.

    Parameters
    ----------
    parent_a, parent_b : PromptChromosome
        Parent individuals.
    crossover_rate : float
        Probability of crossover in [0, 1].
    rng : random.Random
        Seeded RNG.

    Returns
    -------
    tuple of (PromptChromosome, PromptChromosome)
        Two offspring chromosomes.
    """
    if rng.random() > crossover_rate:
        return parent_a.clone(), parent_b.clone()

    cut = rng.randint(1, len(GENE_NAMES) - 1)

    genes_a = {g: getattr(parent_a, g) for g in GENE_NAMES}
    genes_b = {g: getattr(parent_b, g) for g in GENE_NAMES}

    # Build offspring gene dicts
    child_genes_1 = {}
    child_genes_2 = {}
    for i, gene in enumerate(GENE_NAMES):
        if i < cut:
            child_genes_1[gene] = genes_a[gene]
            child_genes_2[gene] = genes_b[gene]
        else:
            child_genes_1[gene] = genes_b[gene]
            child_genes_2[gene] = genes_a[gene]

    child_1 = PromptChromosome(
        **child_genes_1,
        parent_ids=[parent_a.chromosome_id, parent_b.chromosome_id],
    )
    child_2 = PromptChromosome(
        **child_genes_2,
        parent_ids=[parent_a.chromosome_id, parent_b.chromosome_id],
    )

    return child_1, child_2


def uniform_crossover(
    parent_a: PromptChromosome,
    parent_b: PromptChromosome,
    crossover_rate: float,
    rng: random.Random,
    swap_prob: float = 0.5,
) -> tuple[PromptChromosome, PromptChromosome]:
    """
    Uniform crossover: each gene is independently swapped between parents
    with probability ``swap_prob``.

    Parameters
    ----------
    parent_a, parent_b : PromptChromosome
    crossover_rate : float
    rng : random.Random
    swap_prob : float
        Per-gene swap probability (default 0.5).

    Returns
    -------
    tuple of (PromptChromosome, PromptChromosome)
    """
    if rng.random() > crossover_rate:
        return parent_a.clone(), parent_b.clone()

    genes_a = {g: getattr(parent_a, g) for g in GENE_NAMES}
    genes_b = {g: getattr(parent_b, g) for g in GENE_NAMES}

    child_genes_1 = {}
    child_genes_2 = {}
    for gene in GENE_NAMES:
        if rng.random() < swap_prob:
            child_genes_1[gene] = genes_b[gene]
            child_genes_2[gene] = genes_a[gene]
        else:
            child_genes_1[gene] = genes_a[gene]
            child_genes_2[gene] = genes_b[gene]

    child_1 = PromptChromosome(
        **child_genes_1,
        parent_ids=[parent_a.chromosome_id, parent_b.chromosome_id],
    )
    child_2 = PromptChromosome(
        **child_genes_2,
        parent_ids=[parent_a.chromosome_id, parent_b.chromosome_id],
    )

    return child_1, child_2
