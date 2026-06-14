"""
mutation.py
-----------
Prompt mutation operators for the evolutionary framework.

Each operator modifies one or more genes of a PromptChromosome to introduce
variation.  All operators return a new chromosome (the original is not
modified in place).
"""

from __future__ import annotations

import random
from typing import Callable

from evolution import PromptChromosome, _generate_id


# ---------------------------------------------------------------------------
# Adjective pools used by the replace-adjective operator
# ---------------------------------------------------------------------------

ROLE_ADJECTIVES = [
    "thoughtful", "cautious", "wise", "experienced", "balanced",
    "strategic", "empathetic", "objective", "reflective", "analytical",
    "pragmatic", "careful", "nuanced", "insightful", "thorough",
]

ADVISOR_NOUNS = [
    "advisor", "counselor", "mentor", "consultant", "guide",
    "coach", "analyst", "strategist",
]

# ---------------------------------------------------------------------------
# Extra instruction snippets injected by mutation operators
# ---------------------------------------------------------------------------

REASONING_STEPS = [
    "Break the problem into its root causes before recommending a solution.",
    "Identify the key stakeholders and how each is affected.",
    "Consider both short-term relief and long-term consequences.",
    "Map out the chain of events that led to this situation.",
    "Ask yourself what additional information would change your advice.",
]

REFLECTION_INSTRUCTIONS = [
    "Before finalising your answer, reconsider whether you have missed anything important.",
    "Pause after your initial analysis and actively challenge your first conclusions.",
    "Review your advice from the perspective of a critic who disagrees with you.",
    "Ask: what would a reasonable person with the opposite view say?",
    "Re-read the scenario to check for details that might alter your recommendation.",
]

UNCERTAINTY_INSTRUCTIONS = [
    "Acknowledge what is unknown or uncertain in this situation.",
    "Where information is incomplete, state your assumptions explicitly.",
    "Distinguish between what is likely and what is merely possible.",
    "Flag areas where your advice may need to be revised as new information emerges.",
    "Be honest about the limits of the advice you can provide given the available information.",
]

RISK_INSTRUCTIONS = [
    "Explicitly list potential negative consequences before giving recommendations.",
    "Identify early warning signs that the situation may be more serious than it appears.",
    "Assess the severity and likelihood of each identified risk.",
    "Consider worst-case scenarios and how to mitigate them.",
    "Highlight risks the person may not have considered or may be underestimating.",
]

OUTPUT_FORMAT_VARIANTS = [
    "Structure your response with clearly labelled sections: Analysis, Risks, Alternatives, Recommendation.",
    "Present your advice as numbered action steps preceded by a brief situation summary.",
    "Use a concise format: one paragraph of analysis, one paragraph of risks, one paragraph of recommendations.",
    "Organise your response as: (1) Key Issues, (2) Risks to Consider, (3) Options, (4) Recommended Path.",
    "Keep your response structured but conversational, leading with the most critical consideration.",
]


# ---------------------------------------------------------------------------
# Individual mutation operators
# ---------------------------------------------------------------------------

def mutate_replace_adjective(
    chromosome: PromptChromosome, rng: random.Random
) -> PromptChromosome:
    """Replace the adjective in the role gene with a randomly chosen alternative."""
    child = chromosome.clone()
    adj = rng.choice(ROLE_ADJECTIVES)
    noun = rng.choice(ADVISOR_NOUNS)
    child.role = f"You are a {adj} and {rng.choice(ROLE_ADJECTIVES)} {noun}."
    child.parent_ids = [chromosome.chromosome_id]
    return child


def mutate_add_reasoning_step(
    chromosome: PromptChromosome, rng: random.Random
) -> PromptChromosome:
    """Append an additional reasoning step to the reasoning gene."""
    child = chromosome.clone()
    step = rng.choice(REASONING_STEPS)
    child.reasoning = chromosome.reasoning.rstrip(".") + " " + step
    child.parent_ids = [chromosome.chromosome_id]
    return child


def mutate_add_reflection(
    chromosome: PromptChromosome, rng: random.Random
) -> PromptChromosome:
    """Append a reflection instruction to the reflection gene."""
    child = chromosome.clone()
    instr = rng.choice(REFLECTION_INSTRUCTIONS)
    child.reflection = chromosome.reflection.rstrip(".") + " " + instr
    child.parent_ids = [chromosome.chromosome_id]
    return child


def mutate_add_uncertainty(
    chromosome: PromptChromosome, rng: random.Random
) -> PromptChromosome:
    """Append an uncertainty-awareness instruction to the risk_awareness gene."""
    child = chromosome.clone()
    instr = rng.choice(UNCERTAINTY_INSTRUCTIONS)
    child.risk_awareness = chromosome.risk_awareness.rstrip(".") + " " + instr
    child.parent_ids = [chromosome.chromosome_id]
    return child


def mutate_add_risk_detection(
    chromosome: PromptChromosome, rng: random.Random
) -> PromptChromosome:
    """Replace the risk_awareness gene with a stronger variant."""
    child = chromosome.clone()
    child.risk_awareness = rng.choice(RISK_INSTRUCTIONS)
    child.parent_ids = [chromosome.chromosome_id]
    return child


def mutate_remove_redundant(
    chromosome: PromptChromosome, rng: random.Random
) -> PromptChromosome:
    """
    Shorten an overly long gene by trimming appended sentences.
    Targets the longest gene to remove potential redundancy.
    """
    child = chromosome.clone()
    genes = {
        "reasoning": child.reasoning,
        "risk_awareness": child.risk_awareness,
        "reflection": child.reflection,
    }
    longest_key = max(genes, key=lambda k: len(genes[k]))
    original = genes[longest_key]
    sentences = [s.strip() for s in original.split(".") if s.strip()]
    if len(sentences) > 1:
        trimmed = ". ".join(sentences[:-1]) + "."
        setattr(child, longest_key, trimmed)
    child.parent_ids = [chromosome.chromosome_id]
    return child


def mutate_replace_output_format(
    chromosome: PromptChromosome, rng: random.Random
) -> PromptChromosome:
    """Replace the output_format gene with a randomly chosen alternative."""
    child = chromosome.clone()
    child.output_format = rng.choice(OUTPUT_FORMAT_VARIANTS)
    child.parent_ids = [chromosome.chromosome_id]
    return child


# ---------------------------------------------------------------------------
# Mutation dispatcher
# ---------------------------------------------------------------------------

# Registry of all available operators with equal probability by default
MUTATION_OPERATORS: list[Callable[[PromptChromosome, random.Random], PromptChromosome]] = [
    mutate_replace_adjective,
    mutate_add_reasoning_step,
    mutate_add_reflection,
    mutate_add_uncertainty,
    mutate_add_risk_detection,
    mutate_remove_redundant,
    mutate_replace_output_format,
]


def apply_mutation(
    chromosome: PromptChromosome,
    mutation_rate: float,
    rng: random.Random,
) -> PromptChromosome:
    """
    Apply mutation to a chromosome with probability ``mutation_rate``.

    When mutation fires, one operator is chosen uniformly at random from
    ``MUTATION_OPERATORS``.

    Parameters
    ----------
    chromosome : PromptChromosome
        Individual to (potentially) mutate.
    mutation_rate : float
        Probability of mutation in [0, 1].
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    PromptChromosome
        A (possibly mutated) copy of the input chromosome.
    """
    if rng.random() < mutation_rate:
        operator = rng.choice(MUTATION_OPERATORS)
        return operator(chromosome, rng)
    return chromosome.clone()
