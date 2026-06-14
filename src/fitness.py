"""
fitness.py
----------
Blind-Spot Awareness Score (BAS) fitness evaluation.

Uses an LLM-as-a-Judge approach to score generated advice on four dimensions:
  1. Risk Identification (RI)
  2. Alternative Perspectives (AP)
  3. Context Awareness (CA)
  4. Reasoning Completeness (RC)

Each dimension is scored 1–5.  The BAS for a single response is the mean
of the four dimension scores.  The overall fitness of a prompt is the mean
BAS across all evaluation scenarios.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

from llm_interface import LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoreRecord:
    """Stores evaluation scores for one (prompt, scenario) pair."""
    chromosome_id: str
    scenario_id: str
    scenario_category: str
    risk_identification: float
    alternative_perspectives: float
    context_awareness: float
    reasoning_completeness: float
    bas: float
    advice_text: str = ""
    judge_feedback: str = ""


# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of advice given in response to
real-life decision-making scenarios. You evaluate advice strictly and objectively.

Your task is to score the advice on FOUR dimensions, each on a scale from 1 to 5:

1. Risk Identification (RI):
   1 = No risks mentioned
   3 = Some obvious risks acknowledged
   5 = Comprehensive risk identification including non-obvious and nuanced risks

2. Alternative Perspectives (AP):
   1 = Single-track thinking, no alternatives explored
   3 = One or two alternatives mentioned briefly
   5 = Multiple meaningful alternatives thoroughly explored

3. Context Awareness (CA):
   1 = Advice ignores the specific context of the scenario
   3 = Some contextual factors are incorporated
   5 = Advice deeply engages with the full situational context

4. Reasoning Completeness (RC):
   1 = Superficial or oversimplified reasoning
   3 = Adequate reasoning but with notable gaps
   5 = Complete, nuanced reasoning that addresses complexity

Respond ONLY with a valid JSON object in this exact format (no additional text):
{
  "risk_identification": <integer 1-5>,
  "alternative_perspectives": <integer 1-5>,
  "context_awareness": <integer 1-5>,
  "reasoning_completeness": <integer 1-5>,
  "feedback": "<one concise sentence explaining the main strength or weakness>"
}"""

JUDGE_USER_TEMPLATE = """SCENARIO:
{scenario}

ADVICE PROVIDED:
{advice}

Please evaluate the advice according to the four dimensions."""


# ---------------------------------------------------------------------------
# Fitness evaluator
# ---------------------------------------------------------------------------

class FitnessEvaluator:
    """
    Evaluates a PromptChromosome's fitness by generating and judging advice
    across a set of scenarios.

    Parameters
    ----------
    advice_llm : LLMInterface
        LLM used to generate advice (the model being optimised).
    judge_llm : LLMInterface
        LLM used to score the generated advice.
    scenarios : list of dict
        Scenario dictionaries loaded from ``scenarios.json``.
    """

    def __init__(
        self,
        advice_llm: LLMInterface,
        judge_llm: LLMInterface,
        scenarios: list[dict],
    ):
        self.advice_llm = advice_llm
        self.judge_llm = judge_llm
        self.scenarios = scenarios

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, chromosome) -> tuple[float, list[ScoreRecord]]:
        """
        Compute the Blind-Spot Awareness Score for a chromosome.

        Parameters
        ----------
        chromosome : PromptChromosome

        Returns
        -------
        fitness : float
            Mean BAS across all scenarios.
        records : list of ScoreRecord
            Per-scenario scoring details.
        """
        system_prompt = chromosome.assemble()
        records: list[ScoreRecord] = []

        for scenario in self.scenarios:
            try:
                record = self._evaluate_scenario(chromosome, system_prompt, scenario)
                records.append(record)
            except Exception as exc:
                logger.error(
                    "Error evaluating chromosome %s on scenario %s: %s",
                    chromosome.chromosome_id,
                    scenario["id"],
                    exc,
                )

        if not records:
            return 0.0, []

        fitness = sum(r.bas for r in records) / len(records)
        return fitness, records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_scenario(
        self, chromosome, system_prompt: str, scenario: dict
    ) -> ScoreRecord:
        """Generate advice then judge it for one scenario."""
        # Step 1: generate advice
        advice = self.advice_llm.query(
            prompt=scenario["scenario"],
            system_message=system_prompt,
        )

        # Step 2: judge the advice
        judge_prompt = JUDGE_USER_TEMPLATE.format(
            scenario=scenario["scenario"],
            advice=advice,
        )
        raw_judgment = self.judge_llm.query(
            prompt=judge_prompt,
            system_message=JUDGE_SYSTEM_PROMPT,
        )

        scores = self._parse_judgment(raw_judgment)

        bas = (
            scores["risk_identification"]
            + scores["alternative_perspectives"]
            + scores["context_awareness"]
            + scores["reasoning_completeness"]
        ) / 4.0

        return ScoreRecord(
            chromosome_id=chromosome.chromosome_id,
            scenario_id=scenario["id"],
            scenario_category=scenario["category"],
            risk_identification=scores["risk_identification"],
            alternative_perspectives=scores["alternative_perspectives"],
            context_awareness=scores["context_awareness"],
            reasoning_completeness=scores["reasoning_completeness"],
            bas=bas,
            advice_text=advice[:500],  # truncate for storage
            judge_feedback=scores.get("feedback", ""),
        )

    @staticmethod
    def _parse_judgment(raw: str) -> dict:
        """
        Extract scores from the judge's JSON response.

        Falls back to default scores of 1.0 on any parse error.
        """
        defaults = {
            "risk_identification": 1.0,
            "alternative_perspectives": 1.0,
            "context_awareness": 1.0,
            "reasoning_completeness": 1.0,
            "feedback": "",
        }
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(cleaned)
            for key in ["risk_identification", "alternative_perspectives",
                        "context_awareness", "reasoning_completeness"]:
                val = float(data.get(key, 1))
                defaults[key] = max(1.0, min(5.0, val))
            defaults["feedback"] = data.get("feedback", "")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse judge response: %s\nRaw: %s", e, raw[:200])
        return defaults


def load_scenarios(path: str) -> list[dict]:
    """Load scenarios from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]
