"""Mock adapter for testing and development."""

import json
import random
import time
from typing import Any, Optional

from harness.providers.base import ModelProvider, Completion


class MockAdapter(ModelProvider):
    """
    Mock model provider for testing.

    Provides deterministic, seeded, rule-based responses without calling external APIs.
    """

    def __init__(
        self,
        mode: str = "compliant",
        response_map: Optional[dict[str, str]] = None,
        gold_answers: Optional[dict[str, str]] = None,
        accuracy: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize mock adapter.

        Args:
            mode: Response mode - one of:
                - "compliant": Valid JSON matching expected schema
                - "malformed": Various JSON formatting errors
                - "contradictory": Valid JSON but contradictory content
                - "incomplete": Valid JSON but missing required fields
                - "rule_based": Uses gold_answers to generate responses with configurable accuracy
            response_map: Custom mapping of item_id -> response text (optional)
            gold_answers: Mapping of item_id -> correct answer for rule_based mode
            accuracy: Accuracy rate for rule_based mode (0.0 to 1.0)
            seed: Random seed for deterministic behavior
        """
        self.mode = mode
        self.response_map = response_map or {}
        self.gold_answers = gold_answers or {}
        self.accuracy = accuracy
        self.seed = seed
        self.rng = random.Random(seed)

    def complete(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any]
    ) -> Completion:
        """Generate a mock completion."""
        start_time = time.time()

        # Extract item_id and condition_id from params or messages
        item_id = params.get("item_id", "unknown")
        condition_id = params.get("condition_id", "000")

        # Check if we have a custom response
        if item_id in self.response_map:
            response_text = self.response_map[item_id]
        else:
            # Generate response based on mode
            response_text = self._generate_response(item_id, condition_id, messages)

        # Simulate token counts
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
        completion_tokens = len(response_text.split())

        latency_ms = (time.time() - start_time) * 1000

        return Completion(
            text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model="mock-model",
            model_version="1.0",
            latency_ms=latency_ms,
            cached=False,
            metadata={"mode": self.mode, "item_id": item_id}
        )

    def _generate_response(
        self,
        item_id: str,
        condition_id: str,
        messages: list[dict[str, str]]
    ) -> str:
        """Generate response based on mode."""
        if self.mode == "compliant":
            return self._generate_compliant_response(item_id, condition_id)
        elif self.mode == "malformed":
            return self._generate_malformed_response(item_id, condition_id)
        elif self.mode == "contradictory":
            return self._generate_contradictory_response(item_id, condition_id)
        elif self.mode == "incomplete":
            return self._generate_incomplete_response(item_id, condition_id)
        elif self.mode == "rule_based":
            return self._generate_rule_based_response(item_id, condition_id)
        else:
            return self._generate_compliant_response(item_id, condition_id)

    def _generate_compliant_response(self, item_id: str, condition_id: str) -> str:
        """Generate valid JSON matching the expected schema."""
        # Determine answer based on item_id hash for consistency
        choices = ["A", "B", "C", "D"]
        answer_idx = hash(item_id) % len(choices)
        answer = choices[answer_idx]

        if condition_id == "000":
            # Baseline: DirectAnswer
            data = {
                "answer": answer,
                "confidence": 75,
                "reasoning": f"Mock reasoning for {item_id}"
            }
        elif condition_id == "100":
            # E-only: EnumerateOutput
            data = {
                "hypotheses": [
                    {
                        "hypothesis": f"Hypothesis {i}",
                        "supporting_evidence": f"Support {i}",
                        "contradicting_evidence": f"Contradict {i}"
                    }
                    for i in range(3)
                ],
                "answer": answer,
                "confidence": 75,
                "reasoning": f"Mock reasoning for {item_id}"
            }
        elif condition_id == "110":
            # E+Table: TablePlaceboOutput
            data = {
                "hypotheses": [f"Hypothesis {i}" for i in range(3)],
                "evidence_table": [
                    {
                        "evidence": f"Evidence {i}",
                        "summaries": {f"Hypothesis {j}": f"Summary {i}-{j}" for j in range(3)}
                    }
                    for i in range(4)
                ],
                "answer": answer,
                "confidence": 75,
                "reasoning": f"Mock reasoning for {item_id}"
            }
        elif condition_id == "101":
            # E+Prose-Disconfirm: ProseDisconfirmOutput
            data = {
                "hypotheses": [
                    {
                        "hypothesis": f"Hypothesis {i}",
                        "disconfirming_analysis": f"Disconfirming analysis {i}"
                    }
                    for i in range(3)
                ],
                "answer": answer,
                "confidence": 75,
                "reasoning": f"Mock reasoning for {item_id}"
            }
        elif condition_id == "111":
            # Full ACH: FullACHOutput
            hypotheses = [f"Hypothesis {i}" for i in range(3)]
            data = {
                "hypotheses": hypotheses,
                "ach_matrix": [
                    {
                        "evidence": f"Evidence {i}",
                        "consistency_codes": {h: self.rng.choice(["C", "I", "N"]) for h in hypotheses},
                        "diagnostic_value": self.rng.uniform(0, 1)
                    }
                    for i in range(4)
                ],
                "inconsistency_counts": {h: self.rng.randint(0, 4) for h in hypotheses},
                "high_diagnostic_evidence": [f"Evidence {i}" for i in range(2)],
                "answer": answer,
                "confidence": 75,
                "reasoning": f"Mock reasoning for {item_id}"
            }
        elif condition_id == "filter_only":
            # FilterOutput
            data = {
                "relevant_evidence": [f"Relevant {i}" for i in range(3)],
                "contextual_evidence": [f"Contextual {i}" for i in range(2)],
                "irrelevant_evidence": [f"Irrelevant {i}" for i in range(2)],
                "reasoning": f"Mock filtering reasoning for {item_id}"
            }
        elif condition_id == "prism_full":
            # PRISMVerdictOutput
            data = {
                "verdict": answer,
                "confidence": 75,
                "reasoning": f"Mock PRISM reasoning for {item_id}",
                "key_evidence": [f"Key evidence {i}" for i in range(3)]
            }
        else:
            # Default to baseline
            data = {
                "answer": answer,
                "confidence": 75,
                "reasoning": f"Mock reasoning for {item_id}"
            }

        return "```json\n" + json.dumps(data, indent=2) + "\n```"

    def _generate_malformed_response(self, item_id: str, condition_id: str) -> str:
        """Generate JSON with various formatting errors."""
        # Get a base response
        compliant = self._generate_compliant_response(item_id, condition_id)

        # Apply random malformation
        malformations = [
            lambda s: s.replace('"', "'"),  # Single quotes
            lambda s: s.replace(",\n", ",\n  "),  # Extra trailing comma before }
            lambda s: s.replace(": ", ":"),  # Remove space after colon
            lambda s: s[:-10],  # Truncate end
        ]

        malform = self.rng.choice(malformations)
        return malform(compliant)

    def _generate_contradictory_response(self, item_id: str, condition_id: str) -> str:
        """Generate valid JSON but with contradictory content."""
        # Similar to compliant but with intentionally conflicting information
        data = {
            "answer": "A",
            "confidence": 25,  # Low confidence but definite answer
            "reasoning": "All evidence points to B, but I choose A"
        }
        return "```json\n" + json.dumps(data, indent=2) + "\n```"

    def _generate_incomplete_response(self, item_id: str, condition_id: str) -> str:
        """Generate valid JSON but missing required fields."""
        data = {
            "answer": "A",
            # Missing confidence and reasoning
        }
        return "```json\n" + json.dumps(data, indent=2) + "\n```"

    def _generate_rule_based_response(self, item_id: str, condition_id: str) -> str:
        """
        Generate response based on gold answers with configurable accuracy.

        Uses gold_answers to determine correct answer, then randomly decides
        whether to answer correctly based on accuracy parameter.
        """
        # Get gold answer
        gold_answer = self.gold_answers.get(item_id)

        if gold_answer is None:
            # No gold answer available, use compliant mode
            return self._generate_compliant_response(item_id, condition_id)

        # Decide if this will be correct based on accuracy
        is_correct = self.rng.random() < self.accuracy

        if is_correct:
            answer = gold_answer
            confidence = self.rng.randint(70, 95)
            reasoning = f"Based on evidence, {gold_answer} is correct"
        else:
            # Generate wrong answer
            choices = ["A", "B", "C", "D"]
            wrong_choices = [c for c in choices if c != gold_answer]
            answer = self.rng.choice(wrong_choices)
            confidence = self.rng.randint(50, 75)
            reasoning = f"Evidence suggests {answer}, though {gold_answer} is also plausible"

        # Build response based on condition
        if condition_id in ["000", "free_cot"]:
            data = {
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning
            }
        elif condition_id == "100":
            data = {
                "hypotheses": [
                    {
                        "hypothesis": f"Answer is {c}",
                        "supporting_evidence": f"Evidence for {c}",
                        "contradicting_evidence": f"Evidence against {c}"
                    }
                    for c in ["A", "B", "C", "D"]
                ],
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning
            }
        elif condition_id == "110":
            hypotheses = [f"Answer is {c}" for c in ["A", "B", "C", "D"]]
            data = {
                "hypotheses": hypotheses,
                "evidence_table": [
                    {
                        "evidence": f"Evidence {i}",
                        "summaries": {h: f"Summary for {h}" for h in hypotheses}
                    }
                    for i in range(3)
                ],
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning
            }
        elif condition_id == "101":
            data = {
                "hypotheses": [
                    {
                        "hypothesis": f"Answer is {c}",
                        "disconfirming_analysis": f"Analysis against {c}"
                    }
                    for c in ["A", "B", "C", "D"]
                ],
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning
            }
        elif condition_id == "111":
            hypotheses = [f"Answer is {c}" for c in ["A", "B", "C", "D"]]
            data = {
                "hypotheses": hypotheses,
                "ach_matrix": [
                    {
                        "evidence": f"Evidence {i}",
                        "consistency_codes": {h: self.rng.choice(["C", "I", "N"]) for h in hypotheses},
                        "diagnostic_value": self.rng.uniform(0, 1)
                    }
                    for i in range(3)
                ],
                "inconsistency_counts": {h: self.rng.randint(0, 3) for h in hypotheses},
                "high_diagnostic_evidence": [f"Evidence {i}" for i in range(2)],
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning
            }
        elif condition_id == "prism_full":
            data = {
                "verdict": answer,
                "confidence": confidence,
                "reasoning": reasoning,
                "key_evidence": [f"Key evidence {i}" for i in range(2)]
            }
        else:
            data = {
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning
            }

        return "```json\n" + json.dumps(data, indent=2) + "\n```"

    def provider_name(self) -> str:
        """Return provider name."""
        return "mock"

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Mock provider has zero cost."""
        return 0.0
