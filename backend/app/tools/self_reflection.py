# =============================================================================
# MEGA AI — Self-Reflection Tool
# =============================================================================
# Agent re-reads its own previous outputs and finds contradictions.
# =============================================================================

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.tools.base import (
    BaseTool,
    ToolInputError,
)

logger = logging.getLogger(__name__)


@dataclass
class Contradiction:
    """
    A contradiction found between two claims.
    """

    claim_a: str

    claim_b: str

    turn: int

    severity: str = "medium"

    explanation: str = ""


class SelfReflectionTool(BaseTool):
    """
    Self-reflection tool for detecting contradictions.
    """

    name = "self_reflection"

    description = (
        "Re-read previous outputs "
        "and find contradictions"
    )

    timeout_ms = 15000

    max_retries = 0

    # =========================================================================
    # Contradiction Patterns
    # =========================================================================

    _CONTRADICTION_PATTERNS: List[tuple] = [

        # ---------------------------------------------------------------------
        # Direct Negation
        # ---------------------------------------------------------------------

        (
            r"is (\w+)",
            r"is not \1",
        ),

        (
            r"are (\w+)",
            r"are not \1",
        ),

        (
            r"was (\w+)",
            r"was not \1",
        ),

        (
            r"were (\w+)",
            r"were not \1",
        ),

        (
            r"has (\w+)",
            r"has not \1",
        ),

        (
            r"have (\w+)",
            r"have not \1",
        ),

        # ---------------------------------------------------------------------
        # Quantifier Contradictions
        # ---------------------------------------------------------------------

        (
            r"all (\w+)",
            r"some \1 are not",
        ),

        (
            r"none (\w+)",
            r"at least one \1",
        ),

        (
            r"always",
            r"sometimes not",
        ),

        (
            r"never",
            r"sometimes",
        ),

        # ---------------------------------------------------------------------
        # Numeric Contradictions
        # ---------------------------------------------------------------------

        (
            r"(\d+)",
            r"not \1",
        ),
    ]

    def __init__(
        self,
        max_history_turns: int = 10,
    ) -> None:
        """
        Initialize reflection tool.
        """

        super().__init__()

        self.max_history_turns = (
            max_history_turns
        )

    # =========================================================================
    # Claim Extraction
    # =========================================================================

    def _extract_claims(
        self,
        text: str,
    ) -> List[str]:
        """
        Extract factual claims from text.
        """

        sentences = re.split(
            r"[.!?]+",
            text,
        )

        claims = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # -----------------------------------------------------------------
            # Skip Questions
            # -----------------------------------------------------------------

            if sentence.endswith("?"):
                continue

            # -----------------------------------------------------------------
            # Skip Commands
            # -----------------------------------------------------------------

            if sentence.lower().startswith(
                (
                    "please",
                    "can you",
                    "could you",
                )
            ):
                continue

            # -----------------------------------------------------------------
            # Skip Very Short
            # -----------------------------------------------------------------

            if len(sentence.split()) < 3:
                continue

            claims.append(sentence)

        return claims

    # =========================================================================
    # Contradiction Detection
    # =========================================================================

    def _detect_contradictions(
        self,
        outputs: List[str],
    ) -> List[Contradiction]:
        """
        Detect contradictions across outputs.
        """

        contradictions = []

        turn_claims: List[
            List[str]
        ] = []

        for output in outputs:

            claims = self._extract_claims(
                output
            )

            turn_claims.append(claims)

        # ---------------------------------------------------------------------
        # Compare Across Turns
        # ---------------------------------------------------------------------

        for turn_b in range(
            len(turn_claims)
        ):

            for turn_a in range(turn_b):

                for claim_a in (
                    turn_claims[turn_a]
                ):

                    for claim_b in (
                        turn_claims[turn_b]
                    ):

                        contradiction = (
                            self._check_contradiction(
                                claim_a,
                                claim_b,
                                turn_a,
                                turn_b,
                            )
                        )

                        if contradiction:

                            contradictions.append(
                                contradiction
                            )

        return contradictions

    # =========================================================================
    # Contradiction Check
    # =========================================================================

    def _check_contradiction(
        self,
        claim_a: str,
        claim_b: str,
        turn_a: int,
        turn_b: int,
    ) -> Optional[Contradiction]:
        """
        Check whether two claims contradict.
        """

        claim_a_lower = (
            claim_a.lower()
        )

        claim_b_lower = (
            claim_b.lower()
        )

        # ---------------------------------------------------------------------
        # Negation Detection
        # ---------------------------------------------------------------------

        negations = [
            "not ",
            "no ",
            "never ",
            "false",
            "incorrect",
        ]

        a_has_negation = any(
            neg in claim_a_lower
            for neg in negations
        )

        b_has_negation = any(
            neg in claim_b_lower
            for neg in negations
        )

        # ---------------------------------------------------------------------
        # Numeric Contradictions
        # ---------------------------------------------------------------------

        if a_has_negation == b_has_negation:

            nums_a = set(
                re.findall(
                    r"\d+\.?\d*",
                    claim_a_lower,
                )
            )

            nums_b = set(
                re.findall(
                    r"\d+\.?\d*",
                    claim_b_lower,
                )
            )

            if (
                nums_a
                and nums_b
                and nums_a != nums_b
            ):

                subject_a = (
                    claim_a_lower.split()[:5]
                )

                subject_b = (
                    claim_b_lower.split()[:5]
                )

                if any(
                    word in subject_b
                    for word in subject_a
                    if len(word) > 3
                ):

                    return Contradiction(
                        claim_a=claim_a,
                        claim_b=claim_b,
                        turn=turn_b,
                        severity="high",
                        explanation=(
                            "Numeric mismatch: "
                            f"{nums_a} vs {nums_b}"
                        ),
                    )

            return None

        # ---------------------------------------------------------------------
        # Word Overlap
        # ---------------------------------------------------------------------

        words_a = set(
            claim_a_lower.split()
        )

        words_b = set(
            claim_b_lower.split()
        )

        common_words = (
            words_a & words_b
        )

        if len(common_words) >= 3:

            severity = (
                "high"
                if len(common_words) >= 5
                else "medium"
            )

            return Contradiction(
                claim_a=claim_a,
                claim_b=claim_b,
                turn=turn_b,
                severity=severity,
                explanation=(
                    "Opposite claims with "
                    f"{len(common_words)} "
                    "common terms"
                ),
            )

        return None

    # =========================================================================
    # Consistency Score
    # =========================================================================

    def _calculate_consistency_score(
        self,
        contradictions: List[
            Contradiction
        ],
        total_claims: int,
    ) -> float:
        """
        Calculate consistency score.
        """

        if total_claims == 0:
            return 1.0

        severity_weights = {
            "low": 0.3,
            "medium": 0.6,
            "high": 1.0,
        }

        weighted_score = sum(
            severity_weights.get(
                c.severity,
                0.5,
            )
            for c in contradictions
        )

        normalized = min(
            weighted_score
            / (total_claims * 0.2),
            1.0,
        )

        return round(
            max(0.0, 1.0 - normalized),
            2,
        )

    # =========================================================================
    # Recommendation
    # =========================================================================

    def _generate_recommendation(
        self,
        contradictions: List[
            Contradiction
        ],
        consistency_score: float,
    ) -> str:
        """
        Generate recommendation.
        """

        if consistency_score >= 0.8:

            return (
                "Outputs are largely consistent. "
                "Minor review recommended."
            )

        elif consistency_score >= 0.5:

            return (
                "Moderate inconsistencies detected "
                f"({len(contradictions)} issues). "
                "Review flagged claims before synthesis."
            )

        else:

            return (
                "Significant contradictions found "
                f"({len(contradictions)} issues). "
                "Synthesis should prioritize most "
                "recent claims or flag for human review."
            )

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def _execute(
        self,
        outputs: List[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute self-reflection.
        """

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        if not outputs:

            raise ToolInputError(
                "No outputs provided for reflection"
            )

        if not isinstance(outputs, list):

            raise ToolInputError(
                "Outputs must be a list of strings"
            )

        # ---------------------------------------------------------------------
        # Limit History
        # ---------------------------------------------------------------------

        outputs = outputs[
            -self.max_history_turns:
        ]

        # ---------------------------------------------------------------------
        # Detect Contradictions
        # ---------------------------------------------------------------------

        contradictions = (
            self._detect_contradictions(
                outputs
            )
        )

        # ---------------------------------------------------------------------
        # Total Claims
        # ---------------------------------------------------------------------

        total_claims = sum(
            len(
                self._extract_claims(o)
            )
            for o in outputs
        )

        # ---------------------------------------------------------------------
        # Score
        # ---------------------------------------------------------------------

        consistency_score = (
            self._calculate_consistency_score(
                contradictions,
                total_claims,
            )
        )

        # ---------------------------------------------------------------------
        # Recommendation
        # ---------------------------------------------------------------------

        recommendation = (
            self._generate_recommendation(
                contradictions,
                consistency_score,
            )
        )

        return {
            "contradictions_found": [
                {
                    "claim_a": c.claim_a,
                    "claim_b": c.claim_b,
                    "turn": c.turn,
                    "severity": c.severity,
                    "explanation": c.explanation,
                }
                for c in contradictions
            ],

            "consistency_score": (
                consistency_score
            ),

            "recommendation": recommendation,
        }