# =============================================================================
# MEGA AI — Critique Agent
# =============================================================================
# Reviews output of EVERY other agent. Assigns structured confidence score
# PER CLAIM (not whole output). Flags specific text SPANS it disagrees with.
# Output schema: {claim, confidence: 0-1, flagged_span, reason}
# =============================================================================

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agents.base import (
    AgentError,
    AgentOutput,
    BaseAgent,
)

from app.schemas import (
    ContextObject,
    CritiqueResult,
)

logger = logging.getLogger(__name__)


class CritiqueAgent(BaseAgent):
    """
    Critique agent that reviews outputs
    from all other agents.
    """

    agent_id = "critique"

    description = (
        "Reviews outputs of all other agents "
        "with per-claim confidence scoring"
    )

    # =========================================================================
    # Confidence Thresholds
    # =========================================================================

    HIGH_CONFIDENCE: float = 0.8

    MEDIUM_CONFIDENCE: float = 0.5

    LOW_CONFIDENCE: float = 0.3

    # =========================================================================
    # Heuristic Patterns
    # =========================================================================

    RED_FLAG_PATTERNS: List[str] = [
        r"\ball\b",
        r"\bnever\b",
        r"\balways\b",
        r"\bimpossible\b",
        r"\bdefinitely\b",
        r"\bcertainly\b",
        r"\bwithout doubt\b",
    ]

    GREEN_FLAG_PATTERNS: List[str] = [
        r"\baccording to\b",
        r"\bresearch shows\b",
        r"\bstudy found\b",
        r"\bdata indicates\b",
        r"\btypically\b",
        r"\bgenerally\b",
        r"\bin most cases\b",
    ]

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def _execute(
        self,
        ctx: ContextObject,
    ) -> AgentOutput:
        """
        Execute critique pass on all agent outputs.
        """

        agent_outputs = ctx.agent_outputs

        # ---------------------------------------------------------------------
        # No Outputs
        # ---------------------------------------------------------------------

        if not agent_outputs:

            return AgentOutput(
                agent_id=self.agent_id,

                content=json.dumps({
                    "critiques": [],
                    "summary": (
                        "No agent outputs to critique yet"
                    ),
                    "overall_confidence": 0.0,
                }),

                confidence=0.0,
            )

        # ---------------------------------------------------------------------
        # Critique Outputs
        # ---------------------------------------------------------------------

        all_critiques: List[CritiqueResult] = []

        for agent_id, output in agent_outputs.items():

            if agent_id == self.agent_id:
                continue

            critiques = await self._critique_agent_output(
                agent_id,
                output,
                ctx,
            )

            all_critiques.extend(critiques)

        # ---------------------------------------------------------------------
        # Store In Context
        # ---------------------------------------------------------------------

        ctx.critique_results = all_critiques

        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------

        summary = self._generate_summary(
            all_critiques
        )

        # ---------------------------------------------------------------------
        # Output
        # ---------------------------------------------------------------------

        output = {
            "critiques": [
                {
                    "claim": c.claim,
                    "confidence": c.confidence,
                    "flagged_span": c.flagged_span,
                    "reason": c.reason,
                    "agent_source": c.agent_source,
                }
                for c in all_critiques
            ],

            "summary": summary,

            "total_claims_reviewed": len(
                all_critiques
            ),

            "claims_flagged": sum(
                1
                for c in all_critiques
                if c.confidence < self.MEDIUM_CONFIDENCE
            ),

            "avg_confidence": (
                (
                    sum(
                        c.confidence
                        for c in all_critiques
                    )
                    / len(all_critiques)
                )
                if all_critiques
                else 0
            ),
        }

        output_content = json.dumps(
            output,
            indent=2,
        )

        return AgentOutput(
            agent_id=self.agent_id,
            content=output_content,
            confidence=output["avg_confidence"],
            token_count=len(
                output_content.split()
            ),
        )

    # =========================================================================
    # Agent Critique
    # =========================================================================

    async def _critique_agent_output(
        self,
        agent_id: str,
        output: AgentOutput,
        ctx: ContextObject,
    ) -> List[CritiqueResult]:
        """
        Critique a single agent output using LLM-augmented critique or fallback heuristics.
        """

        # 1. Attempt LLM-augmented critique if gateway has providers
        if self.llm and getattr(self.llm, "_providers", None):
            try:
                system_prompt = await self.get_effective_prompt()
                prompt = f"""{system_prompt}

You are reviewing the output of another agent in our multi-agent pipeline.
Agent ID to Critique: {agent_id}
Agent Output Content:
{output.content}

Your task is to critique this output. Identify key factual claims and evaluate them carefully.
For each key factual claim:
1. Assess correctness and reliability, scoring it from 0.0 to 1.0 (where 1.0 is completely verified and robust, and 0.0 is completely false/hallucinated).
2. Identify the specific incorrect or unsupported text span as 'flagged_span' (or set it to null if the claim is fully accurate and supported).
3. Provide a clear reason explaining your confidence score.

Your response MUST be a valid JSON object matching this exact format:
{{
  "critiques": [
    {{
      "claim": "The specific claim extracted.",
      "confidence": 0.85,
      "flagged_span": "Substring of flagged text, or null if fully accurate.",
      "reason": "Justification for confidence."
    }}
  ]
}}

Provide ONLY the raw JSON output. Do NOT include markdown formatting, backticks, or any explanation wrapper.
"""
                response_content = await self._call_llm(prompt, ctx, max_tokens=1536, temperature=0.2)
                cleaned = response_content.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
                    cleaned = re.sub(r"\n```$", "", cleaned)
                    cleaned = cleaned.strip()

                data = json.loads(cleaned)
                critiques = []
                for item in data.get("critiques", []):
                    claim = item.get("claim", "").strip()
                    if not claim:
                        continue
                    confidence = float(item.get("confidence", 0.5))
                    confidence = max(0.0, min(1.0, confidence))
                    flagged_span = item.get("flagged_span")
                    if flagged_span:
                        flagged_span = str(flagged_span).strip()
                        if flagged_span.lower() in ("null", "none", ""):
                            flagged_span = None
                    reason = item.get("reason", "").strip() or "Evaluated by LLM Critique"

                    critiques.append(
                        CritiqueResult(
                            claim=claim,
                            confidence=confidence,
                            flagged_span=flagged_span,
                            reason=reason,
                            agent_source=agent_id,
                        )
                    )
                if critiques:
                    logger.info(f"CritiqueAgent successfully generated {len(critiques)} LLM-augmented critiques for agent {agent_id}")
                    return critiques
            except Exception as e:
                logger.warning(f"LLM-augmented critique failed or returned invalid JSON ({e}). Falling back to heuristic critique.", exc_info=True)

        # 2. Fallback to heuristic critique
        critiques = []
        claims = self._extract_claims(
            output.content
        )

        for claim in claims:

            confidence, flagged_span, reason = (
                self._evaluate_claim(
                    claim,
                    agent_id,
                    output,
                )
            )

            critiques.append(
                CritiqueResult(
                    claim=claim,
                    confidence=confidence,
                    flagged_span=flagged_span,
                    reason=reason,
                    agent_source=agent_id,
                )
            )

        return critiques

    # =========================================================================
    # Claim Extraction
    # =========================================================================

    def _extract_claims(
        self,
        content: str,
    ) -> List[str]:
        """
        Extract factual claims from output.
        """

        try:

            data = json.loads(content)

            if isinstance(data, dict):

                text = data.get(
                    "answer",
                    "",
                )

                if not text:
                    text = data.get(
                        "content",
                        "",
                    )

                if not text:

                    text = json.dumps(
                        data,
                        default=str,
                    )

            else:
                text = str(data)

        except (
            json.JSONDecodeError,
            TypeError,
        ):

            text = content

        sentences = re.split(
            r"[.!?]+",
            text,
        )

        claims = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence.split()) < 3:
                continue

            if sentence.startswith(
                ("{", "}", "[", "]")
            ):
                continue

            if sentence.lower().startswith(
                (
                    "note:",
                    "warning:",
                    "error:",
                )
            ):
                continue

            claims.append(sentence)

        return claims[:20]

    # =========================================================================
    # Claim Evaluation
    # =========================================================================

    def _evaluate_claim(
        self,
        claim: str,
        agent_id: str,
        output: AgentOutput,
    ) -> Tuple[
        float,
        Optional[str],
        str,
    ]:
        """
        Evaluate confidence for claim.
        """

        claim_lower = claim.lower()

        confidence = 0.5

        reasons = []

        flagged_span = None

        # ---------------------------------------------------------------------
        # Red Flags
        # ---------------------------------------------------------------------

        red_flags_found = []

        for pattern in self.RED_FLAG_PATTERNS:

            if re.search(
                pattern,
                claim_lower,
            ):

                confidence -= 0.15

                red_flags_found.append(
                    pattern.replace(
                        r"\b",
                        "",
                    )
                )

        if red_flags_found:

            reasons.append(
                "Red flags detected: "
                f"{', '.join(red_flags_found)}"
            )

            flagged_span = (
                claim[:100]
                if len(claim) > 100
                else claim
            )

        # ---------------------------------------------------------------------
        # Green Flags
        # ---------------------------------------------------------------------

        green_flags_found = []

        for pattern in self.GREEN_FLAG_PATTERNS:

            if re.search(
                pattern,
                claim_lower,
            ):

                confidence += 0.10

                green_flags_found.append(
                    pattern.replace(
                        r"\b",
                        "",
                    )
                )

        if green_flags_found:

            reasons.append(
                "Green flags: "
                f"{', '.join(green_flags_found)}"
            )

        # ---------------------------------------------------------------------
        # Source Confidence
        # ---------------------------------------------------------------------

        source_confidence = output.confidence

        if source_confidence < 0.5:

            confidence -= 0.10

            reasons.append(
                "Source agent low confidence "
                f"({source_confidence:.2f})"
            )

        elif source_confidence > 0.8:

            confidence += 0.10

        # ---------------------------------------------------------------------
        # Specificity
        # ---------------------------------------------------------------------

        word_count = len(
            claim.split()
        )

        if word_count < 5:

            confidence -= 0.05

            reasons.append(
                "Claim is very short — "
                "low specificity"
            )

        elif word_count > 20:

            confidence += 0.05

            reasons.append(
                "Detailed claim — "
                "higher verifiability"
            )

        # ---------------------------------------------------------------------
        # Quantitative Data
        # ---------------------------------------------------------------------

        if re.search(r"\d+", claim):

            confidence += 0.05

            reasons.append(
                "Contains quantitative data"
            )

        # ---------------------------------------------------------------------
        # Clamp
        # ---------------------------------------------------------------------

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        # ---------------------------------------------------------------------
        # Flagging
        # ---------------------------------------------------------------------

        if confidence < self.LOW_CONFIDENCE:

            flagged_span = (
                claim[:150]
                if len(claim) > 150
                else claim
            )

            reasons.append(
                "LOW CONFIDENCE — flag for review"
            )

        elif confidence < self.MEDIUM_CONFIDENCE:

            flagged_span = (
                claim[:100]
                if len(claim) > 100
                else claim
            )

            reasons.append(
                "Medium confidence — some concerns"
            )

        reason_str = (
            "; ".join(reasons)
            if reasons
            else "No specific concerns"
        )

        return (
            round(confidence, 2),
            flagged_span,
            reason_str,
        )

    # =========================================================================
    # Summary
    # =========================================================================

    def _generate_summary(
        self,
        critiques: List[CritiqueResult],
    ) -> Dict[str, Any]:
        """
        Generate critique summary.
        """

        if not critiques:

            return {
                "status": "no_critiques",
                "message": "No claims to critique",
            }

        by_agent: Dict[
            str,
            List[CritiqueResult],
        ] = {}

        for c in critiques:

            agent = c.agent_source

            if agent not in by_agent:
                by_agent[agent] = []

            by_agent[agent].append(c)

        agent_summaries = {}

        for agent, agent_critiques in by_agent.items():

            avg_conf = (
                sum(
                    c.confidence
                    for c in agent_critiques
                )
                / len(agent_critiques)
            )

            flagged = sum(
                1
                for c in agent_critiques
                if c.confidence < self.MEDIUM_CONFIDENCE
            )

            agent_summaries[agent] = {
                "claims_reviewed": len(
                    agent_critiques
                ),

                "avg_confidence": round(
                    avg_conf,
                    2,
                ),

                "flagged_claims": flagged,

                "status": (
                    "needs_review"
                    if flagged > 0
                    else "acceptable"
                ),
            }

        overall_avg = (
            sum(c.confidence for c in critiques)
            / len(critiques)
        )

        total_flagged = sum(
            1
            for c in critiques
            if c.confidence < self.MEDIUM_CONFIDENCE
        )

        return {
            "overall_confidence": round(
                overall_avg,
                2,
            ),

            "total_claims": len(
                critiques
            ),

            "total_flagged": total_flagged,

            "by_agent": agent_summaries,

            "status": (
                "needs_review"
                if total_flagged > 0
                else "acceptable"
            ),

            "recommendation": (
                "Review flagged claims before synthesis"
                if total_flagged > 0
                else "Outputs look acceptable for synthesis"
            ),
        }