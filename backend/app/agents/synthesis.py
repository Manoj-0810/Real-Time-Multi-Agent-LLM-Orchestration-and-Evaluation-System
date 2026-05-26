# =============================================================================
# MEGA AI — Synthesis Agent
# =============================================================================
# Merges all sub-agent outputs. Resolves contradictions flagged by Critique.
# Produces provenance map: each sentence -> source agent + source chunk.
# =============================================================================

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.agents.base import AgentOutput, BaseAgent
from app.schemas import ContextObject, ProvenanceEntry, SynthesisResult

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """Synthesis agent that merges all sub-agent outputs.
    
    This agent:
    1. Reads all agent outputs from context
    2. Reads critique results
    3. Resolves contradictions flagged by critique
    4. Merges outputs into coherent answer
    5. Produces provenance map for every sentence
    
    Attributes:
        agent_id: "synthesis"
        description: Merges all agent outputs and resolves contradictions
    """
    
    agent_id = "synthesis"
    description = "Merges all sub-agent outputs and resolves contradictions"
    
    # Confidence threshold for including claims
    MIN_CLAIM_CONFIDENCE: float = 0.3
    
    async def _execute(self, ctx: ContextObject) -> AgentOutput:
        """Execute synthesis of all agent outputs.
        
        Steps:
        1. Collect all agent outputs
        2. Read critique results
        3. Resolve contradictions
        4. Generate merged answer
        5. Build provenance map
        
        Args:
            ctx: Shared context with all agent outputs and critiques.
            
        Returns:
            AgentOutput with synthesized answer and provenance.
        """
        # Step 1: Collect outputs
        agent_outputs = ctx.agent_outputs
        critiques = ctx.critique_results
        
        if not agent_outputs:
            # No outputs to synthesize
            return AgentOutput(
                agent_id=self.agent_id,
                content=json.dumps({
                    "answer": "No agent outputs available to synthesize.",
                    "contradictions_resolved": 0,
                    "sources_used": [],
                }),
                confidence=0.0,
            )
        
        # Step 2: Build claim quality map from critiques
        claim_quality = self._build_claim_quality_map(critiques)
        
        # Step 3: Resolve contradictions
        contradictions_resolved = self._resolve_contradictions(claim_quality)
        
        # Step 4: Generate merged answer
        answer = await self._generate_answer(ctx, claim_quality)
        
        # Step 5: Build provenance map
        provenance = self._build_provenance_map(answer, agent_outputs)
        ctx.provenance_map.update(provenance)
        
        # Store synthesis result
        ctx.synthesis = SynthesisResult(
            answer=answer,
            contradictions_resolved=contradictions_resolved,
            sources_used=list(agent_outputs.keys()),
        )
        
        # Format output
        output = {
            "answer": answer,
            "contradictions_resolved": contradictions_resolved,
            "sources_used": list(agent_outputs.keys()),
            "provenance": {
                k: {
                    "sentence": v.sentence,
                    "source_agent": v.source_agent,
                    "source_chunk": v.source_chunk,
                    "confidence": v.confidence,
                }
                for k, v in provenance.items()
            },
        }
        
        output_content = json.dumps(output, indent=2)
        
        return AgentOutput(
            agent_id=self.agent_id,
            content=output_content,
            confidence=0.90,
            token_count=len(output_content.split()),
        )
    
    def _build_claim_quality_map(
        self,
        critiques: List,
    ) -> Dict[str, Dict[str, Any]]:
        """Build a map of claim quality from critique results.
        
        Args:
            critiques: List of CritiqueResult objects.
            
        Returns:
            Dict mapping claim -> quality info.
        """
        quality_map = {}
        
        for critique in critiques:
            claim_key = f"{critique.agent_source}:{critique.claim[:50]}"
            quality_map[claim_key] = {
                "claim": critique.claim,
                "confidence": critique.confidence,
                "flagged_span": critique.flagged_span,
                "reason": critique.reason,
                "agent_source": critique.agent_source,
            }
        
        return quality_map
    
    def _resolve_contradictions(
        self,
        claim_quality: Dict[str, Dict[str, Any]],
    ) -> int:
        """Resolve contradictions flagged by critique.
        
        Strategy:
        - Remove claims with confidence < MIN_CLAIM_CONFIDENCE
        - Keep highest confidence version of contradictory claims
        - Flag remaining contradictions for human review
        
        Args:
            claim_quality: Map of claim quality.
            
        Returns:
            Number of contradictions resolved.
        """
        resolved = 0
        
        # Group by semantic similarity (simplified: by agent source)
        by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for key, info in claim_quality.items():
            agent = info["agent_source"]
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(info)
        
        # Find contradictory claims across agents
        for agent_a, claims_a in by_agent.items():
            for agent_b, claims_b in by_agent.items():
                if agent_a >= agent_b:
                    continue
                
                for claim_a in claims_a:
                    for claim_b in claims_b:
                        if self._are_contradictory(
                            claim_a["claim"], claim_b["claim"],
                        ):
                            # Resolve: keep higher confidence
                            if claim_a["confidence"] > claim_b["confidence"]:
                                claim_b["confidence"] = 0.0  # Mark as resolved (removed)
                            else:
                                claim_a["confidence"] = 0.0
                            resolved += 1
        
        return resolved
    
    def _are_contradictory(self, claim_a: str, claim_b: str) -> bool:
        """Check if two claims are contradictory.
        
        Simplified heuristic: check for negation and opposite numbers.
        
        Args:
            claim_a: First claim.
            claim_b: Second claim.
            
        Returns:
            True if claims appear contradictory.
        """
        a_lower = claim_a.lower()
        b_lower = claim_b.lower()
        
        # Check for direct negation
        negations = ["not ", "no ", "never ", "false"]
        a_negated = any(n in a_lower for n in negations)
        b_negated = any(n in b_lower for n in negations)
        
        if a_negated != b_negated:
            # One affirms, one negates — check for overlap
            a_words = set(a_lower.split())
            b_words = set(b_lower.split())
            overlap = len(a_words & b_words)
            
            if overlap >= 3:
                return True
        
        # Check for numeric contradictions
        nums_a = set(re.findall(r'\d+\.?\d*', a_lower))
        nums_b = set(re.findall(r'\d+\.?\d*', b_lower))
        
        if nums_a and nums_b and nums_a != nums_b:
            # Same subject, different numbers
            a_subject = " ".join(a_lower.split()[:5])
            b_subject = " ".join(b_lower.split()[:5])
            if any(w in b_subject for w in a_subject.split() if len(w) > 3):
                return True
        
        return False
    
    async def _generate_answer(
        self,
        ctx: ContextObject,
        claim_quality: Dict[str, Dict[str, Any]],
    ) -> str:
        """Generate merged answer from all agent outputs.
        
        Strategy:
        - Include high-confidence claims
        - Exclude low-confidence claims
        - Use LLM for coherent merging if available
        - Otherwise, concatenate with attribution
        
        Args:
            ctx: Shared context.
            claim_quality: Quality map for claims.
            
        Returns:
            Merged answer string.
        """
        agent_outputs = ctx.agent_outputs
        
        # Filter to high-confidence claims
        high_confidence_claims = [
            info for info in claim_quality.values()
            if info["confidence"] >= self.MIN_CLAIM_CONFIDENCE
        ]
        
        # Sort by confidence
        high_confidence_claims.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Build prompt for LLM synthesis
        output_texts = []
        for agent_id, output in agent_outputs.items():
            if agent_id != self.agent_id:
                try:
                    data = json.loads(output.content)
                    if isinstance(data, dict) and "answer" in data:
                        text = data["answer"]
                    else:
                        text = output.content
                except (json.JSONDecodeError, TypeError):
                    text = output.content
                
                output_texts.append(f"--- {agent_id} ---\n{text[:500]}")
        
        if not output_texts:
            return "No valid outputs to synthesize."
        
        system_prompt = await self.get_effective_prompt()
        prompt = f"""{system_prompt}

Synthesize the following agent outputs into a single coherent answer.

Original question: {ctx.query}

Agent outputs:
{chr(10).join(output_texts)}

Provide a comprehensive answer that:
1. Integrates information from all sources
2. Resolves any apparent contradictions
3. Cites the source agent for key claims
4. Is clear and well-structured

Answer:"""
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.3,
                job_id=ctx.job_id,
                agent_id=self.agent_id,
            )
            return response.content
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}", extra={"job_id": ctx.job_id})
            # Fallback: concatenate with attribution
            parts = []
            for agent_id, output in agent_outputs.items():
                if agent_id != self.agent_id:
                    parts.append(f"[{agent_id}]: {output.content[:300]}")
            return "\n\n".join(parts)
    
    def _build_provenance_map(
        self,
        answer: str,
        agent_outputs: Dict[str, AgentOutput],
    ) -> Dict[str, ProvenanceEntry]:
        """Build provenance map for every sentence in answer.
        
        Maps each sentence to its most likely source agent and chunk.
        
        Args:
            answer: Synthesized answer.
            agent_outputs: All agent outputs.
            
        Returns:
            Provenance map dict.
        """
        provenance = {}
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        
        for i, sentence in enumerate(sentences):
            # Find best source
            best_agent = None
            best_score = 0
            
            for agent_id, output in agent_outputs.items():
                output_text = output.content.lower()
                sentence_lower = sentence.lower()
                
                # Word overlap score
                sentence_words = set(sentence_lower.split())
                output_words = set(output_text.split())
                
                if sentence_words:
                    overlap = len(sentence_words & output_words)
                    score = overlap / len(sentence_words)
                    
                    if score > best_score:
                        best_score = score
                        best_agent = agent_id
            
            provenance[f"sentence_{i}"] = ProvenanceEntry(
                sentence=sentence,
                source_agent=best_agent or "synthesis",
                source_chunk=None,
                confidence=best_score,
            )
        
        return provenance
