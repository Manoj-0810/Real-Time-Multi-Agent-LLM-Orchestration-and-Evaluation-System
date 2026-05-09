# =============================================================================
# MEGA AI — Meta Agent (Self-Improving Loop)
# =============================================================================
# Reads eval failure cases after each run.
# Identifies worst-performing prompt by dimension.
# Proposes rewrite with structured diff + justification.
# NEVER auto-applies — stores for human approval.
# =============================================================================

from __future__ import annotations

import difflib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.base import AgentOutput, BaseAgent
from app.schemas import ContextObject

logger = logging.getLogger(__name__)


class MetaAgent(BaseAgent):
    """Meta agent for self-improving prompt optimization.
    
    This agent:
    1. Reads evaluation results after each run
    2. Identifies worst-performing dimensions
    3. Finds the agent responsible
    4. Proposes prompt rewrites
    5. Generates structured diffs
    6. NEVER auto-applies — stores for human approval
    
    What it DOES:
    - Propose prompt improvements based on data
    - Generate structured diffs
    - Justify each proposal
    - Track performance deltas
    
    What it DOES NOT do:
    - Auto-apply any changes (human approval required)
    - Modify prompts in production without review
    - Make changes without justification
    - Override human decisions
    
    Attributes:
        agent_id: "meta"
        description: Self-improving prompt optimization
    """
    
    agent_id = "meta"
    description = "Proposes prompt improvements from evaluation failures — human approval required"
    
    # Minimum score improvement to propose
    MIN_IMPROVEMENT: float = 0.10
    
    # Agent default prompts (what would be in production)
    AGENT_PROMPTS: Dict[str, str] = {
        "orchestrator": """You are the master orchestrator. Analyze queries and route to sub-agents.
Consider: query complexity, required tools, agent dependencies.
Always provide structured routing with justifications.""",
        "decomposition": """Break down queries into sub-tasks with dependency graphs.
Identify ambiguous terms, extract sub-questions, create typed tasks.""",
        "rag": """Perform multi-hop retrieval. Must retrieve at least 2 chunks.
Cite sources for every claim. Use pgvector for similarity search.""",
        "critique": """Review all agent outputs. Score each claim 0-1.
Flag specific text spans with reasons. Be strict on unsupported claims.""",
        "synthesis": """Merge all agent outputs. Resolve contradictions.
Build provenance map. Attribute every sentence to source.""",
        "compression": """Compress context when budget exceeded.
Preserve structured data losslessly. Remove filler lossily.
Log compression ratio.""",
    }
    
    async def _execute(self, ctx: ContextObject) -> AgentOutput:
        """Execute meta-analysis and propose improvements.
        
        Steps:
        1. Check if evaluation results are in context
        2. Identify worst-performing dimension
        3. Find responsible agent
        4. Propose prompt rewrite
        5. Generate structured diff
        6. Store for human approval
        
        Args:
            ctx: Shared context with evaluation results.
            
        Returns:
            AgentOutput with proposal.
        """
        # Step 1: Check for evaluation results
        eval_results = ctx.metadata.get("eval_results")
        if not eval_results:
            return AgentOutput(
                agent_id=self.agent_id,
                content=json.dumps({
                    "status": "no_eval_data",
                    "message": "No evaluation results in context. Run evaluation first.",
                    "proposals": [],
                }),
                confidence=0.0,
            )
        
        # Step 2: Identify worst dimension
        worst_dimension = self._identify_worst_dimension(eval_results)
        
        if not worst_dimension:
            return AgentOutput(
                agent_id=self.agent_id,
                content=json.dumps({
                    "status": "all_good",
                    "message": "All dimensions above threshold. No improvements needed.",
                    "proposals": [],
                }),
                confidence=0.95,
            )
        
        # Step 3: Find responsible agent
        responsible_agent = self._find_responsible_agent(
            worst_dimension["dimension"],
        )
        
        # Step 4: Propose rewrite
        proposal = await self._propose_rewrite(
            responsible_agent,
            worst_dimension,
        )
        
        # Step 5: Store in context for approval
        if "proposals" not in ctx.metadata:
            ctx.metadata["proposals"] = []
        ctx.metadata["proposals"].append(proposal)
        
        # Format output
        output = {
            "status": "proposal_generated",
            "analysis": {
                "worst_dimension": worst_dimension["dimension"],
                "current_score": worst_dimension["score"],
                "threshold": worst_dimension["threshold"],
                "gap": worst_dimension["gap"],
            },
            "proposal": proposal,
            "disclaimer": (
                "This proposal requires HUMAN APPROVAL before application. "
                "The Meta Agent NEVER auto-applies changes."
            ),
        }
        
        output_content = json.dumps(output, indent=2)
        
        return AgentOutput(
            agent_id=self.agent_id,
            content=output_content,
            confidence=proposal.get("expected_improvement", 0.0),
            token_count=len(output_content.split()),
        )
    
    def _identify_worst_dimension(
        self,
        eval_results: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Identify worst-performing scoring dimension.
        
        Args:
            eval_results: Evaluation results dict.
            
        Returns:
            Dict with dimension, score, threshold, gap — or None if all good.
        """
        # Dimension thresholds
        thresholds = {
            "answer_correctness": 0.6,
            "citation_accuracy": 0.5,
            "contradiction_resolution": 0.6,
            "tool_selection_efficiency": 0.7,
            "context_budget_compliance": 0.7,
            "critique_agreement_rate": 0.6,
        }
        
        # Extract dimension scores
        dimension_scores = eval_results.get("dimension_scores", {})
        if not dimension_scores:
            # Try to compute from test cases
            test_cases = eval_results.get("test_cases", [])
            if test_cases:
                dimension_scores = self._compute_dimension_scores(test_cases)
        
        # Find worst dimension
        worst = None
        worst_gap = 0
        
        for dimension, score in dimension_scores.items():
            threshold = thresholds.get(dimension, 0.6)
            gap = threshold - score
            
            if gap > worst_gap and gap > 0:
                worst_gap = gap
                worst = {
                    "dimension": dimension,
                    "score": score,
                    "threshold": threshold,
                    "gap": gap,
                }
        
        return worst
    
    def _compute_dimension_scores(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute aggregate dimension scores from test cases.
        
        Args:
            test_cases: List of test case results.
            
        Returns:
            Dict mapping dimension -> average score.
        """
        dimension_totals: Dict[str, List[float]] = {}
        
        for case in test_cases:
            scores = case.get("scores", {})
            for dimension, score_data in scores.items():
                if dimension not in dimension_totals:
                    dimension_totals[dimension] = []
                
                if isinstance(score_data, dict):
                    dimension_totals[dimension].append(score_data.get("score", 0))
                elif isinstance(score_data, (int, float)):
                    dimension_totals[dimension].append(score_data)
        
        return {
            dim: sum(scores) / len(scores) if scores else 0.0
            for dim, scores in dimension_totals.items()
        }
    
    def _find_responsible_agent(self, dimension: str) -> str:
        """Find the agent most responsible for a dimension.
        
        Args:
            dimension: Scoring dimension.
            
        Returns:
            Agent ID string.
        """
        dimension_to_agent = {
            "answer_correctness": "synthesis",
            "citation_accuracy": "rag",
            "contradiction_resolution": "synthesis",
            "tool_selection_efficiency": "orchestrator",
            "context_budget_compliance": "orchestrator",
            "critique_agreement_rate": "critique",
        }
        
        return dimension_to_agent.get(dimension, "orchestrator")
    
    async def _propose_rewrite(
        self,
        agent_id: str,
        worst_dimension: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Propose a prompt rewrite for an agent.
        
        Args:
            agent_id: Agent to improve.
            worst_dimension: Worst dimension info.
            
        Returns:
            Proposal dict.
        """
        original_prompt = self.AGENT_PROMPTS.get(agent_id, "")
        
        # Generate improved prompt
        improved_prompt = await self._generate_improved_prompt(
            agent_id, original_prompt, worst_dimension,
        )
        
        # Generate diff
        diff = self._generate_diff(original_prompt, improved_prompt)
        
        # Estimate improvement
        expected_improvement = min(worst_dimension["gap"] * 0.7, 0.20)
        
        proposal = {
            "id": f"rewrite_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "agent_id": agent_id,
            "dimension": worst_dimension["dimension"],
            "original_prompt": original_prompt,
            "proposed_prompt": improved_prompt,
            "diff": diff,
            "justification": (
                f"Dimension '{worst_dimension['dimension']}' scored "
                f"{worst_dimension['score']:.2f} (threshold: {worst_dimension['threshold']:.2f}). "
                f"Agent '{agent_id}' is responsible for this dimension. "
                f"Proposed changes should improve by ~{expected_improvement:.0%}."
            ),
            "expected_improvement": round(expected_improvement, 4),
            "status": "pending_approval",
            "created_at": datetime.utcnow().isoformat(),
        }
        
        return proposal
    
    async def _generate_improved_prompt(
        self,
        agent_id: str,
        original: str,
        worst_dimension: Dict[str, Any],
    ) -> str:
        """Generate an improved prompt.
        
        Uses LLM to generate improved version with specific guidance.
        
        Args:
            agent_id: Agent being improved.
            original: Original prompt.
            worst_dimension: Dimension info.
            
        Returns:
            Improved prompt string.
        """
        dimension = worst_dimension["dimension"]
        
        # Dimension-specific improvements
        improvements = {
            "answer_correctness": "Focus on accuracy. Verify facts before including. Cite sources. Double-check numbers and dates.",
            "citation_accuracy": "Always cite the exact source. Include chunk ID. Verify citation matches content. Never fabricate sources.",
            "contradiction_resolution": "Actively identify conflicts. Prioritize recent data. Flag unresolved contradictions. Explain resolution strategy.",
            "tool_selection_efficiency": "Only use tools when necessary. Prefer simple solutions. Avoid redundant tool calls. Consider cost of each call.",
            "context_budget_compliance": "Track tokens carefully. Compress early. Prioritize essential information. Respect budget limits strictly.",
            "critique_agreement_rate": "Be specific in critiques. Reference exact claims. Provide evidence for disagreements. Score consistently.",
        }
        
        improvement_guidance = improvements.get(dimension, "Improve overall quality and accuracy.")
        
        prompt = f"""Improve the following system prompt for better performance.

Agent: {agent_id}
Dimension to improve: {dimension}
Current score: {worst_dimension['score']:.2f}
Target: {worst_dimension['threshold']:.2f}

Guidance: {improvement_guidance}

Original prompt:
{original}

Provide an improved version that:
1. Addresses the specific weakness
2. Adds clear instructions for the problem area
3. Maintains existing strengths
4. Is concise and actionable

Improved prompt:"""
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.4,
                job_id="meta_rewrite",
                agent_id=self.agent_id,
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM rewrite failed: {e}", extra={"agent_id": agent_id})
            # Fallback: append guidance to original
            return f"{original}\n\nADDITIONAL GUIDANCE: {improvement_guidance}"
    
    def _generate_diff(self, original: str, improved: str) -> str:
        """Generate unified diff between original and improved prompts.
        
        Args:
            original: Original prompt.
            improved: Improved prompt.
            
        Returns:
            Unified diff string.
        """
        original_lines = original.splitlines(keepends=True)
        improved_lines = improved.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            improved_lines,
            fromfile="original",
            tofile="improved",
            lineterm="",
        )
        
        return "".join(diff)
    
    def get_pending_proposals(self, ctx: ContextObject) -> List[Dict[str, Any]]:
        """Get all pending proposals from context.
        
        Args:
            ctx: Shared context.
            
        Returns:
            List of pending proposals.
        """
        proposals = ctx.metadata.get("proposals", [])
        return [p for p in proposals if p.get("status") == "pending_approval"]
