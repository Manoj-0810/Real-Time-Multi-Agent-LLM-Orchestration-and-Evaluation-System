# =============================================================================
# MEGA AI — Evaluation Pipeline
# =============================================================================
# Runs all 15 test cases, scores results, stores in DB, and handles
# re-eval of failed cases with approved prompts.
# =============================================================================

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.evaluation.scorer import EvalScorer
from app.evaluation.test_cases import ALL_TEST_CASES, get_test_cases

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Runs the complete evaluation pipeline.
    
    This pipeline:
    1. Loads all 15 test cases
    2. Runs each case through the system
    3. Scores across 6 dimensions
    4. Stores results in database
    5. Generates diff from previous run
    6. Triggers meta-agent for failed dimensions
    
    Attributes:
        agent_factory: Factory for creating agent instances.
        max_retries: Maximum retries per test case.
        timeout_seconds: Timeout per test case.
    """
    
    def __init__(
        self,
        agent_factory: Any,
        max_retries: int = 2,
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize evaluation pipeline.
        
        Args:
            agent_factory: Factory callable for creating agent instances.
            max_retries: Maximum retries per case.
            timeout_seconds: Timeout per case in seconds.
        """
        self.agent_factory = agent_factory
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.scorer = EvalScorer()
        self.run_history: List[Dict[str, Any]] = []
    
    async def run_evaluation(
        self,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run evaluation on test cases.
        
        Args:
            category: Optional category filter ("baseline", "ambiguous", "adversarial").
            
        Returns:
            Dict with run results and summary.
        """
        run_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting evaluation run {run_id}", extra={"category": category})
        
        # Get test cases
        test_cases = get_test_cases(category)
        
        results = []
        for test_case in test_cases:
            logger.info(
                f"Running test case: {test_case.id}",
                extra={"category": test_case.category},
            )
            
            try:
                result = await self._run_test_case(test_case)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Test case {test_case.id} failed: {e}",
                    exc_info=True,
                )
                results.append({
                    "test_case_id": test_case.id,
                    "status": "error",
                    "error": str(e),
                })
        
        # Compute summary
        total_time = time.time() - start_time
        summary = self.scorer.get_summary()
        
        # Add run metadata
        run_result = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "category": category or "all",
            "total_cases": len(test_cases),
            "completed_cases": len([r for r in results if r.get("status") != "error"]),
            "failed_cases": len([r for r in results if r.get("status") == "error"]),
            "total_time_seconds": round(total_time, 2),
            "summary": summary,
            "results": results,
        }
        
        # Compute diff from previous run
        if self.run_history:
            prev_run = self.run_history[-1]
            run_result["diff_from_previous"] = self._compute_diff(
                prev_run.get("summary", {}),
                summary,
            )
        
        self.run_history.append(run_result)
        
        logger.info(
            f"Evaluation run {run_id} complete: "
            f"{run_result['completed_cases']}/{run_result['total_cases']} cases, "
            f"avg score: {summary.get('overall_average', 0):.3f}",
        )
        
        return run_result
    
    async def _run_test_case(
        self,
        test_case: Any,
    ) -> Dict[str, Any]:
        """Run a single test case.
        
        Args:
            test_case: Test case to run.
            
        Returns:
            Dict with case results.
        """
        case_start = time.time()
        
        # Create fresh agents for isolation
        agents = self.agent_factory()
        orchestrator = agents.get("orchestrator")
        
        if not orchestrator:
            return {
                "test_case_id": test_case.id,
                "status": "error",
                "error": "Orchestrator not available",
            }
        
        try:
            # Execute query through orchestrator
            from app.schemas import ContextObject
            
            ctx = ContextObject(
                job_id=str(uuid.uuid4()),
                query=test_case.query,
            )
            
            # Run orchestrator
            orchestrator_output = await orchestrator.execute(ctx)
            ctx.agent_outputs["orchestrator"] = orchestrator_output
            
            # Execute sub-agents based on routing plan
            await self._execute_agent_chain(ctx, agents)
            
            # Collect outputs
            agent_outputs = ctx.agent_outputs
            tool_calls = [
                {
                    "tool_name": name,
                    "result": str(result),
                }
                for name, result in ctx.tool_results.items()
            ]
            
            # Score results
            budget_violations = ctx.budget_status.violations if ctx.budget_status else []
            
            score = self.scorer.score_test_case(
                test_case=test_case,
                agent_outputs=agent_outputs,
                tool_calls=tool_calls,
                budget_violations=budget_violations,
            )
            
            case_time = time.time() - case_start
            
            return {
                "test_case_id": test_case.id,
                "status": "completed",
                "query": test_case.query,
                "category": test_case.category,
                "execution_time_seconds": round(case_time, 2),
                "scores": score.to_dict(),
                "agents_invoked": list(agent_outputs.keys()),
                "tools_used": tool_calls,
            }
            
        except Exception as e:
            logger.error(f"Test case execution failed: {e}", exc_info=True)
            return {
                "test_case_id": test_case.id,
                "status": "error",
                "error": str(e),
                "query": test_case.query,
                "category": test_case.category,
            }
    
    async def _execute_agent_chain(
        self,
        ctx: Any,
        agents: Dict[str, Any],
    ) -> None:
        """Execute the agent chain based on orchestrator routing.
        
        Args:
            ctx: Context object.
            agents: Dict of agent instances.
        """
        # Get routing from orchestrator output
        orchestrator_output = ctx.agent_outputs.get("orchestrator", {})
        
        try:
            content = json.loads(orchestrator_output.content)
            execution_order = content.get("execution_order", [])
        except (json.JSONDecodeError, AttributeError):
            # Fallback: default order
            execution_order = ["decomposition", "rag", "critique", "synthesis"]
        
        # Execute agents in order
        for agent_id in execution_order:
            agent = agents.get(agent_id)
            if agent and agent_id != "orchestrator":
                try:
                    logger.info(f"Executing agent: {agent_id}", extra={"job_id": ctx.job_id})
                    output = await agent.execute(ctx)
                    ctx.agent_outputs[agent_id] = output
                except Exception as e:
                    logger.error(
                        f"Agent {agent_id} failed: {e}",
                        extra={"job_id": ctx.job_id},
                    )
    
    def _compute_diff(
        self,
        previous: Dict[str, Any],
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute diff between two evaluation runs.
        
        Args:
            previous: Previous run summary.
            current: Current run summary.
            
        Returns:
            Diff dict.
        """
        diff = {
            "overall_average_delta": round(
                current.get("overall_average", 0) - previous.get("overall_average", 0),
                4,
            ),
            "category_deltas": {},
            "dimension_deltas": {},
        }
        
        # Category deltas
        prev_cats = previous.get("category_scores", {})
        curr_cats = current.get("category_scores", {})
        for cat in set(list(prev_cats.keys()) + list(curr_cats.keys())):
            diff["category_deltas"][cat] = round(
                curr_cats.get(cat, 0) - prev_cats.get(cat, 0),
                4,
            )
        
        # Dimension deltas
        prev_dims = previous.get("dimension_scores", {})
        curr_dims = current.get("dimension_scores", {})
        for dim in set(list(prev_dims.keys()) + list(curr_dims.keys())):
            diff["dimension_deltas"][dim] = round(
                curr_dims.get(dim, 0) - prev_dims.get(dim, 0),
                4,
            )
        
        # Regression detection
        diff["regression_detected"] = any(
            v < -0.05 for v in diff["dimension_deltas"].values()
        )
        
        return diff
    
    def get_run_history(self) -> List[Dict[str, Any]]:
        """Get all evaluation run history.
        
        Returns:
            List of run result dicts.
        """
        return self.run_history
    
    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Get the most recent evaluation run.
        
        Returns:
            Latest run result or None.
        """
        if self.run_history:
            return self.run_history[-1]
        return None
