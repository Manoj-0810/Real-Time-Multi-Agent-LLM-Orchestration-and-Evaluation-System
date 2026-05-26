# =============================================================================
# MEGA AI — Base Agent Class
# =============================================================================
# All agents inherit from BaseAgent. Provides structured logging,
# budget management, and tool access.
# =============================================================================

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.context import ContextBudgetManager, hash_content
from app.llm_gateway import LLMGateway
from app.schemas import (
    AgentOutput,
    ContextObject,
    StructuredLogEntry,
)
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# =============================================================================
# Exceptions
# =============================================================================

class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class AgentBudgetExceeded(AgentError):
    """Raised when agent exceeds its token budget."""
    pass


class AgentToolError(AgentError):
    """Raised when all tool retries are exhausted."""
    pass


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """

    # Override in subclass
    agent_id: str = "base"

    description: str = (
        "Base agent — do not use directly"
    )

    def __init__(
        self,
        llm_gateway: LLMGateway,
        tool_registry: ToolRegistry,
        budget_manager: Optional[ContextBudgetManager] = None,
    ) -> None:
        """
        Initialize base agent.
        """

        self.llm = llm_gateway
        self.tools = tool_registry
        self.budget = budget_manager

        self._log_entries: List[StructuredLogEntry] = []

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def execute(
        self,
        ctx: ContextObject,
    ) -> AgentOutput:
        """
        Execute agent with logging and budget tracking.
        """

        job_id = ctx.job_id

        start_time = time.time()

        try:

            # -----------------------------------------------------------------
            # Agent Start
            # -----------------------------------------------------------------

            self._log_event(
                job_id=job_id,
                event_type="agent_start",
                input_payload={
                    "query": ctx.query,
                    "sub_tasks": [
                        t.model_dump()
                        for t in ctx.sub_tasks
                    ],
                },
            )

            logger.info(
                f"Agent started: {self.agent_id}",
                extra={"job_id": job_id},
            )

            # -----------------------------------------------------------------
            # Budget Declaration
            # -----------------------------------------------------------------

            if self.budget:

                default_budgets = {
                    "orchestrator": 2048,
                    "decomposition": 2048,
                    "rag": 4096,
                    "critique": 2048,
                    "synthesis": 4096,
                    "compression": 1024,
                    "meta": 2048,
                }

                budget = default_budgets.get(
                    self.agent_id,
                    2048,
                )

                self.budget.declare_budget(
                    self.agent_id,
                    budget,
                )

            # -----------------------------------------------------------------
            # Budget Validation
            # -----------------------------------------------------------------

            if (
                self.budget
                and self.budget.check_remaining(
                    self.agent_id
                ) <= 0
            ):

                raise AgentBudgetExceeded(
                    f"Agent {self.agent_id} "
                    f"has no remaining budget"
                )

            # -----------------------------------------------------------------
            # Execute Agent Logic
            # -----------------------------------------------------------------

            result = await self._execute(ctx)

            # -----------------------------------------------------------------
            # Token Accounting
            # -----------------------------------------------------------------

            tokens_used = (
                len(result.content.split())
                if result.content
                else 0
            )

            if self.budget:

                self.budget.consume(
                    self.agent_id,
                    tokens_used,
                )

            # -----------------------------------------------------------------
            # Agent End
            # -----------------------------------------------------------------

            latency_ms = int(
                (time.time() - start_time) * 1000
            )

            self._log_event(
                job_id=job_id,
                event_type="agent_end",
                output_payload={
                    "content_hash": hash_content(
                        result.content
                    ),
                    "confidence": result.confidence,
                },
                latency_ms=latency_ms,
                token_count=tokens_used,
            )

            logger.info(
                f"Agent completed: "
                f"{self.agent_id} "
                f"in {latency_ms}ms",
                extra={
                    "job_id": job_id,
                    "tokens": tokens_used,
                },
            )

            return result

        except AgentBudgetExceeded:
            raise

        except Exception as e:

            latency_ms = int(
                (time.time() - start_time) * 1000
            )

            self._log_event(
                job_id=job_id,
                event_type="agent_end",
                output_payload={
                    "error": str(e),
                },
                latency_ms=latency_ms,
                policy_violations=[str(e)],
            )

            logger.error(
                f"Agent failed: "
                f"{self.agent_id} — {e}",
                extra={"job_id": job_id},
            )

            raise AgentError(
                f"Agent {self.agent_id} failed: {e}"
            )

    # =========================================================================
    # Abstract Execution
    # =========================================================================

    @abstractmethod
    async def _execute(
        self,
        ctx: ContextObject,
    ) -> AgentOutput:
        """
        Implement agent logic here.
        """
        raise NotImplementedError

    # =========================================================================
    # Structured Logging
    # =========================================================================

    def _log_event(
        self,
        job_id: str,
        event_type: str,
        input_payload: Optional[Dict[str, Any]] = None,
        output_payload: Optional[Dict[str, Any]] = None,
        latency_ms: int = 0,
        token_count: int = 0,
        policy_violations: Optional[List[str]] = None,
    ) -> None:
        """
        Create structured log entry and save to DB in the background.
        """

        entry = StructuredLogEntry(
            agent_id=self.agent_id,
            job_id=job_id,
            event_type=event_type,
            input_hash=(
                hash_content(input_payload)
                if input_payload
                else None
            ),
            output_hash=(
                hash_content(output_payload)
                if output_payload
                else None
            ),
            latency_ms=latency_ms,
            token_count=token_count,
            policy_violations=(
                policy_violations or []
            ),
        )

        self._log_entries.append(entry)

        # Asynchronously write to database in background
        import asyncio
        from app.db.connection import save_agent_log
        try:
            asyncio.create_task(
                save_agent_log(
                    job_id=job_id,
                    agent_id=self.agent_id,
                    event_type=event_type,
                    input_hash=entry.input_hash,
                    output_hash=entry.output_hash,
                    input_payload=input_payload,
                    output_payload=output_payload,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    policy_violations=entry.policy_violations,
                )
            )
        except Exception as e:
            logger.error(f"Failed to trigger background log save: {e}")

    # =========================================================================
    # Dynamic Prompts
    # =========================================================================

    async def get_effective_prompt(self) -> str:
        """
        Get custom approved prompt from database, or fallback to default prompt.
        """
        from app.db.connection import get_session
        from app.db.models import PromptRewrite
        from sqlalchemy import select
        
        try:
            async with get_session() as session:
                query = (
                    select(PromptRewrite)
                    .where(PromptRewrite.agent_id == self.agent_id)
                    .where(PromptRewrite.status == "approved")
                    .order_by(PromptRewrite.reviewed_at.desc())
                    .limit(1)
                )
                result = await session.execute(query)
                rewrite = result.scalar_one_or_none()
                if rewrite:
                    logger.info(f"Loaded approved custom prompt for agent {self.agent_id} from DB")
                    return rewrite.proposed_prompt
        except Exception as e:
            logger.warning(f"Failed to check database for approved prompt: {e}")
            
        # Fallback to default prompts defined in MetaAgent
        from app.agents.meta import MetaAgent
        return MetaAgent.AGENT_PROMPTS.get(self.agent_id, "")

    # =========================================================================
    # LLM Calls
    # =========================================================================

    async def _call_llm(
        self,
        prompt: str,
        ctx: ContextObject,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Call LLM via gateway.
        """

        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            job_id=ctx.job_id,
            agent_id=self.agent_id,
        )

        return response.content

    # =========================================================================
    # Tool Calls
    # =========================================================================

    async def _call_tool(
        self,
        tool_name: str,
        ctx: ContextObject,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute tool via registry with explicit failure retry and database logging.
        """
        import re
        from app.db.connection import save_tool_call
        from app.schemas import ToolResult
        from app.tools.base import ToolResultStatus
        
        retry_number = 0
        max_retries = 2
        
        last_result = {}
        
        while retry_number <= max_retries:
            # 1. Log the event in memory
            self._log_event(
                job_id=ctx.job_id,
                event_type="tool_call",
                input_payload={
                    "tool": tool_name,
                    "params": kwargs,
                    "retry_number": retry_number,
                },
            )
            
            start_time = time.time()
            
            # 2. Execute tool
            result = await self.tools.execute(
                tool_name=tool_name,
                agent_id=self.agent_id,
                retry_number=retry_number,
                job_id=ctx.job_id,
                **kwargs,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # 3. Acceptance evaluation
            accepted = True
            rejection_reason = None
            
            status = result.get("status")
            if status in ["timeout", "error", "sandbox_violation", "invalid_input"]:
                accepted = False
                rejection_reason = result.get("error", "Tool execution error status")
            elif tool_name == "web_search":
                if not result.get("results") or len(result.get("results", [])) == 0:
                    accepted = False
                    rejection_reason = "No search results returned"
            elif tool_name == "code_execution":
                if result.get("exit_code") != 0:
                    accepted = False
                    rejection_reason = f"Execution failed with non-zero exit code: {result.get('exit_code')}"
            elif tool_name == "nl2sql":
                if result.get("row_count", 0) == 0:
                    accepted = False
                    rejection_reason = "Query returned empty results"
            
            # 4. Log retry details to DB
            try:
                await save_tool_call(
                    job_id=ctx.job_id,
                    agent_id=self.agent_id,
                    tool_name=tool_name,
                    input_data=kwargs,
                    output_data=result,
                    latency_ms=latency_ms,
                    accepted=accepted,
                    rejection_reason=rejection_reason,
                    retry_number=retry_number,
                )
            except Exception as e:
                logger.error(f"Failed to save tool call to database: {e}")
            
            # 5. Populate ctx.tool_results with the trace
            ctx.tool_results[f"{tool_name}_retry_{retry_number}"] = ToolResult(
                tool_name=tool_name,
                input_params=kwargs,
                output=result,
                latency_ms=latency_ms,
                accepted=accepted,
                rejection_reason=rejection_reason,
                retry_number=retry_number,
            )
            
            if accepted:
                ctx.tool_results[tool_name] = ctx.tool_results[f"{tool_name}_retry_{retry_number}"]
                return result
                
            # If rejected, reformulate query/input in explicit code fallback logic
            logger.warning(f"Tool {tool_name} output rejected on attempt {retry_number} in job {ctx.job_id}. Retrying...")
            
            # Fallback input reformulation
            modified_kwargs = kwargs.copy()
            if tool_name == "web_search":
                q = kwargs.get("query", "")
                if len(q.split()) > 4:
                    # Narrow down
                    modified_kwargs["query"] = " ".join(q.split()[:3])
                else:
                    # Broaden
                    modified_kwargs["query"] = f"{q} overview"
            elif tool_name == "code_execution":
                code = kwargs.get("code", "")
                if "print " in code and "print(" not in code:
                    modified_kwargs["code"] = re.sub(r'print\s+(["\'].*?["\'])', r'print(\1)', code)
            elif tool_name == "nl2sql":
                q = kwargs.get("query", "")
                if "count" in q.lower():
                    modified_kwargs["query"] = "total jobs"
                else:
                    modified_kwargs["query"] = "recent jobs"
                    
            kwargs = modified_kwargs
            last_result = result
            retry_number += 1
            
        # Return the last attempt result
        ctx.tool_results[tool_name] = ctx.tool_results[f"{tool_name}_retry_{max_retries}"]
        return last_result

    # =========================================================================
    # Log Access
    # =========================================================================

    def get_logs(self) -> List[StructuredLogEntry]:
        """
        Return log entries.
        """

        return self._log_entries.copy()

    def clear_logs(self) -> None:
        """
        Clear logs.
        """

        self._log_entries.clear()