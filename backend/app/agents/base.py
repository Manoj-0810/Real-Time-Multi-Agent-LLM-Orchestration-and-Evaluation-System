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
        Create structured log entry.
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
        retry_number: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute tool via registry.
        """

        self._log_event(
            job_id=ctx.job_id,
            event_type="tool_call",
            input_payload={
                "tool": tool_name,
                "params": kwargs,
            },
        )

        return await self.tools.execute(
            tool_name=tool_name,
            agent_id=self.agent_id,
            retry_number=retry_number,
            **kwargs,
        )

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