# =============================================================================
# MEGA AI — Context Budget Manager
# =============================================================================
# All context window management happens here.
# Violations are LOGGED, not silently truncated.
# =============================================================================

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from app.config import BUDGET_CONFIG
from app.schemas import BudgetStatus, StructuredLogEntry

logger = logging.getLogger(__name__)


class LogSink(Protocol):
    """Protocol for log sinks — allows pluggable logging backends."""

    async def write(self, entry: StructuredLogEntry) -> None:
        """Write a structured log entry."""
        ...


class ContextBudgetManager:
    """Manages token budgets for each agent in a job.

    This class enforces the "violations are logged, not silently truncated" rule.
    When an agent exceeds its budget, the overflow is logged as a policy violation
    and the compression agent is triggered.

    Attributes:
        job_id: The unique job identifier.
        default_budget: Default token budget per agent.
        agent_budgets: Dict mapping agent_id -> max_tokens.
        agent_consumed: Dict mapping agent_id -> tokens_consumed.
        violations: List of all budget violations for this job.
    """

    def __init__(
        self,
        job_id: str,
        default_budget: int = 8192,
        log_sink: Optional[LogSink] = None,
    ) -> None:
        """Initialize the budget manager for a job.

        Args:
            job_id: Unique identifier for this job.
            default_budget: Default token budget if not explicitly declared.
            log_sink: Optional async log sink for structured logging.
        """
        self.job_id = job_id
        self.default_budget = default_budget
        self.log_sink = log_sink
        self.agent_budgets: Dict[str, int] = {}
        self.agent_consumed: Dict[str, int] = {}
        self.violations: List[Dict[str, Any]] = []

    def declare_budget(self, agent_id: str, max_tokens: int) -> None:
        """Declare the token budget for an agent."""

        self.agent_budgets[agent_id] = max_tokens
        self.agent_consumed[agent_id] = 0

        logger.info(
            "Budget declared",
            extra={
                "job_id": self.job_id,
                "agent_id": agent_id,
                "max_tokens": max_tokens,
            },
        )

    def check_remaining(self, agent_id: str) -> int:
        """Check remaining tokens for an agent."""

        budget = self.agent_budgets.get(
            agent_id,
            self.default_budget,
        )

        consumed = self.agent_consumed.get(
            agent_id,
            0,
        )

        remaining = budget - consumed

        # Compression threshold check
        if (
            budget > 0
            and remaining / budget
            < BUDGET_CONFIG["compression_trigger_threshold"]
        ):

            logger.warning(
                "Budget below compression threshold",
                extra={
                    "job_id": self.job_id,
                    "agent_id": agent_id,
                    "remaining": remaining,
                    "budget": budget,
                    "threshold": BUDGET_CONFIG[
                        "compression_trigger_threshold"
                    ],
                },
            )

        return max(0, remaining)

    def consume(
        self,
        agent_id: str,
        tokens: int,
    ) -> bool:
        """Consume tokens from an agent budget."""

        if agent_id not in self.agent_consumed:
            self.agent_consumed[agent_id] = 0

        if agent_id not in self.agent_budgets:
            self.agent_budgets[agent_id] = self.default_budget

        self.agent_consumed[agent_id] += tokens

        remaining = self.check_remaining(agent_id)

        budget = self.agent_budgets[agent_id]

        within_budget = (
            self.agent_consumed[agent_id]
            <= budget
        )

        if not within_budget:

            overflow = (
                self.agent_consumed[agent_id]
                - budget
            )

            self.log_violation(
                agent_id,
                overflow,
            )

            if BUDGET_CONFIG["log_violations"]:

                logger.error(
                    "Budget exceeded — logging violation",
                    extra={
                        "job_id": self.job_id,
                        "agent_id": agent_id,
                        "consumed": self.agent_consumed[
                            agent_id
                        ],
                        "budget": budget,
                        "overflow": overflow,
                    },
                )

        return within_budget

    def log_violation(
        self,
        agent_id: str,
        overflow: int,
    ) -> None:
        """Log budget violation."""

        violation = {
            "timestamp": (
                datetime.utcnow().isoformat()
            ),
            "job_id": self.job_id,
            "agent_id": agent_id,
            "budget": self.agent_budgets.get(
                agent_id,
                self.default_budget,
            ),
            "consumed": self.agent_consumed.get(
                agent_id,
                0,
            ),
            "overflow": overflow,
            "severity": (
                "critical"
                if overflow > 1000
                else "warning"
            ),
        }

        self.violations.append(violation)

        # Structured database logging for policy violation
        import asyncio
        from app.db.connection import save_agent_log

        try:
            asyncio.create_task(
                save_agent_log(
                    job_id=self.job_id,
                    agent_id=agent_id,
                    event_type="policy_violation",
                    policy_violations=[f"budget_overflow:{overflow}"],
                    output_payload=violation,
                )
            )
        except Exception:
            logger.exception(
                "Failed to write budget violation to database"
            )

    def get_budget_status(
        self,
        agent_id: str,
    ) -> BudgetStatus:
        """Get current budget status."""

        budget = self.agent_budgets.get(
            agent_id,
            self.default_budget,
        )

        consumed = self.agent_consumed.get(
            agent_id,
            0,
        )

        remaining = max(
            0,
            budget - consumed,
        )

        agent_violations = [
            v
            for v in self.violations
            if v["agent_id"] == agent_id
        ]

        return BudgetStatus(
            total_budget=budget,
            tokens_used=consumed,
            tokens_remaining=remaining,
            compression_triggered=(
                remaining / budget
                < BUDGET_CONFIG[
                    "compression_trigger_threshold"
                ]
                if budget > 0
                else False
            ),
            violations=[
                v["overflow"]
                for v in agent_violations
            ],
        )

    def get_total_usage(self) -> Dict[str, int]:
        """Get total usage across all agents."""

        total_budget = sum(
            self.agent_budgets.values()
        )

        total_consumed = sum(
            self.agent_consumed.values()
        )

        return {
            "total_budget": total_budget,
            "total_consumed": total_consumed,
            "total_remaining": max(
                0,
                total_budget - total_consumed,
            ),
        }

    def should_trigger_compression(
        self,
        agent_id: str,
    ) -> bool:
        """Check if compression should trigger."""

        budget = self.agent_budgets.get(
            agent_id,
            self.default_budget,
        )

        if budget == 0:
            return False

        remaining = self.check_remaining(
            agent_id
        )

        ratio = remaining / budget

        return (
            ratio
            < BUDGET_CONFIG[
                "compression_trigger_threshold"
            ]
        )


def hash_content(content: Any) -> str:
    """
    Create SHA-256 hash of content.
    """

    serialized = json.dumps(
        content,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(
        serialized
    ).hexdigest()