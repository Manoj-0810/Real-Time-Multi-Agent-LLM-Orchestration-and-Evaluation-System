# =============================================================================
# MEGA AI — NL2SQL Database Lookup Tool
# =============================================================================
# Converts natural language to SQL and executes against PostgreSQL.
# =============================================================================

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.db.connection import get_engine

from app.tools.base import (
    BaseTool,
    ToolInputError,
    ToolResultStatus,
    ToolTimeoutError,
)

logger = logging.getLogger(__name__)


class NL2SQLTool(BaseTool):
    """
    Natural language to SQL database lookup tool.
    """

    name = "nl2sql"

    description = (
        "Convert natural language to SQL "
        "and query the database"
    )

    timeout_ms = 15000

    max_retries = 2

    # =========================================================================
    # Known Table Schemas
    # =========================================================================

    _TABLE_SCHEMAS: Dict[str, str] = {

        "jobs":
            "id (UUID), query (TEXT), "
            "status (VARCHAR), "
            "created_at (TIMESTAMPTZ), "
            "completed_at (TIMESTAMPTZ)",

        "agent_logs":
            "id (UUID), job_id (UUID), "
            "agent_id (VARCHAR), "
            "event_type (VARCHAR), "
            "latency_ms (INT), "
            "timestamp (TIMESTAMPTZ)",

        "tool_calls":
            "id (UUID), job_id (UUID), "
            "agent_id (VARCHAR), "
            "tool_name (VARCHAR), "
            "accepted (BOOLEAN), "
            "timestamp (TIMESTAMPTZ)",

        "eval_runs":
            "id (UUID), "
            "run_timestamp (TIMESTAMPTZ), "
            "test_cases (JSONB), "
            "summary (JSONB)",

        "prompt_rewrites":
            "id (UUID), "
            "eval_run_id (UUID), "
            "agent_id (VARCHAR), "
            "dimension (VARCHAR), "
            "status (VARCHAR)",
    }

    # =========================================================================
    # Query Patterns
    # =========================================================================

    _QUERY_PATTERNS: List[tuple] = [

        # ---------------------------------------------------------------------
        # Job Queries
        # ---------------------------------------------------------------------

        (
            r"how many jobs",
            "SELECT COUNT(*) as count FROM jobs",
        ),

        (
            r"count of jobs",
            "SELECT COUNT(*) as count FROM jobs",
        ),

        (
            r"total jobs",
            "SELECT COUNT(*) as count FROM jobs",
        ),

        (
            r"jobs (that are )?pending",
            "SELECT * FROM jobs WHERE status = 'pending'",
        ),

        (
            r"jobs (that are )?running",
            "SELECT * FROM jobs WHERE status = 'running'",
        ),

        (
            r"jobs (that are )?completed",
            "SELECT * FROM jobs WHERE status = 'completed'",
        ),

        (
            r"jobs (that are )?failed",
            "SELECT * FROM jobs WHERE status = 'failed'",
        ),

        (
            r"failed jobs",
            "SELECT * FROM jobs WHERE status = 'failed'",
        ),

        (
            r"recent jobs",
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 10",
        ),

        (
            r"latest jobs",
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 10",
        ),

        # ---------------------------------------------------------------------
        # Agent Logs
        # ---------------------------------------------------------------------

        (
            r"agent logs",
            "SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT 10",
        ),

        (
            r"logs for (\w+)",
            (
                "SELECT * FROM agent_logs "
                "WHERE agent_id = '{group_1}' "
                "ORDER BY timestamp DESC LIMIT 10"
            ),
        ),

        (
            r"latency",
            (
                "SELECT agent_id, "
                "AVG(latency_ms) as avg_latency "
                "FROM agent_logs "
                "GROUP BY agent_id"
            ),
        ),

        # ---------------------------------------------------------------------
        # Tool Calls
        # ---------------------------------------------------------------------

        (
            r"tool calls",
            "SELECT * FROM tool_calls ORDER BY timestamp DESC LIMIT 10",
        ),

        (
            r"tool usage",
            (
                "SELECT tool_name, COUNT(*) as call_count "
                "FROM tool_calls "
                "GROUP BY tool_name"
            ),
        ),

        # ---------------------------------------------------------------------
        # Evaluation Queries
        # ---------------------------------------------------------------------

        (
            r"eval runs",
            "SELECT * FROM eval_runs ORDER BY run_timestamp DESC LIMIT 10",
        ),

        (
            r"evaluation results",
            "SELECT * FROM eval_runs ORDER BY run_timestamp DESC LIMIT 10",
        ),

        (
            r"prompt rewrites",
            (
                "SELECT * FROM prompt_rewrites "
                "WHERE status = 'pending'"
            ),
        ),
    ]

    def __init__(self) -> None:
        """
        Initialize NL2SQL tool.
        """

        super().__init__()

    # =========================================================================
    # NL → SQL
    # =========================================================================

    def _nl_to_sql(
        self,
        query: str,
    ) -> tuple[str, float]:
        """
        Convert natural language to SQL.
        """

        query_lower = query.lower().strip()

        # ---------------------------------------------------------------------
        # Pattern Matching
        # ---------------------------------------------------------------------

        for pattern, sql_template in (
            self._QUERY_PATTERNS
        ):

            match = re.search(
                pattern,
                query_lower,
            )

            if match:

                sql = sql_template

                for i, group in enumerate(
                    match.groups(),
                    1,
                ):

                    if group:

                        sql = sql.replace(
                            f"{{group_{i}}}",
                            group,
                        )

                return sql, 0.85

        # ---------------------------------------------------------------------
        # Generic SELECT
        # ---------------------------------------------------------------------

        words = query_lower.split()

        table_name = None

        for word in words:

            if word in self._TABLE_SCHEMAS:

                table_name = word

                break

        if table_name:

            return (
                f"SELECT * FROM {table_name} LIMIT 100",
                0.50,
            )

        return "", 0.0

    # =========================================================================
    # SQL Validation
    # =========================================================================

    def _validate_sql(
        self,
        sql: str,
    ) -> Optional[str]:
        """
        Validate safe SQL.
        """

        sql_upper = sql.strip().upper()

        # ---------------------------------------------------------------------
        # SELECT Only
        # ---------------------------------------------------------------------

        if not sql_upper.startswith("SELECT"):

            return (
                "Only SELECT queries are allowed. "
                f"Query starts with: {sql.split()[0]}"
            )

        # ---------------------------------------------------------------------
        # Dangerous Keywords
        # ---------------------------------------------------------------------

        blocked = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "CREATE",
            "TRUNCATE",
        ]

        for keyword in blocked:

            if keyword in sql_upper:

                return (
                    f"Blocked keyword detected: "
                    f"{keyword}"
                )

        return None

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def _execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute natural language DB query.
        """

        start_time = time.time()

        # ---------------------------------------------------------------------
        # Validate Input
        # ---------------------------------------------------------------------

        if not query or not query.strip():

            raise ToolInputError(
                "Query cannot be empty"
            )

        # ---------------------------------------------------------------------
        # NL → SQL
        # ---------------------------------------------------------------------

        sql, confidence = self._nl_to_sql(
            query
        )

        if not sql:

            available_tables = ", ".join(
                self._TABLE_SCHEMAS.keys()
            )

            raise ToolInputError(
                f"Could not generate SQL for: "
                f"'{query}'. "
                f"Available tables: "
                f"{available_tables}"
            )

        # ---------------------------------------------------------------------
        # Validate SQL
        # ---------------------------------------------------------------------

        validation_error = self._validate_sql(
            sql
        )

        if validation_error:

            raise ToolInputError(
                f"SQL validation failed: "
                f"{validation_error}. "
                f"Generated SQL: {sql}"
            )

        # ---------------------------------------------------------------------
        # Execute SQL
        # ---------------------------------------------------------------------

        try:

            engine = get_engine()

            results = []

            async with engine.connect() as conn:

                result = await conn.execute(
                    text(sql)
                )

                rows = result.mappings().all()

                for row in rows:

                    row_dict = {}

                    for key, value in (
                        dict(row).items()
                    ):

                        if hasattr(
                            value,
                            "isoformat",
                        ):

                            row_dict[key] = (
                                value.isoformat()
                            )

                        else:

                            row_dict[key] = value

                    results.append(
                        row_dict
                    )

            latency_ms = int(
                (
                    time.time()
                    - start_time
                ) * 1000
            )

            return {
                "sql_generated": sql,

                "results": results,

                "row_count": len(results),

                "confidence": confidence,

                "latency_ms": latency_ms,
            }

        except asyncio.TimeoutError:

            raise ToolTimeoutError(
                f"Database query timeout after "
                f"{self.timeout_ms}ms"
            )

        except Exception as e:

            raise ToolInputError(
                f"Database error: {e}. "
                f"SQL: {sql}"
            )

    # =========================================================================
    # Fallback Logic
    # =========================================================================

    def _get_fallback(
        self,
        status: ToolResultStatus,
        error_msg: str,
    ) -> Optional[str]:
        """
        Return fallback instructions.
        """

        if (
            status
            == ToolResultStatus.INVALID_INPUT
        ):

            return (
                "rephrase_query_and_retry"
            )

        elif (
            status
            == ToolResultStatus.TIMEOUT
        ):

            return "simplify_query"

        elif (
            status
            == ToolResultStatus.ERROR
        ):

            if "NO_DATA" in error_msg:

                return (
                    "broaden_search_criteria"
                )

        return "use_different_tool"