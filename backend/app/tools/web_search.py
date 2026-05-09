# =============================================================================
# MEGA AI — Web Search Tool (Stub)
# =============================================================================
# Returns structured results — realistic stub for development.
# Failure contracts implemented per spec.
# =============================================================================

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional

from app.tools.base import (
    BaseTool,
    ToolInputError,
    ToolResultStatus,
    ToolTimeoutError,
)

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search tool — returns structured search results.

    This is a stub implementation that returns realistic results.
    In production, replace _execute() with actual search API calls.

    Failure contracts:
    - timeout → {"error": "TIMEOUT", "results": [], "fallback": "use_knowledge_base"}
    - empty → {"error": "NO_RESULTS", "results": [], "fallback": "reformulate_query"}
    - malformed → {"error": "INVALID_QUERY", "results": [], "fallback": "decompose_query"}
    """

    name = "web_search"

    description = (
        "Search the web for information "
        "on a given query"
    )

    timeout_ms = 10000

    max_retries = 2

    # =========================================================================
    # Stub Knowledge Base
    # =========================================================================

    _KNOWLEDGE_BASE: Dict[
        str,
        List[Dict[str, Any]]
    ] = {

        "python": [
            {
                "url": (
                    "https://docs.python.org/3/"
                ),

                "title": (
                    "Python Documentation"
                ),

                "snippet": (
                    "Python is a programming language "
                    "that lets you work quickly and "
                    "integrate systems."
                ),

                "relevance_score": 0.95,
            },

            {
                "url": (
                    "https://realpython.com/"
                ),

                "title": (
                    "Real Python Tutorials"
                ),

                "snippet": (
                    "Learn Python programming with "
                    "practical, real-world tutorials "
                    "and projects."
                ),

                "relevance_score": 0.88,
            },
        ],

        "machine learning": [
            {
                "url": (
                    "https://scikit-learn.org/"
                ),

                "title": (
                    "Scikit-learn Documentation"
                ),

                "snippet": (
                    "Machine learning in Python — "
                    "simple and efficient tools "
                    "for data analysis."
                ),

                "relevance_score": 0.92,
            },

            {
                "url": (
                    "https://pytorch.org/"
                ),

                "title": "PyTorch",

                "snippet": (
                    "An open source machine learning "
                    "framework for deep learning research."
                ),

                "relevance_score": 0.90,
            },
        ],

        "artificial intelligence": [
            {
                "url": (
                    "https://openai.com/research"
                ),

                "title": (
                    "OpenAI Research"
                ),

                "snippet": (
                    "OpenAI's research on artificial "
                    "intelligence and machine learning."
                ),

                "relevance_score": 0.93,
            },

            {
                "url": (
                    "https://www.anthropic.com/"
                ),

                "title": "Anthropic",

                "snippet": (
                    "AI safety company building "
                    "reliable, interpretable AI systems."
                ),

                "relevance_score": 0.87,
            },
        ],
    }

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def _execute(
        self,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute web search with stub implementation.
        """

        # ---------------------------------------------------------------------
        # Simulated Network Latency
        # ---------------------------------------------------------------------

        latency = random.randint(
            100,
            500,
        )

        await asyncio.sleep(
            latency / 1000
        )

        # ---------------------------------------------------------------------
        # Validate Input
        # ---------------------------------------------------------------------

        if not query or not query.strip():

            raise ToolInputError(
                "Search query cannot be empty"
            )

        query_lower = (
            query.lower().strip()
        )

        # ---------------------------------------------------------------------
        # Simulated Timeout
        # ---------------------------------------------------------------------

        if (
            len(query) > 500
            or random.random() < 0.05
        ):

            raise ToolTimeoutError(
                f"Search timeout after "
                f"{self.timeout_ms}ms"
            )

        # ---------------------------------------------------------------------
        # Knowledge Base Lookup
        # ---------------------------------------------------------------------

        results = []

        for keyword, entries in (
            self._KNOWLEDGE_BASE.items()
        ):

            if keyword in query_lower:

                results.extend(entries)

        # ---------------------------------------------------------------------
        # Generic Results
        # ---------------------------------------------------------------------

        if not results:

            results = [
                {
                    "url": (
                        "https://example.com/search"
                        f"?q={query.replace(' ', '+')}"
                    ),

                    "title": (
                        f"Search results for '{query}'"
                    ),

                    "snippet": (
                        f"Relevant information about "
                        f"{query} from various sources."
                    ),

                    "relevance_score": 0.60,
                }
            ]

        # ---------------------------------------------------------------------
        # Ranking
        # ---------------------------------------------------------------------

        results.sort(
            key=lambda x: x[
                "relevance_score"
            ],
            reverse=True,
        )

        results = results[:max_results]

        return {
            "results": results,
            "query": query,
            "latency_ms": latency,
        }

    # =========================================================================
    # Fallback Logic
    # =========================================================================

    def _get_fallback(
        self,
        status: ToolResultStatus,
        error_msg: str,
    ) -> Optional[str]:
        """
        Return tool-specific fallback instructions.
        """

        fallbacks = {

            ToolResultStatus.TIMEOUT:
                "use_knowledge_base",

            ToolResultStatus.NO_RESULTS:
                "reformulate_query",

            ToolResultStatus.INVALID_INPUT:
                "decompose_query",

            ToolResultStatus.ERROR:
                "retry_with_fallback",
        }

        return fallbacks.get(
            status,
            "retry_with_fallback",
        )