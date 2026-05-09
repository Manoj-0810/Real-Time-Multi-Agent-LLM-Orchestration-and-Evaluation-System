# =============================================================================
# MEGA AI — Decomposition Agent
# =============================================================================
# Breaks ambiguous queries into typed sub-tasks with explicit dependency graph.
# Output schema: {task_id, task_type, description, depends_on: [], status}
# =============================================================================

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from app.agents.base import (
    AgentOutput,
    BaseAgent,
)

from app.schemas import (
    ContextObject,
    SubTask,
)

logger = logging.getLogger(__name__)


class DecompositionAgent(BaseAgent):
    """
    Breaks ambiguous queries into typed sub-tasks
    with explicit dependency tracking.
    """

    agent_id = "decomposition"

    description = (
        "Breaks ambiguous queries into typed "
        "sub-tasks with dependency graph"
    )

    # =========================================================================
    # Task Taxonomy
    # =========================================================================

    TASK_TYPES: List[str] = [
        "information_retrieval",
        "computation",
        "comparison",
        "analysis",
        "synthesis",
        "code_generation",
        "verification",
        "explanation",
    ]

    # =========================================================================
    # Ambiguity Indicators
    # =========================================================================

    AMBIGUITY_PATTERNS: List[str] = [
        "it",
        "this",
        "that",
        "these",
        "those",
        "the impact",
        "the effect",
        "the result",
        "how does",
        "why is",
        "what about",
        "compare",
        "versus",
        "vs",
    ]

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def _execute(
        self,
        ctx: ContextObject,
    ) -> AgentOutput:
        """
        Decompose query into structured sub-tasks.
        """

        query = ctx.query

        # ---------------------------------------------------------------------
        # Detect Ambiguity
        # ---------------------------------------------------------------------

        ambiguity = self._detect_ambiguity(
            query
        )

        # ---------------------------------------------------------------------
        # Extract Sub Questions
        # ---------------------------------------------------------------------

        sub_questions = self._extract_sub_questions(
            query,
            ambiguity,
        )

        # ---------------------------------------------------------------------
        # Create Sub Tasks
        # ---------------------------------------------------------------------

        sub_tasks = self._create_sub_tasks(
            sub_questions
        )

        # ---------------------------------------------------------------------
        # Build Dependency Graph
        # ---------------------------------------------------------------------

        dependencies = self._build_dependencies(
            sub_tasks
        )

        # ---------------------------------------------------------------------
        # Update Context
        # ---------------------------------------------------------------------

        ctx.sub_tasks = sub_tasks

        # ---------------------------------------------------------------------
        # Output
        # ---------------------------------------------------------------------

        output = {
            "original_query": query,

            "ambiguity_detected": (
                ambiguity["detected"]
            ),

            "ambiguity_score": (
                ambiguity["score"]
            ),

            "ambiguity_types": (
                ambiguity["types"]
            ),

            "sub_tasks": [
                {
                    "task_id": st.task_id,
                    "task_type": st.task_type,
                    "description": st.description,
                    "depends_on": st.depends_on,
                    "status": st.status,
                }
                for st in sub_tasks
            ],

            "dependency_graph": dependencies,

            "execution_order": (
                self._topological_sort(
                    sub_tasks
                )
            ),
        }

        output_content = json.dumps(
            output,
            indent=2,
        )

        return AgentOutput(
            agent_id=self.agent_id,
            content=output_content,
            confidence=(
                0.85
                if ambiguity["detected"]
                else 0.95
            ),
            token_count=len(
                output_content.split()
            ),
        )

    # =========================================================================
    # Ambiguity Detection
    # =========================================================================

    def _detect_ambiguity(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Detect ambiguity in query.
        """

        query_lower = query.lower()

        score = 0.0

        types = []

        # ---------------------------------------------------------------------
        # Vague References
        # ---------------------------------------------------------------------

        vague_refs = [
            "it",
            "this",
            "that",
            "these",
            "those",
        ]

        for ref in vague_refs:

            if f" {ref} " in f" {query_lower} ":

                score += 0.15

                if "vague_reference" not in types:
                    types.append(
                        "vague_reference"
                    )

        # ---------------------------------------------------------------------
        # Underspecified Topics
        # ---------------------------------------------------------------------

        underspecified = [
            "the impact",
            "the effect",
            "the result",
            "the problem",
        ]

        for phrase in underspecified:

            if phrase in query_lower:

                score += 0.20

                if (
                    "underspecified_topic"
                    not in types
                ):
                    types.append(
                        "underspecified_topic"
                    )

        # ---------------------------------------------------------------------
        # Broad Questions
        # ---------------------------------------------------------------------

        broad_indicators = [
            "tell me about",
            "explain",
            "what is the",
        ]

        for indicator in broad_indicators:

            if indicator in query_lower:

                score += 0.10

                if "broad_question" not in types:
                    types.append(
                        "broad_question"
                    )

        # ---------------------------------------------------------------------
        # Comparison Requests
        # ---------------------------------------------------------------------

        if any(
            word in query_lower
            for word in [
                "compare",
                "versus",
                "vs",
                "better",
            ]
        ):

            score += 0.15

            types.append(
                "comparison_request"
            )

        # ---------------------------------------------------------------------
        # Multiple Questions
        # ---------------------------------------------------------------------

        question_count = query.count("?")

        if question_count > 1:

            score += min(
                question_count * 0.1,
                0.3,
            )

            types.append(
                "multiple_questions"
            )

        # ---------------------------------------------------------------------
        # Short Queries
        # ---------------------------------------------------------------------

        word_count = len(
            query.split()
        )

        if word_count < 8:

            score += 0.10

            types.append(
                "underspecified_short"
            )

        return {
            "detected": score > 0.3,
            "score": min(score, 1.0),
            "types": types,
        }

    # =========================================================================
    # Sub Question Extraction
    # =========================================================================

    def _extract_sub_questions(
        self,
        query: str,
        ambiguity: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract implicit sub-questions.
        """

        sub_questions = []

        # ---------------------------------------------------------------------
        # Simple Query
        # ---------------------------------------------------------------------

        if not ambiguity["detected"]:

            sub_questions.append({
                "question": query,
                "type": "information_retrieval",
                "priority": 1,
            })

            return sub_questions

        # ---------------------------------------------------------------------
        # Compound Split
        # ---------------------------------------------------------------------

        parts = re.split(
            r"[;!?]|\band\b|\bthen\b",
            query,
        )

        parts = [
            p.strip()
            for p in parts
            if p.strip()
        ]

        for i, part in enumerate(parts):

            part_lower = part.lower()

            # -------------------------------------------------------------
            # Task Type Detection
            # -------------------------------------------------------------

            if any(
                w in part_lower
                for w in [
                    "compare",
                    "versus",
                    "vs",
                    "difference",
                ]
            ):

                task_type = "comparison"

            elif any(
                w in part_lower
                for w in [
                    "calculate",
                    "compute",
                    "sum",
                    "count",
                ]
            ):

                task_type = "computation"

            elif any(
                w in part_lower
                for w in [
                    "why",
                    "explain",
                    "reason",
                ]
            ):

                task_type = "explanation"

            elif any(
                w in part_lower
                for w in [
                    "code",
                    "function",
                    "implement",
                    "script",
                ]
            ):

                task_type = "code_generation"

            elif any(
                w in part_lower
                for w in [
                    "analyze",
                    "pattern",
                    "trend",
                ]
            ):

                task_type = "analysis"

            else:

                task_type = (
                    "information_retrieval"
                )

            sub_questions.append({
                "question": part,
                "type": task_type,
                "priority": i + 1,
            })

        # ---------------------------------------------------------------------
        # Fallback
        # ---------------------------------------------------------------------

        if not sub_questions:

            sub_questions.append({
                "question": query,
                "type": "information_retrieval",
                "priority": 1,
            })

        return sub_questions

    # =========================================================================
    # Sub Task Creation
    # =========================================================================

    def _create_sub_tasks(
        self,
        sub_questions: List[Dict[str, Any]],
    ) -> List[SubTask]:
        """
        Create SubTask objects.
        """

        sub_tasks = []

        for i, sq in enumerate(sub_questions):

            task_id = f"subtask_{i + 1}"

            depends_on = []

            if i > 0:

                depends_on.append(
                    f"subtask_{i}"
                )

            sub_tasks.append(
                SubTask(
                    task_id=task_id,
                    task_type=sq["type"],
                    description=sq["question"],
                    depends_on=depends_on,
                    status="pending",
                )
            )

        return sub_tasks

    # =========================================================================
    # Dependency Graph
    # =========================================================================

    def _build_dependencies(
        self,
        sub_tasks: List[SubTask],
    ) -> Dict[str, List[str]]:
        """
        Build dependency graph.
        """

        dependencies = {}

        for task in sub_tasks:

            dependencies[
                task.task_id
            ] = task.depends_on.copy()

        return dependencies

    # =========================================================================
    # Topological Sort
    # =========================================================================

    def _topological_sort(
        self,
        sub_tasks: List[SubTask],
    ) -> List[str]:
        """
        Topologically sort tasks.
        """

        in_degree = {
            st.task_id: 0
            for st in sub_tasks
        }

        adjacency = {
            st.task_id: []
            for st in sub_tasks
        }

        for st in sub_tasks:

            for dep in st.depends_on:

                if dep in adjacency:

                    adjacency[dep].append(
                        st.task_id
                    )

                    in_degree[
                        st.task_id
                    ] += 1

        queue = [
            tid
            for tid, deg in in_degree.items()
            if deg == 0
        ]

        order = []

        while queue:

            node = queue.pop(0)

            order.append(node)

            for neighbor in adjacency[node]:

                in_degree[
                    neighbor
                ] -= 1

                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order