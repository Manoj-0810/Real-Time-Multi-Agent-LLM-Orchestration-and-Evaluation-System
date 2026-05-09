# =============================================================================
# MEGA AI — Master Orchestrator Agent
# =============================================================================
# Reads query, decides which sub-agents to invoke, in what order, with what
# context budget. NEVER hardcodes a chain — routing via structured reasoning.
# =============================================================================

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from app.agents.base import BaseAgent, AgentOutput
from app.context import ContextBudgetManager
from app.llm_gateway import LLMGateway
from app.schemas import ContextObject, SubTask
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """Master orchestrator that routes queries to sub-agents.

    This agent is the entry point for all queries. It:
    1. Analyzes the query complexity and intent
    2. Decides which sub-agents are needed
    3. Determines execution order with dependency graph
    4. Allocates context budgets per agent
    5. Logs every routing decision with justification

    NEVER hardcodes a chain — routing is always via structured reasoning.
    Agents NEVER call each other directly.
    """

    agent_id = "orchestrator"

    description = (
        "Routes queries to appropriate sub-agents with structured reasoning"
    )

    # =========================================================================
    # Agent Catalog
    # =========================================================================

    AGENT_CATALOG: Dict[str, Dict[str, Any]] = {
        "decomposition": {
            "description": "Breaks ambiguous queries into typed sub-tasks",
            "triggers": [
                "ambiguous query",
                "multi-part question",
                "complex task",
            ],
            "refuses": [
                "simple factual lookup",
                "already decomposed tasks",
            ],
        },

        "rag": {
            "description": (
                "Retrieves relevant documents and answers from knowledge base"
            ),
            "triggers": [
                "needs facts",
                "requires context",
                "knowledge-based question",
            ],
            "refuses": [
                "pure computation",
                "code execution",
            ],
        },

        "critique": {
            "description": (
                "Reviews outputs of other agents for accuracy"
            ),
            "triggers": [
                "after any generation",
                "before synthesis",
            ],
            "refuses": [
                "as first step",
                "without other agent outputs",
            ],
        },

        "synthesis": {
            "description": (
                "Merges all sub-agent outputs into final answer"
            ),
            "triggers": [
                "after all agents complete",
                "multiple outputs to merge",
            ],
            "refuses": [
                "as first step",
                "without outputs to merge",
            ],
        },

        "compression": {
            "description": (
                "Compresses context when budget is exceeded"
            ),
            "triggers": [
                "context budget exceeded",
                "token limit approaching",
            ],
            "refuses": [
                "when context is small",
                "as first step",
            ],
        },

        "meta": {
            "description": (
                "Proposes prompt improvements from evaluation failures"
            ),
            "triggers": [
                "after evaluation run",
                "detected regression",
            ],
            "refuses": [
                "during normal query",
                "as first step",
            ],
        },
    }

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def _execute(self, ctx: ContextObject) -> AgentOutput:
        """
        Execute orchestrator routing logic.
        """

        query = ctx.query

        # Step 1 — Analyze query
        complexity = self._analyze_complexity(query)
        intent = self._classify_intent(query)

        # Step 2 — Routing decision
        routing_decision = await self._make_routing_decision(
            query,
            complexity,
            intent,
        )

        # Step 3 — Allocate budgets
        budget_allocation = self._allocate_budgets(
            routing_decision,
            complexity,
        )

        # Step 4 — Build execution plan
        execution_plan = self._build_execution_plan(
            routing_decision,
            budget_allocation,
        )

        # Log routing decisions
        for decision in routing_decision:

            self._log_event(
                job_id=ctx.job_id,
                event_type="routing_decision",
                input_payload={
                    "query": query,
                    "complexity": complexity,
                },
                output_payload={
                    "agent": decision["agent"],
                    "justification": decision["justification"],
                    "budget": decision.get("budget", 0),
                },
            )

        # Create sub-tasks
        ctx.sub_tasks = self._create_sub_tasks(execution_plan)

        # Output
        output_content = json.dumps(
            {
                "routing_plan": routing_decision,
                "execution_order": execution_plan["order"],
                "budget_allocation": budget_allocation,
                "complexity_analysis": complexity,
                "intent_classification": intent,
                "justification": self._generate_justification(
                    routing_decision,
                    query,
                ),
            },
            indent=2,
        )

        return AgentOutput(
            agent_id=self.agent_id,
            content=output_content,
            confidence=0.90,
            token_count=len(output_content.split()),
        )

    # =========================================================================
    # Complexity Analysis
    # =========================================================================

    def _analyze_complexity(self, query: str) -> Dict[str, Any]:

        word_count = len(query.split())

        sentence_count = (
            query.count(".")
            + query.count("?")
            + query.count("!")
            + 1
        )

        question_marks = query.count("?")

        if word_count < 10 and question_marks <= 1:
            level = "simple"

        elif word_count < 30 and question_marks <= 2:
            level = "moderate"

        else:
            level = "complex"

        return {
            "level": level,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "question_count": question_marks,
        }

    # =========================================================================
    # Intent Classification
    # =========================================================================

    def _classify_intent(self, query: str) -> str:

        query_lower = query.lower()

        code_indicators = [
            "code",
            "python",
            "function",
            "script",
            "program",
            "implement",
        ]

        if any(indicator in query_lower for indicator in code_indicators):
            return "code"

        factual_indicators = [
            "what is",
            "who is",
            "when",
            "where",
            "how many",
            "define",
        ]

        if any(indicator in query_lower for indicator in factual_indicators):
            return "factual"

        analysis_indicators = [
            "analyze",
            "compare",
            "explain",
            "why",
            "impact",
            "effect",
        ]

        if any(indicator in query_lower for indicator in analysis_indicators):
            return "analysis"

        return "general"

    # =========================================================================
    # Routing Decision
    # =========================================================================

    async def _make_routing_decision(
        self,
        query: str,
        complexity: Dict[str, Any],
        intent: str,
    ) -> List[Dict[str, Any]]:

        catalog_str = json.dumps(
            self.AGENT_CATALOG,
            indent=2,
        )

        prompt = f"""
You are the master orchestrator of a multi-agent AI system.

Analyze this query and decide which agents to invoke.

Query: "{query}"

Complexity:
{complexity['level']}

Intent:
{intent}

Available agents:
{catalog_str}
"""

        try:

            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.3,
                job_id="orchestrator_routing",
                agent_id=self.agent_id,
            )

            try:

                routing = json.loads(response.content)

                if not isinstance(routing, list):
                    routing = [routing]

            except json.JSONDecodeError:
                routing = self._fallback_routing(
                    complexity,
                    intent,
                )

        except Exception:

            routing = self._fallback_routing(
                complexity,
                intent,
            )

        valid_agents = set(self.AGENT_CATALOG.keys())

        validated_routing = [
            r for r in routing
            if r.get("agent") in valid_agents
            and r.get("agent") != "orchestrator"
        ]

        has_synthesis = any(
            r.get("agent") == "synthesis"
            for r in validated_routing
        )

        if not has_synthesis:

            validated_routing.append({
                "agent": "synthesis",
                "order": len(validated_routing) + 1,
                "justification": (
                    "Final answer synthesis always required"
                ),
                "budget": 2048,
            })

        validated_routing.sort(
            key=lambda x: x.get("order", 99)
        )

        return validated_routing

    # =========================================================================
    # Fallback Routing
    # =========================================================================

    def _fallback_routing(
        self,
        complexity: Dict[str, Any],
        intent: str,
    ) -> List[Dict[str, Any]]:

        routing = []
        order = 1

        if complexity["level"] in ["moderate", "complex"]:

            routing.append({
                "agent": "decomposition",
                "order": order,
                "justification": (
                    "Query is complex — needs decomposition"
                ),
                "budget": 2048,
            })

            order += 1

        if intent in ["factual", "analysis", "general"]:

            routing.append({
                "agent": "rag",
                "order": order,
                "justification": (
                    f"Query needs knowledge retrieval "
                    f"(intent: {intent})"
                ),
                "budget": 4096,
            })

            order += 1

        routing.append({
            "agent": "critique",
            "order": order,
            "justification": (
                "Review outputs before synthesis"
            ),
            "budget": 2048,
        })

        order += 1

        routing.append({
            "agent": "synthesis",
            "order": order,
            "justification": (
                "Merge all outputs into final answer"
            ),
            "budget": 4096,
        })

        return routing

    # =========================================================================
    # Budget Allocation
    # =========================================================================

    def _allocate_budgets(
        self,
        routing: List[Dict[str, Any]],
        complexity: Dict[str, Any],
    ) -> Dict[str, int]:

        total_budget = (
            16384
            if complexity["level"] == "complex"
            else 8192
        )

        allocation = {}

        for decision in routing:

            agent = decision["agent"]

            requested = decision.get(
                "budget",
                2048,
            )

            allocation[agent] = min(
                requested,
                total_budget // len(routing),
            )

        return allocation

    # =========================================================================
    # Execution Plan
    # =========================================================================

    def _build_execution_plan(
        self,
        routing: List[Dict[str, Any]],
        budgets: Dict[str, int],
    ) -> Dict[str, Any]:

        agents = [r["agent"] for r in routing]

        dependencies: Dict[str, List[str]] = {}

        for i, decision in enumerate(routing):

            agent = decision["agent"]

            if i > 0:

                dependencies[agent] = [
                    routing[j]["agent"]
                    for j in range(i)
                ]

            else:
                dependencies[agent] = []

        return {
            "order": agents,
            "dependencies": dependencies,
            "parallel_groups": self._find_parallel_groups(
                routing
            ),
        }

    # =========================================================================
    # Parallel Groups
    # =========================================================================

    def _find_parallel_groups(
        self,
        routing: List[Dict[str, Any]],
    ) -> List[List[str]]:

        groups = []

        if (
            len(routing) > 1
            and routing[0]["agent"] == "decomposition"
        ):

            groups.append(["decomposition"])

            groups.append([
                r["agent"]
                for r in routing[1:]
                if r["agent"] != "synthesis"
            ])

        else:

            groups.append([
                r["agent"]
                for r in routing
                if r["agent"] != "synthesis"
            ])

        return groups

    # =========================================================================
    # Sub Tasks
    # =========================================================================

    def _create_sub_tasks(
        self,
        execution_plan: Dict[str, Any],
    ) -> List[SubTask]:

        sub_tasks = []

        for agent in execution_plan["order"]:

            depends_on = execution_plan["dependencies"].get(
                agent,
                [],
            )

            sub_tasks.append(
                SubTask(
                    task_id=f"task_{agent}",
                    task_type="agent_execution",
                    description=f"Execute {agent} agent",
                    depends_on=depends_on,
                    status="pending",
                )
            )

        return sub_tasks

    # =========================================================================
    # Human Justification
    # =========================================================================

    def _generate_justification(
        self,
        routing: List[Dict[str, Any]],
        query: str,
    ) -> str:

        agents_invoked = [
            r["agent"]
            for r in routing
        ]

        justifications = [
            r["justification"]
            for r in routing
        ]

        return (
            f"Query '{query[:50]}...' routed through "
            f"{len(agents_invoked)} agents: "
            f"{' -> '.join(agents_invoked)}. "
            f"Key decisions: "
            f"{'; '.join(justifications[:3])}."
        )