# =============================================================================
# MEGA AI — Tool Registry
# =============================================================================
# Central registry for all tools. Tools are discovered and registered here.
# No tool is called directly — all calls go through the registry.
# =============================================================================

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.tools.base import BaseTool, ToolCallRecord
from app.tools.code_exec import CodeExecutionTool
from app.tools.nl2sql import NL2SQLTool
from app.tools.self_reflection import SelfReflectionTool
from app.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all tools in the MEGA AI system.

    This is the single point of access for tools. No agent calls tools
    directly — all tool invocations go through the registry which:
    - Maintains the catalog of available tools
    - Routes tool calls to the correct implementation
    - Tracks usage statistics across all tools
    - Enforces tool availability checks

    Attributes:
        tools: Dict mapping tool name -> tool instance.
        call_history: Aggregated call history across all tools.
    """

    def __init__(self) -> None:
        """Initialize the tool registry and register all tools."""

        self.tools: Dict[str, BaseTool] = {}

        self.call_history: List[ToolCallRecord] = []

        self._register_default_tools()

    # =========================================================================
    # Default Registration
    # =========================================================================

    def _register_default_tools(self) -> None:
        """
        Register all built-in tools.
        """

        tools_to_register: List[BaseTool] = [
            WebSearchTool(),
            CodeExecutionTool(),
            NL2SQLTool(),
            SelfReflectionTool(),
        ]

        for tool in tools_to_register:

            self.register(tool)

        logger.info(
            f"Registered {len(self.tools)} tools: "
            f"{list(self.tools.keys())}"
        )

    # =========================================================================
    # Registration
    # =========================================================================

    def register(
        self,
        tool: BaseTool,
    ) -> None:
        """
        Register a tool in the registry.
        """

        if tool.name in self.tools:

            raise ValueError(
                f"Tool '{tool.name}' is already registered"
            )

        self.tools[tool.name] = tool

        logger.info(
            f"Registered tool: {tool.name}"
        )

    # =========================================================================
    # Retrieval
    # =========================================================================

    def get(
        self,
        tool_name: str,
    ) -> Optional[BaseTool]:
        """
        Get a tool by name.
        """

        tool = self.tools.get(tool_name)

        if not tool:

            logger.warning(
                f"Tool not found: {tool_name}"
            )

        return tool

    # =========================================================================
    # Execution
    # =========================================================================

    async def execute(
        self,
        tool_name: str,
        agent_id: str,
        retry_number: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.
        """

        tool = self.get(tool_name)

        if not tool:

            return {
                "status": "error",
                "error": (
                    f"Tool '{tool_name}' "
                    f"not found in registry"
                ),
                "available_tools": list(
                    self.tools.keys()
                ),
            }

        logger.info(
            f"Tool execution via registry: "
            f"{tool_name} by {agent_id}",
            extra={
                "tool": tool_name,
                "agent_id": agent_id,
            },
        )

        # ---------------------------------------------------------------------
        # Execute Tool
        # ---------------------------------------------------------------------

        result = await tool.execute(
            agent_id=agent_id,
            retry_number=retry_number,
            **kwargs,
        )

        # ---------------------------------------------------------------------
        # Sync Call History
        # ---------------------------------------------------------------------

        self.call_history.extend(
            tool.call_history
        )

        return result.to_dict()

    # =========================================================================
    # Tool Listing
    # =========================================================================

    def list_tools(
        self,
    ) -> List[Dict[str, str]]:
        """
        List all registered tools.
        """

        return [
            {
                "name": name,
                "description": tool.description,
            }
            for name, tool in self.tools.items()
        ]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(
        self,
    ) -> Dict[str, Any]:
        """
        Get aggregated statistics for all tools.
        """

        tool_stats = {}

        for name, tool in self.tools.items():

            tool_stats[name] = tool.get_stats()

        total_calls = sum(
            s["total_calls"]
            for s in tool_stats.values()
        )

        total_successful = sum(
            s["successful"]
            for s in tool_stats.values()
        )

        return {
            "total_tools": len(self.tools),

            "total_calls": total_calls,

            "total_successful": total_successful,

            "total_failed": (
                total_calls - total_successful
            ),

            "tool_breakdown": tool_stats,
        }

    # =========================================================================
    # Availability
    # =========================================================================

    def is_available(
        self,
        tool_name: str,
    ) -> bool:
        """
        Check if a tool is available.
        """

        return tool_name in self.tools


# =============================================================================
# Singleton
# =============================================================================

tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """
    Get the global ToolRegistry singleton.
    """

    return tool_registry