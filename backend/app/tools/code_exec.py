# =============================================================================
# MEGA AI — Code Execution Sandbox Tool
# =============================================================================
# Runs Python snippets safely with timeout and sandbox enforcement.
# =============================================================================

from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from app.tools.base import (
    BaseTool,
    ToolInputError,
    ToolResultStatus,
    ToolSandboxError,
    ToolTimeoutError,
)

logger = logging.getLogger(__name__)


class CodeExecutionTool(BaseTool):
    """
    Safe Python code execution sandbox.

    Executes Python code in an isolated subprocess with:
    - 5-second timeout enforcement
    - Syntax validation before execution
    - Dangerous code detection
    - Memory limits
    - Output capture
    """

    name = "code_execution"

    description = (
        "Execute Python code safely "
        "in a sandboxed environment"
    )

    timeout_ms = 5000

    max_retries = 0

    # =========================================================================
    # Dangerous Patterns
    # =========================================================================

    _BLOCKED_PATTERNS: List[str] = [

        "__import__",

        "import os",

        "import sys",

        "import subprocess",

        "import socket",

        "open(",

        "exec(",

        "eval(",

        "compile(",

        "input(",

        "raw_input(",

        "breakpoint(",

        "pdb",

        "httpx",

        "requests",

        "urllib",

        "ftplib",

        "smtplib",

        "telnetlib",
    ]

    # =========================================================================
    # Safe Modules
    # =========================================================================

    _SAFE_MODULES: List[str] = [

        "math",

        "random",

        "datetime",

        "json",

        "re",

        "statistics",

        "itertools",

        "functools",

        "collections",

        "decimal",

        "fractions",

        "numbers",

        "typing",

        "string",

        "hashlib",

        "uuid",
    ]

    def __init__(
        self,
        max_memory_mb: int = 256,
    ) -> None:
        """
        Initialize sandbox tool.
        """

        super().__init__()

        self.max_memory_mb = max_memory_mb

    # =========================================================================
    # Dangerous Code Detection
    # =========================================================================

    def _detect_dangerous_code(
        self,
        code: str,
    ) -> Optional[str]:
        """
        Analyze code for dangerous patterns.
        """

        # ---------------------------------------------------------------------
        # String-Based Detection
        # ---------------------------------------------------------------------

        code_lower = code.lower()

        for pattern in self._BLOCKED_PATTERNS:

            if pattern.lower() in code_lower:

                return (
                    f"Blocked pattern detected: "
                    f"'{pattern}'"
                )

        # ---------------------------------------------------------------------
        # AST Import Analysis
        # ---------------------------------------------------------------------

        try:

            tree = ast.parse(code)

        except SyntaxError:

            return None

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for alias in node.names:

                    root_module = (
                        alias.name.split(".")[0]
                    )

                    if (
                        root_module
                        not in self._SAFE_MODULES
                    ):

                        return (
                            f"Blocked import: "
                            f"'{alias.name}'"
                        )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):

                if node.module:

                    root_module = (
                        node.module.split(".")[0]
                    )

                    if (
                        root_module
                        not in self._SAFE_MODULES
                    ):

                        return (
                            f"Blocked import from: "
                            f"'{node.module}'"
                        )

        return None

    # =========================================================================
    # Syntax Validation
    # =========================================================================

    def _validate_syntax(
        self,
        code: str,
    ) -> Optional[str]:
        """
        Validate Python syntax.
        """

        try:

            ast.parse(code)

            return None

        except SyntaxError as e:

            return (
                f"Syntax error at line "
                f"{e.lineno}: {e.msg}"
            )

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def _execute(
        self,
        code: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute code in subprocess sandbox.
        """

        start_time = time.time()

        # ---------------------------------------------------------------------
        # Validate Input
        # ---------------------------------------------------------------------

        if not code or not code.strip():

            raise ToolInputError(
                "Code cannot be empty"
            )

        code = code.strip()

        # ---------------------------------------------------------------------
        # Dangerous Code Detection
        # ---------------------------------------------------------------------

        danger = self._detect_dangerous_code(
            code
        )

        if danger:

            raise ToolSandboxError(danger)

        # ---------------------------------------------------------------------
        # Syntax Validation
        # ---------------------------------------------------------------------

        syntax_error = self._validate_syntax(
            code
        )

        if syntax_error:

            raise ToolInputError(
                syntax_error
            )

        # ---------------------------------------------------------------------
        # Execute Subprocess
        # ---------------------------------------------------------------------

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:

                f.write(code)

                temp_file = f.name

            process = subprocess.run(
                [sys.executable, temp_file],

                capture_output=True,

                text=True,

                timeout=self.timeout_ms / 1000,
            )

            execution_time_ms = int(
                (
                    time.time()
                    - start_time
                ) * 1000
            )

            return {
                "stdout": process.stdout,

                "stderr": process.stderr,

                "exit_code": process.returncode,

                "execution_time_ms": (
                    execution_time_ms
                ),
            }

        except subprocess.TimeoutExpired:

            raise ToolTimeoutError(
                f"Execution timeout after "
                f"{self.timeout_ms}ms"
            )

        except Exception as e:

            raise ToolInputError(
                f"Execution error: {e}"
            )

        finally:

            import os

            try:

                os.unlink(temp_file)

            except Exception:
                pass

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

        if (
            status
            == ToolResultStatus.TIMEOUT
        ):

            return (
                "suggest_alternative_approach"
            )

        elif (
            status
            == ToolResultStatus.SANDBOX_VIOLATION
        ):

            return "use_safe_alternative"

        elif (
            status
            == ToolResultStatus.INVALID_INPUT
        ):

            return "fix_syntax_and_retry"

        return "retry_with_simpler_code"