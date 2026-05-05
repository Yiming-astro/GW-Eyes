from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.mcp import MultiMCPTools

CONFIG_DIR = Path(__file__).parent.parent / "src" / "config"
COLLECTOR_CONFIG_PATH = CONFIG_DIR / "collector.yaml"


def load_config_from_yaml(config_path: Path) -> dict[str, Any]:
    """Load config from a YAML file."""
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


DEFAULT_MCP_COMMAND = "python -m GW_Eyes.tools.collector_tools"

class CollectorClient:
    """
    Factory + lifecycle wrapper for the Collector agent and its MCP tools.
    """

    def __init__(
        self,
        mcp_command: Union[str, list[str]] = None,
        mcp_timeout_seconds: int = None,
        model_id: str = None,
        temperature: float = None,
        instructions: str = None,
    ) -> None:
        # Load config from YAML file
        config = load_config_from_yaml(COLLECTOR_CONFIG_PATH)

        # Use provided values or fall back to config file, then to defaults
        if mcp_command is not None:
            # Normalize to list
            self._mcp_commands = [mcp_command] if isinstance(mcp_command, str) else mcp_command
        else:
            config_cmd = config.get("mcp_command", DEFAULT_MCP_COMMAND)
            self._mcp_commands = config_cmd if isinstance(config_cmd, list) else [config_cmd]

        self._mcp_timeout_seconds = (
            mcp_timeout_seconds
            if mcp_timeout_seconds is not None
            else config.get("mcp_timeout_seconds", 1800)
        )
        self._model_id = model_id or os.environ.get("LLM_ID")
        self._temperature = (
            temperature
            if temperature is not None
            else config.get("temperature", 0.7)
        )
        self._instructions = (
            instructions
            if instructions is not None
            else config.get("instructions", "You are a scientific assistant specialized in multi-messenger astronomy.")
        )

        # Create MCPTools instance
        self.mcp_tools = MultiMCPTools(
            commands=self._mcp_commands,
            timeout_seconds=1800)
        self.agent: Optional[Agent] = None

    async def connect(self) -> Agent:
        # Connect all MCP tools
        await self.mcp_tools.connect()

        self.agent = Agent(
            model=OpenAILike(
                id=self._model_id,
                api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ["LLM_BASE_URL"],
                temperature=self._temperature,
            ),
            tools=[self.mcp_tools],
            instructions=self._instructions,
        )

        return self.agent

    async def close(self) -> None:
        # Close all MCP tools
        await self.mcp_tools.close()
