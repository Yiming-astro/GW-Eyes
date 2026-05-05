from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, List, Union

import yaml
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools.mcp import MCPTools

from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.tools.mcp import MultiMCPTools

CONFIG_DIR = Path(__file__).parent.parent / "src" / "config"
EXECUTOR_CONFIG_PATH = CONFIG_DIR / "executor.yaml"


def load_config_from_yaml(config_path: Path) -> dict[str, Any]:
    """Load config from a YAML file."""
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


DEFAULT_MCP_COMMAND = "python -m GW_Eyes.tools.executor_tools"

class ExecutorClient:
    """
    Factory + lifecycle wrapper for the Executor agent and its MCP tools.
    Supports optional RAG knowledge mounting.
    """

    def __init__(
        self,
        mcp_command: Union[str, List[str]] = None,
        mcp_timeout_seconds: int = None,
        model_id: str = None,
        temperature: float = None,
        instructions: str = None,
        enable_rag: bool = False,
        knowledge_path: str = None,
        vectordb_uri: str = None,
        vectordb_table: str = None,
        embedding_model: str = None,
        embedding_dim: int = None,
    ) -> None:
        # Load config from YAML file
        config = load_config_from_yaml(EXECUTOR_CONFIG_PATH)

        # MCP tools settings
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
        # Create MCPTools instance for each command
        self.mcp_tools = MultiMCPTools(commands=self._mcp_commands,
                                       timeout_seconds=self._mcp_timeout_seconds)
        self.agent: Optional[Agent] = None

        # LLM settings
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

        # RAG knowledge settings
        self._enable_rag = enable_rag
        self._knowledge_path = (
            knowledge_path
            if knowledge_path is not None
            else config.get("knowledge_path", "GW_Eyes/knowledge/")
        )
        self._vectordb_uri = (
            vectordb_uri
            if vectordb_uri is not None
            else config.get("vectordb_uri", "GW_Eyes/data/lancedb")
        )
        self._vectordb_table = (
            vectordb_table
            if vectordb_table is not None
            else config.get("vectordb_table", "gw_knowledge")
        )
        self._embedding_model = embedding_model or os.environ.get("LLM_EMBEDDING_ID")
        self._embedding_dim = embedding_dim or int(os.environ.get("LLM_EMBEDDING_DIM", 0))

    def _build_knowledge(self) -> Knowledge:
        embedder = OpenAIEmbedder(
            id=self._embedding_model,
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
            enable_batch=True,
            dimensions=self._embedding_dim,
        )

        vector_db = LanceDb(
            uri=self._vectordb_uri,
            table_name=self._vectordb_table,
            embedder=embedder,
        )

        knowledge = Knowledge(vector_db=vector_db)

        # Insert documents (safe to call repeatedly; LanceDB handles persistence)
        knowledge.insert(path=self._knowledge_path)

        return knowledge

    async def connect(self) -> Agent:
        # Connect all MCP tools
        await self.mcp_tools.connect()
        
        knowledge = None
        add_knowledge_to_context = False

        if self._enable_rag:
            knowledge = self._build_knowledge()
            # search with RAG when necessary
            add_knowledge_to_context = False

        self.agent = Agent(
            model=OpenAILike(
                id=self._model_id,
                api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ["LLM_BASE_URL"],
                temperature=self._temperature,
            ),
            tools=[self.mcp_tools],
            knowledge=knowledge,
            add_knowledge_to_context=add_knowledge_to_context,
            instructions=self._instructions,
        )

        return self.agent

    async def close(self) -> None:
        # Close all MCP tools
        await self.mcp_tools.close()
