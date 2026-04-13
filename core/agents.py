import logging
import os
import subprocess
import streamlit as st
import dspy
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters
import shutil

logger = logging.getLogger(__name__)

MEMPALACE_DIR = "/Users/gedeon/Projects/AI/mempalace"


def _find_uv() -> str | None:
    """Locate the uv binary, checking common install paths since it may not be on PATH."""
    # 1. System PATH first
    uv = shutil.which("uv")
    if uv:
        return uv
    # 2. Common install locations
    candidates = [
        os.path.expanduser("~/.local/bin/uv"),
        os.path.expanduser("~/.cargo/bin/uv"),
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _ensure_palace_initialized():
    """Bootstrap ~/.mempalace/config.json and palace directory if not yet present.

    Checks for existing initialization first — skips the subprocess entirely
    if both the config file and palace directory are already present.
    """
    # Resolve palace_dir the same way MempalaceConfig does:
    # env var > config file default. This ensures the guard is correct
    # even when a custom palace path is configured.
    config_file = os.path.expanduser("~/.mempalace/config.json")
    palace_dir = (
        os.environ.get("MEMPALACE_PALACE_PATH")
        or os.environ.get("MEMPAL_PALACE_PATH")
        or os.path.expanduser("~/.mempalace/palace")
    )

    if os.path.isfile(config_file) and os.path.isdir(palace_dir):
        logger.debug("MemPalace already initialized — skipping.")
        return

    uv_bin = _find_uv()
    if uv_bin is None:
        logger.warning(
            "'uv' not found — MemPalace init skipped. "
            "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
        return
    try:
        result = subprocess.run(
            [uv_bin, "run", "python", "-c",
             "from mempalace.config import MempalaceConfig; cfg = MempalaceConfig(); cfg.init(); "
             "import os; os.makedirs(cfg.palace_path, exist_ok=True); "
             "print('palace_path=' + cfg.palace_path)"],
            cwd=MEMPALACE_DIR,
            capture_output=True,
            text=True,
            timeout=120,  # first run downloads packages; subsequent runs are ~1s
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("palace_path="):
                    logger.info(f"MemPalace initialized at: {line.split('=', 1)[1]}")
        else:
            logger.warning(f"MemPalace init warning: {result.stderr.strip() or result.stdout.strip()}")
    except subprocess.TimeoutExpired:
        logger.warning("MemPalace init timed out — skipping.")
    except Exception as e:
        logger.warning(f"MemPalace init failed: {e}")

class QuerySignature(dspy.Signature):
    """You are a search query optimizer for a RAG document retrieval system.

    Your ONLY job is to rephrase queries that are vague about WHICH document they refer to,
    using the 'processed_document_info' to make them explicit and specific.

    IMPORTANT RULES:
    - If the query is a general instruction, tool invocation, greeting, meta-command, or
      anything NOT about retrieving content from the listed documents, return it EXACTLY
      as-is without any modification.
    - If the query already clearly names what it is looking for, return it as-is.
    - Only modify the query when it contains vague references like 'this document',
      'the uploaded file', 'it', 'the content', 'summarize this', etc.
    - NEVER guess a document name from the examples. ONLY use the document names provided in 'processed_document_info'.

    Examples of queries to return UNCHANGED:
      - "Use mempalace_status to check your memory"
      - "What is the capital of France?"
      - "Hello, how are you?"
      - "Search the web for recent AI news"
      - "How about the food I should eat for muscle building?"

    Examples of queries to MODIFY (assuming 'Book.pdf' is the only processed document):
      - "Summarize this document" → "Summarize the key findings in 'Book.pdf'"
      - "What does it say about methodology?" → "What does 'Book.pdf' say about methodology?"
    """
    original_query = dspy.InputField()
    processed_document_info = dspy.InputField(desc="Information about documents available for search. Only use this to resolve vague references.")
    optimized_query = dspy.OutputField(desc="The original query returned unchanged if it is general/tool/meta, OR a rephrased query with explicit document names if vague document references were present.")

class QueryOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(QuerySignature)

    def forward(self, original_query, processed_document_info):
        return self.generate_query(original_query=original_query, processed_document_info=processed_document_info)

def _get_model(cfg: dict):
    """Helper to return the correct model instance dynamically."""
    if cfg.get("use_cloud") and cfg.get("cloud_provider") == "OpenAI" and cfg.get("cloud_api_key"):
        model_ver = cfg.get("model_version", "gpt-4o-mini")
        return OpenAIChat(id=model_ver, api_key=cfg["cloud_api_key"])
    else:
        # Use the model selected in the UI (e.g. llama3.2, mistral, etc.)
        model_ver = cfg.get("model_version", "llama3.2:latest")
        return Ollama(id=model_ver)

def get_web_search_agent(cfg: dict) -> Agent:
    """Initialize a web search agent."""
    return Agent(
        name="Web Search Agent",
        model=_get_model(cfg),
        tools=[DuckDuckGoTools()],
        instructions="""You are a web search expert. Your primary goal is to find and summarize information from the web that directly answers the user's query.

        Follow these steps:
        1. Use the DuckDuckGo search tools to gather relevant information.
        2. Prefer 'search' or 'duckduckgo_search' for general facts and data.
        3. Extract the most important and factual information pertinent to the user's question.
        4. Synthesize this information into a concise and clear summary or a list of key findings.
        5. Always include the URLs of the sources you used.
        6. If a search tool returns no results, try rephrasing or using a general search before giving up.
        7. If no relevant information can be found at all, explicitly state that you were unable to find an answer.
        """,
        markdown=True,
    )

async def _get_rag_agent_async(mcp_tools: MCPTools | None, cfg: dict) -> Agent:
    """Build the RAG agent inside a live MCP context. Must be called with MCPTools already entered."""
    tools = [mcp_tools] if mcp_tools is not None else []
    return Agent(
        name="Dynamic RAG Agent",
        model=_get_model(cfg),
        tools=tools,
        instructions="""You are an Intelligent Agent providing accurate answers.
        Focus on provided documents or web results. If context is provided, prioritize it.
        If no context is found for a specific document reference, ask the user to be more specific.
        You have access to MemPalace, a memory system, through your tools.
        IMPORTANT: At the start of EVERY session call mempalace_status first to load the memory protocol.
        Use mempalace_search to retrieve context about past decisions before answering.
        Synthesize clearly and cite sources.""",
        markdown=True,
    )


def get_rag_agent(cfg: dict):
    """Return a RAG agent. MCP tools are initialized via asyncio if uv is available.

    NOTE: deepseek-r1 does not support tool calling (Ollama 400 error).
    MCP tools only work with tool-capable models like llama3.1, qwen2.5, mistral-nemo etc.
    """
    _ensure_palace_initialized()

    uv_bin = _find_uv()
    if uv_bin is None:
        logger.warning("uv not found — MemPalace MCP tools will be unavailable.")
        import asyncio
        return asyncio.get_event_loop().run_until_complete(_get_rag_agent_async(None, cfg))

    mcp_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    mcp_tools = MCPTools(
        server_params=StdioServerParameters(
            command=uv_bin,
            args=["run", "python", "-m", "mempalace.mcp_server"],
            cwd=MEMPALACE_DIR,
            env=mcp_env,
        ),
        timeout_seconds=600,
    )

    async def _build():
        async with mcp_tools:
            logger.info(f"MCPTools initialized: {len(mcp_tools.functions)} tools available")
            agent = await _get_rag_agent_async(mcp_tools, cfg)
            response = await agent.arun(_build._prompt)
            return response

    _build._prompt = None  # set by caller

    # Return a lazy wrapper; actual run happens in _compute_response
    return _MCPAgentRunner(mcp_tools, mcp_env, uv_bin, cfg)


class _MCPAgentRunner:
    """Wraps MCPTools lifecycle so the agent is always run inside the async context."""

    def __init__(self, mcp_tools: MCPTools, env: dict, uv_bin: str, cfg: dict):
        self._mcp_tools = mcp_tools
        self._env = env
        self._uv_bin = uv_bin
        self._cfg = cfg

    def run(self, prompt: str):
        import asyncio
        return asyncio.run(self._arun(prompt))

    async def _arun(self, prompt: str):
        async with self._mcp_tools:
            logger.info(f"MCPTools ready: {len(self._mcp_tools.functions)} tools")
            agent = await _get_rag_agent_async(self._mcp_tools, self._cfg)
            return await agent.arun(prompt)


async def _get_memory_agent_async(mcp_tools: MCPTools | None, cfg: dict) -> Agent:
    """Build the Memory Agent inside a live MCP context."""
    tools = [mcp_tools] if mcp_tools is not None else []
    
    # Use cfg to build model safely inside the background thread
    model = _get_model(cfg)

    return Agent(
        name="Memory Saver Agent",
        model=model,
        tools=tools,
        instructions="""You are a Memory Supervisor. Your ONLY job is to save factoids, insights, and summaries into MemPalace.
        You will receive the latest interaction between the User and the Assistant.
        If there are any new facts, decisions, context, or conclusions reached, you MUST call 'mempalace_diary_write' to save them.
        Do NOT reply with a chat message. Just use the tool. If nothing is worth saving, simply return nothing.""",
        markdown=True,
    )


class _MCPMemoryRunner(_MCPAgentRunner):
    def __init__(self, mcp_tools: MCPTools, env: dict, uv_bin: str, cfg: dict):
        super().__init__(mcp_tools, env, uv_bin, cfg)

    async def _arun(self, prompt: str):
        async with self._mcp_tools:
            logger.info(f"MemoryAgent MCPTools ready: {len(self._mcp_tools.functions)} tools")
            agent = await _get_memory_agent_async(self._mcp_tools, self._cfg)
            return await agent.arun(prompt)


def get_memory_agent(cfg: dict) -> _MCPMemoryRunner | None:
    """Return a Memory agent capable of running tools in the background."""
    uv_bin = _find_uv()
    if uv_bin is None:
        return None

    mcp_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    mcp_tools = MCPTools(
        server_params=StdioServerParameters(
            command=uv_bin,
            args=["run", "python", "-m", "mempalace.mcp_server"],
            cwd=MEMPALACE_DIR,
            env=mcp_env,
        ),
        timeout_seconds=300,
    )
    return _MCPMemoryRunner(mcp_tools, mcp_env, uv_bin, cfg)
