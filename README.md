# 🐋 Local Dynamic RAG & Memory Agent

A state-of-the-art, privacy-focused Retrieval-Augmented Generation (RAG) system built with Python, Streamlit, and the **Model Context Protocol (MCP)**. This agent provides a unified interface for document analysis, web search, and long-term semantic memory.

## 🚀 Core Features

*   **Local-First Intelligence**: Default support for local LLMs via **Ollama** (e.g., DeepSeek-R1, Llama 3.2), ensuring data privacy.
*   **MemPalace Semantic Memory**: Integrated long-term memory system using **MCP**. The agent autonomously stores and retrieves past decisions, facts, and conversation context.
*   **Asynchronous Processing**: Non-blocking query pipeline. A background thread handles optimization, retrieval, and generation while keeping the UI responsive.
*   **Recursive Memory Archiving**: A secondary **Memory Saver Agent** listens to every interaction and asynchronously files important insights into the Memory Palace.
*   **Hybrid Retrieval Engine**:
    *   **Vector Search**: Precise semantic retrieval via **LanceDB** with built-in **Self-Healing handle recovery**.
    *   **Resilient Web Search**: Multi-stage web searching using **DuckDuckGo**. Intelligent fallback that handles `403 Forbidden` errors by switching between sources and search snippets.
    *   **Raw Text Registry**: Last-resort full-text lookup to prevents "I don't know" responses.
*   **DSPy Query Optimization**: Uses Chain-of-Thought reasoning to resolve vague references (e.g., "Summarize this pdf") into specific, document-aware search queries.

---

## 🏗️ Architectural Infrastructure

The codebase is designed with strict separation of concerns across a clean directory structure in mind:

### 1. Directory Structure
-   **`rag_agent.py`**: The main UI orchestrator. Manages thread-safe session state and background pipeline coordination.
-   **`core/agents.py`**: Home of the agent definitions (RAG, Web, and Memory agents) and the DSPy optimization layer.
-   **`core/retrieval.py`**: Unified database and vector pipeline management using LanceDB and Langchain.
-   **[mempalace/](mempalace/README.md)**: A self-contained semantic memory project integrated as an MCP server.
-   **`utils/`**: Shared utilities for text cleaning and system compatibility checks.

### 2. The Background Pipeline
To prevent Streamlit UI "freezing" during complex reasoning/retrieval tasks:
1.  **State Snapshots**: The system captures a plain-dict "snapshot" of the current UI state to safely pass into a background thread.
2.  **Thread Worker**: A daemon thread runs the full RAG cycle (Optimize → Retrieve → Compute → Save).
3.  **UI Interleaving**: The main thread polls for results and renders components (Thinking process, Sources, etc.) as they become available.

### 3. Reliability & Self-Healing
To ensure a robust user experience across different models and environments:
-   **MCP Aliasing**: Handles inconsistencies in LLM tool calls. If a model sends a parameter named `q` instead of `query`, the server automatically re-maps it.
-   **RAG Handle Recovery**: The retrieval pipeline automatically detects and repairs lost database handles, ensuring vector search remains active during long sessions.
-   **Anti-Block Web Strategy**: The web agent avoids brittle scraping; it prioritizes search snippets and automatically switches providers if a site blocks the request (e.g., handles AccuWeather 403 errors).

---

## 🔍 Key Workflows

### Semantic Memory Management
When you chat, the agent doesn't just "remember" recent history; it uses **MemPalace** for long-term archiving:
*   **Contextual Retrieval**: Before answering, the agent evaluates if your question relates to your past. If it does, it calls `mempalace_search` or `mempalace_kg_query`.
*   **Automatic Filing**: After every interaction, the **Memory Saver Agent** extracts key factoids and files them in **AAAK format** (a compressed, entity-coded memory dialect) for future use.

### Adaptive Search Bypass
Not every question needs a document search. The system uses a **heuristic bypass** to identify greetings, commands, and general knowledge questions. This avoids wasting compute on RAG for queries like "Hello" or "How are you?".

---

## 🛠️ Installation & Setup

1.  **Dependencies**: Install via `uv` or pip:
    ```bash
    uv pip install -r requirements.txt
    ```
2.  **Ollama**: Ensure Ollama is running and you have pulled a tool-capable model (e.g., `llama3.2` or `qwen2.5`).
3.  **Run**:
    ```bash
    streamlit run rag_agent.py
    ```

---

> [!NOTE]
> This project is designed for users who want total control over their data. It can run 100% air-gapped using local models, while providing the same "smart" memory features found in cloud-based AI services.
