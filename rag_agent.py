import os
import logging
import re
import threading
import time
import dspy
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from core.retrieval import get_lancedb_connection, get_or_create_vector_store, process_pdf, process_web, get_available_documents, register_document, get_document_texts, execute_retrieval_pipeline
from core.agents import QueryOptimizer, get_web_search_agent, get_rag_agent, get_memory_agent
from utils.text_processing import clean_reasoning_output, parse_model_response
from utils.system_checks import is_ollama_installed, get_ollama_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initializes necessary Streamlit session state variables."""
    defaults = {
        'model_version': "llama3.2:latest",
        'vector_store': None,
        'processed_documents': [],
        'history': [],
        'use_web_search': False,
        'force_web_search': False,
        'similarity_threshold': 0.5,
        'rag_enabled': True,
        'use_cloud': False,
        'cloud_api_key': "",
        'cloud_provider': "OpenAI",
        'selected_docs': None,
        # Pipeline thread state
        '_is_processing': False,
        '_stop_event': None,
        '_pipeline_result': None,
        '_optimized_query_info': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Restore processed_documents from the DB registry if the session was cleared (e.g. page refresh)
    if not st.session_state.processed_documents:
        try:
            db = get_lancedb_connection()
            if db is not None:
                docs_in_db = get_available_documents(db)
                if docs_in_db:
                    st.session_state.processed_documents = list(docs_in_db)
                    # Default selected_docs to all known documents if not yet set
                    if st.session_state.selected_docs is None:
                        st.session_state.selected_docs = list(docs_in_db)
                    logger.info(f"Restored {len(docs_in_db)} documents from registry: {docs_in_db}")
        except Exception as e:
            logger.warning(f"Could not restore documents from registry on startup: {e}")

def should_use_vector_search(cfg: dict | None = None) -> bool:
    if cfg is not None:
        return cfg['rag_enabled'] and not cfg['force_web_search'] and cfg['vector_store'] is not None
    return st.session_state.rag_enabled and not st.session_state.force_web_search and st.session_state.vector_store is not None

def should_use_web_fallback(current_context: str, cfg: dict | None = None) -> bool:
    if cfg is not None:
        return (cfg['force_web_search'] or (cfg['rag_enabled'] and not current_context)) and cfg['use_web_search']
    return (st.session_state.force_web_search or (st.session_state.rag_enabled and not current_context)) and st.session_state.use_web_search

def render_model_settings():
    st.sidebar.header("📦 Model Selection")
    
    ollama_installed = is_ollama_installed()
    if not ollama_installed:
        st.sidebar.warning("⚠️ Ollama is not installed. Falling back to Cloud API.")
        ollama_models = []
    else:
        ollama_models = get_ollama_models()
        if not ollama_models:
            st.sidebar.warning("⚠️ Ollama is installed but not running or no models pulled. Falling back to Cloud API.")

    st.session_state.use_cloud = False

    if ollama_models:
        st.session_state.use_cloud = st.sidebar.checkbox("Use Cloud API instead of Ollama", value=False)
        
    if not ollama_models or st.session_state.use_cloud:
        st.session_state.use_cloud = True
        st.session_state.cloud_provider = st.sidebar.selectbox("Select Cloud Provider", options=["OpenAI"])
        st.session_state.cloud_api_key = st.sidebar.text_input(f"{st.session_state.cloud_provider} API Key", type="password")
        if st.session_state.cloud_provider == "OpenAI":
            st.session_state.model_version = st.sidebar.selectbox("Select Model", options=["gpt-4o-mini", "gpt-4o"])
            
        if not st.session_state.cloud_api_key:
            st.sidebar.info("Please enter your API Key to proceed.")
    else:
        # We have local models!
        st.session_state.model_version = st.sidebar.selectbox(
            "Select Local Model",
            options=ollama_models,
            index=0,
            help="These are the models currently pulled in your local Ollama."
        )
        st.sidebar.info(f"Using **{st.session_state.model_version}** locally.")

def render_rag_settings():
    st.sidebar.header("🔍 RAG Configuration")
    st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG Mode", value=st.session_state.rag_enabled)

    # Clear Chat Button
    if st.sidebar.button("🗑️ Clear Chat History"):
        logger.info("Chat history cleared.")
        st.session_state.history = []
        st.rerun()

    # Show Configuration only if RAG is enabled
    if st.session_state.rag_enabled:
        st.sidebar.header("🎯 Search Configuration")
        st.session_state.similarity_threshold = st.sidebar.slider(
            "Document Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Lower values are more strict."
        )

def render_web_settings():
    st.sidebar.header("🌐 Web Search Configuration")
    st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)

def render_sidebar():
    """Renders the settings and configuration sidebar."""
    st.sidebar.header("🤖 Agent Configuration")
    render_model_settings()
    render_rag_settings()
    render_web_settings()

_VAGUE_DOC_REFS = {
    "this document", "the document", "the file", "the pdf", "the uploaded",
    "the content", "it says", "what does it", "summarize this", "summarize it",
    "explain this", "tell me about this", "the text", "the paper", "the report",
}

_GENERAL_PREFIXES = (
    "use ", "call ", "run ", "check ", "show ", "list ", "get ",
    "search the web", "look up", "find on the web",
    "hello", "hi ", "hey ", "how are you",
)

_TOOL_KEYWORDS = (
    "mempalace", "duckduckgo", "web search", "status", "memory",
)


def _is_general_query(prompt: str) -> bool:
    """Return True if the query is obviously NOT about retrieving document content.

    Bypasses DSPy optimization for tool invocations, greetings, and general
    questions that contain no vague document references — preserving the query
    exactly as the user typed it.
    """
    lower = prompt.lower().strip()

    # Contains a vague document reference → needs optimization
    if any(ref in lower for ref in _VAGUE_DOC_REFS):
        return False

    # Starts with a command/tool verb → general instruction, skip
    if any(lower.startswith(p) for p in _GENERAL_PREFIXES):
        return True

    # Mentions specific tool names → skip
    if any(kw in lower for kw in _TOOL_KEYWORDS):
        return True

    return False


def optimize_search_query(prompt: str, cfg: dict | None = None) -> tuple[str, str | None]:
    """Optimizes search query using DSPy.
    
    Returns (optimized_query, optimized_query_info) where the second element
    is a display string to show in the UI (or None). Accepts an optional cfg
    snapshot so it can be safely called from a background thread.
    """
    optimized_query = prompt
    optimized_info = None
    if should_use_vector_search(cfg):
        # Fast bypass: skip DSPy for general commands, tool calls, and greetings
        if _is_general_query(prompt):
            logger.info(f"Query bypass (general): '{prompt[:60]}'")
            return prompt, None
        try:
            # Use selected_docs as the source of truth — only inform the LLM
            # about the documents the user has actually chosen to query against.
            if cfg is not None:
                active_docs = cfg.get('selected_docs') or cfg.get('processed_documents', [])
                use_cloud = cfg.get('use_cloud', False)
                cloud_provider = cfg.get('cloud_provider', '')
                cloud_api_key = cfg.get('cloud_api_key', '')
                model_version = cfg.get('model_version', 'llama3.2:latest')
            else:
                active_docs = getattr(st.session_state, 'selected_docs', None) or st.session_state.processed_documents
                use_cloud = getattr(st.session_state, "use_cloud", False)
                cloud_provider = getattr(st.session_state, "cloud_provider", "")
                cloud_api_key = getattr(st.session_state, "cloud_api_key", "")
                model_version = st.session_state.model_version

            processed_document_info = "No documents have been selected for search."
            if active_docs:
                doc_list = []
                for doc_name in active_docs:
                    if doc_name.startswith("http"):
                        doc_list.append(f"a web page from '{doc_name}'")
                    else:
                        doc_list.append(f"a PDF file named '{doc_name}'")
                processed_document_info = "The user is searching within: " + ", ".join(doc_list) + "."

            # Use selected model for rephrasing
            if use_cloud and cloud_provider == "OpenAI" and cloud_api_key:
                os.environ["OPENAI_API_KEY"] = cloud_api_key
                lm = dspy.LM(f"openai/{model_version}")
            else:
                lm = dspy.LM(f"ollama_chat/{model_version}", api_base="http://localhost:11434")

            with dspy.context(lm=lm):
                optimizer = QueryOptimizer()
                result = optimizer(original_query=prompt, processed_document_info=processed_document_info)
                # Clean the optimized query of reasoning artifacts
                optimized_query = clean_reasoning_output(result.optimized_query)
                optimized_info = f"✨ Optimized Search Query: {optimized_query}"
        except Exception as e:
            logger.error(f"DSPy optimization failed: {e}")
    return optimized_query, optimized_info

def _search_vector_store(query: str, cfg: dict | None = None) -> tuple[str, list]:
    """Helper function to run vector store retrieval."""
    if should_use_vector_search(cfg):
        if cfg is not None:
            selected = cfg.get('selected_docs')
            threshold = cfg.get('similarity_threshold', 0.5)
            vector_store = cfg.get('vector_store')
        else:
            selected = getattr(st.session_state, 'selected_docs', None)
            threshold = st.session_state.similarity_threshold
            vector_store = st.session_state.vector_store
        db_conn = get_lancedb_connection()
        return execute_retrieval_pipeline(vector_store, query, threshold, selected, db_conn, cfg)
    return "", []

def _search_web_fallback(query: str, current_context: str, cfg: dict | None = None) -> str:
    """Helper function to run web fallback if required.
    NOTE: st.info/st.error are NOT called here when cfg is set (background thread).
    """
    context = current_context
    if should_use_web_fallback(current_context, cfg):
        try:
            agent_cfg = cfg if cfg is not None else _snapshot_session_state()
            web_agent = get_web_search_agent(agent_cfg)
            web_results = web_agent.run(query).content
            if web_results:
                context = f"Web Search Results:\n{web_results}"
                if cfg is None:  # main thread only
                    st.info("🌐 Using web search results.")
        except Exception as e:
            if cfg is None:  # main thread only
                st.error(f"❌ Web search error: {str(e)}")
            else:
                logger.error(f"Web search fallback error: {e}")
    return context

def retrieve_agent_context(query: str, cfg: dict | None = None) -> tuple[str, list]:
    """Handles all logic for retrieving context via Vector Store or Web Search."""
    # 1. Vector Store Search
    context, docs = _search_vector_store(query, cfg)
    
    # 2. Web Search Fallback
    context = _search_web_fallback(query, context, cfg)

    return context, docs

def _process_and_register_source(db_conn, source_id: str, texts: list):
    """A generic helper to update state, vector store, and DB after processing text."""
    if texts and db_conn is not None:
        full_text = "\n\n".join([doc.page_content for doc in texts])
        st.session_state.vector_store = get_or_create_vector_store(
            db_conn, texts,
            use_cloud=st.session_state.get('use_cloud', False),
            cloud_provider=st.session_state.get('cloud_provider', ''),
            cloud_api_key=st.session_state.get('cloud_api_key', '')
        )
        st.session_state.processed_documents.append(source_id)
        register_document(db_conn, source_id, full_text)
        
        if st.session_state.selected_docs is not None:
            st.session_state.selected_docs.append(source_id)
        else:
            st.session_state.selected_docs = [source_id]
        
        get_available_documents.clear()
        return True
    return False

def handle_document_upload(db_conn):
    """Handles the sidebar logic for document uploads."""
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")

    # Initialize vector store from existing DB if not yet in session
    if db_conn is not None and st.session_state.vector_store is None:
        st.session_state.vector_store = get_or_create_vector_store(
            db_conn,
            use_cloud=st.session_state.get('use_cloud', False),
            cloud_provider=st.session_state.get('cloud_provider', ''),
            cloud_api_key=st.session_state.get('cloud_api_key', '')
        )

    if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
        with st.spinner('Processing PDF...'):
            texts = process_pdf(uploaded_file)
            if _process_and_register_source(db_conn, uploaded_file.name, texts):
                st.success(f"✅ Added PDF: {uploaded_file.name}")

    if web_url and web_url not in st.session_state.processed_documents:
        with st.spinner('Processing URL...'):
            texts = process_web(web_url)
            if _process_and_register_source(db_conn, web_url, texts):
                st.success(f"✅ Added URL: {web_url}")

    st.sidebar.header("📚 Context Selection")
    if db_conn is not None:
        available_docs = set(get_available_documents(db_conn))
        available_docs.update(st.session_state.processed_documents)
        available_docs = sorted(list(available_docs))
        
        if available_docs:
            st.session_state.selected_docs = st.sidebar.multiselect(
                "Filter RAG Context by Source",
                options=available_docs,
                default=available_docs if st.session_state.selected_docs is None else [d for d in st.session_state.selected_docs if d],
                help="Only text from the selected documents will be used to answer your questions."
            )
        else:
            st.sidebar.info("No documents available in database.")

def _compute_response(context: str, prompt: str, cfg: dict) -> tuple[str, str]:
    """Pure computation — runs in background thread. No st.* calls."""
    rag_agent = get_rag_agent(cfg)
    
    # Format history
    history_str = ""
    if cfg.get('history'):
        history_str = "Conversation History:\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in cfg['history']]) + "\n\n"
        
    full_prompt = history_str
    if context:
        full_prompt += f"Context: {context}\n\n"
    full_prompt += f"Question: {prompt}"
    
    logger.info(f"Final prompt to RAG agent: {full_prompt[:200]}...")
    response = rag_agent.run(full_prompt)
    return parse_model_response(response.content)  # (thinking, answer)


def _render_response(thinking: str, answer: str, docs: list):
    """Render a completed response in the main thread."""
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        if thinking:
            with st.expander("🤔 Thinking Process"):
                st.markdown(thinking)
        st.markdown(answer)
        if docs:
            with st.expander("🔍 Sources"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"**Source {i}**: {doc.page_content[:200]}...")


def _snapshot_session_state() -> dict:
    """Capture a plain-dict snapshot of all session state values needed by the
    background pipeline. This must be called from the main Streamlit thread.
    """
    return {
        'rag_enabled': st.session_state.get('rag_enabled', True),
        'force_web_search': st.session_state.get('force_web_search', False),
        'use_web_search': st.session_state.get('use_web_search', False),
        'similarity_threshold': st.session_state.get('similarity_threshold', 0.5),
        'vector_store': st.session_state.get('vector_store'),
        'selected_docs': st.session_state.get('selected_docs'),
        'processed_documents': list(st.session_state.get('processed_documents', [])),
        'use_cloud': st.session_state.get('use_cloud', False),
        'cloud_provider': st.session_state.get('cloud_provider', ''),
        'cloud_api_key': st.session_state.get('cloud_api_key', ''),
        'model_version': st.session_state.get('model_version', 'llama3.2:latest'),
        'history': list(st.session_state.get('history', [])),
    }


def _run_pipeline(prompt: str, stop_event: threading.Event, cfg: dict):
    """Full query pipeline running in a background daemon thread.
    Checks stop_event between stages. Writes result to session state.
    Uses a pre-captured cfg snapshot instead of touching st.session_state directly.
    """
    try:
        optimized_query, optimized_info = optimize_search_query(prompt, cfg)
        # Safely store the display string back into session_state (write is thread-safe)
        st.session_state._optimized_query_info = optimized_info
        if stop_event.is_set():
            st.session_state._pipeline_result = ('stopped', None)
            return

        context, docs = retrieve_agent_context(optimized_query, cfg)
        if stop_event.is_set():
            st.session_state._pipeline_result = ('stopped', None)
            return

        thinking, answer = _compute_response(context, prompt, cfg)
        if stop_event.is_set():
            st.session_state._pipeline_result = ('stopped', None)
            return

        st.session_state._pipeline_result = ('done', {
            'thinking': thinking,
            'answer': answer,
            'docs': docs,
        })
        
        # Fire off memory save
        if not stop_event.is_set():
            _trigger_memory_update(prompt, answer, cfg)
            
    except Exception as e:
        if not stop_event.is_set():
            logger.error(f"Pipeline error: {e}")
            st.session_state._pipeline_result = ('error', str(e))


def _trigger_memory_update(prompt: str, answer: str, cfg: dict):
    """Fires the memory agent in the background to summarize the latest interaction."""
    mem_agent = get_memory_agent(cfg)
    if mem_agent is None:
        return
        
    summary_prompt = f"""
    New Interaction to process:
    User: {prompt}
    Assistant: {answer}
    """
    logger.info("Triggering background memory supervisor...")
    # This will run synchronously inside the current background thread,
    # but since the UI pipeline result was already set, the user is not blocked.
    try:
        mem_agent.run(summary_prompt)
        logger.info("Background memory supervisor finished.")
    except Exception as e:
        logger.error(f"Background memory supervisor failed: {e}")

def main():
    st.set_page_config(page_title="Deepseek Local RAG", layout="wide")
    st.title("🐋 Local RAG Reasoning Agent")

    init_session_state()
    render_sidebar()

    # UI Layout — disable input while processing
    chat_col, toggle_col = st.columns([0.9, 0.1])
    with chat_col:
        is_disabled = st.session_state._is_processing and st.session_state._pipeline_result is None
        prompt = st.chat_input("Ask me anything...", disabled=is_disabled)
    with toggle_col:
        st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

    # Document Management
    if st.session_state.rag_enabled:
        db_conn = get_lancedb_connection()
        handle_document_upload(db_conn)
    elif not st.session_state.processed_documents:
        st.info("👋 Upload a document or enable RAG mode, or use the toggle to chat directly!")

    # Display Chat History
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Case 1: pipeline finished — render result ──────────────────────────
    if st.session_state._pipeline_result is not None:
        status, data = st.session_state._pipeline_result
        # Clear state before rendering to prevent double-render on rerun
        st.session_state._pipeline_result = None
        st.session_state._is_processing = False
        st.session_state._stop_event = None

        if st.session_state._optimized_query_info:
            st.info(st.session_state._optimized_query_info)
            st.session_state._optimized_query_info = None

        if status == 'done':
            _render_response(data['thinking'], data['answer'], data['docs'])
        elif status == 'stopped':
            st.warning("⏹ Query stopped.")
        elif status == 'error':
            st.error(f"❌ Response generation error: {data}")

    # ── Case 2: pipeline running — show stop button and poll ───────────────
    elif st.session_state._is_processing:
        stop_col, status_col = st.columns([0.15, 0.85])
        with stop_col:
            if st.button("⏹ Stop", type="secondary", key="stop_btn"):
                if st.session_state._stop_event:
                    st.session_state._stop_event.set()
                st.session_state._is_processing = False
                st.session_state._pipeline_result = ('stopped', None)
                st.rerun()
        with status_col:
            st.markdown("🤖 **Generating response…**")
        time.sleep(0.4)
        st.rerun()

    # ── Case 3: new prompt submitted — kick off background thread ──────────
    elif prompt:
        if not st.session_state.history or st.session_state.history[-1]["content"] != prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        stop_event = threading.Event()
        st.session_state._stop_event = stop_event
        st.session_state._is_processing = True
        st.session_state._pipeline_result = None
        st.session_state._optimized_query_info = None

        # Snapshot session state NOW (main thread) so the background thread
        # never needs to read from st.session_state directly.
        cfg = _snapshot_session_state()

        thread = threading.Thread(
            target=_run_pipeline,
            args=(prompt, stop_event, cfg),
            daemon=True,
        )
        add_script_run_ctx(thread, get_script_run_ctx())
        thread.start()
        st.rerun()


if __name__ == "__main__":
    main()
