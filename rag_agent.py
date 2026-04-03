import os
import logging
import re
import dspy
import streamlit as st

from core.retrieval import get_lancedb_connection, get_or_create_vector_store, process_pdf, process_web, get_available_documents, register_document, get_document_texts, execute_retrieval_pipeline
from core.agents import QueryOptimizer, get_web_search_agent, get_rag_agent
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
        'model_version': "deepseek-r1:7b",
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
        'selected_docs': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def should_use_vector_search() -> bool:
    return st.session_state.rag_enabled and not st.session_state.force_web_search and st.session_state.vector_store is not None

def should_use_web_fallback(current_context: str) -> bool:
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

def optimize_search_query(prompt: str) -> str:
    """Optimizes search query using DSPy."""
    optimized_query = prompt
    if should_use_vector_search():
        try:
            # Prepare document info for DSPy
            processed_document_info = "No documents have been processed recently."
            if st.session_state.processed_documents:
                doc_list = []
                for doc_name in st.session_state.processed_documents:
                    if doc_name.endswith('.pdf'):
                        doc_list.append(f"a PDF file named '{doc_name}'")
                    else:
                        doc_list.append(f"a web page from '{doc_name}'")
                processed_document_info = "Recently processed documents include: " + ", ".join(doc_list) + "."

            # Use selected model for rephrasing
            if getattr(st.session_state, "use_cloud", False) and getattr(st.session_state, "cloud_provider", "") == "OpenAI" and getattr(st.session_state, "cloud_api_key", ""):
                os.environ["OPENAI_API_KEY"] = st.session_state.cloud_api_key
                lm = dspy.LM(f"openai/{st.session_state.model_version}")
            else:
                lm = dspy.LM(f"ollama_chat/{st.session_state.model_version}", api_base="http://localhost:11434")

            with dspy.context(lm=lm):
                optimizer = QueryOptimizer()
                result = optimizer(original_query=prompt, processed_document_info=processed_document_info)
                # Clean the optimized query of reasoning artifacts
                optimized_query = clean_reasoning_output(result.optimized_query)
                st.info(f"✨ Optimized Search Query: {optimized_query}")
        except Exception as e:
            logger.error(f"DSPy optimization failed: {e}")
    return optimized_query

def _search_vector_store(query: str) -> tuple[str, list]:
    """Helper function to run vector store retrieval."""
    if should_use_vector_search():
        selected = getattr(st.session_state, 'selected_docs', None)
        threshold = st.session_state.similarity_threshold
        db_conn = get_lancedb_connection()
        return execute_retrieval_pipeline(st.session_state.vector_store, query, threshold, selected, db_conn)
    return "", []

def _search_web_fallback(query: str, current_context: str) -> str:
    """Helper function to run web fallback if required."""
    context = current_context
    if should_use_web_fallback(current_context):
        try:
            web_agent = get_web_search_agent()
            web_results = web_agent.run(query).content
            if web_results:
                context = f"Web Search Results:\n{web_results}"
                st.info("🌐 Using web search results.")
        except Exception as e:
            st.error(f"❌ Web search error: {str(e)}")
    return context

def retrieve_agent_context(query: str) -> tuple[str, list]:
    """Handles all logic for retrieving context via Vector Store or Web Search."""
    # 1. Vector Store Search
    context, docs = _search_vector_store(query)
    
    # 2. Web Search Fallback
    context = _search_web_fallback(query, context)

    return context, docs

def _process_and_register_source(db_conn, source_id: str, texts: list):
    """A generic helper to update state, vector store, and DB after processing text."""
    if texts and db_conn is not None:
        full_text = "\n\n".join([doc.page_content for doc in texts])
        st.session_state.vector_store = get_or_create_vector_store(db_conn, texts)
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
        st.session_state.vector_store = get_or_create_vector_store(db_conn)

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

def generate_and_parse_response(context: str, prompt: str, docs: list):
    """Generates the RAG agent response and updates the UI."""
    try:
        rag_agent = get_rag_agent()
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
        logger.info(f"Final prompt to RAG agent: {full_prompt[:200]}...")
        response = rag_agent.run(full_prompt)

        thinking, answer = parse_model_response(response.content)

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

    except Exception as e:
        st.error(f"❌ Response generation error: {str(e)}")

def main():
    st.set_page_config(page_title="Deepseek Local RAG", layout="wide")
    st.title("🐋 Local RAG Reasoning Agent")

    init_session_state()
    render_sidebar()

    # UI Layout
    chat_col, toggle_col = st.columns([0.9, 0.1])
    with chat_col:
        prompt = st.chat_input("Ask me anything...")
    with toggle_col:
        st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

    # Document Management
    if st.session_state.rag_enabled:
        db_conn = get_lancedb_connection()
        handle_document_upload(db_conn)
    elif not st.session_state.processed_documents:
        # Show prompt when document isn't processed yet
        st.info("👋 Upload a document or enable RAG mode, or use the toggle to chat directly!")
        
    # Display Chat History
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main Chat Logic
    if prompt:
        # Avoid displaying prompt twice if we just typed it, but add to history
        if not st.session_state.history or st.session_state.history[-1]["content"] != prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        with st.spinner("🤖 Processing..."):
            optimized_query = optimize_search_query(prompt)
            context, docs = retrieve_agent_context(optimized_query)
            generate_and_parse_response(context, prompt, docs)

if __name__ == "__main__":
    main()
