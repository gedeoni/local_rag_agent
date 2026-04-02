import os
import logging
import re
import dspy
import streamlit as st

from core.retrieval import get_lancedb_connection, get_or_create_vector_store, process_pdf, process_web
from core.agents import QueryOptimizer, get_web_search_agent, get_rag_agent
from utils.text_processing import clean_reasoning_output

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
    if 'model_version' not in st.session_state:
        st.session_state.model_version = "deepseek-r1:7b"
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'use_web_search' not in st.session_state:
        st.session_state.use_web_search = False
    if 'force_web_search' not in st.session_state:
        st.session_state.force_web_search = False
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.5
    if 'rag_enabled' not in st.session_state:
        st.session_state.rag_enabled = True

def render_sidebar():
    """Renders the settings and configuration sidebar."""
    st.sidebar.header("🤖 Agent Configuration")

    # Model Selection
    st.sidebar.header("📦 Model Selection")
    st.session_state.model_version = st.sidebar.radio(
        "Select Model Version",
        options=["deepseek-r1:1.5b", "deepseek-r1:7b"],
        index=1,
        help="Choose based on your hardware capabilities."
    )
    st.sidebar.info(f"Using {st.session_state.model_version}. Make sure to run `ollama pull {st.session_state.model_version}`")

    # RAG Mode Toggle
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

    st.sidebar.header("🌐 Web Search Configuration")
    st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)

def optimize_search_query(prompt: str) -> str:
    """Optimizes search query using DSPy."""
    optimized_query = prompt
    if st.session_state.rag_enabled and not st.session_state.force_web_search and st.session_state.vector_store:
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
    context = ""
    docs = []
    if st.session_state.rag_enabled and not st.session_state.force_web_search and st.session_state.vector_store:
        try:
            # Threshold-based search
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": st.session_state.similarity_threshold}
            )
            docs = retriever.invoke(query)

            # Fallback to plain similarity search for broad queries
            if not docs:
                logger.info("No documents found with threshold. Falling back to basic similarity search.")
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)
                
                # If the fallback retriever still returns nothing, use raw docs as absolute last resort
                if not docs and 'all_docs_content' in st.session_state:
                    raw_text = st.session_state.all_docs_content
                    # max_chars = 15000 # ~3,500 to 4,000 tokens (safe for DeepSeek 7B window)
                    # if len(raw_text) > max_chars:
                    #     context += "\n\n...[Content Truncated due to context window limits]..."
                    context = raw_text

            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(f"📊 Found {len(docs)} relevant document chunks.")
        except Exception as e:
            st.warning(f"⚠️ Document search issue: {e}")
    return context, docs

def _search_web_fallback(query: str, current_context: str) -> str:
    """Helper function to run web fallback if required."""
    context = current_context
    if (st.session_state.force_web_search or (st.session_state.rag_enabled and not context)) and st.session_state.use_web_search:
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
            if texts and db_conn is not None:
                st.session_state.vector_store = get_or_create_vector_store(db_conn, texts)
                st.session_state.processed_documents.append(uploaded_file.name)
                st.success(f"✅ Added PDF: {uploaded_file.name}")

    if web_url and web_url not in st.session_state.processed_documents:
        with st.spinner('Processing URL...'):
            texts = process_web(web_url)
            if texts and db_conn is not None:
                st.session_state.vector_store = get_or_create_vector_store(db_conn, texts)
                st.session_state.processed_documents.append(web_url)
                st.success(f"✅ Added URL: {web_url}")

    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            st.sidebar.text(f"📄 {source}" if source.endswith('.pdf') else f"🌐 {source}")

def generate_and_parse_response(context: str, prompt: str, docs: list):
    """Generates the RAG agent response and updates the UI."""
    try:
        rag_agent = get_rag_agent()
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
        logger.info(f"Final prompt to RAG agent: {full_prompt[:200]}...")
        response = rag_agent.run(full_prompt)

        # Handle thinking process for DeepSeek R1
        content = response.content
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        else:
            thinking, answer = None, content

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
