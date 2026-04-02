import os
import tempfile
import logging
from datetime import datetime
from typing import List, Optional
import streamlit as st
import bs4
import lancedb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from agno.agent import Agent
from agno.models.ollama import Ollama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.embedder.ollama import OllamaEmbedder


class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-embed-text"):
        """
        Initialize the OllamaEmbeddings with a specific model.

        Args:
            model_name (str): The name of the model to use for embedding.
        """
        self.embedder = OllamaEmbedder(id=model_name, dimensions=768)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        try:
            logger.info(f"Generating embedding for text: {text[:50]}...")
            return self.embedder.get_embedding(text)
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            st.error(f"🔴 Embedding failed: {str(e)}. Make sure ollama is running and model '{self.embedder.id}' is pulled.")
            raise e


# Constants
COLLECTION_NAME = "deepseek_rag_table"
LANCEDB_URI = os.path.abspath("./.lancedb")


# Streamlit App Initialization
st.set_page_config(page_title="Deepseek Local RAG", layout="wide")
st.title("🐋 Deepseek Local RAG Reasoning Agent")

# Session State Initialization
if 'model_version' not in st.session_state:
    st.session_state.model_version = "deepseek-r1:1.5b"
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'exa_api_key' not in st.session_state:
    st.session_state.exa_api_key = ""
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = True


# Sidebar Configuration
st.sidebar.header("🤖 Agent Configuration")

# Model Selection
st.sidebar.header("📦 Model Selection")
st.session_state.model_version = st.sidebar.radio(
    "Select Model Version",
    options=["deepseek-r1:1.5b", "deepseek-r1:7b"],
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

if st.session_state.use_web_search:
    st.session_state.exa_api_key = st.sidebar.text_input(
        "Exa AI API Key", 
        type="password",
        value=st.session_state.exa_api_key,
        help="Optional: Use Exa AI for better results, else DuckDuckGo is used."
    )

    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)", 
        value=",".join(default_domains)
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]


# Utility Functions
@st.cache_resource
def get_lancedb_connection():
    """Initialize and cache LanceDB connection."""
    try:
        # Ensure the directory exists
        if not os.path.exists(LANCEDB_URI):
            logger.info(f"Creating LanceDB directory: {LANCEDB_URI}")
            os.makedirs(LANCEDB_URI, exist_ok=True)

        logger.info(f"Connecting to LanceDB at {LANCEDB_URI}")
        conn = lancedb.connect(LANCEDB_URI)
        logger.info(f"Successfully connected to LanceDB: {type(conn)}")
        return conn
    except Exception as e:
        logger.error(f"LanceDB connection failed: {str(e)}", exc_info=True)
        st.error(f"🔴 LanceDB connection failed: {str(e)}")
        return None


# Document Processing Functions
def process_pdf(file) -> List:
    """Process PDF file."""
    try:
        logger.info(f"Processing PDF: {file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split PDF into {len(chunks)} chunks")
            return chunks
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        st.error(f"📄 PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """Process web URL."""
    try:
        logger.info(f"Processing URL: {url}")
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()

        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split web content into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Web processing error: {str(e)}")
        st.error(f"🌐 Web processing error: {str(e)}")
        return []


# Vector Store Management
def get_or_create_vector_store(db, texts=None):
    """Get existing or create new vector store."""
    embeddings = OllamaEmbeddings()
    try:
        if texts:
            # Create or Append
            logger.info(f"Adding {len(texts)} documents to table '{COLLECTION_NAME}'")
            return LanceDB.from_documents(
                texts,
                embeddings,
                connection=db,
                table_name=COLLECTION_NAME,
                mode="append"
            )
        else:
            # Try to open existing
            logger.info(f"Opening existing table '{COLLECTION_NAME}'")
            return LanceDB(
                connection=db,
                embedding=embeddings,
                table_name=COLLECTION_NAME
            )
    except Exception as e:
        if texts:
            logger.error(f"Vector store creation error: {str(e)}")
            st.error(f"🔴 Vector store creation error: {str(e)}")
        else:
            logger.warning(f"Could not open existing table (might not exist yet): {str(e)}")
        return None

def get_web_search_agent() -> Agent:
    """Initialize a web search agent."""
    web_tools = []
    tool_name = "DuckDuckGo"

    if st.session_state.exa_api_key:
        web_tools.append(ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        ))
        tool_name = "Exa AI"
    else:
        # Revert to standard initialization to avoid Toolkit argument error
        web_tools.append(DuckDuckGoTools())

    return Agent(
        name="Web Search Agent",
        model=Ollama(id="llama3.2"),
        tools=web_tools,
        instructions=f"""You are a web search expert. Your primary goal is to find and summarize information from the web that directly answers the user's query.

        Follow these steps:
        1. Use the {tool_name} search tools.
        2. FOR DUCKDUCKGO: Prefer 'search' or 'duckduckgo_search' over 'search_news' for facts, or historical data.
        3. If you are looking for information about laws in Rwanda use this website https://rwandalii.org as much as possible.
        4. Extract the most important and factual information pertinent to the user's question.
        5. Synthesize this information into a concise and clear summary or a list of key findings.
        6. Always include the URLs of the sources you used.
        7. If a search tool returns no results, try rephrasing or using a general search before giving up.
        8. If no relevant information can be found at all, explicitly state that you were unable to find an answer.
        """,
        # show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    return Agent(
        name="DeepSeek RAG Agent",
        model=Ollama(id=st.session_state.model_version),
        instructions="""You are an Intelligent Agent providing accurate answers.
        Focus on provided documents or web results. Synthesize clearly and cite sources.""",
        # show_tool_calls=True,
        markdown=True,
    )


# UI Layout
chat_col, toggle_col = st.columns([0.9, 0.1])
with chat_col:
    prompt = st.chat_input("Ask me anything...")
with toggle_col:
    st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

# RAG Implementation
if st.session_state.rag_enabled:
    db_conn = get_lancedb_connection()

    # Initialize vector store from existing DB if not yet in session
    if db_conn is not None and st.session_state.vector_store is None:
        st.session_state.vector_store = get_or_create_vector_store(db_conn)

    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")

    if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
        with st.spinner('Processing PDF...'):
            texts = process_pdf(uploaded_file)
            logger.info(f"is text and dbconn set: {bool(texts)} {db_conn is not None}")
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

# Chat Logic
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    context = ""
    docs = []

    with st.spinner("🤖 Processing..."):
        # 1. Document Search
        if st.session_state.rag_enabled and not st.session_state.force_web_search and st.session_state.vector_store:
            try:
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 5, "score_threshold": st.session_state.similarity_threshold}
                )
                docs = retriever.invoke(prompt)
                if docs:
                    context = "\n\n".join([d.page_content for d in docs])
                    st.info(f"📊 Found {len(docs)} relevant document chunks.")
            except Exception as e:
                st.warning(f"⚠️ Document search issue: {e}")

        # 2. Web Search Fallback/Force
        if (st.session_state.force_web_search or (st.session_state.rag_enabled and not context)) and st.session_state.use_web_search:
            try:
                web_agent = get_web_search_agent()
                web_results = web_agent.run(prompt).content
                if web_results:
                    context = f"Web Search Results:\n{web_results}"
                    st.info("🌐 Using web search results.")
            except Exception as e:
                st.error(f"❌ Web search error: {str(e)}")

        # 3. Generate Final Response
        try:
            rag_agent = get_rag_agent()
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
            response = rag_agent.run(full_prompt)

            # Handle thinking process for DeepSeek R1
            content = response.content
            import re
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

elif not st.session_state.processed_documents and st.session_state.rag_enabled:
    st.info("👋 Upload a document to start RAG, or use the toggle to chat directly!")
