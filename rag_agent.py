import os
import tempfile
import logging
import re
from datetime import datetime
from typing import List, Optional
import streamlit as st
import bs4
import lancedb
import dspy

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


# Utility to clean reasoning tags from DeepSeek R1
def clean_reasoning_output(text: str) -> str:
    """Removes <think>...</think> tags and returns the actual content."""
    if not text:
        return ""
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Also remove any markdown headers that might be added by model
    clean_text = re.sub(r'^.*?optimized_query:?\s*', '', clean_text, flags=re.IGNORECASE).strip()
    return clean_text


# DSPy Signatures and Modules for Prompt Fine-tuning
class QuerySignature(dspy.Signature):
    """Refine and expand the user's query for better document retrieval in a RAG system.
    If the query contains vague references like 'this document', 'the uploaded file', or 'the content',
    use the 'processed_document_info' to make the query explicit and specific.
    Identify key terms, entities, and context to create a more effective search query.
    Return a single string that will be used for vector search.
    """
    original_query = dspy.InputField()
    processed_document_info = dspy.InputField(desc="Information about recently processed documents that are available for search.")
    optimized_query = dspy.OutputField(desc="A refined version of the query for search.")

class QueryOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(QuerySignature)

    def forward(self, original_query, processed_document_info):
        return self.generate_query(original_query=original_query, processed_document_info=processed_document_info)


# Constants
COLLECTION_NAME = "deepseek_rag_table"
LANCEDB_URI = os.path.abspath("./.lancedb")


# Streamlit App Initialization
st.set_page_config(page_title="Deepseek Local RAG", layout="wide")
st.title("🐋 Local RAG Reasoning Agent")

# Session State Initialization
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


# Sidebar Configuration
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
        value=0.5,
        help="Lower values are more strict."
    )

st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)


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

            all_content = [doc.page_content for doc in documents]
            st.session_state.all_docs_content = "\n\n".join(all_content)

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
    return Agent(
        name="Web Search Agent",
        model=Ollama(id="llama3.2"),
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
        # show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    return Agent(
        name="DeepSeek RAG Agent",
        model=Ollama(id=st.session_state.model_version),
        instructions="""You are an Intelligent Agent providing accurate answers.
        Focus on provided documents or web results. If context is provided, prioritize it.
        If no context is found for a specific document reference, ask the user to be more specific.
        Synthesize clearly and cite sources.""",
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
        # 1. Query Optimization with DSPy
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
                # Fallback to original prompt

        # 2. Document Search
        if st.session_state.rag_enabled and not st.session_state.force_web_search and st.session_state.vector_store:
            try:
                # 2.1 Threshold-based search
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 5, "score_threshold": st.session_state.similarity_threshold}
                )
                docs = retriever.invoke(optimized_query)

                # 2.2 Fallback to plain similarity search for broad queries
                if not docs:
                    logger.info("No documents found with threshold. Falling back to basic similarity search.")
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    context = st.session_state.all_docs_content

                if docs:
                    context = "\n\n".join([d.page_content for d in docs])
                    st.info(f"📊 Found {len(docs)} relevant document chunks.")
            except Exception as e:
                st.warning(f"⚠️ Document search issue: {e}")

        # 3. Web Search Fallback/Force
        if (st.session_state.force_web_search or (st.session_state.rag_enabled and not context)) and st.session_state.use_web_search:
            try:
                web_agent = get_web_search_agent()
                # Also use optimized_query here if available
                web_results = web_agent.run(optimized_query).content
                if web_results:
                    context = f"Web Search Results:\n{web_results}"
                    st.info("🌐 Using web search results.")
            except Exception as e:
                st.error(f"❌ Web search error: {str(e)}")

        # 4. Generate Final Response
        try:
            rag_agent = get_rag_agent()
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
            logger.info(f"Final prompt to RAG agent: {full_prompt[:200]}...")
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
