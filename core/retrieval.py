import os
import tempfile
import logging
from datetime import datetime
from typing import List
import hashlib

import bs4
import lancedb
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from agno.knowledge.embedder.ollama import OllamaEmbedder

logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "deepseek_rag_table"
LANCEDB_URI = os.path.abspath("./.lancedb")

class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-embed-text"):
        """
        Initialize the OllamaEmbeddings with a specific model.

        Args:
            model_name (str): The name of the model to use for embedding.
        """
        self.embedder = OllamaEmbedder(id=model_name, dimensions=768)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Log once instead of for every chunk
        logger.info(f"Generating embeddings for {len(texts)} document chunks.")
        return [self.embed_query(text, log=False) for text in texts]

    def embed_query(self, text: str, log: bool = True) -> List[float]:
        try:
            if log:
                logger.debug(f"Generating embedding for query: {text[:50]}...")
            return self.embedder.get_embedding(text)
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            st.error(f"🔴 Embedding failed: {str(e)}. Make sure ollama is running and model '{self.embedder.id}' is pulled.")
            raise e

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
        logger.info(f"Successfully connected to LanceDB")
        return conn
    except Exception as e:
        logger.error(f"LanceDB connection failed: {str(e)}", exc_info=True)
        st.error(f"🔴 LanceDB connection failed: {str(e)}")
        return None

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

            # Check if all_docs_content exists because maybe no PDF was processed
            if 'all_docs_content' not in st.session_state:
                st.session_state.all_docs_content = ""
            st.session_state.all_docs_content += "\n\n" + "\n\n".join([d.page_content for d in documents])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split web content into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Web processing error: {str(e)}")
        st.error(f"🌐 Web processing error: {str(e)}")
        return []


def get_or_create_vector_store(db, texts=None):
    """Get existing or create new vector store."""
    if getattr(st.session_state, 'use_cloud', False) and getattr(st.session_state, 'cloud_provider', '') == "OpenAI" and getattr(st.session_state, 'cloud_api_key', ''):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.session_state.cloud_api_key)
        collection_name = "openai_rag_table"
    else:
        embeddings = OllamaEmbeddings()
        collection_name = COLLECTION_NAME
        
    try:
        if texts:
            # Generate deterministic IDs for chunks to avoid duplicating them upon re-upload
            ids = []
            for doc in texts:
                # Use file name or URL as the root identifier
                source_identifier = doc.metadata.get("file_name", doc.metadata.get("url", "unknown_source"))
                # Hash the source and the strict content together
                content_hash = hashlib.md5(f"{source_identifier}_{doc.page_content}".encode("utf-8")).hexdigest()
                ids.append(content_hash)
            
            try:
                # 1. Try to open the existing table and upsert chunks with matching IDs
                logger.info(f"Adding {len(texts)} documents to existing table '{collection_name}'")
                store = LanceDB(
                    connection=db,
                    embedding=embeddings,
                    table_name=collection_name
                )
                store.add_documents(texts, ids=ids)
                return store
            except Exception as inner_e:
                # 2. Table doesn't exist yet, we must initialize it for the first time
                logger.info(f"Creating new table '{collection_name}' with {len(texts)} documents")
                return LanceDB.from_documents(
                    texts,
                    embeddings,
                    connection=db,
                    table_name=collection_name,
                    ids=ids
                )
        else:
            # Try to open existing
            logger.info(f"Opening existing table '{collection_name}'")
            return LanceDB(
                connection=db,
                embedding=embeddings,
                table_name=collection_name
            )
    except Exception as e:
        if texts:
            logger.error(f"Vector store creation error: {str(e)}")
            st.error(f"🔴 Vector store creation error: {str(e)}")
        else:
            logger.warning(f"Could not open existing table (might not exist yet): {str(e)}. Proceeding without it until documents are added.")
        return None

def register_document(db, source: str, content: str = ""):
    """Registers a document source in the dedicated metadata registry table along with its raw text."""
    # Ensure quotes are escaped for DataFusion SQL string
    safe_source = source.replace("'", "''")
    try:
        tables = db.table_names() if hasattr(db, 'table_names') else db.list_tables()
        if "document_registry" not in tables:
            db.create_table("document_registry", data=[{"source": source, "content": content}])
        else:
            table = db.open_table("document_registry")
            # Convert filter string for Datafusion
            df = table.search().where(f"source = '{safe_source}'").to_pandas()
            if df.empty:
                # LanceDB handles schema evolution easily, just pass the dict
                table.add([{"source": source, "content": content}])
    except Exception as e:
        logger.warning(f"Failed to register document in metadata table: {e}")

def get_document_texts(db, sources: List[str]) -> str:
    """Fetches the raw text content for the given list of document sources from the registry."""
    if not sources:
        return ""
    try:
        tables = db.table_names() if hasattr(db, 'table_names') else db.list_tables()
        if "document_registry" in tables:
            table = db.open_table("document_registry")
            
            safe_sources = [s.replace("'", "''") for s in sources]
            in_list = ", ".join(f"'{s}'" for s in safe_sources)
            filter_str = f"source IN ({in_list})"
            
            df = table.search().where(filter_str).to_pandas()
            if not df.empty and "content" in df.columns:
                content_list = df["content"].dropna().tolist()
                return "\n\n".join(content_list)
    except Exception as e:
        logger.warning(f"Failed to fetch content from document_registry: {e}")
    return ""

@st.cache_data
def get_available_documents(_db) -> List[str]:
    """Retrieve distinct document names/urls stored in the database registry."""
    try:
        tables = _db.table_names() if hasattr(_db, 'table_names') else _db.list_tables()
        if "document_registry" in tables:
            table = _db.open_table("document_registry")
            df = table.to_pandas()
            if not df.empty and "source" in df.columns:
                return sorted(list(set(df["source"].tolist())))
    except Exception as e:
        logger.warning(f"Failed to fetch available documents from LanceDB registry: {e}")
    return []
