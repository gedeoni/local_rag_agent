import streamlit as st
import dspy
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools

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
        markdown=True,
    )

def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    # Note: Requires st.session_state.model_version initialized
    model_version = st.session_state.model_version if "model_version" in st.session_state else "deepseek-r1:7b"
    return Agent(
        name="DeepSeek RAG Agent",
        model=Ollama(id=model_version),
        instructions="""You are an Intelligent Agent providing accurate answers.
        Focus on provided documents or web results. If context is provided, prioritize it.
        If no context is found for a specific document reference, ask the user to be more specific.
        Synthesize clearly and cite sources.""",
        markdown=True,
    )
