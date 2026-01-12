# --- 1. IMPORTS ---
import os
import streamlit as st
import tempfile

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

# --- 2. SETUP ---
st.set_page_config(page_title="PDF RAG Demo", layout="wide")
st.title("📄 Answer Questions From PDF")

# --- 3. BACKEND ---
@st.cache_resource
def get_embedding_model():
    # Only loads once. Uses CPU/GPU automatically.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def get_llm():
    # Use Groq for fast, free inference
    return ChatGroq(model="openai/gpt-oss-120b",
    groq_api_key=os.environ["GROQ_API_KEY"])

def process_pdf(pdf_path):
    """Parses PDF -> Markdown -> Chunks"""
    # 1. Parse with Docling
    converter = DocumentConverter()
    result = converter.convert(pdf_path).document
    md_text = result.export_to_markdown()
    
    # 2. Split by Headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"), 
            ("##", "Header 2"), 
            ("###", "Header 3")
        ]
    )
    header_splits = header_splitter.split_text(md_text)
    
    # 3. Recursive Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = text_splitter.split_documents(header_splits)
    return final_chunks

def create_vector_store(chunks, embedding_model):
    """Creates Chroma vector store from text chunks"""
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="pdf_docs"
    )
    return vector_store

# --- 4. DYNAMIC AGENT CREATION ---
def get_agent_for_pdf(vector_store, llm):
    """
    Creates an agent specifically for THIS vector store.
    We define the tool INSIDE here so it can access 'vector_store'.
    """
    
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.max_marginal_relevance_search(query, k=2,lambda_mult=0.5)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_context]
    
    system_prompt = (
        "You have access to a tool that retrieves context from a document. "
        "Use the tool to help answer user queries."
    )
    
    # create_react_agent is the modern LangGraph way to make an agent
    return create_agent(llm, tools, system_prompt=system_prompt)


# --- 5. FRONTEND UI ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file and st.session_state.vector_store is None:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            chunks = process_pdf(tmp_path)
            embeddings = get_embedding_model()
            st.session_state.vector_store = create_vector_store(chunks, embeddings)
            st.success("Agent Ready!")
            os.remove(tmp_path)

user_query = st.chat_input("Ask the Agent...")

if user_query:
    if st.session_state.vector_store is None:
        st.warning("Please upload a PDF first.")
    else:
        # Display User Message
        with st.chat_message("user"):
            st.write(user_query)
            
        # Run Agent
        with st.chat_message("assistant"):
            llm = get_llm()
            # Build the agent on the fly with the current vector store
            agent = get_agent_for_pdf(st.session_state.vector_store, llm)
            
            # Stream the events
            # We use a placeholder to update the text in real-time
            message_placeholder = st.empty()
            final_response = ""
            
            # Stream loop
            for event in agent.stream(
                {"messages": [{"role": "user", "content": user_query}]},
                stream_mode="values"
            ):
                # Get the latest message
                latest_msg = event["messages"][-1]
                
                # If it's the AI's final answer, update the UI
                if isinstance(latest_msg, AIMessage) and latest_msg.content:
                    final_response = latest_msg.content
                    st.write(final_response)
        