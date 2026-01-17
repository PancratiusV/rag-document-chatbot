# RAG Document Chatbot 🤖📄

A LLM chatbot with retrieval-augmented generation (RAG) that lets you upload documents and ask questions about their content in natural language.

![APP Demo](images/app_screenshot.png)

## 🚀 Live Demo
[**Click here to try the app**](https://rag-document-chatbot-pv.streamlit.app/) 

## 🛠️ Tech Stack
*   **Frontend:** Streamlit
*   **Framework:** LangChain
*   **Vector Database:** ChromaDB
*   **Embeddings:** `all-mpnet-base-v2` (Sentence Transformers)
*   **LLM:** `openai/gpt-oss-120b` (via Groq API)

## 💡 How It Works
1.  **Multi-Format Ingestion:** The app accepts various file types (PDF, DOCX, PPTX, etc.) and extracts text using `docling`'s advanced parsing.
2.  **Chunking:** Content is intelligently split into manageable segments to preserve context.
3.  **Embedding:** Text chunks are converted into high-dimensional vector representations.
4.  **Retrieval:** When you ask a question, the system performs a semantic search to find the most relevant information.
5.  **Generation:** The LLM uses the retrieved context to generate a precise, fact-based answer.

