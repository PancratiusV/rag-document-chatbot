# PDF RAG Assistant 🤖📄

A retrieval-augmented generation (RAG) tool that lets you upload PDFs and ask questions about their content in natural language.

## 🚀 Live Demo
[**Click here to try the app**](https://pdf-rag-assistant-pancratiusv.streamlit.app/) 

## 🛠️ Tech Stack
*   **Frontend:** Streamlit
*   **Framework:** LangChain
*   **Vector Database:** ChromaDB
*   **Embeddings:** `all-mpnet-base-v2`
*   **LLM:** `openai/gpt-oss-120b`

## 💡 How It Works
1.  **Ingestion:** The app reads PDF files and splits text into manageable chunks.
2.  **Embedding:** Text chunks are converted into vector representations.
3.  **Retrieval:** When you ask a question, the system finds the most semantically similar chunks in the vector store.
4.  **Generation:** The LLM uses the retrieved context to generate an accurate answer.
