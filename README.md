# RAG-2.0
A secure Retrieval-Augmented Generation (RAG) app that lets users upload documents, index them, and query with natural language. Powered by FAISS, HuggingFace embeddings, and Google Gemini, it delivers safe, context-based answers with semantic reranking.

🚀 Features
------------
* 📄 **Document Ingestion & Chunking** – upload and preprocess text data
* 🔍 **Vector Embeddings & Search** – semantic search powered by FAISS
* 🤖 **Context-Aware Answer Generation** – answers grounded in retrieved documents
* 🎨 **Streamlit UI** – clean and interactive interface for users

## ⚙️ Tech Stack
-----------------
* **LangChain** – orchestration of retrieval & generation
* **FAISS** – efficient vector search and similarity matching
* **HuggingFace Embeddings** – for document embeddings
* **LLMs (Google Generative AI / OpenAI / HuggingFace models)** – for answer generation
* **Streamlit** – frontend UI

How to Run Locally
-------------------
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

🌟 Future Enhancements
* Support for multiple file formats (PDF, CSV, DOCX)
* Multi-query retrieval for better accuracy
* Deployment on **AWS / GCP / Render**


Do you want me to also **write the LinkedIn post again with GitHub repo link included** (so you can just copy-paste)?
