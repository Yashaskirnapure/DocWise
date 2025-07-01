# 📄 DocWise: Intelligent Document Q&A Assistant

**DocWise** is a smart, RAG-based assistant that lets users interact with complex PDF documents through natural language. It combines document parsing, vector-based retrieval, and LLM-powered generation to answer queries, summarize content, and display relevant visual elements like figures and diagrams.

## Features

- 🧠 **Retrieval-Augmented Generation (RAG)**: Combines vector similarity search with LLMs for accurate, grounded answers.
- 🗃️ **PDF Parsing**: Extracts structured text, images, tables, and metadata from uploaded documents.
- 💬 **Dual Mode Query Handling**: Determines if a question requires summarization or detailed retrieval.
- 🖼️ **Visual Context**: Retrieves and validates figures/diagrams relevant to the user's question.
- ⚡ **Fast and Modular**: Uses LangChain, OpenAI GPT-4o, and Qdrant for scalable performance.
- 🌐 **Streamlit Frontend**: Clean and interactive interface for document uploads and chatting.


## Architecture

<!-- You can place your architecture diagram here -->
![Architecture Diagram](/diagrams/new_arch_final.drawio.png)


## Tech Stack

| Layer            | Tools Used                             |
|------------------|----------------------------------------|
| Language Model   | [OpenAI GPT-4o](https://platform.openai.com) |
| Vector DB        | [Qdrant](https://qdrant.tech)           |
| Orchestration    | [LangChain](https://www.langchain.com)  |
| Embeddings       | `text-embedding-3-small` via OpenAI     |
| PDF Parsing      | Adobe PDF Extract API / PyPDFLoader     |
| UI               | Streamlit                               |
| Environment      | Python 3.11+                            |


## Setup

### Prerequisites

- Python 3.11+
- [OpenAI API Key](https://platform.openai.com/account/api-keys)
- [Qdrant Cloud URL + API Key](https://qdrant.tech)
- [Adobe PDF Extract API Credentials](https://acrobatservices.adobe.com/dc-integration-creation-app-cdn/main.html?api=pdf-extract-api)

### Installation

```bash
git clone https://github.com/your-username/docwise.git
cd docwise
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory and add the following:

```env
OPENAI_API_KEY=your_openai_key
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_api_key
PDF_SERVICES_CLIENT_ID=your_adobe_api_client_id
PDF_SERVICES_CLIENT_SECRET=your_adobe_api_client_secret
```
### Running the App

After setting up the environment and installing dependencies, launch the app using:

```bash
streamlit run app.py
```


## Disclaimer

This project was built for **educational and experimental purposes**. While it demonstrates key concepts in retrieval-augmented generation (RAG), document processing, and AI-assisted querying, it is **not intended for production use**. Certain implementation choices were made for learning and rapid prototyping rather than strict adherence to modern industry standards.
