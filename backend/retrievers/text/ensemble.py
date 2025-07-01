from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.retrievers import ParentDocumentRetriever, BM25Retriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

import os
from dotenv import load_dotenv

load_dotenv(override=True)

def batched_add_documents(retriever, documents, batch_size=32):
    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading to Qdrant"):
        batch = documents[i:i+batch_size]
        retriever.add_documents(batch)

def get_ensemble_retriever(documents, name):
    # Splitting setup
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    # Embeddings
    embedding_model = OpenAIEmbeddings()
    collection_name = name.replace(".pdf", "").replace(" ", "_").lower()

    # Qdrant setup with increased timeout
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60.0  # increased timeout
    )

    # Create collection if not present
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    # Vector store + parent retriever
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_model
    )

    docstore = InMemoryStore()
    dense_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )

    # Add documents in batches to avoid timeouts
    batched_add_documents(dense_retriever, documents)

    # BM25 Retriever on parent chunks
    parent_docs = parent_splitter.split_documents(documents)
    sparse_retriever = BM25Retriever.from_documents(parent_docs)
    sparse_retriever.k = 3

    # Ensemble Retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever
