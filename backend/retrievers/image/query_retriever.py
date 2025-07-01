import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

def retrieve_similar_images(query: str):
    embedding_model = OpenAIEmbeddings()
    query_vector = embedding_model.embed_query(query)

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30.0
    )

    results = client.search(
        collection_name="image_captions",
        query_vector=query_vector,
        limit=3
    )

    return [
        {
            "image_id": res.payload.get("image_id"),
            "image_text": res.payload.get("image_text", ""),
            "description": res.payload.get("description", ""),
            "file_path": res.payload.get("file_path", ""),
            "score": res.score
        }
        for res in results if res.payload
    ]
