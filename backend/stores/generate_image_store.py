from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def index_image_captions_to_qdrant(image_captions_file: str):
    with open(image_captions_file, 'r') as f:
        captions = json.load(f)

    embedding_model = OpenAIEmbeddings()

    texts = []
    payloads = []
    ids = []

    for id, meta in captions.items():
        combined_text = f"{meta.get('image_text', '')} {meta.get('description', '')}".strip()
        if not combined_text:
            continue
        texts.append(combined_text)
        payloads.append({
            "image_text": meta.get("image_text"),
            "description": meta.get("description"),
            "file_path": meta.get("filePath"),
            "image_id": id
        })
        ids.append(int(id))

    vectors = embedding_model.embed_documents(texts)
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30.0
    )
    collection_name = "image_captions"

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(vectors))
    ]

    client.upsert(collection_name=collection_name, points=points)