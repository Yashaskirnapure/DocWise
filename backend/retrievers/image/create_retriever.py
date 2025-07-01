import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

def index_image_captions_to_qdrant(image_captions_file: str):
    if not os.path.exists(image_captions_file):
        raise FileNotFoundError(f"{image_captions_file} not found.")

    with open(image_captions_file, "r", encoding="utf-8") as f:
        captions = json.load(f)

    # Prepare text and metadata
    texts, payloads, ids = [], [], []
    for id_str, meta in captions.items():
        caption = (meta.get("image_text") or "").strip()
        desc = (meta.get("description") or "").strip()
        combined = f"{caption} {desc}".strip()

        if not combined:
            continue

        texts.append(combined)
        payloads.append({
            "image_id": int(id_str),
            "image_text": caption,
            "description": desc,
            "file_path": meta.get("filePath", "")
        })
        ids.append(int(id_str))

    if not texts:
        print("[Qdrant] No image captions found. Skipping indexing.")
        return

    # Generate embeddings
    embedding_model = OpenAIEmbeddings()
    vectors = embedding_model.embed_documents(texts)

    # Qdrant setup
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30.0,
    )

    collection_name = "image_captions"

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # Upload points
    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(ids))
    ]

    client.upsert(collection_name=collection_name, points=points)
    print(f"[Qdrant] Indexed {len(points)} image captions into '{collection_name}'")