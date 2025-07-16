import os
import json
import zipfile
import shutil

from backend.parsers.pdf_parser import PDFParser
from backend.parsers.json_parser import parse_json, get_description
from backend.retrievers.text.ensemble import get_ensemble_retriever
from backend.retrievers.text.llm_chain_filters import get_filter_chain
from backend.retrievers.image.create_retriever import index_image_captions_to_qdrant

from langchain_community.document_loaders import PyPDFLoader


def ingest_document(uploaded_file):
    base_dir = "/home/yashas/Desktop/projects/docwise"
    temp_pdf_path = os.path.join(base_dir, "temp.pdf")
    assets_path = os.path.join(base_dir, "assets")
    zip_path = os.path.join(assets_path, "extract.zip")
    extract_path = os.path.join(assets_path, "extract")
    image_captions_path = os.path.join(base_dir, "image_captions.json")

    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    PDFParser(temp_pdf_path)

    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    structured_json = os.path.join(extract_path, "structuredData.json")
    parsed = parse_json(structured_json)

    image_captions = get_description(
        parsed["images"],
        parsed["text"],
        parsed["image_text"]
    )

    with open(image_captions_path, "w", encoding="utf-8") as f:
        json.dump(image_captions, f, indent=2, ensure_ascii=False)

    index_image_captions_to_qdrant(image_captions_path)

    loader = PyPDFLoader(temp_pdf_path)
    text_docs = loader.load()

    retriever = get_ensemble_retriever(text_docs, "text_collection")
    return get_filter_chain(retriever), text_docs
