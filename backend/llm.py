from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import streamlit as st
import json

from dotenv import load_dotenv
load_dotenv(override=True)

@st.cache_resource(show_spinner="Loading LLM...")
def load_model():
    llm = ChatOpenAI(
        model_name="gpt-4o-mini-2024-07-18",
        temperature=0.2
    )
    return llm

def new_custom_prompt_template():
    system_template = (
        "You are an assistance chatbot. Use the pieces of information provided in the context to answer user's question.\n"
        "If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n"
        "Don't provide anything outside the given context and only use english as the language for communication."
    )
    human_template = (
        "Context:\n{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Question:\n{question}\n\n"
        "Start the answer directly. No small talk please."
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    return prompt

def classify_query(user_query: str) -> str:
    llm = load_model()

    classification_prompt = f"""
You are a query classifier for a document-based chatbot.

Classify the following user query as either:

- "summary" → if the user is asking for a general overview, abstract, gist, or high-level summary of the document.
- "vector" → if the user is asking for specific facts, details, or context that would require retrieving precise parts of the document.

Return only the word "summary" or "vector".

User Query:
"{user_query}"
"""

    try:
        response = llm.invoke(classification_prompt)
        classification = (response.content or "").strip().lower()
        if classification not in ["summary", "vector"]:
            raise ValueError(f"Unexpected classification: {classification}")
        return classification
    except Exception as e:
        print(f"[Classifier] Failed to classify query: {e}")
        return "vector"


def validate_images(prompt: str, answer: str, image_results: list, threshold: float = 0.8):
    llm = load_model()

    safe_prompt = (prompt or "").strip()
    safe_answer = (answer or "").strip()

    # Format all image blocks
    image_blocks = []
    for i, img in enumerate(image_results):
        caption = (img.get("image_text") or "").strip()
        desc = (img.get("description") or "").strip()
        file_path = img.get("file_path", "")

        if not caption and not desc:
            continue

        image_blocks.append({
            "index": i,
            "caption": caption,
            "description": desc,
            "file_path": file_path
        })

    if not image_blocks:
        return []

    # Prepare batched prompt
    image_json = json.dumps(image_blocks, indent=2)
    validation_prompt = f"""
You are evaluating images to decide whether each should be included alongside a user's question and answer.

User Question:
"{safe_prompt}"

LLM Answer:
"{safe_answer}"

Below is a list of image metadata:

{image_json}

For each image, determine:

1. If the image is **relevant** to the question and helpful in understanding or illustrating the answer.
2. If relevant, generate a **short caption** for the image.
3. Give a **relevance score** between 0 and 1.

Respond with a list of JSON objects like:
[
  {{
    "index": 0,
    "relevance_score": 0.92,
    "caption": "Diagram showing XYZ"
  }},
  ...
]
Only return the JSON list. No explanations.
"""

    try:
        response = llm.invoke(validation_prompt)
        parsed = json.loads(response.content.strip())

        validated = []
        for res in parsed:
            score = res.get("relevance_score", 0)
            if score >= threshold:
                validated.append({
                    "file_path": image_blocks[res["index"]]["file_path"],
                    "caption": res.get("caption", "").strip(),
                    "score": score
                })

        return validated

    except Exception as e:
        print(f"Batch image validation failed: {e}")
        return []



def generate_image_query(prompt: str, llm_answer: str) -> str:
    llm = load_model()
    safe_prompt = (prompt or "").strip()
    safe_answer = (llm_answer or "").strip()

    query_prompt = f"""
You are generating a search query to retrieve relevant images that support an LLM-generated answer.

User Question:
"{safe_prompt}"

LLM Answer:
"{safe_answer}"

Write a concise search query that describes the key concepts or diagrams most relevant to the answer. Only return the query string.
"""

    try:
        response = llm.invoke(query_prompt)
        return (response.content or "").strip()
    except Exception as e:
        print(f"Failed to generate image query: {e}")
        return safe_prompt  # fallback to original query

