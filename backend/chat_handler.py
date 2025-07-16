from backend.retrievers.image.query_retriever import retrieve_similar_images
from backend.llm import validate_images, generate_image_query, classify_query

def build_chat_history(messages: str):
    history = []
    for i in range(max(0, len(messages) - 10), len(messages) - 1, 2):
        user = messages[i]["content"]
        if i + 1 < len(messages):
            bot = messages[i + 1]["content"]
            history.append((user, bot))
    return history

def should_skip_images(prompt: str) -> bool:
    skip_keywords = [
        "title", "author", "written by", "date", 
        "who wrote", "published", "journal", "year", 
        "conference", "volume", "doi"
    ]

    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in skip_keywords)


def handle_user_query(prompt: str, messages: str, vector_chain, summary):
    chat_history = build_chat_history(messages)
    query_type = classify_query(prompt)

    if query_type == "summary":
        print("summary")
        return summary, []

    text_context_docs = vector_chain.retriever.get_relevant_documents(prompt)
    text_context = "\n\n".join([doc.page_content for doc in text_context_docs])
    prompt_input = {
        "question": prompt,
        "chat_history": chat_history,
        "context": text_context
    }
    response = vector_chain.invoke(prompt_input)
    text_response = response['answer']

    image_query = generate_image_query(prompt, text_response)
    raw_images = retrieve_similar_images(image_query)
    validated_images = validate_images(prompt, text_response, raw_images)

    return text_response, validated_images
