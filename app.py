import streamlit as st
from dotenv import load_dotenv
import os

from backend.retrieval import get_vector_chain, get_summarization_chain
from backend.chat_handler import handle_user_query
from backend.ingest.ingestion import ingest_document
from backend.llm import load_model

load_dotenv(override=True)
llm = load_model()

IMAGE_BASE_PATH='/home/yashas/Desktop/projects/docwise/assets/extract'

def main():
    st.sidebar.header("Upload a PDF to index")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file_id") != file_id:
            st.session_state.messages = []
            with st.spinner("Processing uploaded file..."):
                vector_retriever, summary_retriever = ingest_document(uploaded_file)
                
                vector_chain = get_vector_chain(llm=llm, retriever=vector_retriever)
                summary = get_summarization_chain(llm=llm, docs=summary_retriever)

                st.session_state.vector_chain = vector_chain
                st.session_state.summary = summary
                st.session_state.last_uploaded_file_id = file_id
                st.sidebar.success("File uploaded and indexed!")

    st.title("Get your queries answered here:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("Enter your query")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            if "vector_chain" not in st.session_state or "summary" not in st.session_state:
                st.error("Please upload a PDF first to initialize the system.")
                return

            with st.spinner("Generating answer..."):
                text, images = handle_user_query(prompt, st.session_state.messages, st.session_state.vector_chain, st.session_state.summary)

            st.chat_message("assistant").markdown(text)

            if images:
                st.markdown("### Relevant Images:")
                for img in images:
                    full_path = os.path.join(IMAGE_BASE_PATH, img["file_path"].lstrip("/"))
                    st.image(full_path, caption=img["caption"], use_container_width=True)

            else:
                st.info("No relevant images found for this question.")


            st.session_state.messages.append({"role": "assistant", "content": text})
        except Exception as e:
            st.error(f"Error raised: {str(e)}")

if __name__ == "__main__":
    main()