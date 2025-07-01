from backend.retrievers.text.ensemble import get_ensemble_retriever
from backend.retrievers.text.llm_chain_filters import get_filter_chain
from dotenv import load_dotenv

load_dotenv(override=True)

def process_uploaded_file(uploaded_file):
    ensemble_retriver = get_ensemble_retriever(uploaded_file)
    filtered_retriever = get_filter_chain(ensemble_retriver)

    return filtered_retriever