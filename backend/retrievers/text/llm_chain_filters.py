from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from backend.llm import load_model
from langchain.retrievers.document_compressors import LLMChainExtractor

def get_filter_chain(retriever):
    llm = load_model()
    extractor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=extractor
    )

    return compression_retriever
