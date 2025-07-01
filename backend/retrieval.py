from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from backend.llm import new_custom_prompt_template

def get_vector_chain(llm, retriever):
    prompt = new_custom_prompt_template()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={'prompt': prompt},
        return_source_documents=True,
        verbose=True,
    )

def get_summarization_chain(llm, docs):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)