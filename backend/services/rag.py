"""
RAG service: retrieval + generation pipeline using LangChain.
Supports OpenAI GPT and local Mistral via Ollama.
"""

import os
from typing import List, Dict, Any

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import Document

from backend.services.ingestion import get_vector_store
from dotenv import load_dotenv

load_dotenv()

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))


FINANCE_LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are FinLex, an expert financial and legal document analyst.
Use ONLY the context below to answer the question. Be precise and cite specific 
sections or figures where possible. If the answer is not in the context, say 
"I could not find this information in the provided documents."

Context:
{context}

Question: {question}

Answer (be concise, structured, and cite sources):"""
)


def get_llm():
    """Return LLM — OpenAI or local Ollama based on config."""
    if USE_LOCAL_LLM:
        return Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0.1)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)


def build_rag_chain():
    """Build the RetrievalQA chain."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Max Marginal Relevance for diverse results
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 2},
    )
    llm = get_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": FINANCE_LEGAL_PROMPT},
    )
    return chain


def query(question: str) -> Dict[str, Any]:
    """
    Run a query through the RAG pipeline.
    Returns answer + source documents with metadata.
    """
    chain = build_rag_chain()
    result = chain.invoke({"query": question})

    answer = result["result"]
    source_docs: List[Document] = result["source_documents"]

    # Format sources for response
    sources = []
    seen = set()
    for doc in source_docs:
        src = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        key = f"{src}::{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": src,
                "page": page,
                "snippet": doc.page_content[:200] + "...",
            })

    return {
        "answer": answer,
        "sources": sources,
        "model": OLLAMA_MODEL if USE_LOCAL_LLM else OPENAI_MODEL,
    }
