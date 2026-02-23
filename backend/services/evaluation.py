"""
RAGAS evaluation service.
Scores the RAG pipeline on faithfulness, answer relevancy, and context recall.
"""

from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from backend.services.rag import build_rag_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def run_evaluation(test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Evaluate RAG pipeline using RAGAS metrics.

    test_cases: list of {"question": str, "ground_truth": str}
    Returns scores for faithfulness, answer_relevancy, context_precision.
    """
    chain = build_rag_chain()

    questions, answers, contexts, ground_truths = [], [], [], []

    for tc in test_cases:
        question = tc["question"]
        ground_truth = tc.get("ground_truth", "")

        result = chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result["source_documents"]

        questions.append(question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in source_docs])
        ground_truths.append(ground_truth)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    return {
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevancy": round(scores["answer_relevancy"], 4),
        "context_precision": round(scores["context_precision"], 4),
        "num_test_cases": len(test_cases),
    }


# Default test cases for finance/legal domain
DEFAULT_TEST_CASES = [
    {
        "question": "What is the total revenue reported?",
        "ground_truth": "The total revenue is stated in the financial statements.",
    },
    {
        "question": "What are the key risk factors mentioned?",
        "ground_truth": "Risk factors are outlined in the risk section of the document.",
    },
    {
        "question": "What are the termination clauses in the contract?",
        "ground_truth": "Termination clauses specify conditions under which the contract can be ended.",
    },
]
