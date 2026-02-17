from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os

load_dotenv()

try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import ResponseRelevancy, Faithfulness
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Create evaluator LLM with model gpt-4o-mini
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    
    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    # Define an instance for each metric to evaluate
    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm)
    ]
    
    # Create a single turn sample
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    
    # Wrap in EvaluationDataset
    dataset = EvaluationDataset(samples=[sample])
    
    # Evaluate the response using the metrics
    try:
        results = evaluate(dataset=dataset, metrics=metrics)
        # Convert the Result object to a dictionary of scores
        return results.to_pandas().iloc[0].to_dict()
    except Exception as e:
        return {"error": str(e)}
