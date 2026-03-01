from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import os
import argparse
import json
import rag_client
import llm_client
import pandas as pd

load_dotenv()

try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import (
        ResponseRelevancy, 
        Faithfulness, 
        BleuScore, 
        RougeScore, 
        AnswerCorrectness, 
        ContextPrecision
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def get_metrics(metrics_list: Optional[List[str]], evaluator_llm: Any, evaluator_embeddings: Any) -> List[Any]:
    """Helper to return metric instances based on names. Always includes baseline metrics."""
    baseline_metrics = ["faithfulness", "answer_relevancy"]
    
    # Merge user metrics with baseline
    if metrics_list is None:
        metrics_list = []
    
    # Use a set for names to avoid duplicates
    all_metric_names = set(baseline_metrics)
    for m in metrics_list:
        all_metric_names.add(m.lower())
        
    supported_metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "bleu": BleuScore(),
        "rouge": RougeScore(),
        "answer_correctness": AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "context_precision": ContextPrecision(llm=evaluator_llm)
    }
    
    selected_metrics = []
    for m_name in sorted(list(all_metric_names)):
        if m_name in supported_metrics:
            selected_metrics.append(supported_metrics[m_name])
        else:
            print(f"Warning: Metric '{m_name}' is not supported. Skipping.")
    return selected_metrics

def evaluate_response_quality(
    question: str, 
    answer: str, 
    contexts: List[str], 
    metrics_list: Optional[List[str]] = None, 
    ground_truth: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate response quality using selected RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Create evaluator LLM and embeddings
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    selected_metrics = get_metrics(metrics_list, evaluator_llm, evaluator_embeddings)

    if not selected_metrics:
        return {"error": "No valid metrics selected"}

    # Create a single turn sample
    sample_kwargs = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts
    }
    if ground_truth:
        sample_kwargs["reference"] = ground_truth

    try:
        sample = SingleTurnSample(**sample_kwargs)
        dataset = EvaluationDataset(samples=[sample])
        
        # Evaluate the response using the metrics
        results = evaluate(dataset=dataset, metrics=selected_metrics)
        # Convert the Result object to a dictionary of scores
        scores = results.to_pandas().iloc[0].to_dict()
        
        # Validation: Check for NaNs in required metrics
        for metric in ["faithfulness", "answer_relevancy"]:
            if pd.isna(scores.get(metric)):
                print(f"Warning: Metric '{metric}' resulted in NaN. This often happens if the LLM cannot find evidence in the provided context.")
        
        return scores
    except Exception as e:
        return {"error": str(e)}

def normalize_mission_name(mission: str) -> str:
    """Normalize mission names from JSON to match ChromaDB metadata"""
    m = mission.lower().strip()
    if "apollo 11" in m:
        return "apollo_11"
    if "apollo 13" in m:
        return "apollo_13"
    if "challenger" in m:
        return "challenger"
    return m

def evaluate_from_file(
    file_path: str,
    metrics_list: Optional[List[str]] = None,
    num_questions: Optional[int] = None,
    chroma_dir: str = "chroma_db_openai",
    collection_name: str = "nasa_space_missions_text"
) -> Dict[str, Any]:
    """Load questions from JSON file and run batch evaluation"""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
        
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # Flatten the mission-based structure if necessary
    flat_questions = []
    for item in data:
        mission = item.get("mission", "Unknown")
        normalized_mission = normalize_mission_name(mission)
        for q_pair in item.get("questions", []):
            q_pair["mission"] = normalized_mission
            flat_questions.append(q_pair)
            
    if num_questions:
        flat_questions = flat_questions[:num_questions]
        
    print(f"Loaded {len(flat_questions)} questions from {file_path}")
    
    # Initialize RAG system
    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        return {"error": f"Failed to initialize RAG: {error}"}
        
    openai_key = os.getenv("OPENAI_API_KEY")
    
    samples = []
    print("Processing questions (retrieval + generation)...")
    for i, item in enumerate(flat_questions):
        question = item["question"]
        ground_truth = item.get("answer")
        mission_filter = item.get("mission")
        
        print(f"[{i+1}/{len(flat_questions)}] Processing: {question[:50]}...")
        
        # 1. Retrieve
        docs_result = rag_client.retrieve_documents(collection, question, n_results=5, mission_filter=mission_filter)
        contexts = docs_result["documents"][0] if docs_result and docs_result.get("documents") else []
        
        # 2. Generate
        context_str = ""
        if contexts:
            context_str = rag_client.format_context(contexts, docs_result["metadatas"][0])
        else:
            print(f"  Warning: No contexts retrieved for queston: {question[:30]}")
            
        answer = llm_client.generate_response(openai_key, question, context_str, [])
        
        # 3. Create Sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth
        )
        samples.append(sample)
        
    # Batch Evaluate
    print("Running batch evaluation...")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    selected_metrics = get_metrics(metrics_list, evaluator_llm, evaluator_embeddings)
    
    dataset = EvaluationDataset(samples=samples)
    results = evaluate(dataset=dataset, metrics=selected_metrics)
    
    # Calculate Aggregate Metrics
    df = results.to_pandas()
    
    # Validation: Identify NaNs in baseline metrics
    required = ["faithfulness", "answer_relevancy"]
    # Ensure columns exist before checking
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"Error: Required columns {missing_cols} missing from evaluation results.")
        valid_count = 0
    else:
        valid_mask = df[required].notna().all(axis=1)
        valid_count = int(valid_mask.sum())
    
    if valid_count < len(df):
        print(f"Warning: {len(df) - valid_count} samples had NaN values in required RAGAS metrics.")
        
    # Remove metadata columns if they exist (usually 'user_input', 'response', etc. are kept)
    numeric_cols = df.select_dtypes(include=['number']).columns
    summary = {
        "mean_scores": df[numeric_cols].mean().to_dict(),
        "total_samples": len(df),
        "valid_samples": valid_count,
        "detailed_results": df.to_dict(orient="records")
    }
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG response quality using RAGAS.")
    parser.add_argument("--question", type=str, help="The question asked.")
    parser.add_argument("--answer", type=str, help="The generated answer.")
    parser.add_argument("--contexts", type=str, nargs="+", help="The retrieved contexts.")
    parser.add_argument("--ground_truth", type=str, help="The reference answer (optional).")
    parser.add_argument("--metrics", type=str, nargs="+", default=["faithfulness", "answer_relevancy"],
                        help="List of metrics to use: faithfulness, answer_relevancy, bleu, rouge, answer_correctness, context_precision")
    parser.add_argument("--test-set", type=str, help="Path to test_questions.json for batch evaluation.")
    parser.add_argument("--num-questions", type=int, help="Limit the number of questions to evaluate from the test set.")
    parser.add_argument("--chroma-dir", type=str, default="chroma_db_openai", help="ChromaDB directory.")
    parser.add_argument("--collection-name", type=str, default="nasa_space_missions_text", help="ChromaDB collection name.")
    
    args = parser.parse_args()

    if args.test_set:
        summary = evaluate_from_file(
            args.test_set, 
            args.metrics, 
            args.num_questions,
            args.chroma_dir,
            args.collection_name
        )
        if "error" in summary:
            print(f"Error: {summary['error']}")
        else:
            print("\n" + "="*50)
            print("BATCH EVALUATION SUMMARY")
            print("="*50)
            print(f"Total Samples: {summary['total_samples']}")
            print(f"Valid Samples (all baseline metrics present): {summary['valid_samples']}")
            print("\nMean Scores:")
            for metric, score in summary['mean_scores'].items():
                print(f"  {metric}: {score:.4f}")
            print("="*50)
            
            # Save detailed results to a JSON file
            output_file = "evaluation_results.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Detailed results saved to {output_file}")
            
    elif args.question and args.answer and args.contexts:
        scores = evaluate_response_quality(
            args.question, 
            args.answer, 
            args.contexts, 
            args.metrics, 
            args.ground_truth
        )
        print(json.dumps(scores, indent=2))
    else:
        # Default test run if no arguments provided
        print("Running default evaluation test...")
        test_q = "What was the primary goal of the Apollo 11 mission?"
        test_a = "The primary goal was to land two astronauts on the Moon."
        test_c = ["Apollo 11 mission goal was landing on the moon."]
        test_gt = "The primary goal was to land two astronauts on the Moon."
        
        scores = evaluate_response_quality(test_q, test_a, test_c, args.metrics, test_gt)
        print(json.dumps(scores, indent=2))
