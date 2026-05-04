import json, math, os
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    _AnswerRelevancy as AnswerRelevancy,
    _AnswerCorrectness as AnswerCorrectness,
    _ContextPrecision as ContextPrecision,
    _ContextRecall as ContextRecall,
    _Faithfulness as Faithfulness,
)
from rag import run_pipeline
from dotenv import load_dotenv
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE_NAME") or os.getenv("AWS_PROFILE")
EVAL_MODEL_ID = (
    os.getenv("EVAL_BEDROCK_MODEL_ID")
    or os.getenv("BEDROCK_MODEL_PREMIUM")
    or os.getenv("BEDROCK_MODEL_ID")
)
EVAL_EMBEDDINGS_MODEL_ID = os.getenv("EVAL_EMBEDDINGS_MODEL_ID")

if not EVAL_MODEL_ID:
    raise RuntimeError("Set BEDROCK_MODEL_ID or BEDROCK_MODEL_PREMIUM in .env before running evaluate.py")
if not EVAL_EMBEDDINGS_MODEL_ID:
    raise RuntimeError("Set EVAL_EMBEDDINGS_MODEL_ID in .env before running evaluate.py")

def build_ragas_models():
    llm = ChatBedrock(
        model=EVAL_MODEL_ID,
        region=AWS_REGION,
        credentials_profile_name=AWS_PROFILE,
        temperature=0,
        **{"max_" + "to" + "kens": 2048},
    )
    embeddings = BedrockEmbeddings(
        model_id=EVAL_EMBEDDINGS_MODEL_ID,
        region_name=AWS_REGION,
        credentials_profile_name=AWS_PROFILE,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

def average_score(values):
    valid = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value))
    ]
    return round(sum(valid) / len(valid), 4) if valid else None

def run_eval():
    with open("test_set.json") as f:
        test_set = json.load(f)

    rows = []
    print(f"Running {len(test_set)} questions through RAG pipeline...")
    print(f"Using Bedrock judge model: {EVAL_MODEL_ID}")
    print(f"Using Bedrock embeddings model: {EVAL_EMBEDDINGS_MODEL_ID}")

    for i, item in enumerate(test_set):
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:60]}...")
        result = run_pipeline(item["question"])
        rows.append({
            "question": item["question"],
            "user_input": item["question"],
            "answer": result["answer"],
            "response": result["answer"],
            "contexts": result["contexts"],
            "retrieved_contexts": result["contexts"],
            "ground_truth": item["ground_truth"],
            "reference": item["ground_truth"],
        })

    dataset = Dataset.from_list(rows)
    ragas_llm, ragas_embeddings = build_ragas_models()
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    scores = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    score_lists = {
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_precision": scores["context_precision"],
        "context_recall": scores["context_recall"],
        "answer_correctness": scores["answer_correctness"],
    }

    for i, row in enumerate(rows):
        row["scores"] = {
            name: values[i]
            for name, values in score_lists.items()
            if i < len(values)
        }

    report = {
        "faithfulness": average_score(score_lists["faithfulness"]),
        "answer_relevancy": average_score(score_lists["answer_relevancy"]),
        "context_precision": average_score(score_lists["context_precision"]),
        "context_recall": average_score(score_lists["context_recall"]),
        "answer_correctness": average_score(score_lists["answer_correctness"]),
        "n_questions": len(test_set),
        "details": rows
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n📊 RAGAS Evaluation Results:")
    print(f"  Faithfulness:       {report['faithfulness']}")
    print(f"  Answer Relevancy:   {report['answer_relevancy']}")
    print(f"  Context Precision:  {report['context_precision']}")
    print(f"  Context Recall:     {report['context_recall']}")
    print(f"  Answer Correctness: {report['answer_correctness']}")
    print(f"\n✅ Full report saved to reports/eval_report.json")

if __name__ == "__main__":
    run_eval()
