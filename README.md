Production RAG systems fail silently...

# Bedrock RAG Eval Starter

A small, hand-labeled RAG evaluation project for AWS Bedrock documentation. It ingests 12 AWS documentation URLs, answers questions with a Bedrock Claude model, and grades the output with Ragas so failures are visible instead of hidden.

## What This Measures

This repo evaluates whether a Bedrock documentation RAG pipeline can retrieve the right context, answer from that context, and avoid confident wrong answers.

Ragas metrics used:

- faithfulness
- answer_relevancy
- context_precision
- context_recall
- answer_correctness

## Ingested URLs

- https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-invoke.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-invoke-stream.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/security.html

## Latest Scores

The latest report is visible at `reports/latest.html` and the raw JSON is in `reports/eval_report.json`.

```json
{
  "faithfulness": 0.9762,
  "answer_relevancy": 0.5706,
  "context_precision": 0.6417,
  "context_recall": 0.5333,
  "answer_correctness": 0.5285,
  "n_questions": 20
}
```

## Adversarial Case

Does Bedrock support GPT-4o? -> RAG answered: "I don't know based on the provided context."

Ground truth: "No. GPT-4o is an OpenAI model and is not listed as a supported Amazon Bedrock foundation model in the ingested AWS Bedrock documentation."

This is wrong/incomplete for the golden answer. The evaluator caught it: faithfulness = 1.0000, answer_relevancy = 0.0000, context_precision = 0.0000, context_recall = 1.0000, answer_correctness = 0.2239.

Faithfulness stayed high because the model did not hallucinate beyond the retrieved context, but answer_correctness dropped because the answer did not match the expected production answer.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` should contain an AWS profile plus model configuration:

```bash
AWS_REGION=us-east-1
AWS_PROFILE_NAME=your-profile-name
BEDROCK_MODEL_ID=your-bedrock-answer-model-id
BEDROCK_MODEL_PREMIUM=your-bedrock-judge-model-id
EVAL_EMBEDDINGS_MODEL_ID=your-bedrock-embeddings-model-id
```

## Usage

```bash
python ingest.py
python rag.py
python evaluate.py
```

## Project Structure

```text
.
|-- data/urls.txt          # 12 ingested AWS documentation URLs
|-- ingest.py              # Load docs, chunk, embed, and write ChromaDB
|-- rag.py                 # Retrieve context and answer with Bedrock
|-- evaluate.py            # Run Ragas evaluation with Bedrock judge model
|-- test_set.json          # 20 hand-labeled Q&A pairs
|-- reports/latest.html    # Visible latest Ragas score report
|-- reports/eval_report.json
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Notes

- Answer model: configured with `BEDROCK_MODEL_ID`.
- Judge model: configured with `BEDROCK_MODEL_PREMIUM`.
- Vector store: local ChromaDB.
- Golden set: 20 hand-labeled questions, including adversarial cases.

## What these scores mean (honest read)

I deliberately shipped raw numbers, not pretty ones. Here is the diagnosis
my own Ragas report produced:

| Metric              | Score   | Reading                                    |
| ------------------- | ------- | ------------------------------------------ |
| faithfulness        | 0.9762  | Answers stay grounded. Low hallucination.  |
| answer_relevancy    | 0.5706  | Answers drift off the actual question.     |
| context_precision   | 0.6417  | ~35% of retrieved chunks are off-topic.    |
| context_recall      | 0.5333  | Retriever MISSES ~47% of needed context.   |
| answer_correctness  | 0.5285  | Downstream of the retrieval gap.           |

The dominant failure is retrieval, not generation. A starter chunking config
(starter-sized chunks, light overlap, default Bedrock embeddings) leaves real recall gaps
on AWS docs specifically, because Bedrock tables and nested subsection
structure defeat naive splitting.

Concrete next iterations this repo would need to hit 0.8+ recall:
1. Split on <h2>/<h3> semantic boundaries before character-level chunking
2. Add table-aware extraction (current trafilatura loses column context)
3. Hybrid retrieval (BM25 + dense) over pure vector
4. Query rewriting for multi-hop questions

## The adversarial case: "Does Bedrock support GPT-4o?"

Expected: the system should say "No, Bedrock does not host OpenAI models."
Actual: answer_correctness = 0.2239. Faithfulness stayed at 0.97.

Translation: the RAG didn't hallucinate random facts, it *deflected into
adjacent retrieved content* about the Bedrock model catalog. A user asking
the question would walk away uncertain, which in an enterprise support bot
is its own class of failure.

This is why single-metric evals are dangerous. A dashboard showing only
"faithfulness: 0.97 passed" hides the real problem.

If your production RAG is drifting and you don't know it, I run $500 AI Reliability Audits. DM me on LinkedIn.
