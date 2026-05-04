import boto3, json, chromadb
from dotenv import load_dotenv
import os

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE_NAME") or os.getenv("AWS_PROFILE")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

if not BEDROCK_MODEL_ID:
    raise RuntimeError("Set BEDROCK_MODEL_ID in .env before running rag.py")

session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
bedrock = session.client("bedrock-runtime", region_name=AWS_REGION)

def retrieve(query: str, n=3) -> list[str]:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("bedrock_docs")
    results = collection.query(query_texts=[query], n_results=n)
    return results["documents"][0]  # list of chunk strings

def generate(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided context."

Context:
{context_block}

Question: {question}
Answer:"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_" + "to" + "kens": 512,
        "messages": [{"role": "user", "content": prompt}]
    })

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

def run_pipeline(question: str) -> dict:
    contexts = retrieve(question)
    answer = generate(question, contexts)
    return {
        "question": question,
        "contexts": contexts,
        "answer": answer
    }

if __name__ == "__main__":
    test = "What is Amazon Bedrock?"
    result = run_pipeline(test)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"\nContexts used:\n" + "\n---\n".join(result['contexts'][:1]))
