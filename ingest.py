import os, requests, chromadb
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

URLS_PATH = Path("data/urls.txt")

def load_urls():
    return [
        line.strip()
        for line in URLS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

def scrape(url):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.find("div", {"id": "main-content"}) or soup.find("body")
    return main.get_text(separator="\n", strip=True)

def build_index():
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("bedrock_docs")
    except ValueError:
        pass
    collection = client.create_collection("bedrock_docs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80
    )

    urls = load_urls()
    for i, url in enumerate(urls):
        print(f"Ingesting {i+1}/{len(urls)}: {url}")
        text = scrape(url)
        chunks = splitter.split_text(text)
        if not chunks:
            print(f"Skipping empty page: {url}")
            continue
        collection.add(
            documents=chunks,
            ids=[f"doc{i}_chunk{j}" for j, _ in enumerate(chunks)],
            metadatas=[{"source": url} for _ in chunks]
        )
    print(f"✅ Indexed {collection.count()} chunks")

if __name__ == "__main__":
    build_index()
