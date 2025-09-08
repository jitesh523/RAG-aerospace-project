import argparse, os, uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from src.index.faiss_index import build_faiss
from src.index.milvus_index import insert_rows
from src.config import Config

def load_pdfs(input_dir: str):
    docs = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, f))
                docs.extend(loader.load())
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def to_milvus_rows(chunks, embeddings):
    rows = []
    texts = [c.page_content for c in chunks]
    embs = embeddings.embed_documents(texts)
    for emb, doc in zip(embs, chunks):
        rid = str(uuid.uuid4())[:32]
        src = doc.metadata.get("source", "")
        page = int(doc.metadata.get("page", -1))
        rows.append((rid, emb, doc.page_content[:65000], src, page))
    return rows

def main(input_dir: str, batch_size: int):
    docs = load_pdfs(input_dir)
    chunks = chunk_docs(docs)
    # Build a hot FAISS index for API service warm start (optional persist w/ .save_local)
    faiss_index = build_faiss(chunks)
    faiss_index.save_local("./faiss_store")

    # Persist everything to Milvus
    embeddings = OpenAIEmbeddings(model=Config.EMBED_MODEL, api_key=Config.OPENAI_API_KEY)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        rows = to_milvus_rows(batch, embeddings)
        insert_rows(rows)
        print(f"[ingest] inserted {i+len(batch)}/{len(chunks)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with PDFs")
    parser.add_argument("--batch_size", type=int, default=200)
    args = parser.parse_args()
    main(args.input, args.batch_size)
