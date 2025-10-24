"""
BasisQuest.py

BasisQuest is an AI-powered knowledge assistant built to explore and understand SAP Basis and enterprise automation documentation.

It combines retrieval-augmented generation (RAG), semantic search, and LLM reasoning to help engineers find precise answers from SAP technical manuals, whitepapers, and API references — all through a simple conversational interface.

Designed for both demonstration and real-world scalability, BasisQuest embodies the future of intelligent enterprise support.
"""

# ===============================
# Imports (sorted alphabetically)
# ===============================
import faiss
import numpy as np
import os
import re
from bs4 import BeautifulSoup
from collections import deque
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from tqdm import tqdm
from typing import Any, Deque, Dict, List, Tuple


# ===============================
# Configuration
# ===============================
CHUNK_SIZE: int = 4          # paragraphs per chunk
CHUNK_OVERLAP: int = 1       # overlapping paragraphs
TOP_K: int = 6               # FAISS top results
RERANK_TOP_M: int = 3        # re-ranked top results

EMBED_MODEL: str = "text-embedding-3-small"
CHAT_MODEL_FAST: str = "gpt-4o-mini"
CHAT_MODEL_STRICT: str = "gpt-4o"

DOCS_FOLDER: str = "docs"

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ===============================
# Preprocessing Utilities
# ===============================
def clean_text(text: str) -> str:
    """Clean up raw text before embedding."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def split_paragraphs(text: str, chunk_size: int = 3,
                     chunk_overlap: int = 1) -> List[str]:
    """Split text into overlapping chunks of paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(paragraphs), step):
        chunk = " ".join(paragraphs[i:i + chunk_size])
        chunks.append(clean_text(chunk))
    return chunks


# ===============================
# Document Loading
# ===============================
def load_documents(folder: str) -> List[Dict[str, str]]:
    """Load .txt, .pdf, .docx, .html files into text chunks."""
    docs: List[Dict[str, str]] = []

    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            ext = file.lower().split(".")[-1]
            text = ""

            try:
                if ext in ("txt", "md"):
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                elif ext == "pdf":
                    reader = PdfReader(path)
                    text = "\n".join(page.extract_text() or ""
                                     for page in reader.pages)
                elif ext == "docx":
                    doc = Document(path)
                    text = "\n".join(p.text for p in doc.paragraphs)
                elif ext == "html":
                    with open(path, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f, "html.parser")
                        text = soup.get_text(separator="\n")
            except Exception as e:
                print(f"Failed to read {file}: {e}")
                continue

            if text.strip():
                for chunk in split_paragraphs(text, CHUNK_SIZE, CHUNK_OVERLAP):
                    docs.append({"content": chunk, "source": file})

    print(f"Documents loaded, total chunks: {len(docs)}")
    return docs


# ===============================
# Embedding Generation (Batch)
# ===============================
def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    """Generate normalized embeddings for multiple texts."""
    if not texts:
        return np.empty((0, 1536), dtype="float32")

    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    embs = [np.array(d.embedding, dtype="float32") for d in response.data]
    embs = [e / np.linalg.norm(e) for e in embs]
    return np.vstack(embs)


# ===============================
# Index Building
# ===============================
def build_index(docs: List[Dict[str, str]]
                ) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Create a FAISS cosine-similarity index."""
    texts = [d["content"] for d in docs]
    embeddings = []

    BATCH_SIZE = 50
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch = texts[i:i + BATCH_SIZE]
        batch_embs = get_embeddings_batch(batch)
        embeddings.append(batch_embs)

    if not embeddings:
        raise ValueError("No embeddings generated — check your documents.")

    all_embs = np.vstack(embeddings)
    dim = all_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embs)
    return index, all_embs


# ===============================
# Semantic Re-ranking
# ===============================
def rerank(query_emb: np.ndarray, doc_embs: np.ndarray,
           top_indices: np.ndarray, docs: List[Dict[str, str]],
           top_m: int = 3) -> List[Dict[str, str]]:
    """Re-rank retrieved chunks by cosine similarity."""
    selected_embs = doc_embs[top_indices]
    sims = np.dot(selected_embs, query_emb)
    reranked_idx = [top_indices[i] for i in np.argsort(-sims)[:top_m]]
    return [docs[i] for i in reranked_idx]


# ===============================
# Model Selection
# ===============================
def choose_model(query: str, context: str) -> str:
    """Select model based on input length."""
    return CHAT_MODEL_FAST if len(query + context) < 3000 else CHAT_MODEL_STRICT


# ===============================
# Conversation Memory Management
# ===============================
def summarize_history(full_history: str) -> str:
    """Summarize chat history for memory compression."""
    if not full_history.strip():
        return ""

    response = client.chat.completions.create(
        model=CHAT_MODEL_FAST,
        messages=[
            {"role": "system",
             "content": ("You are a summarization assistant. Summarize the "
                         "conversation concisely in 3-5 sentences.")},
            {"role": "user", "content": full_history},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def update_history_queue(history_queue: Deque[str], query: str,
                         answer: str, max_turns: int = 5
                         ) -> Tuple[Deque[str], str]:
    """Keep a rolling conversation memory and summarize older parts."""
    history_queue.append(f"User: {query}\nAssistant: {answer}")
    if len(history_queue) > max_turns:
        old_history = "\n".join(list(history_queue)[:-max_turns])
        summary = summarize_history(old_history)
        history_queue.clear()
        history_queue.append(f"Summary of prior discussion: {summary}")
    return history_queue, "\n".join(list(history_queue))


# ===============================
# Chat Handling
# ===============================
def chat(query: str, context: str, history: str = "") -> str:
    """Generate an answer based on context and chat history."""
    model = choose_model(query, context)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": ("You are a helpful assistant that answers questions "
                         "based on provided documents. Respond concisely, "
                         "use bullet points when listing information, "
                         "and stay consistent with prior context.")},
            {"role": "user",
             "content": (f"Previous conversation:\n{history}\n\n"
                         f"Context:\n{context}\n\n"
                         f"Question: {query}\n\n"
                         f"Provide a helpful, structured answer "
                         "(use bullet points when appropriate).")},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ===============================
# Main Loop
# ===============================
def run_chatbot() -> None:
    """Main chatbot loop with summarizing memory."""
    docs = load_documents(DOCS_FOLDER)
    index, all_embs = build_index(docs)

    print("\nChatbot with auto-summarizing memory ready! Type 'exit' to quit.\n")
    history_queue: Deque[str] = deque()

    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break

        q_emb = get_embeddings_batch([query])[0]
        _, I = index.search(np.array([q_emb]), TOP_K)
        top_indices = I[0]
        top_docs = rerank(q_emb, all_embs, top_indices, docs, RERANK_TOP_M)

        combined_context = "\n".join(d["content"] for d in top_docs)
        full_history = "\n".join(list(history_queue))
        answer = chat(query, combined_context, full_history)
        sources = ", ".join(sorted(set(d["source"] for d in top_docs)))

        print(f"\nChatbot: {answer}\nSources: {sources}\n")

        history_queue, _ = update_history_queue(history_queue, query, answer)


# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run_chatbot()
