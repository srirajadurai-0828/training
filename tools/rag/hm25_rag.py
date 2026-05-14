import os
import json
import logging
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from pydantic import BaseModel
from typing import Optional

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Document loading & splitting
# ─────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_PDF_PATH = os.path.join(_DIR, "Horizon_Bank_FAQ_Knowledge_Base.pdf")

loader = PyPDFLoader(_PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)

for i, doc in enumerate(docs):
    doc.metadata["doc_id"] = i

# ─────────────────────────────────────────────
# Embeddings + Pinecone vector store
# ─────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_RAG_INDEX_NAME")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
vectorstore.add_documents(docs)

# ─────────────────────────────────────────────
# BM25 sparse retriever
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
docs_tokenized = [tokenizer.tokenize(doc.page_content) for doc in docs]
bm25 = BM25Okapi(docs_tokenized)


# ─────────────────────────────────────────────
# LLM-as-Judge  — structured output schema
# ─────────────────────────────────────────────
class ChunkJudgement(BaseModel):
    doc_id: int
    relevance_score: float   # 0.0 – 1.0
    reasoning: str           # one-sentence justification


class JudgeResult(BaseModel):
    judgements: list[ChunkJudgement]


_JUDGE_SYSTEM_PROMPT = """You are a relevance judge for a banking FAQ retrieval system.

Given a user query and a list of document chunks (each with an id and content),
score each chunk for relevance to the query on a scale from 0.0 to 1.0:
  1.0  — directly and fully answers the query
  0.7  — closely related, partially answers
  0.4  — loosely related background info
  0.0  — irrelevant

Respond ONLY with valid JSON that matches this schema (no markdown, no preamble):
{
  "judgements": [
    {"doc_id": <int>, "relevance_score": <float>, "reasoning": "<one sentence>"},
    ...
  ]
}"""


def _llm_judge(query: str, candidate_docs: list, threshold: float = 0.5) -> list:
    """
    Send candidate chunks to an LLM judge.
    Returns docs whose relevance_score >= threshold, sorted best-first.

    Falls back to Anthropic → OpenAI (mirrors your SafeLLM pattern).
    Raises only if both providers fail.
    """
    if not candidate_docs:
        return []

    # Build the user message: numbered chunks
    chunks_text = "\n\n".join(
        f"[chunk {doc.metadata['doc_id']}]\n{doc.page_content}"
        for doc in candidate_docs
    )
    user_message = f"Query: {query}\n\nChunks:\n{chunks_text}"

    messages = [
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    raw_response = None

    # Primary: Anthropic
    try:
        judge_llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            temperature=0,
            max_tokens=1024,
        )
        raw_response = judge_llm.invoke(messages).content
        logger.info("LLM judge: Anthropic succeeded")
    except Exception as anthropic_err:
        logger.warning(f"LLM judge: Anthropic failed ({anthropic_err}), trying OpenAI...")
        try:
            judge_llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                max_tokens=1024,
            )
            raw_response = judge_llm.invoke(messages).content
            logger.info("LLM judge: OpenAI fallback succeeded")
        except Exception as openai_err:
            logger.error(f"LLM judge: both providers failed. Anthropic={anthropic_err} | OpenAI={openai_err}")
            # Graceful degradation: return all candidates unfiltered
            logger.warning("LLM judge: returning all candidates unfiltered.")
            return candidate_docs

    # Parse the JSON response
    try:
        parsed = json.loads(raw_response)
        judgements = parsed.get("judgements", [])
    except json.JSONDecodeError as e:
        logger.error(f"LLM judge: failed to parse JSON response: {e}\nRaw: {raw_response}")
        return candidate_docs  # graceful fallback

    # Build a score map and log results
    score_map: dict[int, float] = {}
    for j in judgements:
        doc_id = j.get("doc_id")
        score = j.get("relevance_score", 0.0)
        reason = j.get("reasoning", "")
        score_map[doc_id] = score
        logger.info(f"  [doc_id={doc_id}] score={score:.2f} | {reason}")

    # Filter by threshold and sort best-first
    accepted = [
        doc for doc in candidate_docs
        if score_map.get(doc.metadata["doc_id"], 0.0) >= threshold
    ]
    accepted.sort(
        key=lambda d: score_map.get(d.metadata["doc_id"], 0.0),
        reverse=True,
    )

    logger.info(
        f"LLM judge: {len(candidate_docs)} candidates → "
        f"{len(accepted)} passed threshold={threshold}"
    )
    return accepted


# ─────────────────────────────────────────────
# Public retriever tool
# ─────────────────────────────────────────────
def hm25_retriever_tool(
    query: str,
    k: int = 5,
    judge_threshold: float = 0.5,
    use_judge: bool = True,
) -> str:
    """
    Hybrid BM25 + vector retrieval with optional LLM-as-judge reranking.

    Pipeline:
      1. BM25 sparse retrieval   → top k doc_ids
      2. Vector similarity search → top k doc_ids
      3. Union of both sets       → candidate pool (up to 2k chunks)
      4. LLM judge scores each candidate for relevance
      5. Return chunks that pass `judge_threshold`, best-first
         (falls back to unfiltered union if the judge fails)
    """
    # Step 1 — BM25
    tokenized_query = tokenizer.tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_k = np.argsort(bm25_scores)[::-1][:k]

    # Step 2 — Vector
    vector_results = vectorstore.similarity_search_with_score(query, k=k)
    vector_top_k = [item[0].metadata["doc_id"] for item in vector_results]

    # Step 3 — Union
    combined_ids = list(
        set([int(i) for i in bm25_top_k] + [int(i) for i in vector_top_k])
    )
    candidate_docs = [docs[i] for i in combined_ids]

    logger.info(
        f"Retriever: BM25={list(bm25_top_k)} | "
        f"Vector={vector_top_k} | "
        f"Union={combined_ids}"
    )

    # Step 4 — LLM judge reranking
    if use_judge:
        reranked = _llm_judge(query, candidate_docs, threshold=judge_threshold)
    else:
        reranked = candidate_docs[:k]

    # Step 5 — Return top-k as a single string
    final_docs = reranked[:k] if reranked else candidate_docs[:k]
    return "\n\n".join(doc.page_content for doc in final_docs)