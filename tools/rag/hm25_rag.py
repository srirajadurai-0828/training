import os
import numpy as np

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

load_dotenv()

_DIR = os.path.dirname(os.path.abspath(__file__))
_PDF_PATH = os.path.join(_DIR, "Banking_FAQ_Knowledge_Base.pdf")

loader = PyPDFLoader(_PDF_PATH)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

for i, doc in enumerate(docs):
    doc.metadata["doc_id"] = i

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "horizon-bank-faq"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

vectorstore.add_documents(docs)

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

docs_tokenized = [
    tokenizer.tokenize(doc.page_content)
    for doc in docs
]

bm25 = BM25Okapi(docs_tokenized)

def hm25_retriever_tool(query: str, k: int = 5):

    tokenized_query = tokenizer.tokenize(query)

    bm25_results = bm25.get_scores(tokenized_query)

    bm25_top_k = np.argsort(bm25_results)[::-1][:k]

    vector_results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    vector_top_k = [
        item[0].metadata["doc_id"]
        for item in vector_results
    ]

    combined_top_k = list(
        set([int(i) for i in bm25_top_k] + [int(i) for i in vector_top_k])
    )[:k]

    retrieved_docs = [docs[i] for i in combined_top_k]

    return "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )