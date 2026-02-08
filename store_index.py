from dotenv import load_dotenv
import os
import glob

from src.helper import filter_to_minimal_docs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing Pinecone API key")


# ✅ Load all PDFs from folder
docs = []

for pdf in glob.glob("data/*.pdf"):
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())

print(f"Loaded {len(docs)} pages")


# Optional filtering
filter_data = filter_to_minimal_docs(docs)


# ✅ Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

text_chunks = splitter.split_documents(filter_data)

print(f"Created {len(text_chunks)} chunks")


# ✅ Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ✅ Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

existing_indexes = [i.name for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# ✅ Upload to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Upload complete!")
