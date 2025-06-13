import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # ✅ use Chroma

DATA_PATH = "data/"
CHROMA_DB_PATH = "vectorstore/db_chroma"  # ✅ update path

def load_pdffiles(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(chunks, embedding_model, db_path):
    db = Chroma.from_documents(chunks, embedding_model, persist_directory=db_path)
    db.persist()

if __name__ == "__main__":
    documents = load_pdffiles(DATA_PATH)
    chunks = create_chunks(documents)
    embeddings = get_embedding_model()
    create_vector_store(chunks, embeddings, CHROMA_DB_PATH)
    print("Chroma vector DB created.")
