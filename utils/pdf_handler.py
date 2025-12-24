import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

class RAGPipeline:
    def __init__(self):
        # Load local SentenceTransformer model (Resume Claim)
        print("Loading SentenceTransformer model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

    def process_pdf(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # 1. Load & Split (Resume Claim: "preprocessing and chunking")
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # 2. Index (Resume Claim: "FAISS indexing")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            return f"Indexed {len(splits)} chunks."
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_retriever(self):
        if not self.vector_store:
            return None
        return self.vector_store.as_retriever(search_kwargs={"k": 3})