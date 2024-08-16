import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

OPENAI_KEY=os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")
HF_KEY=os.environ.get("HF_TOKEN")


def select_embeddings(em_service):

    source = ""

    if em_service == "openai":
        source = "openai"
        embeddings = OpenAIEmbeddings(
            model='text-embedding-3-large',
            openai_api_key=OPENAI_KEY,
            disallowed_special=())

    if em_service == "google":
        source = "gemini"
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )
    if em_service == "huggingface":
        source = "hf"
        embeddings = HuggingFaceEmbeddings(
            # model_name="sentence-transformers/all-mpnet-base-v2"
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )

    return embeddings, source 


corpus_path = "corpus/pdf-corpus"
text_corpus_path = "corpus/text-corpus"
github_corpus_path = "corpus/github-corpus"


def main(em_service):

    """
    Main function to generate the specified FAISS vector indices.
    """
    pdf_loader = PyPDFDirectoryLoader(corpus_path)
    pdf_docs = pdf_loader.load()
    text_loader = DirectoryLoader(text_corpus_path, glob="*", loader_cls=TextLoader)
    text_docs = text_loader.load()
    github_loader = DirectoryLoader(github_corpus_path, glob="*/*", loader_cls=TextLoader)
    github_docs = github_loader.load()

    # Combine PDF, text, and URL documents
    docs = pdf_docs + text_docs + github_docs
    print(f"pdfs: {len(pdf_docs)}, texts: {len(text_docs)}, github: {len(github_docs)} total: {len(docs)}")

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250,
                                                   separators=['.', '\n', ' '])

    chunked_documents = text_splitter.split_documents(docs)
    print(len(chunked_documents))

    embeddings, source = select_embeddings(em_service)
    store_path = f"{source}-faiss-index"
    isExist = os.path.exists(store_path)

    if not isExist:
        db = FAISS.from_documents(chunked_documents, embeddings)
        db.save_local(store_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate FAISS indices from 3 corpus directories."
    )
    # Add command-line argument for embedding service
    parser.add_argument('--em_service', choices=["huggingface", "google", "openai"], default="google",
            help='Embedding service')
    args = parser.parse_args()

    main(args.em_service)

