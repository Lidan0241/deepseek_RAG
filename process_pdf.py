# Document processing and retrieval  
from langchain_community.document_loaders import PyMuPDFLoader  # Extracts text from PDF files for processing
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into smaller chunks for better embedding and retrieval
from langchain.vectorstores import Chroma  # Handles storage and retrieval of vector embeddings using ChromaDB
from langchain_community.embeddings import OllamaEmbeddings  
# Define the function that processes the PDF
def process_pdf(pdf_bytes):
    # If PDF files are empty, return None — This prevents errors from trying to process an empty input.
    if pdf_bytes is None:
        return None, None, None
    # PyMuPDFLoader initializes the PDF file
    loader = PyMuPDFLoader(pdf_bytes) 
    # .load() method reads the content of the PDF and extracts its text
    data = loader.load()
    # RecursiveCharacterTextSplitter splits the PDF into chunks of 500 characters, keeping 100 characters overlap to keep context 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # Splits the documents into chunks and stores them in chunks object
    chunks = text_splitter.split_documents(data)
    # Create embeddings using OllamaEmbeddings 
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    # Create a vector database which allows us to store the chunks and their embeddings
    vectorstore=Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")  # Example directory
    # This creates a retriever that enables searching through the vectorstore.
    retriever = vectorstore.as_retriever()
    """
    The function returns 3 objects
        text_splitter → (Used to split new text in the same way as before)
        vectorstore → (Holds the processed document chunks)
        retriever → (Used to fetch relevant document chunks when answering questions)
    """
    
    return text_splitter, vectorstore, retriever

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)