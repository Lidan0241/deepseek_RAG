from langchain_community.document_loaders import PyMuPDFLoader  # Extracts text from PDF files for processing
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into smaller chunks for better embedding and retrieval
from langchain.vectorstores import Chroma 

def ollama_llm(question, context):

    # Format the prompt with the question and context to provide structured input for the AI
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    # Send the structured prompt to the Ollama model for processing
    response = ollama.chat(
        model="deepseek-r1:1.5b",  # Specifies the AI model to use
        messages=[{'role': 'user', 'content': formatted_prompt}]  # Formats the user input
    )
    # Extract the AI-generated response content
    response_content = response['message']['content']
    # Remove content inside <think>...</think> tags to clean up AI reasoning traces
    final_answer = re.sub(r'<think>.*?</think>', # We're searching for think tags
                          '', # We'll replace them with empty spaces
                          response_content, # In response_content
                          flags=re.DOTALL).strip() # (dot) should match newlines (\n) as well.
    # Return the final cleaned response
    return final_answer

# Define rag_chain function for Retrieval Augmented Generation
def rag_chain(question, text_splitter, vectorstore, retriever):
    """
    This function takes as input:
        - The question we want to ask the model
        - The text_splitter object to split the PDF and read into chunks
        - The vectorstore for retrieving embeddings 
        - The retriever objects which retrieves data from the vectorstore
    """
    retrieved_docs = retriever.invoke(question) # In this step, we will find the part of the document most relevant to the question
    formatted_content = combine_docs(retrieved_docs) # We will then combine the retrieved parts of the document 
    return ollama_llm(question, formatted_content) # Run the model on the question, and the relevant context from the document

# Put it all together — Create a function that performs the logic expected by the Chatbot  
def ask_question(pdf_bytes, question): 
    text_splitter, vectorstore, retriever = process_pdf(pdf_bytes) # Process the PDF
    if text_splitter is None:
        return None  # No PDF uploaded    
    result = rag_chain(question, text_splitter, vectorstore, retriever) # Return the results with RAG
    return {result}

