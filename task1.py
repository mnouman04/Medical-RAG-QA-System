# Medical RAG QA System - Core Implementation
# Install required packages first:
# pip install langchain langchain-google-genai chromadb pandas sentence-transformers

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Step 1: Set up API Key
# Get your free API key from https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = "AIzaSyBGqAZObVGNHXQ8fCjnWb2rPm7uCE74snk"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Step 2: Load and preprocess the medical dataset
def load_medical_data(csv_path):
    """Load medical transcriptions from CSV file"""
    print("Loading medical transcriptions...")
    df = pd.read_csv(csv_path)
    
    # Basic cleaning
    df = df.dropna(subset=['transcription'])
    df['transcription'] = df['transcription'].astype(str)
    
    print(f"Loaded {len(df)} medical records")
    return df

# Step 3: Create document chunks with metadata
def create_documents(df):
    """Convert dataframe to LangChain documents with metadata"""
    print("Creating documents with metadata...")
    documents = []
    
    for idx, row in df.iterrows():
        # Combine relevant fields for better context
        content = f"Medical Specialty: {row.get('medical_specialty', 'Unknown')}\n"
        content += f"Description: {row.get('description', '')}\n"
        content += f"Transcription: {row['transcription']}"
        
        # Create metadata for each document
        metadata = {
            'specialty': str(row.get('medical_specialty', 'Unknown')),
            'description': str(row.get('description', ''))[:200],
            'doc_id': idx
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    print(f"Created {len(documents)} documents")
    return documents

# Step 4: Split documents into chunks
def split_documents(documents):
    """Split large documents into smaller chunks"""
    print("Splitting documents into chunks...")
    
    # I'm using RecursiveCharacterTextSplitter because it tries to keep
    # related content together by splitting on paragraphs first, then sentences
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk is about 1000 characters
        chunk_overlap=200,  # 200 char overlap to maintain context
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

# Step 5: Create vector store with embeddings
def create_vector_store(chunks):
    """Create Chroma vector database with embeddings"""
    print("Creating embeddings and vector store...")
    print("This might take a few minutes...")
    
    # Using Google's embedding model (free with Gemini API)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create Chroma vector store
    # persist_directory saves the database to disk so we don't have to recreate it
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./medical_chroma_db"
    )
    
    print("Vector store created successfully!")
    return vectorstore

# Step 6: Load existing vector store (if already created)
def load_vector_store():
    """Load existing vector store from disk"""
    print("Loading existing vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory="./medical_chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

# Step 7: Create RAG chain
def create_rag_chain(vectorstore):
    """Create the RAG QA chain with retriever and LLM"""
    print("Setting up RAG chain...")
    
    # Create a retriever that fetches top 4 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create custom prompt template for medical QA
    prompt_template = """You are a helpful medical assistant. Use the following pieces of medical transcription context to answer the question. 
If you don't know the answer based on the context, say "I don't have enough information in the medical records to answer this question."

Always cite which medical specialty or type of consultation the information comes from.

Context: {context}

Question: {question}

Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,  # Lower temperature for more factual responses
        convert_system_message_to_human=True
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" means put all retrieved docs into prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# Step 8: Ask questions
def ask_question(qa_chain, question):
    """Ask a question and get an answer with sources"""
    print(f"\nQuestion: {question}")
    print("-" * 80)
    
    result = qa_chain({"query": question})
    
    answer = result['result']
    sources = result['source_documents']
    
    print(f"Answer: {answer}\n")
    print(f"Retrieved from {len(sources)} source documents:")
    for i, doc in enumerate(sources, 1):
        specialty = doc.metadata.get('specialty', 'Unknown')
        description = doc.metadata.get('description', '')[:100]
        print(f"\n  Source {i}:")
        print(f"    Specialty: {specialty}")
        print(f"    Description: {description}...")
    
    print("=" * 80)
    return answer, sources

# Main execution
if __name__ == "__main__":
    print("Medical RAG QA System")
    print("=" * 80)
    
    # Path to your downloaded CSV file
    CSV_PATH = "mtsamples.csv"
    
    # Choose whether to create new vector store or load existing one
    CREATE_NEW = True  # Set to False if you already created the vector store
    
    if CREATE_NEW:
        # Full pipeline: load data, create chunks, build vector store
        df = load_medical_data(CSV_PATH)
        documents = create_documents(df)
        chunks = split_documents(documents)
        vectorstore = create_vector_store(chunks)
    else:
        # Load existing vector store (much faster)
        vectorstore = load_vector_store()
    
    # Create RAG chain
    qa_chain = create_rag_chain(vectorstore)
    
    # Example questions
    questions = [
        "What are the common symptoms of diabetes?",
        "How is hypertension typically treated?",
        "What are the causes of chest pain?",
        "Explain the procedure for a colonoscopy",
        "What medications are used for asthma?"
    ]
    
    print("\nAsking sample medical questions...\n")
    
    for question in questions:
        ask_question(qa_chain, question)
    
    # Interactive mode
    print("\n\nInteractive Mode - Enter your questions (type 'quit' to exit)")
    print("=" * 80)
    
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using Medical RAG QA System!")
            break
        
        if user_question.strip():
            ask_question(qa_chain, user_question)


            # import chromadb

# client = chromadb.CloudClient(
#   api_key='ck-3723EHDVJKWiuPXWCEZNa3kdCAVKrPh2BsjCNDve2dyj',
#   tenant='8915e904-a4aa-4201-bcf8-832c521d3f5b',
#   database='Task1'
# )