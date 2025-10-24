# chat_with_pdf.py
"""
INFO 5940 Assignment 1: RAG Application
Student: [xc572]
Description: Main application file for the RAG system with Streamlit UI.
"""

import streamlit as st
import os
from openai import OpenAI
from os import environ

# LangChain components matching instructor's notebook
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize OpenAI client with Cornell endpoint 
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.ai.it.cornell.edu",
)


def process_uploaded_files(uploaded_files):
    """
    Process uploaded TXT and PDF files using instructor's notebook loading approach
    """
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        # Create temporary file (teacher uses file paths for loaders)
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Use exact same loader pattern as instructor's notebook
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(temp_filename)
            else:  # .txt files
                loader = TextLoader(temp_filename)
            
            documents = loader.load()
            
            # Use instructor's exact text splitting parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,    # Reference to instructor's notebook
                chunk_overlap=0    # Reference to instructor's notebook
            )
            
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f" Processed {uploaded_file.name}: {len(documents)} -> {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error processing {uploaded_file.name}: {e}")
            raise e
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    return all_chunks

def setup_vector_store(chunks):
    """
    Create vector store using instructor's exact Chroma and embedding setup
    """
    # Match instructor's embedding model exactly
    embeddings = OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        openai_api_key=os.environ["API_KEY"],
        openai_api_base="https://api.ai.it.cornell.edu"
    )
    
    # Create vector store exactly like instructor's notebook
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings
    )
    
    print(f" Vector store created with {len(chunks)} chunks")
    return vectorstore


def format_docs(docs):       
    """
    Format documents for context presentation in RAG responses.
    
    Integration with the instructor's code:
    Directly reuse the  format_docs  function from the instructor's notebook.
    Maintain identical document formatting logic

    Design purpose:
    Convert retrieved document chunks into a context string readable by the LLM
    Use delimiters to clearly distinguish between different document chunks
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_rag_answer(question, vectorstore, k=5):
    """
    Implement the core RAG workflow following teacher's minimal approach.
    
    Integration Methods with the Instructor's Code:
    Reuse the similarity search process from the instructor's notebook.
    Maintain the same context construction pattern.
    Use the instructor-recommended retrieval parameter k=5.

    Design Objectives:
    Encapsulate the teacher's notebook workflow into callable functions.
    Integrate similarity search + context construction + LLM calls.
    Provide a unified Q&A interface for the Streamlit application.
    
    
    """
    try:
        # 1) Retrieve - using instructor's similarity search approach
        docs = vectorstore.similarity_search(question, k=k)
        print(f"Retrieved {len(docs)} relevant document chunks")
        
        # 2) Build context following instructor's pattern
        context = format_docs(docs)
        
        # 3) Create system instructions based on teacher's concise approach
        system_instructions = (
            "You are a helpful assistant for question answering.\n"
            "Use ONLY the provided context to answer concisely (<=3 sentences).\n"
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}"
        )
        
        # 4) Call API using instructor's chat completion pattern
        stream = client.chat.completions.create(
            model="openai.gpt-4o",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": question}
            ],
            stream=True,
            temperature=0.2
        )
        
        return stream, docs
        
    except Exception as e:
        print(f"RAG workflow error: {e}")
        raise e


def initialize_session_state():
    """
    Initialize Streamlit session state for managing application state.
    
    Design Objectives:
    Manage application state including chat history and vector storage.
    Ensure state persistence after page refresh.
    
    Integration Methods with the instructor's code:
    The instructor utilizes st.session_state.messages to manage chat history.
    Here we extend the state management for vector storage, complying with the
    multi-document requirements of the assignment.

    """
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload documents and ask questions about them!"}]
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

def main():
    """
    Main Streamlit application integrating all RAG components.
    
    Integration Approach with Instructor's Code:
    Reuse the basic interface structure from the instructor's chat_with_pdf.py.
    Extend it to support multi-file uploads and a complete RAG workflow, meeting assignment requirements
    
    Design Objectives:
    Integrate the functionalities of all preceding steps into a unified web interface.
    Provide a complete user experience for file upload → processing → Q&A interactions.
    Implement multi-document support and a chat interface as required by the assignment specifications.
    
    
    """
    st.set_page_config(
        page_title="Document RAG Chat System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Document RAG Chat System")
    st.markdown("Upload your documents and chat with them using AI!")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for file upload:Extend the instructor's upload function to support multiple files
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # Multi-file upload 
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload .txt or .pdf files to chat with"
        )
        
        # Process uploaded files
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    all_chunks = process_uploaded_files(uploaded_files)
                    
                    if all_chunks:
                        vectorstore = setup_vector_store(all_chunks)
                        
                        # Store state and reset chat
                        st.session_state.vectorstore = vectorstore
                        st.session_state.messages = [{"role": "assistant", "content": "Documents processed! Ask me anything about them."}]
                        
                        st.success(f"Processed {len(uploaded_files)} files with {len(all_chunks)} text chunks!")
                    else:
                        st.error("No content could be extracted from the files.")
                        
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
        
        # API Key reminder 
        st.markdown("---")
        st.subheader("Setup Required")
        st.markdown("""
        Before using this app, set your OpenAI API key in terminal:
        ```bash
        export API_KEY="your-api-key-here"
        ```
        """)
    
    # Main chat interface - Expanding the instructor's Chat Interface
    st.subheader("💬 Chat with Your Documents")
    
    # Display chat history - Reusing the instructor's message display logic
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Chat input - Intelligent disabling based on document status
    question = st.chat_input(
        "Ask something about your documents",
        disabled=st.session_state.vectorstore is None
    )
    
    # Handle user questions - Integrate our STEP 3 RAG functionality
    if question and st.session_state.vectorstore:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            try:
                # Retrieve answers using our STEP 3 RAG function
                stream, source_docs = get_rag_answer(question, st.session_state.vectorstore)
                response = st.write_stream(stream)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display source documents:Enhanced functionality to facilitate verification of answer accuracy.
                with st.expander("📎 View Source Documents"):
                    for i, doc in enumerate(source_docs, 1):
                        st.write(f"**Source {i}:**")
                        st.write(doc.page_content)
                        st.write("---")
                        
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.write(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Application entry point
if __name__ == "__main__":
    main()
