import streamlit as st
from dotenv import load_dotenv
import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
import chromadb
import tempfile

# Import functions from qa_bot.py
from qa_bot import create_index, evaluate_faithfulness, evaluate_relevancy, load_documents

# Load environment variables and setup
load_dotenv()
Settings.llm = OpenAI(model="gpt-4o-mini")

def save_uploaded_file(uploaded_file):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to the temporary directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Use imported load_documents function instead
        documents = load_documents(temp_dir)
        return documents

# Streamlit UI
st.title("Document Q&A System")

# File upload
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])

if uploaded_file:
    # Initialize session state for the index if it doesn't exist
    if "index" not in st.session_state:
        with st.spinner("Processing document..."):
            # Save and process the uploaded file
            documents = save_uploaded_file(uploaded_file)
            
            # Create or get the database
            db = chromadb.PersistentClient(path="./chroma_db")
            
            # Create the index using imported function
            index = create_index(db, documents, "custom_files")
            
            # Store the index in session state
            st.session_state.index = index
        
        st.success("Document processed successfully!")

    # Query input
    query = st.text_input("Ask a question about your document:")
    
    if query:
        with st.spinner("Generating answer..."):
            # Create query engine and get response
            query_engine = st.session_state.index.as_query_engine()
            response = query_engine.query(query)
            
            # Display response
            st.write("Answer:", response.response)

            # Optional: Display evaluation metrics
            with st.expander("View Response Evaluation"):
                # Evaluate faithfulness using imported function
                faithfulness_score, faithfulness_passing = evaluate_faithfulness(query, response)
                st.write(f"Faithfulness Score: {faithfulness_score}")
                st.write(f"Faithfulness Test Passed: {faithfulness_passing}")
                
                # Evaluate relevancy using imported function
                relevancy_score, relevancy_passing = evaluate_relevancy(query, response)
                st.write(f"Relevancy Score: {relevancy_score}")
                st.write(f"Relevancy Test Passed: {relevancy_passing}") 