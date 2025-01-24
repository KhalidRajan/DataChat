import streamlit as st
import requests
import json

API_BASE_URL = "http://127.0.0.1:5000"

# Streamlit UI
st.title("DataChat")

# Add tabs for different input methods
input_method = st.tabs(["Upload Document", "Enter URL"])

with input_method[0]:
    # Existing file upload logic
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    
    if uploaded_file:
        # Initialize session state for the index if it doesn't exist
        if "index_id" not in st.session_state:
            with st.spinner("Processing document..."):
                file = {"file": uploaded_file}
                response = requests.post(f"{API_BASE_URL}/process_document", files=file)
                
                if response.status_code == 200:
                    data = response.json()
                    if "status" in data and data["status"] == "error":
                        st.error(f"Error processing document: {data['message']}")
                    else:
                        st.session_state.index_id = data["collection_name"]
                        st.success("Document processed successfully!")
                else:
                    st.error(f"Failed to process document. Status code: {response.status_code}")

with input_method[1]:
    # New URL input section
    url = st.text_input("Enter webpage URL:")
    
    if url:
        # Initialize session state for the index if it doesn't exist
        if "index" not in st.session_state:
            with st.spinner("Processing webpage..."):
                
                response = requests.post(f"{API_BASE_URL}/process_url", json={"url": url})
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.index_id = data["collection_name"]
                    st.success("Webpage processed successfully!")
                else:
                    st.error("Failed to process webpage. Please try again.")

# Move query section outside of tabs so it appears for both input methods
if "index_id" in st.session_state:
    # Existing query input and response logic
    query = st.text_input("Ask a question about your document:")
    
    if query:
        with st.spinner("Generating answer..."):
            
            response = requests.post(f"{API_BASE_URL}/query", json={"query": query, "index_id": st.session_state.index_id})

            if response.status_code == 200:
                data = response.json()
                st.write(data["response"])

                # Optional: Display evaluation metrics
                with st.expander("View Response Evaluation"):
                    # Evaluate faithfulness using imported function
                    eval_data = data["evaluation"]
                    st.write(f"Faithfulness Score: {eval_data['faithfulness_score']}")
                    st.write(f"Faithfulness Test Passed: {eval_data['faithfulness_passing']}")
                    st.write(f"Relevancy Score: {eval_data['relevancy_score']}")
                    st.write(f"Relevancy Test Passed: {eval_data['relevancy_passing']}")
                
            else:
                st.error("Failed to generate answer. Please try again.")

            

