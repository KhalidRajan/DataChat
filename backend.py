from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import chromadb
import tempfile

from qa_bot import (
    create_index,
    evaluate_faithfulness,
    evaluate_relevancy, 
    load_documents,
    scrape_webpage,
    generate_random_uuid
)

load_dotenv()
Settings.llm = OpenAI(model = "gpt-4o-mini")

app = Flask(__name__)
CORS(app)

indices = {}

# Create or get the database
db = chromadb.PersistentClient(path="./chroma_db")

@app.route('/process_document', methods=['POST'])
def process_document():
    try:
        uploaded_file = request.files['file']

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file to the temporary directory
            file_path = os.path.join(temp_dir, uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # Use imported load_documents function instead
            documents = load_documents(file_path)
            print(documents)
            
            # Use random UUID for collection name
            collection_name = generate_random_uuid()
            index = create_index(db, documents, collection_name)

            indices[collection_name] = index

            return jsonify({"status": "success", "collection_name": collection_name})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/process_url', methods=['POST'])
def process_url():
    try:
        url = request.json['url']

        documents = scrape_webpage(url)

            
        # Use random UUID for collection name
        collection_name = generate_random_uuid()
        index = create_index(db, documents, collection_name)

        indices[collection_name] = index

        return jsonify({"status": "success", "collection_name": collection_name})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query_text = data['query']
        index_id = data['index_id']

        if index_id not in indices:
            return jsonify({"status": "error", "message": "Index not found"})
        
        index = indices[index_id]

        query_engine = index.as_query_engine()
        response = query_engine.query(query_text)

        faithfulness_score, faithfulness_passing = evaluate_faithfulness(query_text, response)
        relevancy_score, relevancy_passing = evaluate_relevancy(query_text, response)

        return jsonify({
            "status": "success", 
            "response": str(response),
            "evaluation": {
                "faithfulness_score": faithfulness_score,
                "faithfulness_passing": faithfulness_passing,
                "relevancy_score": relevancy_score,
                "relevancy_passing": relevancy_passing
            }
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)