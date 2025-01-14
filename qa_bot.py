from dotenv import load_dotenv
import os
import uuid
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.readers.web import SimpleWebPageReader

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
Settings.llm = OpenAI(model = "gpt-4o-mini")
llm = OpenAI(model="gpt-4o-mini")

def load_documents(directory):
    documents = SimpleDirectoryReader(directory).load_data()
    return documents

def create_index(db, documents, collection_name):
    chroma_collection = db.get_or_create_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents, storage_context = storage_context
    )
    return index

def evaluate_faithfulness(query, response) -> tuple[float, bool]:
    faithfulness_evaluator = FaithfulnessEvaluator(llm = llm)
    eval_result = faithfulness_evaluator.evaluate_response(query = query, response=response)
    return (eval_result.score, eval_result.passing)

def evaluate_relevancy(query, response) -> tuple[float, bool]:
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    eval_result = relevancy_evaluator.evaluate_response(query = query, response=response)
    return (eval_result.score, eval_result.passing)

def generate_random_uuid():
    return str(uuid.uuid4())

def scrape_webpage(url):
    documents = SimpleWebPageReader(html_to_text=True).load_data(
        urls = [url]
    )
    return documents