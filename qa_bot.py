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
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
Settings.llm = OpenAI(model = "gpt-4o-mini")
llm = OpenAI(model="gpt-4o-mini")

def load_documents(file_path):
    try:
        if os.path.isfile(file_path):
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            if not documents:
                raise ValueError("No content could be extracted from the file")
            return documents
        else:
            raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading document: {str(e)}")

def create_index(db, documents, collection_name):
    chroma_collection = db.get_or_create_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection = chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents, storage_context = storage_context, store_nodes_override=True
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

def create_bm25_retriever(index):
    bm25_retriever = BM25Retriever.from_defaults(
        docstore = index.docstore,
        similarity_top_k = 2
    )
    return bm25_retriever

def create_fusion_retriever(vector_retriever, bm25_retriever):
    retriever = QueryFusionRetriever(
        retrievers = [vector_retriever, bm25_retriever],
        similarity_top_k = 2,
        num_queries = 4,
        mode = "reciprocal_rerank",
        use_async = True,
        verbose = True
    )
    return retriever

def create_fusion_query_engine(retriever):
    query_engine = RetrieverQueryEngine.from_args(retriever)
    return query_engine
