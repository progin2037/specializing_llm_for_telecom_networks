from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import chromadb
import pandas as pd
import re
from tqdm import tqdm

from utils import create_dir_with_sampled_docs, remove_release_number, create_empty_directory


def rag_inference_on_train(data: pd.DataFrame,
                           engine: RetrieverQueryEngine):
    """
    Perform RAG inference on train data and save results. The output will be used in fine-tuning.

    Args:
        data (pd.DataFrame): A DataFrame with data about questions and their options
        engine (RetrieverQueryEngine): RAG query engine
    """
    create_empty_directory('results')
    context_all_train = []
    for idx, row in tqdm(data[['question', 'answer']].reset_index().iterrows()):
        question = row['question']
        answer = row['answer']  # answer is added to simplify output examination
        response = engine.query(question)
        try:
            response_1 = response.source_nodes[0].text
        except:
            response_1 = ''
        response_1 = re.sub('\s+', ' ', response_1)
        context_all_train.append([question,
                                  response_1,
                                  answer])
        # Print output every 50 questions
        if idx % 50 == 0:
            print(question)
            print(f'\n{response_1}')
            print(f'\nAnswer:\n{answer}')
    # Convert to DataFrame and save data
    context_all_train_df = pd.DataFrame(context_all_train, columns=['Question', 'Context_1', 'Answer'])
    context_all_train_df.to_csv('results/context_all_train.csv', index=False)
    context_all_train_df.to_pickle('results/context_all_train.pkl')


DOCS_PATH = 'data/rel18'
VECTOR_PATH = 'data/rag_vector/index'
SAMPLE_DOCS = False
RAG_INFERENCE = True

# Import embedding model from Hugging Face (full list: https://huggingface.co/spaces/mteb/leaderboard)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
Settings.chunk_size = 128
Settings.chunk_overlap = 20

# Read documents
if SAMPLE_DOCS:
    # Read sampled documents
    SAMPLED_DOCS_PATH = 'data/sampled_rel18'
    SAMPLE_FRAC = 0.5
    create_dir_with_sampled_docs(DOCS_PATH,
                                 SAMPLED_DOCS_PATH,
                                 SAMPLE_FRAC)
    documents = SimpleDirectoryReader(SAMPLED_DOCS_PATH).load_data()
else:
    # Read all documents from the directory
    documents = SimpleDirectoryReader(DOCS_PATH).load_data()

# Initialize client and save vectorized documents
db = chromadb.PersistentClient(path=VECTOR_PATH)
chroma_collection = db.get_or_create_collection("rel18")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents,
                                        storage_context=storage_context)

# Perform RAG inference on train data and save obtained text chunks for fine-tuning
if RAG_INFERENCE:
    # Set number of chunks to retrieve
    top_k = 1
    # Configure retriever
    retriever = VectorIndexRetriever(index=index,
                                     similarity_top_k=top_k)
    # Assemble query engine
    query_engine = RetrieverQueryEngine(retriever=retriever,
                                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])
    # Read train data
    train = pd.read_json('data/TeleQnA_training.txt').T
    # Get question ID column (a number of the question)
    train['Question_ID'] = train.index.str.split(' ').str[-1]
    # Remove [3GPP Release <number>] from question
    train = remove_release_number(train, 'question')
    # Get context for each question from train
    rag_inference_on_train(train, query_engine)
