from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


from utils import create_dir_with_sampled_docs

DOCS_PATH = 'data/rel18'
VECTOR_PATH = 'data/rag_vector/index'
SAMPLE_DOCS = False

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
