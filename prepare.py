import os
from dotenv import load_dotenv
load_dotenv()


#Loading

from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="kiro-steering")
documents = reader.load_data()

print(f"Loaded {len(documents)} documents")

#Chunking

from llama_index.core.node_parser import SentenceSplitter

node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

nodes = node_parser.get_nodes_from_documents(
    documents = documents, show_progress=True
)

print(f"Created {len(nodes)} nodes")

#Embedding

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

from llama_index.embeddings.cohere import CohereEmbedding

# with input_typ='search_query'
embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_document",
)
texts = [node.get_text() for node in nodes]
embeddings = embed_model.get_text_embedding_batch(texts)

print(len(embeddings))
print(embeddings[:5])

# Indexing and Saving

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pc.Index("kiro")

pinecone_vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace="kiro-steering"
)

storage_context = StorageContext.from_defaults(
    vector_store=pinecone_vector_store
)

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)

print("Saved to Pinecone!")
print(pinecone_index.describe_index_stats())