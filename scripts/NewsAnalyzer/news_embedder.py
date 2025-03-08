import os
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

class SentenceTransformerEmbeddings:
    """
    Custom embeddings class that uses sentence-transformers directly
    """
    def __init__(self, model_name="intfloat/e5-large-v2"):
        self.model = SentenceTransformer(model_name)
        print(f"Initialized embedding model: {model_name}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a query."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

def create_pinecone_index(
    pinecone_api_key: str,
    index_name: str = "news-index",
    dimension: int = 1024  # E5-Large embedding dimension
) -> None:
    """
    Create a Pinecone index for news article embeddings.
    
    Args:
        pinecone_api_key: Your Pinecone API key
        index_name: Name of the Pinecone index
        dimension: Dimension of the embeddings
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists, if not create it
    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        
        # Create the index
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Using existing Pinecone index: {index_name}")

def upload_news_to_pinecone(
    document_chunks: List[Document],
    pinecone_api_key: str,
    index_name: str = "newsdata",
    batch_size: int = 100,
    model_name: str = "intfloat/e5-large-v2"
) -> None:
    """
    Upload news document chunks to Pinecone.
    
    Args:
        document_chunks: List of document chunks to embed
        pinecone_api_key: Your Pinecone API key
        index_name: Name of the Pinecone index
        batch_size: Size of batches for processing
        model_name: Name of the embedding model to use
    """
    # Initialize embedding model
    embedding_model = SentenceTransformerEmbeddings(model_name=model_name)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    
    # Process documents in batches
    for i in range(0, len(document_chunks), batch_size):
        batch = document_chunks[i:i+batch_size]
        
        # Extract text and metadata
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [f"news_chunk_{i+j}" for j in range(len(batch))]
        
        # Get embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # Create records for Pinecone
        records = []
        for j, (id, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
            # Clean metadata to handle null values
            cleaned_metadata = {}
            for key, value in metadata.items():
                # Skip null values
                if value is None:
                    cleaned_metadata[key] = ""  # Replace None with empty string
                # Handle lists that might contain None
                elif isinstance(value, list):
                    cleaned_metadata[key] = [item if item is not None else "" for item in value]
                # Keep other valid values
                else:
                    cleaned_metadata[key] = value
            
            # Add the text content to metadata
            cleaned_metadata["text"] = texts[j][:1000]  # Store truncated text in metadata
            
            records.append({
                "id": id,
                "values": embedding,
                "metadata": cleaned_metadata
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=records)
        
        print(f"Uploaded batch {i//batch_size + 1}, total chunks so far: {min(i+batch_size, len(document_chunks))}")
    
    print(f"Successfully uploaded {len(document_chunks)} chunks to Pinecone")

if __name__ == "__main__":
    # Example usage
    from news_fetcher import fetch_news
    from news_processor import process_news_articles
    
    # Get Pinecone API key
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # 1. Fetch news articles
    articles = fetch_news(
        query="artificial intelligence",
        from_date="2025-02-20",
        to_date="2025-02-27"
    )
    
    # 2. Process articles into chunks
    chunks = process_news_articles(articles)
    
    # 3. Create Pinecone index (if not exists)
    create_pinecone_index(
        pinecone_api_key=PINECONE_API_KEY,
        index_name="newsdata"
    )
    
    # 4. Upload chunks to Pinecone
    upload_news_to_pinecone(
        document_chunks=chunks,
        pinecone_api_key=PINECONE_API_KEY,
        index_name="newsdata"
    )