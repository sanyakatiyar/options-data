import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class NewsRetriever:
    """Class for retrieving news from Pinecone vector database"""
    
    def __init__(
        self, 
        pinecone_api_key: str = None,
        index_name: str = "news-index",
        model_name: str = "intfloat/e5-large-v2"
    ):
        """
        Initialize the news retriever.
        
        Args:
            pinecone_api_key: Your Pinecone API key (defaults to env var if None)
            index_name: Name of the Pinecone index
            model_name: Name of the embedding model to use
        """
        # Get API key from env var if not provided
        if pinecone_api_key is None:
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.index_name = index_name
        
        print(f"Initialized NewsRetriever with index: {index_name}")
    
    def search(self, query: str, top_k: int = 5, filter: Dict = None) -> List[Dict[str, Any]]:
        """
        Search for news articles matching the query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            filter: Pinecone filter to apply
            
        Returns:
            List of matching articles with metadata and relevance score
        """
        # Create query embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "score": match.score,
                "title": match.metadata.get("title", "Unknown Title"),
                "source": match.metadata.get("source", "Unknown Source"),
                "published_at": match.metadata.get("published_at", "Unknown Date"),
                "url": match.metadata.get("url", ""),
                "content_preview": match.metadata.get("text", "")[:200] + "...",
                "metadata": match.metadata
            })
        
        return formatted_results
    
    def search_by_date_range(
        self, 
        query: str, 
        from_date: str, 
        to_date: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles within a specific date range.
        
        Args:
            query: Query string
            from_date: Start date in ISO format (YYYY-MM-DD)
            to_date: End date in ISO format (YYYY-MM-DD)
            top_k: Number of results to return
            
        Returns:
            List of matching articles with metadata and relevance score
        """
        # Create filter for date range
        # Note: This assumes published_at is stored in the metadata in this format
        filter = {
            "published_at": {"$gte": from_date, "$lte": to_date}
        }
        
        return self.search(query, top_k, filter)
    
    def search_by_source(
        self, 
        query: str, 
        sources: List[str], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles from specific sources.
        
        Args:
            query: Query string
            sources: List of source names to filter by
            top_k: Number of results to return
            
        Returns:
            List of matching articles with metadata and relevance score
        """
        # Create filter for sources
        filter = {
            "source": {"$in": sources}
        }
        
        return self.search(query, top_k, filter)

if __name__ == "__main__":
    # Example usage
    
    # Initialize the retriever
    retriever = NewsRetriever()
    
    # Basic search
    print("\n--- Basic Search ---")
    results = retriever.search("Latest developments in artificial intelligence")
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
        print(f"Title: {result['title']}")
        print(f"Source: {result['source']}")
        print(f"Published: {result['published_at']}")
        print(f"Preview: {result['content_preview']}")
    
    # Search by date range
    print("\n--- Search by Date Range ---")
    date_results = retriever.search_by_date_range(
        "cryptocurrency regulations", 
        from_date="2025-02-20", 
        to_date="2025-02-27"
    )
    
    # Print date-filtered results
    for i, result in enumerate(date_results):
        print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
        print(f"Title: {result['title']}")
        print(f"Source: {result['source']}")
        print(f"Published: {result['published_at']}")