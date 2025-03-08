import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import components from other modules
from news_fetcher import fetch_news
from news_processor import process_news_articles
from news_embedder import create_pinecone_index, upload_news_to_pinecone
from news_retriever import NewsRetriever

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='News Analyzer System')
    
    # Main operation mode
    parser.add_argument('--mode', type=str, choices=['fetch', 'query', 'full'], default='full',
                       help='Operation mode: fetch (fetch and embed only), query (search only), or full (fetch, embed, and query)')
    
    # Fetch parameters
    parser.add_argument('--query', type=str, default='None',
                       help='Search query for news articles')
    parser.add_argument('--sources', type=str, default=None,
                       help='Comma-separated list of news sources')
    parser.add_argument('--from-date', type=str, default=None,
                       help='Start date for article search (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=str, default=None,
                       help='End date for article search (YYYY-MM-DD)')
    parser.add_argument('--language', type=str, default='en',
                       help='Language of news articles')
    parser.add_argument('--category', type=str, default=None,
                       help='News category (for top headlines)')
    
    # Vector DB parameters
    parser.add_argument('--index-name', type=str, default='newsdata',
                       help='Name of the Pinecone index')
    parser.add_argument('--model', type=str, default='intfloat/e5-large-v2',
                       help='Embedding model name')
    
    # Query parameters
    parser.add_argument('--search-query', type=str, default=None,
                       help='Query for searching the database (if different from fetch query)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results to return')
    
    return parser.parse_args()

def main():
    """
    Main function to run the news analyzer pipeline.
    """
    args = parse_arguments()
    
    # Get Pinecone API key
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Set default dates if not provided
    if args.from_date is None:
        args.from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    if args.to_date is None:
        args.to_date = datetime.now().strftime('%Y-%m-%d')
    
    # If mode is 'fetch' or 'full', fetch and embed news
    if args.mode in ['fetch', 'full']:
        print(f"\n===== Fetching news for query: '{args.query}' =====")
        print(f"Date range: {args.from_date} to {args.to_date}")
        if args.sources:
            print(f"Sources: {args.sources}")
        
        # 1. Fetch news articles
        articles = fetch_news(
            query=args.query,
            sources=args.sources,
            from_date=args.from_date,
            to_date=args.to_date,
            language=args.language,
            category=args.category
        )
        
        if not articles:
            print("No articles found. Exiting.")
            return
        
        # 2. Process articles into chunks
        chunks = process_news_articles(articles)
        
        # 3. Create Pinecone index (if not exists)
        create_pinecone_index(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=args.index_name,
            dimension=1024  # E5-Large dimension
        )
        
        # 4. Upload chunks to Pinecone
        upload_news_to_pinecone(
            document_chunks=chunks,
            pinecone_api_key=PINECONE_API_KEY,
            index_name=args.index_name,
            model_name=args.model
        )
    
    # If mode is 'query' or 'full', search the database
    if args.mode in ['query', 'full']:
        # Use search_query if provided, otherwise use the fetch query
        search_query = args.search_query if args.search_query else args.query
        
        print(f"\n===== Searching for: '{search_query}' =====")
        
        # Initialize the retriever
        retriever = NewsRetriever(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=args.index_name,
            model_name=args.model
        )
        
        # Perform the search
        results = retriever.search(search_query, args.top_k)
        
        # Display results
        if results:
            print(f"\nFound {len(results)} relevant articles:")
            
            for i, result in enumerate(results):
                print(f"\n--- Article {i+1} (Relevance: {result['score']:.4f}) ---")
                print(f"Title: {result['title']}")
                print(f"Source: {result['source']}")
                print(f"Published: {result['published_at']}")
                print(f"URL: {result['url']}")
                print(f"Preview: {result['content_preview']}")
        else:
            print("No relevant articles found.")

if __name__ == "__main__":
    main()