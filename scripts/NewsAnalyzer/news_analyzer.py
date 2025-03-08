import os
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Any
import time

# Import components from other modules
from news_fetcher import fetch_news
from news_processor import process_news_articles
from news_embedder import create_pinecone_index, upload_news_to_pinecone
from news_retriever import NewsRetriever
from llm_interface import LLMInterface
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='News Analyzer with RAG System')
    
    # Main operation parameters
    parser.add_argument('--query', type=str, required=True,
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
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip fetching new articles and use existing database')
    parser.add_argument('--index-name', type=str, default='newsdata',
                       help='Name of the Pinecone index')
    parser.add_argument('--model', type=str, default='intfloat/e5-large-v2',
                       help='Embedding model name')
    parser.add_argument('--llm-model', type=str, default='deepseek/deepseek-chat:free',
                       help='LLM model to use via OpenRouter')
    
    return parser.parse_args()

def clear_pinecone_index(api_key: str, index_name: str):
    """Clear all vectors from a Pinecone index."""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Delete all vectors
    index.delete(delete_all=True)
    print(f"Cleared all vectors from Pinecone index: {index_name}")
    
    # Give Pinecone some time to process the deletion
    time.sleep(2)

def main():
    """
    Main function to run the news analyzer with RAG.
    """
    args = parse_arguments()
    
    # Get API keys
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Initialize retriever and LLM interface
    retriever = NewsRetriever(
        pinecone_api_key=PINECONE_API_KEY,
        index_name=args.index_name,
        model_name=args.model
    )
    
    try:
        llm = LLMInterface(
            api_key=OPENROUTER_API_KEY,
            model=args.llm_model
        )
    except Exception as e:
        print(f"Warning: Error initializing LLM interface: {str(e)}")
        print("You can still retrieve articles, but LLM responses may not work.")
        llm = None
    
    # Fetch and process news if not skipped
    if not args.skip_fetch:
        print(f"\n===== Fetching news for query: '{args.query}' =====")
        
        # Fetch news articles
        articles = fetch_news(
            query=args.query,
            sources=args.sources,
            from_date=args.from_date,
            to_date=args.to_date,
            language=args.language,
            category=args.category
        )
        
        if not articles:
            print("No articles found.")
            return
        
        # Process articles into chunks
        chunks = process_news_articles(articles)
        
        # Make sure index exists
        create_pinecone_index(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=args.index_name
        )
        
        # Clear existing vectors before adding new ones
        print(f"Clearing existing vectors from index: {args.index_name}")
        clear_pinecone_index(PINECONE_API_KEY, args.index_name)
        
        # Upload chunks to Pinecone
        upload_news_to_pinecone(
            document_chunks=chunks,
            pinecone_api_key=PINECONE_API_KEY,
            index_name=args.index_name,
            model_name=args.model
        )
    
    # Start interactive loop
    print("\n===== News Analysis System Ready =====")
    print("Type 'exit' to quit the program")
    
    while True:
        # Get user question
        user_question = input("\nEnter your question about the news: ")
        
        if user_question.lower() in ['exit', 'quit', 'q']:
            break
        
        # Retrieve relevant articles
        print(f"Searching for relevant articles...")
        search_results = retriever.search(user_question, top_k=10)
        
        # Display retrieved articles
        if search_results:
            print(f"\nFound {len(search_results)} relevant articles:")
            
            for i, result in enumerate(search_results):
                print(f"\n--- Article {i+1} (Relevance: {result['score']:.4f}) ---")
                print(f"Title: {result['title']}")
                print(f"Source: {result['source']}")
                print(f"Published: {result.get('published_at', 'Unknown date')}")
        else:
            print("No relevant articles found.")
            continue
        
        # Query LLM with retrieved context
        print("\nGenerating answer based on retrieved articles...")
        if llm is not None:
            llm_response = llm.query(user_question, search_results)
            
            # Display LLM response
            print("\n===== Answer =====")
            print(llm_response)
        else:
            print("\nLLM interface is not available. Here's a summary of the articles instead:")
            for i, result in enumerate(search_results[:3]):
                print(f"\n--- Summary of Article {i+1} ---")
                print(f"Title: {result['title']}")
                print(f"Source: {result['source']}")
                content = result.get('content_preview', result.get('text', ''))
                if content:
                    print(f"Content: {content[:300]}...")
            
            print("\nTo get LLM-generated answers, please check your OpenRouter API key and internet connection.")

if __name__ == "__main__":
    main()