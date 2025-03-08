from newsapi import NewsApiClient
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

def fetch_news(
    query: str = None,
    sources: str = None,
    domains: str = None,
    from_date: str = None,
    to_date: str = None,
    language: str = 'en',
    sort_by: str = 'publishedAt',
    category: str = None
) -> List[Dict[str, Any]]:
    """
    Fetch news articles from NewsAPI.
    
    Args:
        query: Keywords or phrases to search for
        sources: Comma-separated string of news sources or blogs
        domains: Comma-separated string of domains
        from_date: A date in ISO 8601 format (e.g. "2023-12-25")
        to_date: A date in ISO 8601 format (e.g. "2023-12-31")
        language: The 2-letter ISO-639-1 code of the language (default: 'en')
        sort_by: The order to sort articles ("relevancy", "popularity", "publishedAt")
        category: Category for top headlines (business, entertainment, health, etc.)
        
    Returns:
        List of news articles with their metadata
    """
    # Initialize NewsAPI client
    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        raise ValueError("NEWSAPI_KEY environment variable not set")
    
    newsapi = NewsApiClient(api_key=api_key)
    
    # Set default dates if not provided
    if from_date is None:
        # Default to 7 days ago
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch articles
    if category:
        # Use top headlines for category-based queries
        response = newsapi.get_top_headlines(
            q=query,
            sources=sources,
            category=category,
            language=language,
            page_size=100  # Fetch more articles at once
        )
        articles = response.get('articles', [])
    else:
        # Use everything endpoint for more flexible queries
        response = newsapi.get_everything(
            q=query,
            sources=sources,
            domains=domains,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by=sort_by,
            page_size=100  # Fetch more articles at once
        )
        articles = response.get('articles', [])
    
    # Add unique IDs and improve metadata
    for i, article in enumerate(articles):
        article['id'] = f"news_{i}"
        # Convert publishedAt to proper datetime if needed
        if 'publishedAt' in article:
            try:
                pub_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                article['published_date'] = pub_date.strftime("%Y-%m-%d")
            except:
                article['published_date'] = article['publishedAt']
    
    print(f"Fetched {len(articles)} news articles")
    return articles

if __name__ == "__main__":
    # Example usage
    articles = fetch_news(
        query="artificial intelligence",
        sources="techcrunch,wired,the-verge",
        from_date="2025-02-20",
        to_date="2025-02-27",
        language="en",
        sort_by="publishedAt"
    )
    
    # Print a summary of fetched articles
    for i, article in enumerate(articles[:5]):  # Just show first 5 for brevity
        print(f"\nArticle {i+1}:")
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']['name']}")
        print(f"Published: {article['publishedAt']}")
        print(f"URL: {article['url']}")