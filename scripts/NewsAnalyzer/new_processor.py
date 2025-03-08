from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_document_from_article(article: Dict[str, Any]) -> Document:
    """
    Convert a news article into a LangChain Document.
    
    Args:
        article: News article data from NewsAPI
        
    Returns:
        LangChain Document with content and metadata
    """
    # Extract the content from the article
    title = article.get('title', '')
    description = article.get('description', '')
    content = article.get('content', '')
    
    # Combine the text fields
    full_content = f"TITLE: {title}\n\nDESCRIPTION: {description}\n\nCONTENT: {content}"
    
    # Create metadata from article fields - ensure no None values
    metadata = {
        'source': article.get('source', {}).get('name', 'Unknown'),
        'author': article.get('author', 'Unknown') if article.get('author') is not None else "",
        'published_at': article.get('publishedAt', ''),
        'url': article.get('url', ''),
        'title': title if title else "No Title",
        'document_type': 'news_article'
    }
    
    return Document(page_content=full_content, metadata=metadata)

def process_news_articles(articles: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """
    Process news articles into document chunks.
    
    Args:
        articles: List of news articles from NewsAPI
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of document chunks with metadata
    """
    # Convert articles to LangChain Documents
    documents = [create_document_from_article(article) for article in articles]
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the documents into chunks
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    
    print(f"Created {len(chunks)} chunks from {len(articles)} news articles")
    return chunks

if __name__ == "__main__":
    # Example usage
    from news_fetcher import fetch_news
    
    # Fetch news articles
    articles = fetch_news(
        query="artificial intelligence",
        from_date="2025-02-20",
        to_date="2025-02-27"
    )
    
    # Process articles into chunks
    chunks = process_news_articles(articles)
    
    # Preview the first chunk
    if chunks:
        print("\nFirst chunk preview:")
        print(f"Content: {chunks[0].page_content[:150]}...")
        print(f"Metadata: {chunks[0].metadata}")