import streamlit as st
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import math
from typing import List, Dict, Any
from newsapi import NewsApiClient
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Set environment variables to fix PyTorch compatibility issues
os.environ["STREAMLIT_WATCH_MODULE_PATH"] = "false"
import torch  # This must be imported after setting the environment variable

# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="News Analyzer with AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'fetched_articles' not in st.session_state:
    st.session_state['fetched_articles'] = []
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 1
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to change page
def change_page(direction):
    if direction == "next":
        st.session_state['current_page'] += 1
    else:
        st.session_state['current_page'] -= 1

# App title and description
st.title("📰 News Analyzer with AI")
st.markdown("""
This app fetches news articles, analyzes them with AI, and answers your questions.
Use the form below to fetch news on a specific topic, then ask questions to analyze the content.
""")

# Check for required API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY environment variable not set. Please set it in a .env file.")
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY environment variable not set. Please set it in a .env file.")
if not NEWSAPI_KEY:
    st.error("NEWSAPI_KEY environment variable not set. Please set it in a .env file.")

#############################################
# News Fetcher Component
#############################################

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
    
    return articles

#############################################
# News Processor Component
#############################################

def create_document_from_article(article: Dict[str, Any]) -> Document:
    """
    Convert a news article into a LangChain Document.
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
    
    return chunks

#############################################
# Embedding and Vector DB Component
#############################################

class SentenceTransformerEmbeddings:
    """
    Custom embeddings class that uses sentence-transformers directly
    """
    def __init__(self, model_name="intfloat/e5-large-v2"):
        self.model = SentenceTransformer(model_name)
        
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
    index_name: str = "newsdata",
    dimension: int = 1024  # E5-Large embedding dimension
) -> None:
    """
    Create a Pinecone index for news article embeddings.
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists, if not create it
    if index_name not in pc.list_indexes().names():
        # Create the index
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        return f"Created Pinecone index: {index_name}"
    else:
        return f"Using existing Pinecone index: {index_name}"

def clear_pinecone_index(api_key: str, index_name: str):
    """Clear all vectors from a Pinecone index."""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Delete all vectors
    index.delete(delete_all=True)
    
    # Give Pinecone some time to process the deletion
    time.sleep(2)
    return f"Cleared all vectors from Pinecone index: {index_name}"

def upload_news_to_pinecone(
    document_chunks: List[Document],
    pinecone_api_key: str,
    index_name: str = "newsdata",
    batch_size: int = 100,
    model_name: str = "intfloat/e5-large-v2"
) -> str:
    """
    Upload news document chunks to Pinecone.
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
    
    return f"Successfully uploaded {len(document_chunks)} chunks to Pinecone"

#############################################
# News Retriever Component
#############################################

class NewsRetriever:
    """Class for retrieving news from Pinecone vector database"""
    
    def __init__(
        self, 
        pinecone_api_key: str = None,
        index_name: str = "newsdata",
        model_name: str = "intfloat/e5-large-v2"
    ):
        """
        Initialize the news retriever.
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
    
    def search(self, query: str, top_k: int = 10, filter: Dict = None) -> List[Dict[str, Any]]:
        """
        Search for news articles matching the query.
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
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata
            })
        
        return formatted_results

#############################################
# LLM Interface Component
#############################################

class LLMInterface:
    """Class for interacting with LLMs through OpenRouter"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek/deepseek-chat:free",
        site_url: str = "http://localhost",
        site_name: str = "NewsAnalyzer"
    ):
        """
        Initialize the LLM interface.
        """
        # Get API key from env var if not provided
        if api_key is None:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        self.model = model
        self.extra_headers = {
            "HTTP-Referer": site_url,
            "X-Title": site_name
        }
    
    def query(self, user_question: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Query the LLM with optional context from retrieved documents.
        """
        # Prepare messages
        messages = []
        
        # Add context if provided
        if context and len(context) > 0:
            # Create system message with context
            context_text = self._format_context(context)
            system_message = f"""You are a news analysis assistant that answers questions based on the latest news articles.
            
Use the following news articles as context for answering the user's question:

{context_text}

Answer the user's question based on the information in these articles. If the information needed is not in the articles, say so and provide a general response based on your knowledge. Always cite the source of information when possible."""
            
            messages.append({"role": "system", "content": system_message})
        else:
            # No context, just use a simple system message
            messages.append({
                "role": "system", 
                "content": "You are a news analysis assistant that answers questions based on the latest information."
            })
        
        # Add user question
        messages.append({"role": "user", "content": user_question})
        
        try:
            # Call the API
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=self.model,
                messages=messages
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error when trying to generate a response: {str(e)}\n\nPlease check your OpenRouter API key and internet connection."
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a string for context."""
        context_parts = []
        
        for i, doc in enumerate(documents):
            source = doc.get("source", "Unknown")
            title = doc.get("title", "No Title")
            content = doc.get("content_preview", doc.get("text", ""))
            url = doc.get("url", "")
            date = doc.get("published_at", "")
            
            context_parts.append(f"ARTICLE {i+1}:")
            context_parts.append(f"Title: {title}")
            context_parts.append(f"Source: {source}")
            if date:
                context_parts.append(f"Published: {date}")
            context_parts.append(f"Content: {content}")
            if url:
                context_parts.append(f"URL: {url}")
            context_parts.append("")  # Empty line between articles
        
        return "\n".join(context_parts)

#############################################
# Initialize Retriever and LLM with caching
#############################################

@st.cache_resource
def get_retriever(index_name, model_name):
    try:
        retriever = NewsRetriever(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=index_name,
            model_name=model_name
        )
        return retriever
    except Exception as e:
        st.error(f"Error initializing retriever: {str(e)}")
        return None

@st.cache_resource
def get_llm(model_name):
    try:
        llm = LLMInterface(
            api_key=OPENROUTER_API_KEY,
            model=model_name
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM interface: {str(e)}")
        return None

#############################################
# Sidebar Configuration
#############################################

st.sidebar.header("Configuration")

# Pinecone settings
index_name = st.sidebar.text_input("Pinecone Index Name", "newsdata")
embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["intfloat/e5-large-v2", "intfloat/e5-base-v2", "BAAI/bge-large-en-v1.5", "BAAI/bge-base-v1.5"],
    index=0
)

# LLM settings
llm_model = st.sidebar.selectbox(
    "LLM Model",
    ["deepseek/deepseek-chat:free", "google/gemini-pro", "anthropic/claude-3-sonnet", "meta-llama/llama-3-8b-instruct"],
    index=0
)

# Initialize services based on configuration
retriever = get_retriever(index_name, embedding_model)
llm = get_llm(llm_model)

#############################################
# Main Application
#############################################

# Set up the tabs
tab1, tab2 = st.tabs(["Fetch News", "Ask Questions"])

# Tab 1: Fetch News
with tab1:
    st.header("Fetch News Articles")
    
    # Form for fetching news
    with st.form(key="news_fetch_form"):
        # News query parameters
        query = st.text_input("Search Query (required)", "artificial intelligence")
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "Category (for top headlines)",
                [None, "business", "entertainment", "general", "health", "science", "sports", "technology"],
                index=0
            )
            language = st.selectbox("Language", ["en", "es", "fr", "de", "it"], index=0)
            
        with col2:
            from_date = st.date_input("From Date", datetime.now() - timedelta(days=7))
            to_date = st.date_input("To Date", datetime.now())
            
        sources = st.text_input("News Sources (comma-separated)", "")
        
        # Submit button
        fetch_submit = st.form_submit_button("Fetch News")
    
    # Process form submission (outside the form)
    if fetch_submit:
        if not query:
            st.error("Search query is required.")
        else:
            # Fetch news articles
            with st.spinner(f"Fetching news for query: '{query}'..."):
                try:
                    articles = fetch_news(
                        query=query,
                        sources=sources if sources else None,
                        from_date=from_date.strftime('%Y-%m-%d'),
                        to_date=to_date.strftime('%Y-%m-%d'),
                        language=language,
                        category=category
                    )
                    
                    # Store in session state
                    st.session_state['fetched_articles'] = articles
                    st.session_state['current_page'] = 1
                    
                    if not articles:
                        st.warning("No articles found.")
                    else:
                        st.success(f"Fetched {len(articles)} news articles.")
                except Exception as e:
                    st.error(f"Error fetching news: {str(e)}")
    
    # Display fetched articles (always show if available)
    if st.session_state['fetched_articles']:
        st.subheader("Fetched News Articles")
        
        # Pagination
        articles = st.session_state['fetched_articles']
        total_articles = len(articles)
        articles_per_page = 10
        total_pages = max(1, math.ceil(total_articles / articles_per_page))
        
        # Navigation buttons
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            prev_disabled = st.session_state['current_page'] <= 1
            if st.button("← Previous", disabled=prev_disabled, key="prev_btn", on_click=change_page, args=("prev",)):
                pass
        
        with col2:
            st.markdown(f"<div style='text-align: center'>Page {st.session_state['current_page']} of {total_pages}</div>", 
                      unsafe_allow_html=True)
        
        with col3:
            next_disabled = st.session_state['current_page'] >= total_pages
            if st.button("Next →", disabled=next_disabled, key="next_btn", on_click=change_page, args=("next",)):
                pass
        
        # Current page content
        current_page = st.session_state['current_page']
        start_idx = (current_page - 1) * articles_per_page
        end_idx = min(start_idx + articles_per_page, total_articles)
        
        # Display articles
        for i in range(start_idx, end_idx):
            try:
                article = articles[i]
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Image
                    if article.get('urlToImage'):
                        try:
                            st.image(article['urlToImage'], use_container_width=True)
                        except:
                            st.image("https://via.placeholder.com/150x100?text=Image+Error", use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/150x100?text=No+Image", use_container_width=True)
                
                with col2:
                    # Article details
                    st.markdown(f"### {article.get('title', 'No Title')}")
                    
                    source_name = "Unknown Source"
                    if 'source' in article and isinstance(article['source'], dict):
                        source_name = article['source'].get('name', 'Unknown Source')
                    
                    published = article.get('publishedAt', 'Unknown Date')
                    st.markdown(f"**Source:** {source_name} | **Published:** {published}")
                    st.markdown(article.get('description', 'No description available'))
                    
                    if article.get('url'):
                        st.markdown(f"[Read more]({article['url']})")
                
                st.divider()
            except Exception as e:
                st.error(f"Error displaying article {i}: {str(e)}")
        
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_articles} articles")
        
        # Vector DB upload section
        st.subheader("Store In Vector Database")
        st.write("Process the fetched articles and store them in the vector database for querying.")
        
        if st.button("Process and Store in Vector Database"):
            try:
                # Process articles
                with st.spinner("Processing articles..."):
                    chunks = process_news_articles(articles)
                st.success(f"Created {len(chunks)} text chunks from the articles.")
                
                # Set up Vector DB
                with st.spinner("Setting up vector database..."):
                    message = create_pinecone_index(
                        pinecone_api_key=PINECONE_API_KEY,
                        index_name=index_name
                    )
                    st.info(message)
                
                # Clear existing data
                with st.spinner("Clearing previous data from vector database..."):
                    message = clear_pinecone_index(PINECONE_API_KEY, index_name)
                    st.info(message)
                
                # Upload to Vector DB
                with st.spinner("Uploading to vector database..."):
                    message = upload_news_to_pinecone(
                        document_chunks=chunks,
                        pinecone_api_key=PINECONE_API_KEY,
                        index_name=index_name,
                        model_name=embedding_model
                    )
                    st.success(message)
                
                st.success("✅ News articles have been processed and stored successfully!")
                st.info("🔍 Now you can go to the 'Ask Questions' tab to analyze these articles!")
            
            except Exception as e:
                st.error(f"Error processing and storing articles: {str(e)}")

# Tab 2: Ask Questions
with tab2:
    st.header("Ask Questions About the News")
    
    # Display chat history
    for i, (question, answer, sources) in enumerate(st.session_state['chat_history']):
        st.chat_message("user").write(question)
        
        with st.chat_message("assistant"):
            st.write(answer)
            
            # Display sources if available
            if sources:
                with st.expander("View Sources", expanded=False):
                    for j, src in enumerate(sources):
                        st.markdown(f"**Source {j+1}:** {src['title']} ({src['source']})")
                        st.markdown(f"Relevance: {src['score']:.4f}")
                        if 'url' in src and src['url']:
                            st.markdown(f"[Read more]({src['url']})")
                        st.divider()
    
    # Handle new questions
    user_question = st.chat_input("Ask a question about the news")
    
    if user_question:
        # Show user question
        st.chat_message("user").write(user_question)
        
        # Process and show answer
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            answer_placeholder.info("Searching for relevant articles...")
            
            if retriever is None:
                answer_placeholder.error("Retriever is not available. Please check your Pinecone API key and connection.")
                sources = []
                answer = "I couldn't access the news database. Please make sure you've fetched news articles in the 'Fetch News' tab and check your API keys."
            else:
                try:
                    # Search for relevant articles
                    search_results = retriever.search(user_question, top_k=10)
                    
                    if not search_results:
                        answer_placeholder.warning("No relevant articles found.")
                        sources = []
                        answer = "I couldn't find any relevant information in the news articles. Try asking a different question or fetch more news articles related to your topic."
                    else:
                        sources = search_results
                        
                        if llm is None:
                            answer_placeholder.warning("LLM is not available. Showing article summaries instead.")
                            summaries = []
                            for i, result in enumerate(search_results[:3]):
                                title = result.get('title', 'Untitled')
                                source = result.get('source', 'Unknown')
                                content = result.get('content_preview', result.get('text', ''))
                                summaries.append(f"**Article {i+1}:** {title} from {source}\n\n{content[:200]}...")
                                
                            answer = "Here are the most relevant articles I found:\n\n" + "\n\n".join(summaries)
                        else:
                            answer_placeholder.info("Generating answer based on retrieved articles...")
                            answer = llm.query(user_question, search_results)
                        
                        # Show sources
                        with st.expander("View Sources", expanded=False):
                            for i, result in enumerate(search_results):
                                col1, col2 = st.columns([1, 4])
                                
                                with col1:
                                    # Score indicator
                                    score = result['score']
                                    st.progress(score)
                                    st.caption(f"Relevance: {score:.4f}")
                                
                                with col2:
                                    st.markdown(f"**{result['title']}**")
                                    st.caption(f"Source: {result['source']}")
                                    if 'url' in result and result['url']:
                                        st.markdown(f"[Read original article]({result['url']})")
                                
                                st.markdown(result.get('content_preview', result.get('text', ''))[:300] + "...")
                                st.divider()
                                
                except Exception as e:
                    answer_placeholder.error(f"An error occurred: {str(e)}")
                    sources = []
                    answer = f"Sorry, I encountered an error while trying to answer your question: {str(e)}"
            
            # Display the answer
            answer_placeholder.write(answer)
        
        # Add to chat history
        st.session_state['chat_history'].append((user_question, answer, sources))

# Footer
st.sidebar.divider()
st.sidebar.caption("News Analyzer with RAG and LLM")
st.sidebar.caption("Data fetched from NewsAPI")