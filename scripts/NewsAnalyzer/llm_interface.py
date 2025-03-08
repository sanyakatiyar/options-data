from openai import OpenAI
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

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
        
        Args:
            api_key: OpenRouter API key (defaults to env var if None)
            model: Model identifier to use on OpenRouter
            site_url: Site URL for OpenRouter statistics
            site_name: Site name for OpenRouter statistics
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
        
        print(f"Initialized LLM interface with model: {model}")
    
    def query(self, user_question: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Query the LLM with optional context from retrieved documents.
        
        Args:
            user_question: The user's question
            context: Optional list of documents to provide as context
            
        Returns:
            LLM response
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
            print("Sending request to OpenRouter API...")
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=self.model,
                messages=messages
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenRouter API: {str(e)}")
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

if __name__ == "__main__":
    # Example usage
    llm = LLMInterface()
    
    # Example with no context
    response = llm.query("explain futures and options difference simply")
    print("\nResponse without context:")
    print(response)
    
    # Example with mock context
    mock_context = [
        {
            "title": "Understanding Financial Derivatives",
            "source": "Financial Times",
            "published_at": "2025-02-25",
            "content_preview": "Futures contracts obligate the buyer to purchase an asset at a predetermined future date and price. Options contracts, on the other hand, give the buyer the right, but not the obligation, to buy or sell an asset at a specified price during a certain period of time.",
            "url": "https://example.com/article1"
        },
        {
            "title": "Markets Outlook: Derivatives Trading Volume Increases",
            "source": "Bloomberg",
            "published_at": "2025-02-26",
            "content_preview": "Options trading has seen a 25% increase year-over-year, while futures trading has remained relatively stable. Analysts attribute this to increased market volatility and investors seeking hedging strategies.",
            "url": "https://example.com/article2"
        }
    ]
    
    response_with_context = llm.query("explain futures and options difference simply", mock_context)
    print("\nResponse with context:")
    print(response_with_context)