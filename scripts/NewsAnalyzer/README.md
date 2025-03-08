# News Analyzer with RAG and LLM

A powerful application that combines news retrieval, embedding-based search, and AI-powered analysis to help you explore and understand news content. This application fetches news articles from multiple sources, processes them using advanced natural language processing techniques, and allows you to ask questions about the content using a large language model.

![News Analyzer Screenshot](https://via.placeholder.com/800x400?text=News+Analyzer+Dashboard)

## Features

- **Real-time News Retrieval**: Fetch the latest news from multiple sources using NewsAPI
- **Powerful Search**: Use semantic search to find relevant articles based on meaning, not just keywords
- **AI-Powered Analysis**: Ask questions about the news and get intelligent responses
- **Visual Content Display**: View articles with images in a clean, paginated interface
- **Vector Database Storage**: Store processed news articles for efficient retrieval
- **Multiple Language Support**: Search for news in different languages
- **Category Filtering**: Focus on specific news categories like business, technology, sports, etc.
- **Interactive Chat Interface**: Have a conversation about the news with AI assistance

## System Architecture

The application consists of several key components:

1. **News Fetcher**: Retrieves articles from NewsAPI
2. **News Processor**: Converts articles into document chunks suitable for embedding
3. **Embedding Engine**: Creates vector representations of text using state-of-the-art models
4. **Vector Database**: Stores embeddings in Pinecone for efficient similarity search
5. **News Retriever**: Finds relevant articles based on semantic similarity
6. **LLM Interface**: Connects to OpenRouter/OpenAI for AI-powered analysis
7. **Web Interface**: Streamlit-based UI for interacting with the system

## Installation

### Prerequisites

- Python 3.8+
- Pinecone account (for vector database)
- NewsAPI key
- OpenRouter or OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news-analyzer.git
   cd news-analyzer
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   NEWSAPI_KEY=your_newsapi_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## Usage

### Web Interface (Streamlit)

The easiest way to use the application is through the Streamlit web interface:

```bash
streamlit run app.py
```

This will open a browser window with the application interface, where you can:

1. Enter a search query to find news articles
2. Select news categories, sources, and date ranges
3. View articles with images and descriptions
4. Process and store articles in the vector database
5. Ask questions about the news content
6. See relevant source articles for each answer

### Command Line Interface

You can also use the command-line interface for automated news analysis:

```bash
# Fetch, process, and store news articles
python news_analyzer.py --query "artificial intelligence" --category "technology" --from-date "2025-02-20" --to-date "2025-02-27"

# Use existing database without fetching new articles
python news_analyzer.py --query "artificial intelligence" --skip-fetch
```

After running this command, you'll enter an interactive mode where you can ask questions about the news.

## Configuration Options

### Web Interface Settings

The Streamlit app provides several configuration options in the sidebar:

- **Pinecone Index Name**: Name of the vector database index (default: "newsdata")
- **Embedding Model**: Model used for creating vector embeddings
  - Options include E5-Large, E5-Base, BGE-Large, BGE-Base
- **LLM Model**: AI model used for answering questions
  - Options include various models available through OpenRouter

### Command Line Arguments

The `news_analyzer.py` script accepts the following arguments:

- `--query`: Search query for news articles (required)
- `--sources`: Comma-separated list of news sources
- `--from-date`: Start date for article search (YYYY-MM-DD)
- `--to-date`: End date for article search (YYYY-MM-DD)
- `--language`: Language code (default: "en")
- `--category`: News category (business, entertainment, health, etc.)
- `--skip-fetch`: Skip fetching new articles and use existing database
- `--index-name`: Name of the Pinecone index (default: "newsdata")
- `--model`: Embedding model name (default: "intfloat/e5-large-v2")
- `--llm-model`: LLM model via OpenRouter (default: "deepseek/deepseek-chat:free")

## File Structure

- `app.py`: Streamlit web application
- `news_analyzer.py`: Command-line interface
- `news_fetcher.py`: Functions for retrieving news from NewsAPI
- `news_processor.py`: Functions for processing news articles
- `news_embedder.py`: Functions for creating embeddings and storing in Pinecone
- `news_retriever.py`: Class for retrieving news from vector database
- `llm_interface.py`: Class for interacting with LLMs via OpenRouter
- `main.py`: Original entry point with simplified functionality

## Vector Database Setup

The application uses Pinecone for storing and retrieving vector embeddings. The default index name is "newsdata" with a dimension of 1024 (for E5-Large embeddings). The system will automatically:

1. Check if the index exists
2. Create it if necessary
3. Clear previous vectors before adding new ones (to avoid mixing news from different queries)
4. Upload embeddings in batches to avoid memory issues

## LLM Integration

The application connects to OpenRouter to access various large language models. When you ask a question:

1. The system retrieves the 10 most relevant articles
2. It formats these articles as context for the LLM
3. The LLM generates a response based on the articles and your question
4. Sources are provided so you can verify the information

## Requirements

- streamlit==1.32.0
- python-dotenv==1.0.0
- newsapi-python==0.2.7
- langchain==0.1.12
- sentence-transformers==2.5.0
- pinecone-client==3.1.0
- openai==1.22.0
- torch==2.2.0
- numpy==1.26.4

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your API keys are correctly set in the `.env` file
2. **No Articles Found**: Try broadening your search query or date range
3. **PyTorch Compatibility Issues**: The app handles PyTorch compatibility with Streamlit automatically
4. **Vector Database Connection**: Check your Pinecone API key and network connection
5. **LLM Response Errors**: Verify your OpenRouter API key and selected model

## Extending the Application

You can extend this application in several ways:

1. **Add more news sources**: Integrate with additional news APIs
2. **Implement custom embeddings**: Use different embedding models
3. **Add visualization features**: Create graphs of news trends
4. **Enable sentiment analysis**: Analyze the sentiment of news articles
5. **Implement user accounts**: Save searches and conversations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NewsAPI](https://newsapi.org/) for providing access to news articles
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Pinecone](https://www.pinecone.io/) for vector database services
- [OpenRouter](https://openrouter.ai/) for LLM API access
- [Streamlit](https://streamlit.io/) for the web interface framework
- [LangChain](https://python.langchain.com/) for document processing utilities