# NeoStats - Smart AI Chatbot with RAG

A powerful Retrieval-Augmented Generation (RAG) enabled chatbot built with Streamlit. Ask questions about your PDF documents with intelligent web search fallback and multi-LLM support.

## 🌟 Features

- **PDF Document Processing**: Upload and process PDF files for intelligent Q&A
- **RAG Pipeline**: Advanced retrieval-augmented generation with semantic search
- **Intelligent Fallback**:
  - PDF RAG → Web Search → LLM-only → Local Extraction
- **Multi-Provider LLM Support**:
  - Primary: xAI Grok API
  - Fallback: Groq
  - Compatible with OpenAI-format APIs
- **Web Search Integration**: DuckDuckGo for real-time information retrieval
- **Response Modes**: Toggle between concise and detailed responses
- **Session State Management**: Persistent chat history and document context
- **Vector Store Caching**: FAISS-based efficient similarity search

## 🛠️ Tech Stack

| Component      | Technology                                    |
| -------------- | --------------------------------------------- |
| UI Framework   | **Streamlit** (1.28.0+)                       |
| Vector DB      | **FAISS** (CPU)                               |
| Embeddings     | **SentenceTransformers** (all-MiniLM-L6-v2)   |
| RAG Framework  | **LangChain** (0.1.0+)                        |
| LLM Providers  | **xAI Grok**, **Groq**, **OpenAI-compatible** |
| PDF Processing | **PyPDF**                                     |
| Web Search     | **DuckDuckGo Search**                         |
| Language       | **Python 3.8+**                               |

## 📋 Prerequisites

- Python 3.8 or higher
- API Key for at least one LLM provider:
  - **xAI Grok** (recommended) - [Get key](https://console.x.ai)
  - **Groq** - [Get key](https://console.groq.com)
  - **OpenAI-compatible API**

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd NeoStats
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# LLM Configuration
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=grok-beta
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=3

# Web Search
WEB_SEARCH_RESULTS=3
```

## 🎯 Usage

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Using the Chat Interface

1. **Upload PDF** (Optional):
   - Use the sidebar file uploader to add PDF documents
   - Documents are processed and indexed automatically

2. **Ask Questions**:
   - Type your query in the chat input
   - Choose response mode (Concise/Detailed)
   - Get intelligent responses with context from documents or web

3. **View Sources**:
   - See the right sidebar for relevant document chunks
   - Track which sources were used for answers

## 📁 Project Structure

```
NeoStats/
├── app.py                      # Main Streamlit application
├── config/
│   └── config.py              # Environment & API configuration
├── models/
│   ├── llm.py                 # LLM integration & response generation
│   └── embeddings.py          # SentenceTransformers wrapper
├── utils/
│   ├── rag.py                 # RAG pipeline (PDF→Vector store)
│   ├── web_search.py          # DuckDuckGo integration
│   └── helpers.py             # Utilities & helpers
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## 🔄 RAG Pipeline

The system implements a sophisticated fallback strategy:

```
User Query
    ├─→ [1] PDF RAG Search
    │   └─→ Found relevant context? → Answer with citations
    │
    ├─→ [2] Web Search (if no PDF context)
    │   └─→ Found results? → Answer with web sources
    │
    ├─→ [3] LLM-only (if no context available)
    │   └─→ Generate answer from training
    │
    └─→ [4] Local PDF Extraction (fallback)
        └─→ Extract text directly from uploaded PDFs
```

## 🔧 Configuration

### LLM Models

**Recommended for xAI Grok:**

```env
OPENAI_API_KEY=your_grok_api_key
LLM_MODEL=grok-beta
# API base URL will be set to xAI endpoint
```

**For Groq (Fallback):**

```env
OPENAI_API_KEY=your_groq_api_key
LLM_MODEL=mixtral-8x7b-32768
```

### RAG Parameters

| Parameter         | Default          | Purpose                     |
| ----------------- | ---------------- | --------------------------- |
| `CHUNK_SIZE`      | 500              | Tokens per document chunk   |
| `CHUNK_OVERLAP`   | 50               | Overlap between chunks      |
| `RETRIEVAL_K`     | 3                | Top-K documents to retrieve |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence embedding model    |

### Streamlit Config

Edit `.streamlit/config.toml` for UI customization:

- Theme settings
- Layout configuration
- Caching options

## 📊 Features Deep-Dive

### Context Relevance Calculation

The system scores retrieved documents based on:

- Semantic similarity to the query
- Chunk quality and completeness
- Fallback strategy effectiveness

### Session Management

- **Chat History**: Persists across page reloads
- **Vector Store**: Cached embeddings for fast retrieval
- **UI State**: Maintains sidebar and response settings

### Error Handling

- Graceful API fallback when LLM unavailable
- Local PDF extraction when embedding fails
- Web search as safety net

## 🔐 Security & Best Practices

- ✅ API keys stored in `.env` (never in code)
- ✅ Input sanitization for user queries
- ✅ FAISS runs locally (no external vector DB exposure)
- ✅ PDF processing isolated and error-safe
- ✅ Rate limiting for web search

## 📈 Performance Tips

1. **Optimize Chunk Size**: Larger chunks (up to 1000 tokens) for better context, smaller for precision
2. **Adjust Retrieval-K**: Higher K (5-10) for broader search, lower (1-3) for focused answers
3. **Use Production Embeddings**: all-MiniLM-L6-v2 balances speed and quality
4. **Cache Management**: Vector store caches between sessions automatically

## 🐛 Troubleshooting

### API Key Issues

```
Error: "Could not authenticate with API"
→ Check .env file and API key validity
```

### PDF Processing Fails

```
Error: "Failed to process PDF"
→ Ensure PDF is readable (not corrupted/encrypted)
→ Check file permissions
```

### No Web Search Results

```
Warning: "Web search had no results for query"
→ This is normal - fallback to LLM-only response
```

### Slow Embeddings

```
Performance: "Generating embeddings is slow"
→ First run loads model to cache (1-2 min normal)
→ Subsequent runs are instant
```

## 📖 Documentation

For detailed architecture and design patterns, see [INTERVIEW_PREP.md](INTERVIEW_PREP.md):

- Complete system design
- Technology deep-dives
- Common interview questions
- Best practices

## 🤝 Contributing

To contribute improvements:

1. Create a feature branch
2. Make your changes
3. Test with `streamlit run app.py`
4. Commit with clear messages
5. Submit a pull request

## 📝 License

This project is provided as-is for educational and professional use.

## 🎓 Learning Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [LangChain Docs](https://python.langchain.com/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [SentenceTransformers](https://www.sbert.net/)

## 💡 Future Enhancements

- [ ] Multi-modal RAG (images, tables)
- [ ] Fine-tuned embeddings
- [ ] Persistent vector store (SQLite/PostgreSQL)
- [ ] User authentication
- [ ] Response streaming
- [ ] Cost tracking and analytics

## ❓ FAQ

**Q: Can I use local LLMs?**
A: Yes, with OpenAI-compatible APIs like Ollama or LM Studio.

**Q: How many PDFs can I upload?**
A: Limited by available memory. Typically 100+ PDFs depending on size.

**Q: Is data stored after the session ends?**
A: No, everything is session-based. Restart = clean slate.

**Q: Can I export chat history?**
A: Currently stored in session state. You can extend Streamlit with export features.

---

Made with ❤️ using Streamlit and LangChain
