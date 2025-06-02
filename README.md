# DORA Regulation Assistant Chatbot
A Retrieval-Augmented Generation (RAG) chatbot for answering questions about DORA (Digital Operational Resilience Act) regulations. Uses multiple retrieval methods and Azure OpenAI for intelligent responses.

## What This Chatbot Does
- Answers questions about DORA regulations using legal documents
- Provides source citations for all answers
- Uses advanced retrieval methods (Dense, Sparse, Hybrid, HyDE)
- Interactive web interface with chat functionality
- Multiple configuration options for different retrieval strategies

## Requirements
### Paid Services Required
- **Azure OpenAI API** - You need a paid Azure account with OpenAI access
- Get your API keys from Azure Portal

### System Requirements
- Python 3.10.16
- Docker

## Installation
### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Docker
- Download Docker Desktop from https://docker.com
- Install and start Docker Desktop

### 3. Create Environment File
Create a file named `.env.local` in the project folder:
```bash
# Azure OpenAI Configuration (REQUIRED - PAID SERVICE)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=2023-05-15
GENERATOR_DEPLOYMENT_NAME=your_gpt_deployment_name
```

### 4. Start Qdrant Vector Database
```bash
docker-compose up -d
```

**Option 2: Basic Docker Run**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Option 3: Docker Run with Persistent Storage**

**Windows**
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v %cd%\qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant
```

**Linux/Mac**  
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant
```

**Start Qdrant later**
```bash
docker start qdrant
```

**Verify Qdrant is Running:**
- Dashboard: http://localhost:6333/dashboard#/collections  
- Health check: http://localhost:6333/health

### 5. Load Embeddings to Qdrant
```bash
python qdrant.py
```
*This will load the document embeddings into the vector database*

### 6. Run the Chatbot
```bash
streamlit run app.py
```
Open your browser and go to: `http://localhost:8501`

## How to Use
1. **Select Retriever Type**: Choose Dense, Sparse, Hybrid, or HyDE
2. **Set K Value**: Number of documents to retrieve (recommended: 10-15)
3. **Configure Options**: Enable/disable filtering and source display
4. **Ask Questions**: Type questions about DORA regulations
5. **View Sources**: Click source buttons to see retrieved documents

## Configuration Options
- **Dense Retriever**: Uses vector similarity search
- **Sparse Retriever**: Uses keyword-based BM25 search  
- **Hybrid Retriever**: Combines dense and sparse methods
- **HyDE Retriever**: Enhanced queries with hypothetical documents
- **Show Chunks**: Display source documents used
- And many more configuration options available

## Troubleshooting

**Azure API errors:**
- Check your API keys in `.env.local`
- Verify your Azure subscription has OpenAI access
- Check API rate limits

**App not loading:**
```bash
# Check if all dependencies installed
pip install -r requirements.txt

# Restart Qdrant
docker-compose restart
```

## File Structure
```
DORA_CHATBOT/
├── app.py                    # Main chatbot interface
├── requirements.txt          # Python dependencies  
├── docker-compose.yml        # Qdrant database setup
├── .env.local                # Your API keys (create this)
├── qdrant.py                 # Load embeddings to database
├── config.py                 # Configuration settings
└── [other retrieval and processing files]
```

## Cost Information
This chatbot uses **Azure OpenAI API** which is a **paid service**:
- Charges per API call/token
- Embedding generation costs
- Chat completion costs
