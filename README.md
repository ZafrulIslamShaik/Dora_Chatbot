# DORA Regulation Assistant Chatbot
A Retrieval-Augmented Generation (RAG) chatbot for answering questions about DORA (Digital Operational Resilience Act) and its related regulations.
Powered by Azure OpenAI and advanced retrieval techniques for intelligent responses.

## What This Chatbot Does
- Answers questions about DORA and its related regulations.
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
AZURE_OPENAI_ENDPOINT=https://placeholder-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=XXXXX
GENERATOR_DEPLOYMENT_NAME=your_gpt_deployment_name
```

### 4. Start Qdrant Vector Database
Make sure Docker and Docker Compose are installed and running on your machine.

### Option 1: Using Docker Compose

1. Open your terminal or command prompt.

2. Navigate to the folder containing the `docker-compose.yml` file:

    ```bash
    cd path/to/your/docker-compose-folder
    ```

3. Start Qdrant in detached mode:

    ```bash
    docker-compose up -d
    ```

This will launch the Qdrant container in the background.

---

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

## Setup Instructions
1. **Download embeddings**: Download the pre-computed embeddings from [this Google Drive link](https://drive.google.com/drive/folders/1ztlPmfwEeUUin1yKHhSIdTcKKv33xmYI?usp=drive_link)

**Two embedding types available:**
- **Large embeddings** (~1000 tokens per chunk): Optimized chunking with sentence boundaries - tested and recommended
- **Small embeddings** (~3 sentences per chunk): Sentence-wise splitting - good evaluation metrics but not fully tested in chatbot

Both can be loaded using Qdrant.

2. **Place embeddings**: Extract and place all downloaded embedding files in the `document_embeddings/` folder

3. **Load to database**: Run `python qdrant.py` to load embeddings into the vector database

4. **Start chatbot**: Run `streamlit enhanced_app.py` to launch the interface


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
DORA_CHATBOT/
├── enhanced_app.py           # Main chatbot interface
├── dense_retriever.py    
├── sparse_retriever.py      
├── hybrid_retriever.py     
├── hyde.py
├── relevance_filtering.py   
├── generator.py           
├── reference_identifiers.json 
├── requirements.txt        
├── docker-compose.yml          
├── qdrant.py                            
└── document_embeddings/
   ├── embedding_generator.py
   ├── large_embeddings.npy 
   ├── small_embeddings.npy  
   ├── large_documents.json  
   ├── small_documents.json  
   ├── cross_references.json  
   ├── cross_references.npy  
   ├── .env.local 
   └── document_urls.json


## Cost Information
This chatbot uses **Azure OpenAI API** which is a **paid service**:
- Charges per API call/token
- Embedding generation costs
- Chat completion costs
