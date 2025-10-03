# LangChain

A comprehensive hands-on exploration of LangChain framework covering data ingestion, text processing, embeddings, and vector storage solutions.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org)

## Overview

This project demonstrates practical implementation of LangChain concepts through hands-on tutorials and examples. It covers simple complete pipeline from data ingestion to vector storage, showing various text processing techniques and embedding strategies.

## Features

### Data Ingestion

- **Multi-format Document Loading**: Support for PDF, TXT, XML, and web content
- **Web Scraping**: Extract content from web pages with BeautifulSoup integration
- **Academic Papers**: ArXiv paper loading capabilities
- **Wikipedia Integration**: Direct Wikipedia content retrieval

### Text Processing & Splitting

- **Character-based Splitting**: Simple character-level text segmentation
- **Recursive Character Splitting**: Intelligent text chunking preserving semantic structure
- **HTML Text Extraction**: Clean text extraction from HTML documents
- **JSON Document Splitting**: Structured data processing

### Embeddings & Vectorization

- **OpenAI Embeddings**: High-quality text embeddings using OpenAI's models
- **HuggingFace Integration**: Open-source embedding models
- **Ollama Local Models**: Local embedding generation
- **Multiple Model Support**: Various embedding strategies

### Vector Storage Solutions

- **ChromaDB**: Persistent vector database with SQLite backend
- **FAISS**: Facebook's efficient similarity search library
- **Vector Indexing**: Optimized storage and retrieval
- **Similarity Search**: Fast nearest neighbor queries

## Installation

### Prerequisites

- Python 3.10+
- Conda or pip package manager

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/greatdaveo/langc
   cd langc
   ```

2. **Create virtual environment**

   ```bash
   conda create -p venv python==3.10 -y
   conda activate ./venv
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file with your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   HF_TOKEN=your_huggingface_token
   ```

5. **Jupyter Kernel Setup** (if needed)
   ```bash
   python -m pip install --upgrade ipykernel
   python -m ipykernel install --user --name venv --display-name "Python 3.10 (venv)"
   ```

## Usage

### Running the Notebooks

1. **Start Jupyter Lab/Notebook**

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. **Select the appropriate kernel**: "Python 3.10 (venv)"

3. **Navigate through the notebooks** in order:
   - Start with `DataIngestion/DataIngestion.ipynb`
   - Move to `DataTransformer/` for text processing
   - Explore `Embeddings/` for vectorization
   - Finish with `VectoreStore/` for storage solutions

### Quick Start Example

```python
# Load and process a document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load document
loader = TextLoader('speech.txt')
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in vector database
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

# Query the database
results = vectorstore.similarity_search("your query here")
```

## Key Components

### Data Ingestion Module

- **TextLoader**: Load plain text files
- **PyPDFLoader**: Extract text from PDF documents
- **WebBaseLoader**: Scrape web content with BeautifulSoup
- **ArxivLoader**: Load academic papers from ArXiv
- **WikipediaLoader**: Fetch Wikipedia articles

### Text Processing Module

- **CharacterTextSplitter**: Simple character-based splitting
- **RecursiveCharacterTextSplitter**: Intelligent recursive splitting
- **HTMLHeaderTextSplitter**: HTML-aware text extraction
- **RecursiveJsonSplitter**: JSON document processing

### Embeddings Module

- **OpenAI Embeddings**: High-quality commercial embeddings
- **HuggingFace Embeddings**: Open-source model integration
- **Ollama Embeddings**: Local model support

### Vector Storage Module

- **ChromaDB**: Persistent vector database
- **FAISS**: High-performance similarity search
- **Vector Indexing**: Optimized storage strategies

## Technologies Used

- **LangChain**: Core framework for LLM applications
- **OpenAI**: Commercial embedding models
- **HuggingFace**: Open-source ML models
- **ChromaDB**: Vector database
- **FAISS**: Similarity search library
- **BeautifulSoup**: Web scraping
- **PyPDF**: PDF processing
- **Jupyter**: Interactive development environment

## Acknowledgments

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [HuggingFace](https://huggingface.co/)
- [ChromaDB](https://www.trychroma.com/)
- [FAISS](https://faiss.ai/)

---

## 👨‍💻 Developed By
> Olowomeye David [GitHub](https://github.com/greatdaveo) [LinkedIn](https://linkedin.com/in/greatdaveo)

---