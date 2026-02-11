# LawQA - Legal Question Answering with RAG

A Retrieval-Augmented Generation (RAG) system that answers legal questions by combining semantic search over real legal advice data with Llama-2 for response generation.

## Architecture

```
pile-of-law/r_legaladvice dataset
        |
        v
  Data Preprocessing (Preparing_Dataset.ipynb)
  Extract: Title, Question, Answer, URL
        |
        v
  Embedding (sentence-transformers/all-MiniLM-L6-v2)
  384-dimensional vectors, cosine similarity
        |
        v
  Vector Store (Pinecone)
  100,000 documents indexed in batches of 32
        |
        v
  Query Pipeline (LangChain RetrievalQA)
  User query -> top-3 semantic search -> context + query -> Llama-2 -> answer
        |
        v
  LLM (meta-llama/Llama-2-7b-chat-hf)
  4-bit quantized via BitsAndBytes for efficient inference
```

## Tech Stack

- **LLM:** Llama-2-7b-chat (4-bit quantized with BitsAndBytes)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Database:** Pinecone
- **Orchestration:** LangChain (RetrievalQA chain)
- **Dataset:** [pile-of-law/r_legaladvice](https://huggingface.co/datasets/pile-of-law/pile-of-law) from Hugging Face

## Project Structure

```
lawqa.py                  # Complete RAG pipeline
lawqa.ipynb               # Notebook version with step-by-step walkthrough
Preparing_Dataset.ipynb   # Dataset preprocessing and exploration
```

## Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (minimum ~16GB VRAM for 4-bit quantized Llama-2-7b)
- [Pinecone](https://www.pinecone.io/) account and API key
- [Hugging Face](https://huggingface.co/) account with Llama-2 access approved

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Set the following environment variables before running:

```bash
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_ENV="your-pinecone-environment"
export HF_AUTH_TOKEN="your-huggingface-token"
```

### Run

```bash
python lawqa.py
```

## Example Query

> "I never got any eviction notice from my landlord. One day he came to my home and told me to leave and that if I didn't he was gonna sue me. Can he really do that?"

The system retrieves the 3 most relevant legal advice documents from the vector store and generates an answer grounded in those sources.
