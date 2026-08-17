# LawQA: Legal Question Answering with RAG

[![Deploy to HuggingFace Space](https://github.com/Ali-Rafiaei/LawQA-RAG/actions/workflows/deploy-to-space.yml/badge.svg)](https://github.com/Ali-Rafiaei/LawQA-RAG/actions/workflows/deploy-to-space.yml)

A Retrieval-Augmented Generation (RAG) system that answers legal questions by combining semantic search over real legal advice data with an LLM for response generation.

🔗 **Live demo:** [HuggingFace Space](https://huggingface.co/spaces/AliRF/lawqa)

This repo holds two versions of the same idea:
- **`lawqa.py`** - the original pipeline: a fully local, from-scratch RAG stack running a 4-bit quantized Llama-2-7b, built to work through the mechanics end to end.
- **`app.py` / `lawqa_demo.py`** - the deployed live demo: swaps the local LLM for the Gemini API so it can run on free hosting with no GPU, and adds conversation memory and structured, product-style answers on top of the same retrieval pipeline.

## What makes the live demo different

- **Conversation memory:** ask a follow-up ("what if it keeps happening?") and the assistant resolves it against earlier turns before searching, instead of treating every question as a cold start. Memory is scoped per browser session, so multiple visitors can use the public demo at once without their conversations mixing.
- **Structured answers:** every response follows a consistent Summary / Key points / What you can do / Important-caveat format, driven entirely by prompt design, no extra parsing code.
- **Continuous deployment:** every merge to `main` automatically redeploys the live Space via GitHub Actions, no manual upload step.

## Architecture

Original pipeline (`lawqa.py`):
```
pile-of-law/r_legaladvice dataset
    |
    v
ingest.py (run once)
Preprocess → Embed (all-MiniLM-L6-v2, 384-dim) → Upsert to Pinecone
100,000 documents, batched at 32
    |
    v
lawqa.py (RAG inference)
Query → top-3 semantic retrieval → prompt + context → Llama-2 → answer
```

Live demo (`app.py` / `lawqa_demo.py`) replaces the last step with a history-aware retrieval chain and Gemini:
```
Question + chat history
    |
    v
history-aware retriever (rewrites follow-ups into standalone queries, then retrieves)
    |
    v
structured-answer prompt + Gemini
    |
    v
answer + sources, shown in a per-session Gradio chat
```

## Tech Stack

- **LLM (original):** Llama-2-7b-chat (4-bit quantized via BitsAndBytes)
- **LLM (live demo):** Gemini (`gemini-3.6-flash`, via API)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Database:** Pinecone
- **Orchestration:** LangChain (LCEL pipeline / history-aware retrieval chain)
- **Interface (live demo):** Gradio
- **Deployment:** HuggingFace Spaces, auto-deployed via GitHub Actions
- **Dataset:** [pile-of-law/r_legaladvice](https://huggingface.co/datasets/pile-of-law/pile-of-law)

## Project Structure

```
ingest.py               - one-time data ingestion: embed and index 100k documents
lawqa.py                - original local RAG inference pipeline (Llama-2)
requirements.txt        - dependencies for the original local pipeline
lawqa_demo.py            - live demo pipeline (Gemini, history-aware retrieval)
app.py                   - Gradio interface for the live demo
requirements-space.txt   - dependencies for the deployed HuggingFace Space
.github/workflows/       - CI: deploys app.py, lawqa_demo.py, requirements-space.txt
                           to the HuggingFace Space on every merge to main
```

## Setup (original local pipeline)

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (~16GB VRAM for 4-bit quantized Llama-2-7b)
- [Pinecone](https://www.pinecone.io/) account and API key
- [Hugging Face](https://huggingface.co/) account with [Llama-2 access](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) approved

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your-pinecone-api-key
HF_AUTH_TOKEN=your-huggingface-token
```

### Run

```bash
# First run only, embeds and indexes 100k documents into Pinecone
python ingest.py

# RAG inference pipeline
python lawqa.py
```

`ingest.py` checks whether the index is already populated and skips re-ingestion if so.

## Running the live demo locally

No GPU needed, since inference goes through the Gemini API.

```bash
pip install -r requirements-space.txt
```

Add to your `.env`:

```
PINECONE_API_KEY=your-pinecone-api-key
GOOGLE_API_KEY=your-google-ai-studio-key
```

```bash
python app.py
```

Opens a Gradio chat UI on `localhost` with the same conversational, structured-answer pipeline that's deployed on the live Space.

## Example

**Query:**
> I never got any eviction notice from my landlord. One day he came to my home and
told me to leave and that if I didn't he was gonna sue me. Can he really do that?

The pipeline returns two things: a generated answer grounded in the retrieved
documents, and the source URLs those documents came from so every answer is
traceable back to real cases.

**Answer** *(generated by Llama-2 from retrieved context)*:
> Yes, he can. In most states, a landlord can give a tenant notice to quit, which is a formal way of telling them they must vacate the property by a certain date. If the tenant fails to do so, the landlord can file an eviction lawsuit and potentially win a court order to have the tenant removed. It's important to note that the specific laws regarding eviction vary by state and locality, so it's best to consult with a lawyer or a tenant rights organization for more information.

**Sources** *(documents retrieved from the vector store and used as context)*:
- https://www.reddit.com/r/legaladvice/comments/3kllgh/landlord_is_threatening_eviction_and_saying_ill/
- https://www.reddit.com/r/legaladvice/comments/eedcjw/roommate_is_a_toxic_little_shirt_his_name_is_on/
- https://www.reddit.com/r/legaladvice/comments/bxwsjh/can_landlord_kick_me_out_all_of_a_sudden/

## Limitations

Retrieval quality degrades on questions that don't closely match the phrasing of indexed documents.
The embedding model is strong but the dataset is conversational Reddit text, so formal legal
queries sometimes miss relevant results. Both LLMs can occasionally state specific statute numbers
or case citations that don't exist, which is a known risk when grounding legal text on a smaller or
general-purpose model; the system prompt tells the model to say so when the retrieved context is
insufficient rather than guess, but that's a mitigation, not a guarantee. A production version of
this would run ingestion as a background job with checkpointing.
