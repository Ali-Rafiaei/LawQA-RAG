import os
import time
import pandas as pd

from datasets import load_dataset
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from torch import cuda

load_dotenv()

EMBED_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_NAME = 'lawqa-llama2'
BATCH_SIZE = 32
MAX_DOCS = 100_000


def load_and_preprocess():
    print("Loading dataset...")
    train_data = load_dataset("pile-of-law/pile-of-law", 'r_legaladvice', split="train")
    validation_data = load_dataset("pile-of-law/pile-of-law", 'r_legaladvice', split="validation")

    train_set = pd.DataFrame(train_data, columns=["text", "created_timestamp", "downloaded_timestamp", "url"])
    validation_set = pd.DataFrame(validation_data, columns=["text", "created_timestamp", "downloaded_timestamp", "url"])

    df = pd.concat([train_set, validation_set]).reset_index(drop=True)
    df = df.drop(['created_timestamp', 'downloaded_timestamp'], axis=1)
    df["text"] = df["text"].str.replace("\n", "")
    df["Title"] = df['text'].apply(lambda x: x[7:x.find("Question")])
    df["Question"] = df['text'].apply(lambda x: x[x.find("Question")+9:x.find("Answer")])
    df["Answer"] = df['text'].apply(lambda x: x[x.find("Answer")+11:])
    df.columns = ["Text", "Url", "Title", "Question", "Answer"]
    print(f"Dataset loaded: {len(df)} documents.")
    return df


def init_pinecone():
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Index created.")

    return pc.Index(INDEX_NAME)


def ingest(index, df, embed_model):
    data = df.iloc[:MAX_DOCS]
    print(f"Upserting {len(data)} documents in batches of {BATCH_SIZE}...")

    for i in range(0, len(data), BATCH_SIZE):
        batch = data.iloc[i:i + BATCH_SIZE]
        ids = [f"id-{idx}" for idx in batch.index]
        texts = batch['Text'].tolist()
        embeds = embed_model.embed_documents(texts)
        metadata = [
            {"text": row['Text'][:2000], "source": row['Url']}
            for _, row in batch.iterrows()
        ]
        index.upsert(vectors=list(zip(ids, embeds, metadata)))

        if (i // BATCH_SIZE) % 100 == 0:
            print(f"  {i + len(batch)}/{len(data)} documents upserted.")

    print("Ingestion complete.")


if __name__ == "__main__":
    index = init_pinecone()

    stats = index.describe_index_stats()
    if stats['total_vector_count'] > 0:
        print(f"Index already contains {stats['total_vector_count']} vectors. Skipping ingestion.")
        print("To re-ingest, delete the index from the Pinecone dashboard first.")
    else:
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': BATCH_SIZE}
        )

        df = load_and_preprocess()
        ingest(index, df, embed_model)
