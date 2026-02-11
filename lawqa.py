import transformers
import pandas as pd
import pinecone
import os
import time

from torch import cuda, bfloat16
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

#----------------------Defining the embedding model----------------------
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)


#---------------Initialing the pinecone instance------------------
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

#-----------------Creating the Pinecone database if it doesn't exist already-------------
index_name = 'lawqa-llama2'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=384,
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pinecone.Index(index_name)
index.describe_index_stats()


#----------------------importing and preprocessing the data-----------------
train_data = load_dataset("pile-of-law/pile-of-law", 'r_legaladvice', split="train")
validation_data = load_dataset("pile-of-law/pile-of-law", 'r_legaladvice', split="validation")

train_set = pd.DataFrame(train_data, columns=["text", "created_timestamp", "downloaded_timestamp", "url"])
validation_set = pd.DataFrame(validation_data, columns=["text", "created_timestamp", "downloaded_timestamp", "url"])

df = pd.concat([train_set, validation_set])
df = df.drop(['created_timestamp','downloaded_timestamp'], axis=1)
df["text"] = df["text"].str.replace("\n","")
df["Title"] = df['text'].apply(lambda x: x[7:x.find("Question")])
df["Question"] = df['text'].apply(lambda x: x[x.find("Question")+9:x.find("Answer")])
df["Answer"] = df["text"].apply(lambda x: x[x.find("Answer")+11:])
df.columns = ["Text", "Url", "Title", "Question", "Answer"]
df.drop(["text"], axis=1, inplace=True)


#----------------------Storing the data on the database --------------------
data = df.iloc[:100000]

batch_size = 32

for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"id-{i}" for i, x in batch.iterrows()]
    texts = [x['Text'] for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # setting the metadata
    metadata = [
        {"text": x['Text'][:2000],
         'source': x['Url']} for i, x in batch.iterrows()
    ]
    # storing in the Pineconce database
    index.upsert(vectors=zip(ids, embeds, metadata))

#---------------------- Defining the model by using the name on Hugging Face ----------------------
model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

#----------------------the `bitsandbytes` library is required to be able to load the Llama 2 model----------------------
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

#---------------------- Downloading Llama2 model from Hugging Face using auth token ----------------------
hf_auth = os.getenv("HF_AUTH_TOKEN")
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)


model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

#---------------------- Downloading the tokenizer from Hugging Face using auth token ----------------------
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

#---------------------- Creating the text generation pipeline-----------------
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    # parameters for the model are set here
    temperature=0.000000000000000000000000001,  # Controling the randomness of the output
    max_new_tokens=512,  # maximum number of tokens in the output
    repetition_penalty=1.1  # to prevent repetition
)

#---------------------- Creating the database instance ----------------------
text_field = 'text'  # field in metadata that contains the text

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

#---------------------- Testing the similarity search ----------------------
query = "I never got any eviction notice from my landlord. One day he came to my home and told me to leave and that id I didn't he was gonna sue me. My question is that can he really do that? please provide sources for your answer."

vectorstore.similarity_search(
    query,
    k=3  # returns top 3 most relevant documents
)

#---------------------- Setting the LLM model ----------------------
llm = HuggingFacePipeline(pipeline=generate_text)

#---------------------- Initializing the RetrievalQA chain ----------------------
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)


#---------------------- Testing the model ----------------------
query = "I never got any eviction notice from my landlord. One day he came to my home and told me to leave and that id I didn't he was gonna sue me. My question is that can he really do that? please provide sources for your answer."
result = rag_pipeline({"query": query})

print(result)
