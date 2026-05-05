import os
import transformers
import sys

from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient
from torch import cuda, bfloat16
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore

load_dotenv()

EMBED_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'
INDEX_NAME = 'lawqa-llama2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

#---------------------- Embedding model ----------------------
embed_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

#---------------------- Pinecone vector store ----------------------
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embed_model, text_key='text')

#---------------------- Llama-2 model ----------------------
hf_auth = os.getenv("HF_AUTH_TOKEN")

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(MODEL_ID, token=hf_auth)
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, token=hf_auth)

#---------------------- Generation pipeline ----------------------
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    do_sample=False,
    max_new_tokens=512,
    repetition_penalty=1.1
)

#---------------------- RAG chain ----------------------
llm = HuggingFacePipeline(pipeline=generate_text)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template(
    "Use the following legal Q&A context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(source_documents=retriever, question=RunnablePassthrough())
    .assign(context=lambda x: format_docs(x["source_documents"]))
    .assign(answer=prompt | llm | StrOutputParser())
)

#---------------------- Query ----------------------
sample_query = "I never got any eviction notice from my landlord. One day he came to my home and told me to leave and that if I didn't he was gonna sue me. Can he really do that?"
query = sys.argv[1] if len(sys.argv) > 1 else sample_query
result = rag_chain.invoke(query)

print("Answer:", result["answer"])
print("\nSources:")
for doc in result["source_documents"]:
    print(" -", doc.metadata.get("source", "unknown"))
