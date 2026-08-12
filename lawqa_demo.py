import os

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

load_dotenv()
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "lawqa-llama2"

embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
pc = PineconeClient(
    api_key=os.environ.get("PINECONE_API_KEY"),
)
index = pc.Index(INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embed_model, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", google_api_key=os.environ.get("GOOGLE_API_KEY")
)

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question, which might reference "
    "context in the chat history, rewrite it as a standalone question that can "
    "be understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed, otherwise return it as-is."
)

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

SYSTEM_PROMPT = """You are a legal assistant answering questions based on real legal
advice from r/legaladvice. Structure every response exactly as follows:

**Summary:** One sentence direct answer.

**Key points:**
- Point 1
- Point 2
- Point 3 (if relevant)

**What you can do:** Concrete next steps the person can take.

**Important:** This advice is based on crowd-sourced legal discussions, not professional
legal counsel. For serious matters, consult a qualified lawyer in your jurisdiction.

Use only information from the provided context. If the context does not contain enough
information to answer, say so clearly rather than guessing.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def ask(question: str, chat_history: list) -> tuple[str, list[str]]:
    result = rag_chain.invoke({"input": question, "chat_history": chat_history})
    sources = list({doc.metadata.get("source", "Unknown") for doc in result["context"]})
    return result["answer"], sources


if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage

    answer1, sources1 = ask(
        "My landlord entered my apartment without giving me notice. What are my rights?",
        chat_history=[],
    )
    print("ANSWER 1:\n", answer1)
    print("SOURCES 1:\n", sources1)

    chat_history = [
        HumanMessage(
            content="My landlord entered my apartment without giving me notice. What are my rights?"
        ),
        AIMessage(content=answer1),
    ]

    from langchain_core.output_parsers import StrOutputParser

    rewrite_chain = contextualize_prompt | llm | StrOutputParser()
    rewritten = rewrite_chain.invoke(
        {"input": "What if it keeps happening?", "chat_history": chat_history}
    )
    print("REWRITTEN QUERY:\n", rewritten)

    answer2, sources2 = ask("What if it keeps happening?", chat_history=chat_history)

    print("ANSWER 2:\n", answer2)
    print("SOURCES 2:\n", sources2)
