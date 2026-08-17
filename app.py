try:
    import spaces
except ImportError:
    spaces = None

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from lawqa_demo import ask

if spaces is not None:

    @spaces.GPU
    def zerogpu_startup_probe():
        pass


MAX_HISTORY_MESSAGES = 6


def to_chat_history(history: list[dict]) -> list:
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    messages = []
    for turn in trimmed:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))

    return messages


def respond(message: str, history: list[dict]) -> str:
    chat_history = to_chat_history(history)
    answer, sources = ask(message, chat_history)

    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

    return answer


demo = gr.ChatInterface(
    respond,
    title="LawQA: Legal Question Assistant",
    description=(
        "Ask legal questions and follow up naturally. The assistant remembers your "
        "conversation, with answers grounded in real cases from r/legaladvice (100k+ posts)."
    ),
    examples=["My landlord entered my apartment without notice. What are my rights?"],
    concurrency_limit=3,
)

if __name__ == "__main__":
    demo.launch()
