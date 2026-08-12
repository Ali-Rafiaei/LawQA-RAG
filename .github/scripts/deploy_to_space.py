import os

from huggingface_hub import HfApi

REPO_ID = "alirf/lawqa"
FILES = {
    "app.py": "app.py",
    "lawqa_demo.py": "lawqa_demo.py",
    "requirements_space.txt": "requirements.txt",
}

SPACE_README = """---
title: LawQA: A Legal Question Assistant
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.15.2
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Conversational legal Q&A grounded in r/legaladvice, with memory and structured answers
---

Full write-up, source, and original local-Llama-2 version: https://github.com/Ali-Rafiaei/LawQA-RAG
"""


def main():
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    for local_path, repo_path in FILES.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="space",
        )

    api.upload_file(
        path_or_fileobj=SPACE_README.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space",
    )


if __name__ == "__main__":
    main()
