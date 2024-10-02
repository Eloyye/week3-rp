from langchain_openai import ChatOpenAI


def get_ollama_model(port=11434, model="llama3.2"):
    llm = ChatOpenAI(
        api_key="ollama",
        model=model,
        base_url=f"http://localhost:{port}/v1",
    )
    return llm