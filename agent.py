from langchain_anthropic import ChatAnthropic
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from configuration import setup_env


def print_message(agent_executor, config, message: str):
    for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=message)]}, config
    ):
        print(chunk)
        print("----")

def main():
    setup_env()
    chat_model = ChatAnthropic(model_name='claude-3-5-sonnet-20240620')
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
    tools = [wikipedia]

    system_prompt = 'You are an LLM Agent that uses Wikipedia as a reliable source of information. Make sure to answer questions as thoroughly as possible. If the page is not relevant then say that you do not have enough information to answer the question. Respond with no mention of Wikipedia.'
    agent_exec = create_react_agent(chat_model, tools, checkpointer=MemorySaver(), state_modifier=system_prompt)
    config = {"configurable": {"thread_id": "123"}}
    print_message(agent_exec, config,"Summarize: \"FDA v. Alliance for Hippocratic Medicine\"")


if __name__ == '__main__':
    main()