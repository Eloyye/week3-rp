import argparse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from configuration import setup_env
from utils import get_ollama_model

CLAUDE_SONNET = 'claude-3-5-sonnet-20240620'

def get_sql_generator_tool():
    template = [
        ('system', """
            You are a SQL specialized API tool. 
            You are given a request in natural language but your job is to only create the table or tables that satisfy
            that requirement.
            Only provide the SQL response and nothing else.
        """),
        ("human", "{user_query}")
    ]
    prompt = ChatPromptTemplate(template)
    model = ChatAnthropic(model=CLAUDE_SONNET)
    chain = prompt | model | StrOutputParser()
    tool_description = """
        This tool provides SQL context generation. 
        Use this tool when client does not provide necessary context like a SQL table or schema.
    """
    return chain.as_tool(name='SQL_Context_Generator', description=tool_description)

class OutputAgentResponse(BaseModel):
    context: str = Field(..., description= "context of SQL comprising of SQL CREATE TABLE statements.")
    query: str = Field(..., description="Contains the output query")

def get_main_agent():
    # model = ChatAnthropic(model=CLAUDE_SONNET)
    agent_system_message = """
            You are an LLM agent specialized in making SQL queries based on natural language. 
            If you are not given any context, then you need to use generate table by performing a tool call. 
            Output your response to both the generated context tables and your SQL query like:
             ```json
             {
                context: "CREATE TABLE",
                query: "SELECT ",
             }
             ```
        """
    sql_generator_tool = get_sql_generator_tool()
    model = get_ollama_model(model='sqllama')
    app = create_react_agent(model=model, tools=[sql_generator_tool], state_modifier=agent_system_message)
    return app

def get_args():
    parser = argparse.ArgumentParser(
        prog='multimodal llm',
        description='llm that calls visionAPI for vision based ',
        epilog='Text at the bottom of help')
    parser.add_argument('query')
    args = parser.parse_args()
    result = {"query": args.query}
    return result


def debug(app, content):
    config = {"configurable": {"thread_id": "42139a"}}
    for chunk in app.stream({'messages': [HumanMessage(content=content)]}, config=config):
        print(chunk)
        print('-'*10)

def main():
    setup_env()

    args = get_args()

    app = get_main_agent()

    # debug(app, args["query"])

    chain = app

    result = chain.invoke({"messages": [
        HumanMessage(content=args["query"])
    ]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    main()