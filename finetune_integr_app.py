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

def create_model_as_tool(tool_name="", system_message="", human_input="", model=None, tool_description="", arg_types=None):
    assert model
    template = [
        ('system', system_message),
        ("human", human_input)
    ]
    prompt = ChatPromptTemplate(template)
    chain = prompt | model | StrOutputParser()
    return chain.as_tool(name=tool_name, description=tool_description, arg_types=arg_types)

def get_sql_query_generator():
    system_prompt = """
                You are a SQL specialized API tool. 
                You are given a request in natural language but your job is to create the queries corresponding to a given context
                Only provide the SQL response and nothing else.
            """
    model = get_ollama_model(model='sqllama')
    tool_description = """
            This tool provides SQL query generation given question in natural language and context in sql. 
            Use this tool to generate a query and must be used in a final tool call.
        """
    return create_model_as_tool("SQL_query_generator", system_prompt, "Here is the question:\n{question}\n\nHere is the context in SQL:\n{context}", model, tool_description, arg_types={"question": str, "context": str})

def get_sql_generator_tool():
    system_prompt = """
            You are a SQL specialized API tool. 
            You are given a request in natural language but your job is to only create the table or tables that satisfy
            that requirement.
            Only provide the SQL response and nothing else.
        """
    model = ChatAnthropic(model=CLAUDE_SONNET)
    tool_description = """
        This tool provides SQL context generation. 
        Use this tool when client does not provide necessary context like a SQL table or schema.
    """
    return create_model_as_tool("SQL_context_Generator", system_prompt, '{user_query}', model, tool_description)

class OutputAgentResponse(BaseModel):
    context: str = Field(..., description= "context of SQL comprising of SQL CREATE TABLE statements.")
    query: str = Field(..., description="Contains the output query")

def get_main_agent():
    # model = ChatAnthropic(model=CLAUDE_SONNET)
    agent_system_message = """
            You are a Q&A model that specializes in answering questions about SQL. If the client provides zero context in terms of SQL tables, then you must make a tool call to create that SQL context 
            . Finally, You would then need to make a tool call
            to generate query:
            the output should be of the following schema:
            {
                context: string,
                query: string
            }
        """
    sql_context_generator = get_sql_generator_tool()
    sql_query_generator = get_sql_query_generator()
    # tool calling
    model = ChatAnthropic(model=CLAUDE_SONNET)
    app = create_react_agent(model=model, tools=[sql_context_generator, sql_query_generator], state_modifier=agent_system_message)
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

    debug(app, args["query"])

    # chain = app
    #
    # result = chain.invoke({"messages": [
    #     HumanMessage(content=args["query"])
    # ]})

    # print(result['messages'][-1].content)

if __name__ == '__main__':
    main()