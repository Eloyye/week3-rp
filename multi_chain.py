import getpass
import os

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from env import ANTHROPHIC_API


class WikipediaConcepts(BaseModel):
    headline: str = Field(description='headline of a given query')

def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPHIC_API
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=5000)
    wikipedia_runner = WikipediaQueryRun(api_wrapper=api_wrapper)

    wikipedia_template = """Given the following query:
    <query>
    {query}
    </query>
    Write the keyword or title that most relate the query. You should only output the keyword or title and nothing else.
    """
    query_wikipedia_prompt = PromptTemplate.from_template(wikipedia_template)

    query_wikipedia_chain = query_wikipedia_prompt | llm | StrOutputParser() | wikipedia_runner


    summarization_template = """Search Wikipedia to find relevant information and then summarize it.
    <WikipediaResult>
    {wikipedia_result}
    </WikipediaResult>
    Please provide a concise summary of the information found:
    """

    prompt = PromptTemplate.from_template(template=summarization_template)

    summarization_chain = {"wikipedia_result": query_wikipedia_chain} | prompt | llm | StrOutputParser()

    output_prompt = PromptTemplate.from_template(template="""Given the initial query:
    <query>
    {query}
    </query>

    Here is a summary from wikipedia which may or may not be relevant to the query:
    <output>
    {output}
    </output>

    Determine whether the output summary is sufficient to answer the query. If the output from wikipedia is relevant to the question, please
    give a response that most answers the question. If it is not relevant, then answer that you could not answer the question. Just respond with 
    an answer and do not mention about query or if it is sufficient in the response output.
    """)

    result_chain = {"output": summarization_chain, "query": lambda x: x["query"]} | output_prompt | llm | StrOutputParser()

    result = result_chain.invoke({"query": "What is a load balancer"})

    print(result)



if __name__ == '__main__':
    main()
