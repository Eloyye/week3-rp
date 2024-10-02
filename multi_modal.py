import argparse
import base64
from operator import itemgetter
from typing import Literal

import httpx
from attr.validators import max_len
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from configuration import setup_env



def encode_image(image_link: str):
    return base64.b64encode(httpx.get(image_link).content).decode("utf-8")

class ImageRecognitionArgs(BaseModel):
    image_url: str = Field(..., description='image url')
    image_file_extension: Literal['jpeg', 'gif', 'png', 'webp'] = Field(..., max_length=4, description='image file extension')

def image_recognition_llm_tool(llm):

    template = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Vision Language Model called visionAPI. Your job is to take an image and with as much detail as possible, describe the image being shown.
        """),
        MessagesPlaceholder(variable_name="image_message")
    ])
    image_binary_runnable = {"image_message": {"image_url": itemgetter("image_url"), "image_file_extension": itemgetter("image_file_extension")} | RunnableLambda(create_image_messages)}
    chain = image_binary_runnable | template | llm | StrOutputParser()
    return chain.as_tool(name='visionAPI', description='Use this tool when given a url with either a .jpeg, .gif, .png, .webp file. To call this tool ONLY provide image_url and image_file_extension (without the period)', args_schema=ImageRecognitionArgs)


def create_image_messages(inputs):
    file_extension, image_url = inputs["image_file_extension"], inputs["image_url"]
    url_to_image_binary = f"data:image/{file_extension};base64,{encode_image(image_url)}"
    image_message = [HumanMessage(
        content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": url_to_image_binary},
                    }
                ]
        )
    ]

    return image_message



def talk_to_agent(content: str):
    setup_env()
    llm = ChatAnthropic(model='claude-3-5-sonnet-20240620')
    image_recognition_tool = image_recognition_llm_tool(llm)
    agent = create_react_agent(llm, [image_recognition_tool], state_modifier="""You are an agent chatbot such that whenever user inserts a link, you must call visionAPI tool.
        Then, using that description of the image answer the user's questions. Do not make up new url links.
        """)
    # config = {"configurable": {"thread_id": "abc1234"}}
    result = agent.invoke({"messages": [('human', content)]})
    return result["messages"][-1].content

def main():
    parser = argparse.ArgumentParser(
        prog='multimodal llm',
        description='llm that calls visionAPI for vision based ',
        epilog='Text at the bottom of help')
    parser.add_argument('query')
    args = parser.parse_args()
    print(f"Q: {args.query}")
    print(f'A: {talk_to_agent(args.query)}')
if __name__ == '__main__':
    main()