import argparse
import base64
import io
import os
import random
import string

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from langchain_google_vertexai.vision_models import VertexAIImageGeneratorChat

from configuration import setup_env
from utils import get_ollama_model

from PIL import Image


class ImageClassifierOutput(BaseModel):
    is_requesting_image_generation: bool = Field(..., description="This field is true if the user_prompt is requesting image generation, false otherwise.")
    image_prompt: str = Field(..., description="This field rephrases the user's prompt like the following example: \"Generate an image with a man holding an apple.\" ")

def request_image_classifier(model):
    prompt = PromptTemplate.from_template("Here is the following user prompt: {user_prompt}\n\n Determine whether this prompt is requesting for image and get the outputted image_prompt.")
    chain = prompt | model.with_structured_output(ImageClassifierOutput)
    return chain


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def save_image(image_res):
    img_base64 = image_res["image_url"]["url"].split(",")[-1]
    OUTPUT_PATH = './outputs/images/'
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    image_name = generate_random_string(10)
    image_path = os.path.join(OUTPUT_PATH, image_name) + '.jpg'
    print(f'image path: {image_path}')
    with Image.open(io.BytesIO(base64.decodebytes(bytes(img_base64, "utf-8")))) as img:
        img.save(fp=image_path)

def get_pass_chain(inps):
    def get_image(prompt):
        return chain.invoke({'user_filtered_prompt': prompt}).content[0]
    generator = VertexAIImageGeneratorChat()
    prompt = ChatPromptTemplate.from_messages([("user", "{user_filtered_prompt}")])
    chain = prompt | generator
    image_response = get_image(inps['user_filtered_prompt'])
    save_image(image_response)
    model = ChatAnthropic(model='claude-3-5-sonnet-20240620')
    summary_chain = model | StrOutputParser()
    output_prompt = PromptTemplate.from_template("""
        Given the following prompt:
        {user_prompt}
        And given image summary:
        {image_summary}
    """)
    output_chain = {"image_summary": summary_chain, "user_prompt": lambda x: inps['original_prompt']} | output_prompt | model | StrOutputParser()
    return output_chain.invoke([HumanMessage(content=[image_response])])


def get_non_request_chain(inps):
    model = ChatAnthropic(model='claude-3-5-sonnet-20240620')
    prompt = ChatPromptTemplate.from_messages([('user', '{prompt}')])
    chain = prompt | model | StrOutputParser()
    return chain.invoke({'prompt': inps['original_prompt']})

def route(args):
    print(args)
    assert 'img_classifier_out' in args
    ico: ImageClassifierOutput = args['img_classifier_out']
    pass_chain = get_pass_chain
    non_request_chain = get_non_request_chain
    assert ico.is_requesting_image_generation and isinstance(ico.is_requesting_image_generation, bool)
    new_chain = pass_chain if ico.is_requesting_image_generation else non_request_chain
    inputs_ = {"original_prompt": args['user_prompt'], "user_filtered_prompt": ico.image_prompt}
    return new_chain(inputs_)


def main():
    setup_env()
    parser = argparse.ArgumentParser(
        prog='multimodal llm',
        description='llm that calls visionAPI for vision based ',
        epilog='Text at the bottom of help')
    parser.add_argument('query')
    args = parser.parse_args()
    image_classifier_model = ChatAnthropic(model='claude-3-5-sonnet-20240620')
    image_classification_check = {"user_prompt": lambda x: x['user_prompt'], "img_classifier_out": request_image_classifier(image_classifier_model)} | RunnableLambda(route)
    for chunk in image_classification_check.stream({'user_prompt': args.query}):
        print(chunk)
        print('-'*10)

if __name__ == '__main__':
    main()