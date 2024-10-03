import os
import shutil
import io
import base64

import fitz
from PIL import Image
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from configuration import setup_env
from fastapi import FastAPI, File, UploadFile, HTTPException

from utils import get_ollama_model

app = FastAPI()
OUTPUT_DIRECTORY = "outputs"
setup_env()
if not os.path.isdir(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)


def pdf_pages_to_base64(pdf_path: str):
    pdf_document = fitz.open(pdf_path)
    for page in pdf_document:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        yield base64.b64encode(buffer.getvalue()).decode("utf-8")
    pdf_document.close()

def is_multi_modal(model) -> bool:
    model_name = model.model
    print(model.model)
    if model_name == 'claude-3-5-sonnet-20240620':
        return True
    return False

async def parse_pdf(file_path: str) -> str:
    loader_local = UnstructuredPDFLoader(file_path=file_path, strategy='fast')
    docs = []
    for doc in loader_local.lazy_load():
        docs.append(doc.page_content)
    result = "\n\n".join(docs)
    return result


async def handle_multi_modal(file_path, model):
    system_message = SystemMessage(content=get_system_prompt())
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page_bin}"},
            } for page_bin in pdf_pages_to_base64(file_path)
        ],
    )
    chain = model | StrOutputParser()
    return await chain.ainvoke([system_message, message])

def get_system_prompt():
    return """
        You will be provided with the contents of a pdf.
    As a professional summarizer, create a concise and comprehensive summary of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines:

    Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.

    Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.

    Rely strictly on the provided text, without including external information.

    Format the summary in paragraph form for easy understanding.

    Conclude your notes with [End of Notes, Message #X] to indicate completion, where "X" represents the total number of messages that I have sent. In other words, include a message counter where you start with #1 and add 1 to the message counter every time I send a message.

By following this optimized prompt, you will generate an effective summary that encapsulates the essence of the given text in a clear, concise, and reader-friendly manner.
        """

async def get_summary(file_path: str, model) -> str:
    if is_multi_modal(model):
        return await handle_multi_modal(file_path, model)
    pdf_contents = await parse_pdf(file_path)
    template = [
        ('system', get_system_prompt()),
        ('user', '{contents}')
    ]
    prompt = ChatPromptTemplate(template)
    chain = prompt | model | StrOutputParser()
    return await chain.ainvoke({'contents': pdf_contents})


def get_model(model_name: str):
    model = None
    if model_name == 'claude_sonnet':
        model = ChatAnthropic(model='claude-3-5-sonnet-20240620')
    elif model_name == 'llama3.2':
        model = get_ollama_model(model='llama3.2')
    elif model_name == 'mistral':
        model = get_ollama_model(model='mistral')
    elif model_name == 'gemma2b':
        model = get_ollama_model(model='gemma2:2b')
    else:
        assert False, f"Got invalid model: ${model_name}"
    return model


def copy_contents_to_file(request_file, file_path):
    with open(file_path, mode="wb") as f:
        shutil.copyfileobj(request_file.file, f)

@app.post("/uploadfile/")
async def upload_file_summarize(request_file: UploadFile = File(...)):
    # if content_type != "application/pdf":
    #     raise HTTPException(status_code=400, detail=f"Expected application/pdf but got ${request_file.content_type}")
    file_output = os.path.join(OUTPUT_DIRECTORY, request_file.filename)
    try:
        if not os.path.isfile(file_output):
            copy_contents_to_file(request_file, file_output)
    except Exception as e:
        return {"message": f"There was an error uploading the file:\n\n{e}\n\n"}
    finally:
        request_file.file.close()
    model = get_model('claude_sonnet')
    summary = await get_summary(file_output, model)
    return {"filename": request_file.filename, "summary": summary}

def main():
    setup_env()
