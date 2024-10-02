import os
import shutil
from typing import Literal

from langchain_community.chat_models import ChatAnthropic

from configuration import setup_env
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()
OUTPUT_DIRECTORY = "outputs"
setup_env()
if not os.path.isdir(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)


async def get_summary(file_output: str) -> str:

    pass


def get_model(model_name: Literal['claude_sonnet']):
    model = None
    if model_name == 'claude_sonnet':
        return ChatAnthropic(model='claude-3-5-sonnet-20240620')
    elif model_name == 'llama3.2':
        


@app.post("/uploadfile/")
async def upload_file_summarize(request_file: UploadFile = File(...)):
    # if content_type != "application/pdf":
    #     raise HTTPException(status_code=400, detail=f"Expected application/pdf but got ${request_file.content_type}")
    file_output = os.path.join(OUTPUT_DIRECTORY, request_file.filename)
    try:
        if not os.path.isfile(file_output):
            with open(file_output, mode="wb") as f:
                shutil.copyfileobj(request_file.file, f)
        else:
            raise HTTPException(status_code=400, detail="file already exists")
    except Exception as e:
        return {"message": f"There was an error uploading the file:\n\n{e}\n\n"}
    finally:
        request_file.file.close()
    model = get_model('claude_sonnet')
    summary = await get_summary(file_output, model)
    return {"filename": request_file.filename, "summary": summary}

def main():
    setup_env()
