import asyncio
import io
import base64

import fitz
from PIL import Image
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from configuration import setup_env


async def main() -> str:
    # if is_multi_modal(model):
    #     return await handle_multi_modal(file_path, model)
    setup_env()
    template = [
        ('user', '{contents}')
    ]
    model = ChatAnthropic(model='claude-3-5-sonnet-20240620')
    prompt = ChatPromptTemplate(template)
    chain = prompt | model | StrOutputParser()
    result = await chain.ainvoke({'contents': "hi, how are you" })
    print(result)

if __name__ == '__main__':
    asyncio.run(main())
