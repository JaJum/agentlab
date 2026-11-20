import os

from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 加载密钥，禁止把 API Key 硬编码进源码
load_dotenv()

def get_response():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    completion = client.chat.completions.create(
        model="qwen3.8-flash",  # 此处以qwen3.8-flash为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[{'role': 'system', 'content': "你是一个专业的 assistant."},
                  {'role': 'user', 'content': '你是谁？'}]
        )
    # json 数据
    #print(completion.model_dump_json())
    print(completion.choices[0].message.content)

if __name__ == '__main__':
    get_response()