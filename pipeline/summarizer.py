import openai
import os
from dotenv import load_dotenv

# 加载环境变量（保护你的 API Key 不被泄露）
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_summary(transcript_path, prompt_path):
    """
    读取转录文本，并调用 GPT-4o-mini 生成笔记
    """
    # 1. 读取 Whisper 生成的文本 (假设是 txt 或 json 中的 text 字段)
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return "错误：找不到转录文件，请检查 ASR 环节。"

    # 2. 读取你的“秘密武器”——Prompt 模板
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_instructions = f.read()

    # 3. 发送给 GPT-4o-mini 
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": f"这是视频的转录原文，请开始你的工作：\n\n{content}"}
            ],
            temperature=0.3, # 越低越严谨，不会瞎编
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用 AI 出错啦: {str(e)}"