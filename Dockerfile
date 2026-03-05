FROM python:3.12-slim

# 1. 安装系统级依赖 (ffmpeg 是必须的)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. 升级 pip 并直接安装核心包（这样最保险）
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir streamlit yt-dlp openai-whisper openai faster-whisper pydub moviepy

# 3. 复制剩余文件
COPY . .

# 4. 暴露端口
EXPOSE 8501

# 5. 启动
ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]