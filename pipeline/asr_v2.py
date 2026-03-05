# pipeline/asr_v2.py
import streamlit as st
from faster_whisper import WhisperModel
import torch
from pathlib import Path

# ==============================
# 模型加载：使用 st.cache_resource
# ==============================
# 这个装饰器确保模型只会被加载到内存一次，所有用户共享这一个模型
@st.cache_resource
def load_whisper_model(model_size="medium"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU 环境下 int8 最快，GPU 环境下 float16 最快
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"🚀 正在加载模型: {model_size} ({device})...")
    model = WhisperModel(
        model_size, 
        device=device, 
        compute_type=compute_type,
        download_root="./models" # 将模型下载到本地目录
    )
    return model

# ==============================
# 转录逻辑：使用 st.cache_data
# ==============================
# 这个装饰器确保：只要同一个音频文件没变，直接返回上次的结果，不再重复转录
@st.cache_data(show_spinner=False)
def transcribe_audio(audio_path: str, model_size="small"):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return None

    # 调用缓存好的模型实例
    model = load_whisper_model(model_size)

    # 这里的参数是提速关键
    segments_generator, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,  # 核心提速：跳过静音，减少模型压力
        vad_parameters=dict(min_silence_duration_ms=500),   
    )

    segments = []
    full_text = []
    for seg in segments_generator:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        full_text.append(seg.text.strip())

    return {
        "text": "".join(full_text),
        "segments": segments
    }