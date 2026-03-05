import os
from pathlib import Path
from faster_whisper import WhisperModel

# 全局变量，用于模型单例缓存
_model = None

def _get_model(model_size: str = "small", device: str = "cpu"):
    """
    获取（或加载）Faster-Whisper 模型
    
    优化说明：
    - 针对云端 CPU 环境：强制使用 int8 量化，极致节省内存
    - 模型下载路径：指定缓存目录，避免权限问题
    """
    global _model

    if _model is None:
        # 使用 faster-whisper 的初始化方式
        # compute_type="int8" 是在 CPU 上运行的“保命插件”，内存占用减半
        _model = WhisperModel(
            model_size, 
            device=device, 
            compute_type="int8",
            download_root="./models" 
        )
    return _model


def transcribe_audio(
    audio_path: str,
    language: str = "zh",
    model_size: str = "small",  # 云端建议默认用 small 或 base
) -> dict:
    """
    使用【Faster-Whisper】进行 ASR
    """

    # 1️⃣ 输入校验
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 2️⃣ 获取模型 (强制 CPU 环境适配)
    model = _get_model(model_size=model_size, device="cpu")

    # 3️⃣ 调用 Faster-Whisper 转录
    # vad_filter=True: 自动过滤静音片段，极大提升 B 站长视频的处理速度
    # beam_size=5: 搜索宽度，5 是准确度与速度的平衡点
    segments_generator, info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        beam_size=5,
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # 4️⃣ 提取带时间戳的 segments
    # 注意：faster-whisper 返回的是生成器，需要迭代出来
    segments = []
    full_text_list = []
    
    for seg in segments_generator:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        full_text_list.append(seg.text.strip())

    # 5️⃣ 返回统一的“数据契约”
    return {
        "text": "".join(full_text_list),  # 中文建议直接拼接，不加空格
        "segments": segments,
        "meta": {
            "backend": "faster_whisper",
            "model_size": model_size,
            "language": language,
            "probability": round(info.language_probability, 2), # 额外记录语言置信度
            "duration": round(info.duration, 2)                 # 记录音频时长
        },
    }
