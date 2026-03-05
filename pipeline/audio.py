# pipeline/audio.py

import subprocess
from pathlib import Path


def normalize_audio(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
) -> str:
    """
    音频标准化：
    - 转 wav
    - mono
    - 16kHz（默认）
    
    返回：标准化后的音频路径
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                    # 覆盖输出
        "-i", str(input_path),   # 输入
        "-ac", "1",              # mono
        "-ar", str(sample_rate), # 采样率
        "-vn",                   # 不要视频
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio normalization failed: {e}")

    return str(output_path)