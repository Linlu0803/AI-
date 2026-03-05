# pipeline/downloader.py
import yt_dlp
from pathlib import Path
from typing import List, Union


def download_audio(
    url: str,
    output_dir: str = "temp",
    allow_playlist: bool = False,
    playlist_items: Union[str, None] = None,
) -> List[str]:
    """
    下载音频（默认只下载一个视频）

    参数：
    - allow_playlist: 是否允许下载 playlist
    - playlist_items: 指定下载 playlist 的哪些条目，如 "2"、"1-3"

    返回：
    - 音频文件路径列表（即使只有一个，也用 list）
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_template = output_dir / "%(id)s.%(ext)s"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_template),
        "quiet": True,
        "noplaylist": not allow_playlist,
        
        # --- ✨ 新增伪装配置 ---
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
        "nocheckcertificate": True, # 忽略证书校验，提高通过率
        # ---------------------
        
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    # 只有你明确允许 playlist，才加这个
    if allow_playlist and playlist_items:
        ydl_opts["playlist_items"] = playlist_items

    audio_files = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # 如果是 playlist
            if "entries" in info:
                for entry in info["entries"]:
                    filename = ydl.prepare_filename(entry)
                    audio_files.append(
                        str(Path(filename).with_suffix(".mp3"))
                    )
            else:
                filename = ydl.prepare_filename(info)
                audio_files.append(
                    str(Path(filename).with_suffix(".mp3"))
                )

        return audio_files

    except Exception as e:
        raise RuntimeError(f"Audio download failed: {e}")
    
    