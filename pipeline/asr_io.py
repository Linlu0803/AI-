# pipeline/asr_io.py, 整理格式、添加时间戳标签、创建文件夹。

import json
from pathlib import Path
from datetime import datetime


def save_asr_result(
    asr_result: dict,
    output_path: str,
    source_url: str | None = None,
):
    """
    将 ASR 结果保存为 JSON 文件（中间数据契约）

    参数说明：
    - asr_result: 来自 transcribe_audio() 的返回结果
    - output_path: 保存路径，例如 temp/asr.json
    - source_url: 原始视频链接（用于追溯）
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ==============================
    # 1️⃣ 构造顶层 meta 信息
    # ==============================
    # 这里的 meta 是“系统级 meta”
    # 不等同于 asr_result 里的 meta（那是 ASR 自己的）
    data = {
        "meta": {
            "created_at": datetime.utcnow().isoformat(),
            "source": "bilibili" if source_url else None,
            "video_url": source_url,
            "asr_backend": asr_result.get("meta", {}).get("backend"),
            "model_size": asr_result.get("meta", {}).get("model_size"),
            "language": asr_result.get("meta", {}).get("language"),
        },

        # ==============================
        # 2️⃣ 全文文本（粗粒度）
        # ==============================
        # 用于：
        # - 快速预览
        # - 作为 summarizer 的全局上下文
        "text": asr_result.get("text", ""),

        # ==============================
        # 3️⃣ 带时间戳的 segments（核心资产）
        # ==============================
        # 每个 segment 加一个 id
        # 后面 OCR / 总结 / 标注 都会用到这个 id
        "segments": [],
    }

    for idx, seg in enumerate(asr_result.get("segments", [])):
        data["segments"].append({
            "id": idx,                 # 稳定引用用
            "start": seg["start"],     # 秒
            "end": seg["end"],         # 秒
            "text": seg["text"],       # 转写文本
        })

    # ==============================
    # 4️⃣ 写入 JSON 文件
    # ==============================
    # ensure_ascii=False：
    #   保证中文不被转成 \uXXXX
    # indent=2：
    #   方便你肉眼查看 / Debug
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)