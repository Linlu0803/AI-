# app.py


import os
import tempfile
from pathlib import Path
import streamlit as st
from pipeline.downloader import download_audio
from pipeline.audio import normalize_audio
from pipeline.asr_v2 import transcribe_audio  
from pipeline.asr_io import save_asr_result
from pipeline.summarizer import generate_summary 

# 终端要先运行虚拟环境.venv: source .venv/bin/activate

# 终端运行命令：streamlit run app.py



# ==============================
# 1. 页面基础配置
# ==============================
st.set_page_config(
    page_title="LearnFlow AI (MVP)",
    layout="centered",
    page_icon="🎧"
)

# ==============================
# 2. 初始化 Session State (核心：防止刷新丢失)
# ==============================
# 我们把需要持久化的数据存入 st.session_state
if "asr_result" not in st.session_state:
    st.session_state.asr_result = None
if "final_note" not in st.session_state:
    st.session_state.final_note = None
if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""

# ==============================
# 3. UI 标题区
# ==============================
st.title("🎧 LearnFlow AI")
st.write("把视频转成**可学习的文字笔记**")

# ==============================
# 4. 输入区
# ==============================
st.sidebar.header("输入设置")
input_mode = st.sidebar.radio("选择输入来源", ["视频链接 (URL)", "上传本地文件"])

source_path = None  # 最终交给 ASR 的路径

if input_mode == "视频链接 (URL)":
    url = st.text_input("🔗 粘贴视频链接 (ilibili)", placeholder="https://www.bilibili.com/video/...")
    is_ready = True if url else False

    # 如果用户输入了新 URL，清空之前的状态，防止看错
    if url != st.session_state.processed_url:
        st.session_state.asr_result = None
        st.session_state.final_note = None
else:
    uploaded_file = st.file_uploader("📂 上传音频/视频", type=["mp3", "m4a", "wav", "mp4"])
    url = None
    is_ready = True if uploaded_file else False



# ==============================
# 5. 执行按钮逻辑
# ==============================
if st.button("🚀 开始榨取知识", type="primary"):
    try:
        with st.status("🚀 正在全力处理中...", expanded=True) as status:
            # 1️⃣ 下载音频
            if input_mode == "视频链接 (URL)":
                status.write("⬇️ 正在从链接解析并下载音频...")
                # 这里调用你原来的 yt-dlp 下载逻辑
                audio_files = download_audio(url) 
                raw_audio_path = audio_files[0]
            else:
                status.write("💾 正在处理上传的文件...")
                # 将上传的文件流保存为临时文件，方便 FFmpeg 读取
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    raw_audio_path = tmp.name

            # 2️⃣ 标准化音频
            status.write("🎼 标准化音频...")
            # 建议：文件名可以带上 URL 哈希，防止多人冲突，这里先沿用你的 temp/audio.wav
            wav_audio = normalize_audio(raw_audio_path, "temp/audio.wav")

            # 3️⃣ ASR (这里调用的是带缓存的版本)
            status.write("🧠 语音转文字（ASR 高速版）...")
            asr_result = transcribe_audio(wav_audio)

            # 4️⃣ 保存结果 (保存到本地作为备份)
            status.write("💾 保存转录文本...")
            save_asr_result(
                asr_result,
                output_path="temp/asr.json",
                source_url=url,
            )

            # 5️⃣ AI 总结
            status.write("✨ 正在提取知识精华 (GPT-4o-mini)...")
            final_note = generate_summary(
                "temp/asr.json",
                "prompts/summary_cn.txt"
            )

            # 【关键：存入 State】
            st.session_state.asr_result = asr_result
            st.session_state.final_note = final_note
            st.session_state.processed_url = url

            status.update(label="✅ 知识榨取完成！", state="complete", expanded=False)

    except Exception as e:
        st.error(f"处理失败，错误信息: {e}")

# ==============================
# 6. 展示结果区 (放在按钮逻辑外)
# ==============================
# 只要 session_state 里有数据，无论怎么刷新页面，这里都会显示
if st.session_state.final_note and st.session_state.asr_result:
    st.divider() # 视觉分割线
    
    tab1, tab2 = st.tabs(["📝 深度学习笔记", "📄 转录原文"])
    
    with tab1:
        st.markdown(st.session_state.final_note)
        
        # 下载按钮：现在点击它，页面刷新后内容依然在！
        st.download_button(
            label="💾 保存笔记为 Markdown",
            data=st.session_state.final_note,
            file_name="learning_note.md",
            mime="text/markdown",
            key="download_btn" # 给个固定 key 也是好习惯
        )

    with tab2:
        st.text_area(
            "全文文本",
            st.session_state.asr_result["text"],
            height=400
        )
        st.info("提示：如果需要更精细的时间戳，可以查看 temp/asr.json")