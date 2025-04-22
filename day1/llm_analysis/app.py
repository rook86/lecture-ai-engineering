# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot & Data Analyzer", layout="wide")

# LLMモデルのロード（キャッシュを利用）
@st.cache_resource
def load_model():
    return llm.load_model()
pipe = load_model()

# --- Streamlit アプリ ---
st.title("🤖 Gemma 2 Chatbot & Data Analyzer")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
if 'page' not in st.session_state:
    st.session_state.page = "チャット"
page = st.sidebar.radio(
    "ページ選択",
    ["データ分析"],
    index=["データ分析"].index(st.session_state.page),
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector),
    key="page_selector"
)
st.sidebar.markdown("---")
st.sidebar.info("開発者: Your Name")

# --- 各ページ表示 ---
if page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("モデルの読み込みに失敗しました。")
elif page == "履歴閲覧":
    ui.display_history_page()
elif page == "サンプルデータ管理":
    ui.display_data_page()
elif page == "データ分析":
    if pipe:
        ui.display_data_analysis_page(pipe)
    else:
        st.error("モデルの読み込みに失敗しました。")
