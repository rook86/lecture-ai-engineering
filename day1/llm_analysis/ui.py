# ui.py
import streamlit as st
import pandas as pd
import llm

def display_chat_page(pipe):
    st.header("チャット")
    # 既存のチャットUI呼び出し
    # （省略）

def display_history_page():
    st.header("履歴閲覧")
    # 既存の履歴閲覧UI
    # （省略）

def display_data_page():
    st.header("サンプルデータ管理")
    # 既存のサンプルデータ管理UI
    # （省略）

def display_data_analysis_page(pipe):
    st.header("データ分析")
    st.write("CSVまたはExcelをアップロードして、LLMに分析させます。")
    uploaded = st.file_uploader(
        "ファイルを選択してください（.csv, .xls, .xlsx）",
        type=["csv", "xls", "xlsx"]
    )
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {e}")
            return

        st.subheader("データプレビュー")
        st.dataframe(df.head())

        if st.button("LLMで分析する"):
            with st.spinner("分析中…"):
                analysis = llm.analyze_dataframe(pipe, df)
            st.subheader("分析結果")
            st.write(analysis)
