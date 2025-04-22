# llm.py
import os
import torch
from transformers import pipeline
from config import MODEL_NAME

@st.cache_resource
def load_model():
    """LLMモデルをロードして返す"""
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    return pipe

def analyze_dataframe(pipe, df):
    """
    DataFrameの簡単な要約とサンプルをプロンプトに組み込み、
    LLMに分析を実行させる。
    """
    # カラム一覧とサンプル数を取得
    cols = df.columns.tolist()
    sample = df.head(5).to_dict(orient="records")
    prompt = (
        f"以下のテーブルデータについて分析してください。\n"
        f"1) カラム一覧: {cols}\n"
        f"2) 最初の5行のサンプル: {sample}\n"
        "・データの傾向や特徴量の重要性を述べてください。\n"
        "・異常値や欠損値があれば指摘してください。\n"
        "・必要に応じて追加の前処理や可視化の提案をしてください。\n"
    )
    # 生成
    result = pipe(prompt, max_length=512, do_sample=False)[0]["generated_text"]
    return result.replace(prompt, "").strip()
