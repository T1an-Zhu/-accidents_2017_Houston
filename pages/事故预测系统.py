import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sentence_transformers import SentenceTransformer
import os
import requests
import zipfile

st.title("🚗 城市当天小时级事故预测")

st.markdown("输入天气、前一天或当天预警信息，预测24小时事故数量折线图。")

# 用户输入
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("温度 (°F)", -10.0, 110.0, 68.0, 0.5)
    visibility = st.slider("能见度 (km)", 0.0, 10.0, 5.0, 0.1)
with col2:
    month = st.selectbox("月份", list(range(1,13)), index=5)
    weekday = st.selectbox("星期几 (1=周一, 7=周日)", list(range(1,8)), index=2)

alert_text = st.text_area("🌧️ 预警文本信息", "Heat index values are expected to climb into the 100-107 range late morning through this evening.")

# 加载模型
MODEL_FILES = {
    "daily_model": {
        "url": "https://huggingface.co/T1anyz/accidents_models/resolve/main/models/daily_accident_model.pkl",
        "local_path": "./models/daily_accident_model.pkl"
    },
    "hourly_model": {
        "url": "https://huggingface.co/T1anyz/accidents_models/resolve/main/models/hourly_accident_model.pkl",
        "local_path": "./models/hourly_accident_model.pkl"
    },
    "scaler_struct": {
        "url": "https://huggingface.co/T1anyz/accidents_models/resolve/main/models/struct_scaler_model1.pkl",
        "local_path": "./models/struct_scaler_model1.pkl"
    },
    "scaler_total": {
        "url": "https://huggingface.co/T1anyz/accidents_models/resolve/main/models/scaler_total_model2.pkl",
        "local_path": "./models/scaler_total_model2.pkl"
    },
    "embed_model": {
        "url": "https://huggingface.co/T1anyz/accidents_models/resolve/main/models/fine_tuned_minilm_daily.zip",
        "local_path": "./models/fine_tuned_minilm_daily"
    }
}

os.makedirs("./models", exist_ok=True)


def download_if_not_exists(url, local_path): 
    if url.endswith(".zip"): # 下载并解压 zip 文件 
        import zipfile 
        r = requests.get(url, stream=True) 
        zip_path = local_path + ".zip" 
        with open(zip_path, "wb") as f: 
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk) 
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: 
            zip_ref.extractall(local_path) 
        os.remove(zip_path) 
    else: # 直接下载 pkl 文件 
        r = requests.get(url, stream=True)
        with open(local_path, "wb") as f: 
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk) 


# 下载所有文件
for key, val in MODEL_FILES.items():
    download_if_not_exists(val["url"], val["local_path"])



model1 = joblib.load('./models/daily_accident_model.pkl')
model2 = joblib.load('./models/hourly_accident_model.pkl')
scaler_struct = joblib.load('./models/struct_scaler_model1.pkl')
scaler_total = joblib.load('./models/scaler_total_model2.pkl')
embed_model = SentenceTransformer('./models/fine_tuned_minilm_daily/fine_tuned_minilm_daily')


# 预测逻辑
if st.button("开始预测 🚀"):
    text_emb = embed_model.encode([alert_text])[0]
    X_daily_input = np.concatenate([text_emb, [temperature, visibility, month, weekday]])
    X_daily_input[-4:] = scaler_struct.transform([X_daily_input[-4:]])[0]
    X_daily_input = X_daily_input.reshape(1,-1)
    total_pred = model1.predict(X_daily_input)[0]

    # 小时分布
    hours = np.arange(24)
    X_hour_input = np.array([[total_pred, weekday, h] for h in hours])
    y_hour_pred = model2.predict(X_hour_input)
    y_hour_pred = np.clip(y_hour_pred, 0, None)

    hour_ratios = y_hour_pred / y_hour_pred.sum()
    y_hour_pred = hour_ratios * total_pred

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hours, y_hour_pred, marker='o', color='#4A90E2')
    for x, y in zip(hours, y_hour_pred):
        ax.text(x, y + 0.3, f"{y:.0f}", ha='center', va='bottom', fontsize=9, color='red')
    ax.set_title("Predicted Hourly Accidents")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Accident Count")

    ax.grid(True)
    st.pyplot(fig)

    st.success("预测完成！")
    st.markdown(f"**当天预测事故总数：** {total_pred:.1f}")
    st.markdown(f"**最高事故时段：** {np.argmax(y_hour_pred)} 时")
