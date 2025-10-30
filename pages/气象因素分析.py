import streamlit as st

st.title("🌦️ 气象因素分析")

image_files = [
    './visualization/tables/Visibilitymi_severity.png',
    './visualization/tables/TemperatureF_severity.png',
    './visualization/tables/Humidity%_severity.png',
    './visualization/tables/Wind_Speedmph_severity.png'
]

for img in image_files:
    st.image(img, use_container_width=True)
