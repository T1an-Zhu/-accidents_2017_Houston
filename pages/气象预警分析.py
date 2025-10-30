import streamlit as st

st.title("⚠️ 气象预警关联分析")

image_files = [
    './visualization/advisory_accident_counts.png',
    './visualization/advisory_distance_mean.png'
]

for img in image_files:
    st.image(img, use_container_width=True)
