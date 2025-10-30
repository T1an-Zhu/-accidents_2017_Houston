import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.title("🗺️ 空间分布分析")

# 加载数据
df = pd.read_csv('./data/USAccidents_2017_Houston_cleaned.csv')
df_houston = df[df['City'].str.contains('Houston', case=False, na=False)]

# 创建 Folium 地图
m = folium.Map(location=[29.76, -95.36], zoom_start=10, tiles='CartoDB positron')
heat_data = [[row['Start_Lat'], row['Start_Lng'], row['Severity']] for _, row in df_houston.iterrows()]
HeatMap(heat_data, radius=10, blur=15).add_to(m)

# 渲染
st_folium(m, width=900, height=600)
