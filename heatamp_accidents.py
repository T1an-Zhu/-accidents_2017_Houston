import pandas as pd
import folium
from folium.plugins import HeatMap

# 读取数据
df = pd.read_csv('./data/USAccidents_2017_Houston_cleaned.csv')

# 筛选休斯顿市范围数据
df_houston = df[df['City'].str.contains('Houston', case=False)]

# 创建地图中心点（休斯顿市中心经纬度）
houston_center = [29.76, -95.36]  # 可以根据实际数据调整

# 创建 Folium 地图
m = folium.Map(location=houston_center, zoom_start=10, tiles='CartoDB positron')

# 构建热力图数据 [[lat, lng, weight], ...]，可用Severity作为权重
heat_data = [[row['Start_Lat'], row['Start_Lng'], row['Severity']] for index, row in df_houston.iterrows()]

# 添加热力图
HeatMap(heat_data, radius=10, blur=15, max_zoom=12).add_to(m)

# 保存地图为 HTML
m.save('./visualization/houston_accident_heatmap.html')
m
