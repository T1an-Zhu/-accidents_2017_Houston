import pandas as pd
import os

# 1. 读取原始数据集
input_file = "./data/base/US_Accidents_March23.csv"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"未找到文件：{input_file}")

print("读取原始数据集...")
df = pd.read_csv(input_file, low_memory=False)

# 2. 选择实验所需字段
selected_columns = [
    'ID', 'Severity',
    'Start_Time', 'End_Time', 'Distance(mi)',
    'Start_Lat', 'Start_Lng', 'City','County',
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition'
]

missing_cols = [col for col in selected_columns if col not in df.columns]
if missing_cols:
    print("以下字段在原始数据中未找到：", missing_cols)
else:
    print("所有所需字段均存在。")

df_subset = df[selected_columns].copy()

# 3. 转换时间格式
df_subset['Start_Time'] = pd.to_datetime(df_subset['Start_Time'], errors='coerce')
df_subset['End_Time'] = pd.to_datetime(df_subset['End_Time'], errors='coerce')

# 4. 删除 Start/End 时间缺失行
df_subset = df_subset.dropna(subset=['Start_Time', 'End_Time'])
print(f"\n删除时间列缺失行后，数据行数：{len(df_subset):,}")

# 5. 过滤 2017 年全年数据
df_2017 = df_subset[df_subset['Start_Time'].dt.year == 2017].copy()
print(f"\n筛选 2017 年全年数据后，数据行数：{len(df_2017):,}")

# 6. 缺失值处理
# 数值列：用中位数填充
numeric_median_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)']
for col in numeric_median_cols:
    if col in df_2017.columns:
        df_2017[col] = df_2017[col].fillna(df_2017[col].median())

# 风速和降水：空值填 0
for col in ['Wind_Speed(mph)', 'Precipitation(in)']:
    if col in df_2017.columns:
        df_2017[col] = df_2017[col].fillna(0)

# Weather_Condition：空值填众数
if 'Weather_Condition' in df_2017.columns:
    df_2017['Weather_Condition'] = df_2017['Weather_Condition'].fillna(df_2017['Weather_Condition'].mode()[0])

# 7. 按城市统计事故数量
city_counts = df_2017['City'].value_counts().reset_index()
city_counts.columns = ['City', 'Accident_Count']

top_city = city_counts.iloc[0]['City']
top_count = city_counts.iloc[0]['Accident_Count']
print(f"\n2017 年事故最多的城市是：{top_city}（事故数：{top_count:,}）")

# 8. 提取该城市的所有事故记录
df_top_city = df_2017[df_2017['City'] == top_city].copy()

# 9. 保存清理后的数据 
output_dir = "./data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, f"USAccidents_2017_{top_city.replace(' ', '_')}_cleaned.csv")
df_top_city.to_csv(output_file, index=False)
print(f"\n数据清理完成（2017年），已保存为：{output_file}")
print(f"{top_city} 事故记录数：{len(df_top_city):,}")
