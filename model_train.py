import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# 配置路径
DATA_FILE = './data/weather_accident_merged_houston.csv'
TEXT_EMBED_FILE = './data/accident_text_embeddings_daily.npy'  # 日级文本 embedding
OUTPUT_MODEL_DAILY = './models/daily_accident_model.pkl'
OUTPUT_MODEL_HOURLY = './models/hourly_accident_model.pkl'
OUTPUT_SCALER_STRUCT = './models/struct_scaler_model1.pkl'  # 新增保存标准化器
OUTPUT_SCALER_TOTAL = './models/scaler_total_model2.pkl'     # 小时级模型总量标准化器
FIG_SAVE_PATH = './visualization/hourly_prediction_example.png'

# 1. 读取数据
df = pd.read_csv(DATA_FILE)
df['accident_date'] = pd.to_datetime(df['accident_date'].str.strip())

# 2. 聚合为按天数据（日级模型）
daily_df = df.groupby('accident_date').agg({
    'temperature':'mean',
    'visibility':'mean',
    'accident_id':'count'
}).reset_index()
daily_df.rename(columns={'accident_id':'accident_count'}, inplace=True)

# 添加时间特征
daily_df['month'] = daily_df['accident_date'].dt.month
daily_df['weekday'] = daily_df['accident_date'].dt.weekday

# 3. 加载日级文本 embedding
X_text = np.load(TEXT_EMBED_FILE)
if len(X_text) != len(daily_df):
    raise ValueError(f"Embedding数量 {len(X_text)} ≠ 天数 {len(daily_df)}")

# 4. 构建日级特征 + 标准化
X_struct = daily_df[['temperature','visibility','month','weekday']].values
scaler_struct = StandardScaler()
X_struct_scaled = scaler_struct.fit_transform(X_struct)
joblib.dump(scaler_struct, OUTPUT_SCALER_STRUCT)  # 保存标准化器

X_daily = np.hstack([X_text, X_struct_scaled])
y_daily = daily_df['accident_count'].values

X_train, X_test, y_train, y_test = train_test_split(X_daily, y_daily, test_size=0.2, random_state=42)

model_daily = RandomForestRegressor(n_estimators=200, random_state=42)
model_daily.fit(X_train, y_train)

y_pred = model_daily.predict(X_test)
print("\n==== 模型 1（日事故预测） ====")
print(f"R² = {r2_score(y_test, y_pred):.4f}")
print(f"MAE = {mean_absolute_error(y_test, y_pred):.2f}")

# 5. 构建小时级数据
hourly_df = df.groupby(['accident_date','accident_hour']).agg({'accident_id':'count'}).reset_index()
hourly_df.rename(columns={'accident_id':'hourly_count'}, inplace=True)

# 将日级总事故数和星期几加入
hourly_df = hourly_df.merge(daily_df[['accident_date','accident_count','weekday']], on='accident_date', how='left')

# 小时级特征标准化总事故数
scaler_total = StandardScaler()
hourly_df['accident_count_scaled'] = scaler_total.fit_transform(hourly_df[['accident_count']])
joblib.dump(scaler_total, OUTPUT_SCALER_TOTAL)  # 保存小时级总量标准化器

X_hour = hourly_df[['accident_count_scaled','weekday','accident_hour']].values
y_hour = hourly_df['hourly_count'].values

Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_hour, y_hour, test_size=0.2, random_state=42)

model_hourly = RandomForestRegressor(n_estimators=150, random_state=42)
model_hourly.fit(Xh_train, yh_train)

yh_pred = model_hourly.predict(Xh_test)
print("\n==== 模型 2（小时级分布预测） ====")
print(f"R² = {r2_score(yh_test, yh_pred):.4f}")
print(f"MAE = {mean_absolute_error(yh_test, yh_pred):.2f}")

# 6. 可视化一个样例日期的预测结果
unique_dates = hourly_df['accident_date'].dropna().unique()
example_date = np.random.choice(unique_dates)
example_date_ts = pd.Timestamp(example_date)
example_day = hourly_df[hourly_df['accident_date']==example_date].copy()
X_example = example_day[['accident_count_scaled','weekday','accident_hour']].values
example_day['predicted'] = model_hourly.predict(X_example)

plt.figure(figsize=(10,5))
plt.plot(example_day['accident_hour'], example_day['hourly_count'], label='真实值', marker='o')
plt.plot(example_day['accident_hour'], example_day['predicted'], label='预测值', marker='x')
plt.title(f"{example_date_ts.date()} 当天事故小时分布预测")
plt.xlabel("小时")
plt.ylabel("事故数量")
plt.legend()
plt.grid(True)

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)
plt.savefig(FIG_SAVE_PATH, dpi=200)
plt.show()
print(f"\n折线图已保存至: {FIG_SAVE_PATH}")

# 7. 保存模型
os.makedirs(os.path.dirname(OUTPUT_MODEL_DAILY), exist_ok=True)
joblib.dump(model_daily, OUTPUT_MODEL_DAILY)
joblib.dump(model_hourly, OUTPUT_MODEL_HOURLY)
print(f"\n模型已保存：\n - 日级模型: {OUTPUT_MODEL_DAILY}\n - 小时模型: {OUTPUT_MODEL_HOURLY}")
