import pandas as pd
import altair as alt

# 读取数据
df = pd.read_csv('./data/USAccidents_2017_Houston_cleaned.csv')
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

# 1.按月份统计
df['Month'] = df['Start_Time'].dt.month
monthly_counts = df.groupby('Month')['ID'].count().reset_index()
monthly_counts = monthly_counts[(monthly_counts['Month'] >= 1) & (monthly_counts['Month'] <= 12)]

chart_month = alt.Chart(monthly_counts).mark_bar(
    color='steelblue',
    cornerRadiusTopLeft=5,
    cornerRadiusTopRight=5
).encode(
    x=alt.X('Month:O', title='月份', axis=alt.Axis(labelFontSize=16, titleFontSize=18, labelAngle=0)),
    y=alt.Y('ID:Q', title='事故数量', axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
    tooltip=['Month', 'ID']
).properties(
    title=alt.TitleParams(text='Houston地区2017年事故数量随月份变化', fontSize=20),
    width=600,
    height=400
)
chart_month.save('./visualization/monthly_accidents_altair.html')


# 2.按小时统计
df['Hour'] = df['Start_Time'].dt.hour
hourly_counts = df.groupby('Hour')['ID'].count().reset_index()

chart_hour = alt.Chart(hourly_counts).mark_bar(
    color='orange',
    cornerRadiusTopLeft=5,
    cornerRadiusTopRight=5
).encode(
    x=alt.X('Hour:O', title='小时', axis=alt.Axis(labelFontSize=16, titleFontSize=18, labelAngle=0)),
    y=alt.Y('ID:Q', title='事故数量', axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
    tooltip=['Hour', 'ID']
).properties(
    title=alt.TitleParams(text='Houston地区2017年事故数量随小时变化', fontSize=20),
    width=600,
    height=400
)
chart_hour.save('./visualization/hourly_accidents_altair.html')


# 3.按星期统计
df['Weekday'] = df['Start_Time'].dt.weekday  # 0=周一, 6=周日
weekday_counts = df.groupby('Weekday')['ID'].count().reset_index()
weekday_counts['Weekday'] = weekday_counts['Weekday'].map(
    {0:'周一',1:'周二',2:'周三',3:'周四',4:'周五',5:'周六',6:'周日'}
)

chart_weekday = alt.Chart(weekday_counts).mark_bar(
    color='seagreen',
    cornerRadiusTopLeft=5,
    cornerRadiusTopRight=5
).encode(
    x=alt.X('Weekday:O', title='星期', sort=['周一','周二','周三','周四','周五','周六','周日'],
            axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
    y=alt.Y('ID:Q', title='事故数量', axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
    tooltip=['Weekday', 'ID']
).properties(
    title=alt.TitleParams(text='Houston地区2017年事故数量随星期变化', fontSize=20),
    width=600,
    height=400
)
chart_weekday.save('./visualization/weekday_accidents_altair.html')

print("三个 HTML 图表已生成：月份、小时、星期（文字已放大，图表尺寸不变）")
