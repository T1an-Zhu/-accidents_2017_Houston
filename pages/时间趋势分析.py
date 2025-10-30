import streamlit as st
import pandas as pd
import altair as alt

st.title("📊 时间趋势分析")

# 加载数据
df = pd.read_csv('./data/USAccidents_2017_Houston_cleaned.csv')
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Month'] = df['Start_Time'].dt.month
df['Hour'] = df['Start_Time'].dt.hour
df['Weekday'] = df['Start_Time'].dt.weekday
df['Weekday_Name'] = df['Weekday'].map({0:'周一',1:'周二',2:'周三',3:'周四',4:'周五',5:'周六',6:'周日'})

# 月份分布
monthly_counts = df.groupby('Month')['ID'].count().reset_index()
chart_month = alt.Chart(monthly_counts).mark_bar(color='steelblue').encode(
    x=alt.X('Month:O', 
            title='月份',
            axis=alt.Axis(labelFontSize=16, titleFontSize=18, labelAngle=0)),  # <- labelAngle=0
    y='ID:Q',
    tooltip=['Month','ID']
).properties(width=600, height=300, title="月份事故数量分布")
st.altair_chart(chart_month, use_container_width=True)

# 小时分布
hourly_counts = df.groupby('Hour')['ID'].count().reset_index()
chart_hour = alt.Chart(hourly_counts).mark_bar(color='orange').encode(
    x=alt.X('Hour:O',
            title='小时',
            axis=alt.Axis(labelFontSize=16, titleFontSize=18, labelAngle=0)),
    y='ID:Q',
    tooltip=['Hour','ID']
).properties(width=600, height=300, title="小时事故数量分布")
st.altair_chart(chart_hour, use_container_width=True)

# 星期分布
weekday_counts = df.groupby('Weekday')['ID'].count().reset_index()
weekday_counts['Weekday'] = weekday_counts['Weekday'].map({0:'周一',1:'周二',2:'周三',3:'周四',4:'周五',5:'周六',6:'周日'})
chart_weekday = alt.Chart(weekday_counts).mark_bar(color='seagreen').encode(
    x=alt.X('Weekday:O',
            title='星期',
            sort=['周一','周二','周三','周四','周五','周六','周日'],  # <- 指定顺序
            axis=alt.Axis(labelFontSize=16, titleFontSize=18, labelAngle=0)),
    y='ID:Q',
    tooltip=['Weekday','ID']
).properties(width=600, height=300, title="星期事故数量分布")
st.altair_chart(chart_weekday, use_container_width=True)
