# 安装依赖：pip install dash plotly pandas

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import altair as alt
import folium
from folium.plugins import HeatMap
import flask
import os

# 1️⃣ 数据准备
df = pd.read_csv('./data/USAccidents_2017_Houston_cleaned.csv')
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Month'] = df['Start_Time'].dt.month
df['Hour'] = df['Start_Time'].dt.hour
df['Weekday'] = df['Start_Time'].dt.weekday  # 0=周一
df['Weekday_Name'] = df['Weekday'].map({0:'周一',1:'周二',2:'周三',3:'周四',4:'周五',5:'周六',6:'周日'})

# 2️⃣ 初始化 Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# 注册 Flask 静态目录，让 /visualization/ 指向本地 ./visualization/
visualization_path = os.path.join(os.getcwd(), "visualization")

@server.route("/visualization/<path:filename>")
def serve_visualization(filename):
    return flask.send_from_directory(visualization_path, filename)

# ---------------------- 一级页面布局 ----------------------
app.layout = html.Div([
    html.H1("Houston交通事故交互式分析大屏", style={'textAlign':'center'}),

    html.Div([
        html.Button("时间趋势分析", id='btn_time', n_clicks=0),
        html.Button("空间分布分析", id='btn_space', n_clicks=0),
        html.Button("气象因素分析", id='btn_weather', n_clicks=0),
        html.Button("气象预警关联分析", id='btn_alert', n_clicks=0),
    ], style={'display':'flex', 'justifyContent':'space-around', 'margin':'20px'}),

    html.Div(id='page-content')  # 二级页面内容
])

# ---------------------- 二级页面回调 ----------------------
@app.callback(
    Output('page-content', 'children'),
    [Input('btn_time', 'n_clicks'),
     Input('btn_space', 'n_clicks'),
     Input('btn_weather', 'n_clicks'),
     Input('btn_alert', 'n_clicks')]
)
def display_page(btn_time, btn_space, btn_weather, btn_alert):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div("请选择板块查看详细内容")
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

# ---------------------- 时间趋势分析 ----------------------
    if button_id == 'btn_time':
        # 1. 按月份统计
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
            title=alt.TitleParams(text='Houston地区事故数量随月份变化', fontSize=20),
            width=480,
            height=280
        )

        # 2. 按小时统计
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
            title=alt.TitleParams(text='Houston地区事故数量随小时变化', fontSize=20),
            width=480,
            height=280
        )

        # 3. 按星期统计
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
            title=alt.TitleParams(text='Houston地区事故数量随星期变化', fontSize=20),
            width=480,
            height=280
        )

        # 创建自定义HTML内容，强制禁用滚动
        def create_chart_html(chart):
            base_html = chart.to_html()
            # 在生成的HTML中插入CSS来禁用滚动
            styled_html = base_html.replace(
                '<head>',
                '''<head>
                <style>
                    body { margin: 0; padding: 0; overflow: hidden !important; }
                    #altair-chart { overflow: hidden !important; }
                    .vega-embed { overflow: hidden !important; }
                    .vega-embed-wrapper { overflow: hidden !important; }
                </style>
                '''
            )
            return styled_html

        return html.Div([
            html.H2("时间趋势分析", style={'textAlign': 'center', 'margin': '20px 0'}),
            
            html.Div([
                html.H3("月份分布", style={'textAlign': 'center', 'margin': '10px 0'}),
                html.Iframe(
                    srcDoc=create_chart_html(chart_month),
                    style={
                        'border': 'none',
                        'display': 'block',
                        'margin': '0 auto',
                        'overflow': 'hidden',
                        'width': '600px',
                        'height': '400px',
                        'scrolling': 'no'
                    }
                )
            ], style={'margin': '30px 0', 'overflow': 'hidden'}),
            
            html.Div([
                html.H3("小时分布", style={'textAlign': 'center', 'margin': '10px 0'}),
                html.Iframe(
                    srcDoc=create_chart_html(chart_hour),
                    style={
                        'border': 'none',
                        'display': 'block',
                        'margin': '0 auto',
                        'overflow': 'hidden',
                        'width': '600px',
                        'height': '400px',
                        'scrolling': 'no'
                    }
                )
            ], style={'margin': '30px 0', 'overflow': 'hidden'}),
            
            html.Div([
                html.H3("星期分布", style={'textAlign': 'center', 'margin': '10px 0'}),
                html.Iframe(
                    srcDoc=create_chart_html(chart_weekday),
                    style={
                        'border': 'none',
                        'display': 'block',
                        'margin': '0 auto',
                        'overflow': 'hidden',
                        'width': '600px',
                        'height': '400px',
                        'scrolling': 'no'
                    }
                )
            ], style={'margin': '30px 0', 'overflow': 'hidden'})
        ], style={'overflow': 'hidden', 'padding': '20px', 'height': '100%', 'minHeight': '100vh'})

    # ---------------------- 空间分布分析 ----------------------
    elif button_id == 'btn_space':
        # 筛选休斯顿市范围数据
        df_houston = df[df['City'].str.contains('Houston', case=False, na=False)]
        
        # 创建地图中心点（休斯顿市中心经纬度）
        houston_center = [29.76, -95.36]
        
        # 创建 Folium 地图
        m = folium.Map(location=houston_center, zoom_start=10, tiles='CartoDB positron')
        
        # 构建热力图数据 [[lat, lng, weight], ...]，可用Severity作为权重
        heat_data = [[row['Start_Lat'], row['Start_Lng'], row['Severity']] for index, row in df_houston.iterrows()]
        
        # 添加热力图
        HeatMap(heat_data, radius=10, blur=15, max_zoom=12).add_to(m)
        
        # 直接获取地图的HTML内容
        map_html = m._repr_html_()
        
        return html.Div([
            html.H2("空间分布分析", style={'textAlign': 'center', 'margin': '20px 0'}),
            html.Iframe(
                srcDoc=map_html,
                style={
                    'width': '100%',
                    'height': 'calc(100vh - 100px)',  # 自适应高度
                    'border': 'none',
                    'display': 'block',
                    'overflow': 'hidden'
                }
            )
        ], style={'overflow': 'hidden', 'padding': '20px', 'height': '100vh'})

    
    # ---------------------- 气象因素分析 ----------------------
    elif button_id == 'btn_weather':
        # 图片文件列表
        image_files = [
            'tables/Visibilitymi_severity.png',
            'tables/TemperatureF_severity.png', 
            'tables/Humidity%_severity.png',
            'tables/Wind_Speedmph_severity.png'
        ]
        
        # 创建图片显示组件
        image_components = []
        for img_file in image_files:
            image_components.extend([
                html.Img(src=f"/visualization/{img_file}", 
                        style={'width': '80%', 'display': 'block', 'margin': '20px auto'}),
            ])
        
        return html.Div([
            html.H2("气象因素分析", style={'textAlign': 'center', 'margin': '20px 0'}),
            *image_components
        ], style={'overflow': 'auto', 'padding': '20px', 'height': '100vh'})

    # ---------------------- 气象预警关联分析 ----------------------
    elif button_id == 'btn_alert':
        # 图片文件列表
        image_files = [
            'advisory_accident_counts.png',
            'advisory_distance_mean.png'
        ]
        
        image_components = [
            html.Img(
                src=f"/visualization/{img_file}",
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            )
            for img_file in image_files
        ]
        
        return html.Div([
            html.H2("气象预警关联分析", style={'textAlign': 'center', 'margin': '20px 0'}),
            html.Div(image_components, style={
                'height': 'calc(100vh - 100px)',
                'overflowY': 'auto',   # 允许滚动
                'padding': '20px'
            })
        ])


# ---------------------- 启动服务器 ----------------------
if __name__ == '__main__':
    app.run(debug=True)
