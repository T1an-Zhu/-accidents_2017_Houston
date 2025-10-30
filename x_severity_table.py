import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取数据
df = pd.read_csv('./data/USAccidents_2017_Houston_cleaned.csv')
df['Severity'] = df['Severity'].astype(int)

output_dir = './visualization/tables'
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_table(feature, bins, labels, title, output_file):
    # 去掉重复边界
    unique_bins = sorted(list(set(bins)))
    if len(unique_bins) != len(bins):
        bins = unique_bins
    
    # 划分区间
    df[f'{feature}_bin'] = pd.cut(df[feature], bins=bins, labels=labels[:len(bins)-1], include_lowest=True, duplicates='drop')
    
    # 生成计数表格
    count_table = df.groupby(['Severity', f'{feature}_bin'], observed=True)['ID'].count().unstack(fill_value=0)
    
    # 总数
    total_counts = df.groupby(f'{feature}_bin', observed=True)['ID'].count()
    total_counts_str = [f"{label}\n(N={total_counts[label]})" for label in labels[:len(bins)-1]]
    count_table.columns = total_counts_str

    # 转换为比例
    proportion_table = count_table.div(count_table.sum(axis=0), axis=1).round(2)

    # 在左侧增加 Severity 列
    cell_text = []
    row_labels = proportion_table.index.tolist()
    for sev, row in zip(row_labels, proportion_table.values):
        cell_text.append([sev] + list(row))
    
    col_labels = ['Severity'] + proportion_table.columns.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='center',
        colLoc='center',
        loc='center',
        colWidths=[0.08] + [0.92/len(proportion_table.columns)]*len(proportion_table.columns)
    )

    # 调整字体和行高
    tbl.auto_set_font_size(False)   # 关闭自动字体大小
    tbl.set_fontsize(24)            # 设置表格字体大小，原来是18，调大到24
    tbl.scale(1, 4)                 # 放大单元格，横向2倍，纵向4倍

    plt.title(title, fontsize=40, weight='bold')  # 标题字体调大到20
        
    # 单独调左上角 Severity 字体
    tbl[(0, 0)].set_fontsize(16)  

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"已保存表格图片: {output_file}")

# 配置要生成的表格
features_info = [
    ('Visibility(mi)', [0,2,4,6,8,10,df['Visibility(mi)'].max()], ['0-2','2-4','4-6','6-8','8-10','10+'], '能见度 vs 事故严重度占比'),
    ('Temperature(F)', [0,40,60,80,100], ['0-40°F','40-60°F','60-80°F','80-100°F'], '温度 vs 事故严重度占比'),
    ('Humidity(%)', [0,30,50,70,100], ['0-30%','30-50%','50-70%','70-100%'], '湿度 vs 事故严重度占比'),
    ('Wind_Speed(mph)', [0,5,10,20,50], ['0-5 mph','5-10 mph','10-20 mph','20-50 mph'], '风速 vs 事故严重度占比')
]

for feature, bins, labels, title in features_info:
    output_file = os.path.join(output_dir, f"{feature.replace('(', '').replace(')', '').replace(' ', '_')}_severity.png")
    generate_table(feature, bins, labels, title, output_file)
