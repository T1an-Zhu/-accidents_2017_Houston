import os
import matplotlib.pyplot as plt
import pandas as pd

def visualize_selected_weather_accident_data(df, output_dir="./visualization"):
    """
    可视化选定的气象预警与交通事故数据
    生成图表：
    1. 小时事故分布
    2. 星期事故分布
    3. 不同预警下事故数量
    4. 不同预警下事故平均距离
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #  解决中文和英文显示问题 
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    print("开始生成可视化图表...")


    #   不同预警下事故数量 & 平均距离 
    advisory_cols = [col for col in df.columns if col.startswith('advisory_')]
    if len(advisory_cols) > 0:
        # 每种预警下事故数量
        advisory_accident_counts = df[advisory_cols].sum().sort_values(ascending=False)
        plt.figure(figsize=(10,5))
        plt.bar(advisory_accident_counts.index, advisory_accident_counts.values, color=plt.cm.plasma(range(len(advisory_accident_counts))))
        plt.xlabel("Advisory Type")
        plt.ylabel("Number of Accidents")
        plt.title("Accidents by Advisory Type")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "advisory_accident_counts.png"))
        plt.close()

        # 每种预警下事故平均距离
        advisory_distance_mean = {}
        for col in advisory_cols:
            mean_distance = df.loc[df[col]==1, 'distance'].mean()
            advisory_distance_mean[col] = mean_distance
        advisory_distance_mean = pd.Series(advisory_distance_mean).sort_values(ascending=False)
        plt.figure(figsize=(10,5))
        plt.bar(advisory_distance_mean.index, advisory_distance_mean.values, color=plt.cm.inferno(range(len(advisory_distance_mean))))
        plt.xlabel("Advisory Type")
        plt.ylabel("Average Distance (mi)")
        plt.title("Average Distance by Advisory Type")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "advisory_distance_mean.png"))
        plt.close()

    print(f"可视化图表已保存到文件夹: {output_dir}")
