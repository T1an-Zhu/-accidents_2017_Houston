# 基于多源数据融合的城市交通事故动态分析与可视化研究——以休斯顿为例

## 项目背景
本项目旨在通过对城市交通事故数据与气象数据的融合分析，探索城市交通事故的动态特征及其与环境因素的关系。研究以休斯顿市2017年的交通事故为案例，结合时间、地点、事故严重程度及天气预警信息，开展多维度分析与可视化。

项目符合课程“城市动态系统多源数据融合分析与应用”的要求，突出开放性和探索性，鼓励通过多源数据融合方法解决实际城市问题。

## 数据来源
- **交通事故数据**：US Accidents 数据集，包含事故时间、地点、事故严重程度、距离等信息。
- **气象数据**：事故发生时的温度、湿度、能见度、风速、降水等。
- **天气预警数据**：各种气象预警信息，用于分析不同预警下事故数量及严重程度。

## 数据处理
1. 筛选2017年的数据。
2. 清理缺失值：
   - 时间列缺失行删除。
   - 数值列使用中位数填充。
   - 风速、降水缺失值填0。
   - 天气条件使用众数填充。
3. 提取所需字段：
   - 时间、地点、事故严重程度、事故距离、气象特征、天气预警信息。
4. 生成衍生字段：
   - `Start_Time`（事故发生完整时间）
   - `Month`、`Hour`、`Weekday`
   - `advisory_count`（预警数量统计）

## 分析与可视化
项目主要分析包括：

1. **事故时间分布**
   - 小时事故分布
   - 星期事故分布（横坐标1-7表示周一至周日）
   - 月份事故分布

2. **事故与预警关系**
   - 不同预警下事故数量
   - 不同预警下事故平均距离

3. **事故与环境因素关系**
   - 能见度、温度、湿度、风速等分组事故严重度占比表格

## 可视化工具
- 使用 **Matplotlib**、**Seaborn** 生成静态柱状图、折线图、表格图。
- 支持中文显示，并调整图表美观度（字体、行高、列宽等）。

## 文件说明
| 文件名 | 说明 |
|--------|------|
| `data` | 存储数据文件（包含数据源文件和处理后的文件）|
| `data/base` | 存储数据源文件|
| `visualization` | 存储生成的可视化图片（包括png & html格式）|
| `visualization/tables` | 存储生成的可视化表格（png格式）|
| `models` | 存储模型和标准化器 |
| `pages` | 存储网站代码文件 |
| `data_pp.py` | 结构化数据预处理代码 |
| `text_pp.py` | 非结构化数据（气候预警信息）预处理代码 |
| `fusion.py` | 融合数据处理代码 |
| `fusion_visualisation.py` | 融合数据可视化代码（基于fusion.py运行） |
| `heatamp_accidents.py` | 事故热力图生成代码 |
| `time_accidents.py` | 事故时间分布图生成代码（小时、星期、月份） |
| `x_severity_table.py` | 多因素事故严重程度占比表格生成代码 |
| `embedding_train.py` | 日级embedding训练与微调代码 |
| `model_train.py` | 两层随机森林模型模型代码 |
| `home.py` | 可视化网站（streamlit实现） |
| `requirements.txt` | 部署依赖包说明 |
| `download_model.md` | embedding_train.py使用说明 |
| `README.md` | 项目说明文档（本文件） |

## 使用方法

***重要说明***：为了保证轻量化，本项目在NLP部分使用了all-MiniLM-L12-v2模型，已部署并微调完整，因此**不需要**重复==embedding_train.py==程序。关于该程序的使用说明，详情请阅读==download_model.md==。


1. 安装依赖：
```bash
pip install pandas os matplotlib numpy re folium altair tabulate json streamlit huggingface_hub requests
```
2. 具体说明：
- 复现时，理论上来说仅需保留./data/base/目录下的文件，其余非代码文件可全部清除，但不建议删除./models/fine_tuned_minilm_daily并重新部署微调模型
**注：若使用已微调好的参数（不运行embedding_train.py），则需保留./data/accident_text_embeddings_daily.npy文件和model/fine_tuned_minilm_daily/文件夹**；
- 1）运行 ==data_pp.py== 和 ==text_pp.py== 对数据进行预处理
- 2）运行 ==fusion.py== , ==heatamp_accidents.py== , ==monthly_accidents.py== , ==x_severity_table.py== 得到analysis文件和可视化图表
- 3）运行 ==embedding_train.py== 得到微调后日级 embedding **（可选）**
- 4）运行 ==model_train.py== 得到两层随机森林模型
- 5）将models文件夹上传到Hugging Face（公开）；将==home.py==，==data==、==pages==、==visualization==文件夹上传到GitHub（data/base可不上传）;登录Streamlit Cloud 平台（https://authkit.streamlit.io/）采用Deploy from GitHub完成部署。
https://2017-houston-traffic-analysis-zytzzn.streamlit.app/ 