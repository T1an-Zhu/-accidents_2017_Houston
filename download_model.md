# 小型文本 Embedding 自监督微调程序使用说明

## 1. 项目简介

本程序用于对日级的天气事故文本数据进行自监督微调，并生成文本的向量表示（embedding）。  
使用 [SentenceTransformers](https://www.sbert.net/) 的 `all-mpnet-base-v2` 模型，通过对比学习 (MultipleNegativesRankingLoss) 对小时级文本进行自监督微调。

**主要功能：**

1. 按日聚合文本数据
2. 自监督数据增强（轻微随机扰动）
3. 微调 SentenceTransformer 模型
4. 生成微调后的文本向量
5. 保存 embedding 到本地文件

---

## 2. 环境依赖

### Python 版本
- Python 3.8 及以上

### 必要 Python 包
```bash
pip install pandas numpy tqdm
pip install torch torchvision torchaudio 
pip install sentence-transformers
pip install huggingface_hub
```
### 登录huggingface
https://huggingface.co/
- 完成登录、邮箱验证
- 点击 *setting* 选择 *create access token*
- 在终端登录huggingface
```bash
huggingface-cli login
```
- 将token粘贴到终端

**现在就可以运行embedding_train.py了**

