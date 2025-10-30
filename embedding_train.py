import pandas as pd
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from tqdm import tqdm

# 配置参数
MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'
INPUT_FILE = './data/weather_accident_merged_houston.csv'
OUTPUT_FILE = './data/accident_text_embeddings_daily.npy'
TEXT_COLUMN = 'day_one_text'
OUTPUT_MODEL_DIR = './models/fine_tuned_minilm_daily'
EPOCHS = 5
REPEAT_TIMES = 20
DROP_PROB = 0.1
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 读取数据并按天聚合，只保留有效天
df = pd.read_csv(INPUT_FILE)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
df['Date'] = pd.to_datetime(df['accident_date'].str.strip(), errors='coerce')
df = df.dropna(subset=['Date'])  # 删除非法日期行

# 只聚合有事故文本的天
daily_df = df.groupby('Date').agg({
    TEXT_COLUMN: lambda x: " ".join(x)
}).reset_index()

daily_df = daily_df[daily_df[TEXT_COLUMN].str.strip() != ""]
texts = daily_df[TEXT_COLUMN].tolist()
print(f"Day-level samples: {len(texts)}") 

# 2. 构建 SentenceTransformer
word_embedding_model = models.Transformer(MODEL_NAME)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=DEVICE)

# 3. 文本增强函数
def augment_text(text, drop_prob=DROP_PROB):
    words = text.split()
    if len(words) == 0:
        return text
    new_words = [w for w in words if random.random() > drop_prob]
    if len(new_words) == 0:
        new_words = words
    return " ".join(new_words)

# 4. 构造训练样本
train_examples = []
for t in texts:
    for _ in range(REPEAT_TIMES):
        view1 = augment_text(t)
        view2 = augment_text(t)
        train_examples.append(InputExample(texts=[view1, view2]))
print(f"Total training examples: {len(train_examples)}")

# 5. Dataset 和 DataLoader
class SBertDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx].texts

def collate_fn(batch):
    return batch

train_dataset = SBertDataset(train_examples)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 6. 定义 loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# 7. 训练模型
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for batch in tqdm(train_dataloader):
        texts1 = [b[0] for b in batch]
        texts2 = [b[1] for b in batch]

        sent_feat1 = model.tokenize(texts1)
        sent_feat2 = model.tokenize(texts2)

        sent_feat1 = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in sent_feat1.items()}
        sent_feat2 = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in sent_feat2.items()}

        sentence_features = [sent_feat1, sent_feat2]
        labels = torch.arange(len(texts1), dtype=torch.long, device=DEVICE)

        optimizer.zero_grad()
        loss_value = train_loss(sentence_features, labels)
        loss_value.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done.")

# 8. 保存微调模型
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
model.save(OUTPUT_MODEL_DIR)
print(f"微调模型已保存到 {OUTPUT_MODEL_DIR}")

# 9. 生成天级 embedding
model.eval()
all_embeddings = []
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, batch_size=BATCH_SIZE)
    all_embeddings.append(batch_embeddings)
all_embeddings = np.vstack(all_embeddings)

# 10. 保存 embedding
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
np.save(OUTPUT_FILE, all_embeddings)
print(f"已保存 {all_embeddings.shape[0]} 条天级 embedding 到 {OUTPUT_FILE}")
