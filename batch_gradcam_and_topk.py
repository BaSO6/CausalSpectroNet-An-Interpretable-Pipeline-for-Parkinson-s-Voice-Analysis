# batch_token_topk.py
# 用于根据 TokenCNN 模型得分排序上的 top-k PD/HC 样本，用来描述 token 分布及 LLM 描述之间的关联

import os
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from token_model_training import TokenCNNClassifier as TokenCNN  # ✅ 使用训练时的模型结构

# === 配置 ===
CSV_PATH = "audio_vq_multimodal_with_llm.csv"
MODEL_PATH = "token_cnn_model.pth"
TOPK = 5
SEQ_LEN = 100
NUM_TOKENS = 30

# === 工具函数 ===
def tokenize(row):
    tokens = eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
    x = torch.tensor(tokens[:SEQ_LEN], dtype=torch.long)
    if len(x) < SEQ_LEN:
        pad = torch.zeros(SEQ_LEN - len(x), dtype=torch.long)
        x = torch.cat([x, pad])
    return x

def plot_topk(scores, title):
    paths = [os.path.basename(r["path"]) for r in scores]
    values = [r["score"] for r in scores]
    plt.figure(figsize=(10, 3))
    plt.barh(paths, values, color="tomato" if title.startswith("PD") else "steelblue")
    plt.xlabel("Model Confidence (PD)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fname = f"topk_{title.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=300)
    print(f"✅ Saved plot: {fname}")

# === 主逻辑 ===
def run_topk_token_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenCNN().to(device)  # ✅ 使用训练时一致的 TokenCNNClassifier
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    df = pd.read_csv(CSV_PATH)
    records = df.to_dict("records")

    results = []
    for row in tqdm(records, desc="Scoring samples"):
        x = tokenize(row).unsqueeze(0).to(device)
        y = row["label"]
        with torch.no_grad():
            logits = model(x)
            conf = torch.softmax(logits, dim=1)[0, 1].item()
        results.append({
            "path": row["file_path"],
            "score": conf,
            "label": y,
            "llm": row.get("llm_description", "")
        })

    pd_scores = [r for r in results if r["label"] == 1]
    hc_scores = [r for r in results if r["label"] == 0]
    pd_topk = sorted(pd_scores, key=lambda x: -x["score"])[:TOPK]
    hc_topk = sorted(hc_scores, key=lambda x: -x["score"])[:TOPK]

    print("\n=== Top-K PD ===")
    for r in pd_topk:
        print(f"{r['path']} | score={r['score']:.4f}\nLLM: {r['llm']}\n")

    print("\n=== Top-K HC ===")
    for r in hc_topk:
        print(f"{r['path']} | score={r['score']:.4f}\nLLM: {r['llm']}\n")

    # 可视化 Top-K 结果
    plot_topk(pd_topk, "PD Top-K Samples")
    plot_topk(hc_topk, "HC Top-K Samples")

    # 保存所有得分
    pd.DataFrame(results).to_csv("token_topk_scores.csv", index=False)
    print("\u2705 Saved CSV: token_topk_scores.csv")

# === 启动入口 ===
if __name__ == "__main__":
    run_topk_token_model()