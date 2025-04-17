# run_all_experiments.py
# 一键运行所有核心实验（Token CNN、Transformer、融合模型、多模态解释）

import os
import subprocess

print("\n===== [1] 训练 Token CNN 模型 =====")
subprocess.run(["python", "token_model_training.py"])

print("\n===== [2] 训练 Token Transformer 模型 =====")
with open("token_model_training.py", "r") as f:
    content = f.read()

content = content.replace("TokenCNNClassifier", "TokenTransformerClassifier")
with open("token_model_transformer_temp.py", "w") as f:
    f.write(content)
subprocess.run(["python", "token_model_transformer_temp.py"])

print("\n===== [3] 混淆矩阵评估（CNN 模型） =====")
from token_model_extensions import TokenCNNClassifier, evaluate_with_confusion_matrix
from token_model_training import TokenSequenceDataset
import torch
from torch.utils.data import DataLoader

csv_path = "audio_vq_multimodal_with_llm.csv"
dataset = TokenSequenceDataset(csv_path)
total_len = len(dataset)
val_len = int(0.2 * total_len)
_, val_set = torch.utils.data.random_split(dataset, [total_len - val_len, val_len])
val_loader = DataLoader(val_set, batch_size=16)

model = TokenCNNClassifier()
model.load_state_dict(torch.load("token_cnn_model.pth", map_location="cpu"))
evaluate_with_confusion_matrix(model, val_loader, device=torch.device("cpu"))

print("\n===== [4] 多模态模型训练（Token + Spectrogram） =====")
subprocess.run(["python", "multi_modal_training.py"])

print("\n===== [5] GradCAM 批量处理与Top-K分析 =====")
subprocess.run(["python", "batch_gradcam_and_topk.py"])

print("\n===== [6] GradCAM Top-K 拼图展示 =====")
subprocess.run(["python", "gradcam_summary_figure.py"])

print("\n✅ 所有实验已完成，你可以开始撰写论文结果分析部分。")
