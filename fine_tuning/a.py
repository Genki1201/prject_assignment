import torch
import torch.nn as nn


#multitaskの学習済みモデルを読み込む
multitask_path = "D:\project_assignment\deep_fashion_model\multitask.pth"
multitask_model = torch.jit.load(multitask_path, map_location='cpu')
#classifier層を削除
multitask_features = nn.Sequential(*list(multitask_model.children())[:-2])
print(multitask_features)