import torch
from PIL import Image
import numpy as np

# 保存したモデルのパス
multitask_model_path = "D:\project_assignment\deep_fashion_model\multitask.pth"

# モデルを読み込む
mutlitask_model_script = torch.jit.load(multitask_model_path, map_location='cpu')
#モデルを評価モードに
mutlitask_model_script.eval()

print(mutlitask_model_script)

from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch

class FineTuningModel(nn.Module):
    def __init__(self, num_category_classes, num_fabric_classes):
        super(FineTuningModel, self).__init__()
        #simple_tempの学習済みモデルを読み込む
        simple_temp_path = "D:\\project_assignment\\simple_temp_model\\max_model.pth"
        simple_temp_model = torch.jit.load(simple_temp_path, map_location='cpu')
        #classifier層を削除
        self.simple_temp_features = nn.Sequential(*list(simple_temp_model.children())[:-1])


        #multitaskの学習済みモデルを読み込む
        multitask_path = "D:\project_assignment\deep_fashion_model\multitask.pth"
        multitask_model = torch.jit.load(multitask_path, map_location='cpu')
        #classifier層を削除
        self.multitask_features = nn.Sequential(*list(multitask_model.children())[:-2])
#ここから
        #特徴量の総数
        num_features = 960 * 1 * 1

        #categoryの出力層
        self.category_classifier = nn.Sequential(
            nn.Flatten(), #channel*height*wide
            nn.Linear(in_features=num_features, out_features=num_category_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

        #fabricの出力層
        self.fabric_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_features, out_features=num_fabric_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        #classifier以外
        x = self.features(x)
        
        # category
        output_category = self.category_classifier(x)

        # fabric
        output_fabric = self.fabric_classifier(x)

        return output_category, output_fabric

