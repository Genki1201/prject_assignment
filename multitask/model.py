from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch

class MultiTaskModel(nn.Module):
    def __init__(self, num_category_classes, num_fabric_classes):
        super(MultiTaskModel, self).__init__()
        #mobilenetの学習済みモデルを読み込む
        mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        #classifier層とgolobal avarage poolingを削除
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])

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
