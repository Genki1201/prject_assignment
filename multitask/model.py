from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch

#モデルのすべてのパラメータを表示
mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
#mobilenetの最終層手前までを取得
features = mobilenet.features

print(features)

class MultiTaskModel(nn.Module):
    def __init__(self, num_category_classes, num_fabric_classes):
        super(MultiTaskModel, self).__init__()
        #mobilenetの学習済みモデルを読み込む
        mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        #mobilenetの最終層手前までを取得
        self.features = mobilenet.features

        #categoryの出力層
        self.category_classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=num_category_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

        #fabricの出力層
        self.fabric_classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=num_fabric_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        #出力層以外
        x = self.features(x)

        #grobal avarage pooling
        x = x.mean([2, 3])

        # category
        output_category = self.category_classifier(x)

        # タスク2のforward処理
        output_fabric = self.fabric_classifier(x)

        return output_category, output_fabric



#誤差関数を定義
criterion = nn.NLLLoss()

