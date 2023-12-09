from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch

#イメージネットですでに学習済みのモデルをロード
TempModel = models.mobilenet_v3_large(weights='IMAGENET1K_V1')

#mobilenet_v3_large
TempModel.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=1, bias=True),
    nn.Identity() #線形関数
    )

#モデルのすべてのパラメータを表示
num = 0
for name, param in TempModel.named_parameters():
    #print(name)
    num+=1
print("layer is: ", num)

#全部学習
print("learn all parameters")
lr = 0.001
optimizer = optim.Adam(TempModel.parameters(), lr=lr)
print("lr: ", lr)

#print(TempModel)

#誤差関数を定義(二乗和誤差)
criterion = nn.MSELoss()

