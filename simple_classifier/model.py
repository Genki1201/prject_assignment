from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch

#イメージネットですでに学習済みのモデルをロード
#CateModel = models.vgg16(weights='IMAGENET1K_V1')
CateModel = models.mobilenet_v3_large(weights='IMAGENET1K_V1')

#モデルのすべての層を表示
#print(CateModel)

"""
#出力数を調整(vgg)
CateModel.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=12, bias=True),
        nn.LogSoftmax(dim=1)
    )
"""
#category 12class fabric 12class

#mobilenet_v3_large
CateModel.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=12, bias=True),
    nn.LogSoftmax(dim=1)
    )
"""
#efficientnet
CateModel.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1280, out_features=12, bias=True),
    nn.LogSoftmax(dim=1)
)
"""
#モデルのすべての層を表示
#print(CateModel)

#モデルのすべてのパラメータを表示
num = 0
for name, param in CateModel.named_parameters():
    #print(name)
    num+=1
print("layer is: ", num)

#全部学習させたければ1そうでなければ0
i = 1
if i == 0:
    #再学習させたい範囲を指定
    learning_range = 6
    print('update param', learning_range)

    #更新するパラメータのリスト
    learning_params = []
    i=0
    #更新するパラメータ以外を固定
    for name, param in CateModel.named_parameters():
        if i in range(num-learning_range, num):
            param.requires_grad=True
            learning_params.append(param)
        else:
            param.requires_grad=False
        i+=1

    #最適化アルゴリズムを定義（lrは学習率）
    optimizer = optim.Adam(learning_params)

elif i == 1:
    #全部学習
    print("learn all parameters")
    lr = 0.001
    optimizer = optim.Adam(CateModel.parameters(), lr=lr)
    print("lr: ", lr)


#誤差関数を定義
criterion = nn.NLLLoss()

