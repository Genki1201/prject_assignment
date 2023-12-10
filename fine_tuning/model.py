from torchvision import models
import torch.optim as optim
import torch.nn as nn
import torch

class FineTuningModel(nn.Module):
    def __init__(self):
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

        #特徴量の総数
        num_features = 960 * 2 * 1 * 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_features, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=640, bias=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=640, out_features=320, bias=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=320, out_features=1, bias=True),
            nn.Identity() #線形関数
        )

    def forward(self, x):
        #multitaskのclassifier以外
        y = self.multitask_features(x)

        #simple_tempのclassifier以外
        z = self.simple_temp_features(x)
        #simple_tempの情報にmultitaskの情報を結合
        z = torch.cat((z, y), dim=1) 
        #classifier層
        output = self.classifier(z)

        return output

#これ以降はmainに書く

model = FineTuningModel()
#モデルのすべてのパラメータを表示
num = 0
for name, param in model.named_parameters():
    print(name)
    num+=1
print("layer is: ", num)


#再学習させたい範囲を指定
learning_range = 8
print('update param', learning_range)

#更新するパラメータのリスト
learning_params = []
i=0
#更新するパラメータ以外を固定
for name, param in model.named_parameters():
    if i in range(num-learning_range, num):
        param.requires_grad=True
        learning_params.append(param)
        print(name)
    else:
        param.requires_grad=False
    i+=1

#最適化アルゴリズムを定義（lrは学習率）
optimizer = optim.Adam(learning_params)

