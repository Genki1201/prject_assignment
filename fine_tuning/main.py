from torch.utils.data import random_split, DataLoader, Subset
import torch
from sklearn.utils import shuffle
import torch.optim as optim
from collections import defaultdict
import torch.nn as nn
from sklearn.model_selection import KFold



#GPUが利用できることの確認
print("can use GPU: ", torch.cuda.is_available())
#GPUが利用可能ならデバイスをgpuに指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#学習を行う関数
def train(model, device, data_loader, optim, loss_def):
    #モデルを学習モードに
    model.train()

    total_loss=0 #1エポックの損失

    iter = 0
    for input, label in data_loader: #1イテレータ分
        input = input.to(device)
        label = label.to(device)
        label = label.unsqueeze(1) #sizeを合わせる

        output = model(input) #1イテレータの出力
        #損失関数の計算（lossは1イテレータの平均）
        loss = loss_def(output, label)
        total_loss += float(loss) * input.size(0)

        #勾配の初期化
        optim.zero_grad()
        #勾配計算
        loss.backward()
        #パラメータの更新
        optim.step()

        #print(iter)
        iter += 1

    #損失関数と正解率の計算
    avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss

#検証を行う関数
def validation(model, device, data_loader, loss_def):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        for input, label in data_loader:
            input = input.to(device)
            label = label.to(device)
            label = label.unsqueeze(1) #sizeを合わせる

            output=model(input)

            loss = loss_def(output, label)
            total_loss += float(loss) * input.size(0)

    avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss

#実際の学習

#カスタムデータセットのインポート
from dataset import MyDataset, transformer, rote_transformer, scale_transformer

tem = "max"
print(tem, " temprature")
#最低気温
train_dataset = MyDataset(csv_path="D:\\project_assignment\\temp_label\\temp_label_small_train.csv", 
                           transform=transformer,
                           aug_transform=rote_transformer,
                           temp=tem)

val_dataset = MyDataset(csv_path="D:\\project_assignment\\temp_label\\temp_label_small_val.csv", 
                           transform=transformer,
                           aug_transform=rote_transformer,
                           temp=tem)

print("train size is ", len(train_dataset))

#データローダの作成
train_dl = DataLoader(dataset=train_dataset,
                    batch_size=128,
                    shuffle=False
                    )

print("validation size is ", len(val_dataset))
val_dl = DataLoader(dataset=val_dataset,
                    batch_size=64,
                    shuffle=False)


#モデルを取得
from model import FineTuningModel

fineTuningModel = FineTuningModel()

#誤差関数を定義(二乗和誤差)
criterion = nn.MSELoss()
print("criterion is: ", criterion)

#optimizerを定義
#モデルのパラメーターの数を取得
num = 0
for name, param in fineTuningModel.named_parameters():
    #print(name)
    num+=1
#再学習させたい範囲を指定
learning_range = 8
print('update param', learning_range)
#更新するパラメータのリスト
learning_params = []
i=0
#更新するパラメータ以外を固定
for name, param in fineTuningModel.named_parameters():
    if i in range(num-learning_range, num):
        param.requires_grad=True
        learning_params.append(param)
        #print(name)
    else:
        param.requires_grad=False
    i+=1
#最適化アルゴリズムを定義（lrは学習率）
optimizer = optim.Adam(learning_params)

#モデルをGPUに転送
fineTuningModel = fineTuningModel.to(device)

num_epochs = 10

history = defaultdict(list)
for epoch in range(num_epochs):
    #学習
    train_loss = train(fineTuningModel, 
                        device, 
                        train_dl, 
                        optimizer, 
                        criterion)
    history["train_loss"].append(train_loss)

    #評価
    validation_loss = validation(fineTuningModel,
                                 device,
                                 val_dl,
                                 criterion)
    history["valid_loss"].append(validation_loss)

    print('--------------------------------------------------------------------')
    print(f"epoch {epoch + 1} ")
    print(f"[train] loss: {train_loss:.6f}")
    print(f"[validation] loss: {validation_loss:.6f}")

#モデルの保存
model_path = 'D:\\project_assignment\\finetuning_model\\max_model.pth'
model_scripted = torch.jit.script(fineTuningModel)
#model_scripted.save(model_path)
