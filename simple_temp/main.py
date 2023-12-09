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
        total_correct = 0
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

#最低気温
train_dataset = MyDataset(csv_path="D:\\project_assignment\\temp_label\\temp_label_small_train.csv", 
                           transform=transformer,
                           aug_transform=scale_transformer,
                           temp=min)

val_dataset = MyDataset(csv_path="D:\\project_assignment\\temp_label\\temp_label_small_val.csv", 
                           transform=transformer,
                           aug_transform=scale_transformer,
                           temp=min)

print("train size is ", len(train_dataset))

#データローダの作成
train_dl = DataLoader(dataset=train_dataset,
                    batch_size=128,
                    shuffle=True
                    )

print("validation size is ", len(val_dataset))
val_dl = DataLoader(dataset=val_dataset,
                    batch_size=64,
                    shuffle=False)


#モデルを取得
from model import TempModel, optimizer, criterion
print("criterion is: ", criterion)
#モデルをGPUに転送
TempModel = TempModel.to(device)

num_epochs = 50

history = defaultdict(list)
for epoch in range(num_epochs):
    #学習
    train_loss = train(TempModel, 
                        device, 
                        train_dl, 
                        optimizer, 
                        criterion)
    history["train_loss"].append(train_loss)

    #評価
    validation_loss = validation(TempModel,
                                 device,
                                 val_dl,
                                 criterion)
    history["valid_loss"].append(validation_loss)

    print(f"epoch {epoch + 1} ")
    print(f"[train] loss: {train_loss:.6f}")
    print(f"[validation] loss: {validation_loss:.6f}")
    print('--------------------------------------------------------------------')

#モデルの保存
model_path = 'D:\\project_assignment\\dee_fashion_model\\simple_temp_model.pth'
torch.save(TempModel.to('cpu'), model_path)
