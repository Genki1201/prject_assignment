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
    total_correct=0 #1エポックの精度

    iter = 0
    for input, label in data_loader: #1イテレータ分
        input = input.to(device)
        label = label.to(device)

        output = model(input) #1イテレータの出力

        #損失関数の計算（lossは1イテレータの平均）
        #print(label.size())
        loss = loss_def(output, label)
        total_loss += float(loss) * input.size(0)

        #勾配の初期化
        optim.zero_grad()
        #勾配計算
        loss.backward()
        #パラメータの更新
        optim.step()

        #正解数を計算
        pred_label = output.argmax(dim=1)
        total_correct += int((pred_label == label).sum())

        #print(iter)
        iter += 1

    #損失関数と正解率の計算
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)

    return avg_loss, accuracy

#検証を行う関数
def validation(model, device, data_loader, loss_def):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        for input, label in data_loader:
            input = input.to(device)
            label = label.to(device)

            output=model(input)

            loss = loss_def(output, label)
            total_loss += float(loss) * input.size(0)

            pred_label = output.argmax(dim=1)
            total_correct += int((pred_label == label).sum())
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)

    return avg_loss, accuracy

#実際の学習

#カスタムデータセットのインポート
from dataset1 import CategoryDataset, transformer, scale_transformer, rote_transformer
"""
#category
train_dataset = CategoryDataset(csv_path="D:\\project_assignment\\deep_fashion_label\\final_label\\category_train.csv", 
                           transform=transformer,
                           aug_transform=rote_transformer)

val_dataset = CategoryDataset(csv_path="D:\\project_assignment\\deep_fashion_label\\final_label\\category_val.csv", 
                           transform=transformer,
                           aug_transform=rote_transformer)

"""
#fabric
train_dataset = CategoryDataset(csv_path="D:\\project_assignment\\deep_fashion_label\\final_label\\fabric_train.csv", 
                           transform=transformer,
                           aug_transform=scale_transformer)

val_dataset = CategoryDataset(csv_path="D:\\project_assignment\\deep_fashion_label\\final_label\\fabric_val.csv", 
                           transform=transformer,
                           aug_transform=scale_transformer)

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
from model import CateModel, optimizer, criterion
print("criterion is: ", criterion)
#モデルをGPUに転送
CateModel = CateModel.to(device)

num_epochs = 10

history = defaultdict(list)
for epoch in range(num_epochs):
    #学習
    train_loss, train_accuracy = train(CateModel, 
                                        device, 
                                        train_dl, 
                                        optimizer, 
                                        criterion)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    #評価
    validation_loss, validation_accuracy = validation(CateModel,
                                            device,
                                            val_dl,
                                            criterion)
    history["valid_loss"].append(validation_loss)
    history["valid_accuracy"].append(validation_accuracy)

    print(
            f"epoch {epoch + 1} "
            f"[train] loss: {train_loss:.6f}, accuracy: {train_accuracy:.0%} "
            f"[validation] loss: {validation_loss:.6f}, accuracy: {validation_accuracy:.0%}"
        )

#モデルの保存
model_path = 'D:\\project_assignment\\dee_fashion_model\\simple_category_model.pth'
torch.save(CateModel.to('cpu'), model_path)
