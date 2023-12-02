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
    for input, fabric_label, category_label in data_loader: #1イテレータ分
        input = input.to(device)
        category_label = category_label.to(device)
        fabric_label = fabric_label.to(device)

        category_output, fabric_output = model(input) #1イテレータの出力

        #損失関数の計算（lossは1イテレータの平均）
        category_loss = loss_def(category_output, category_label)
        fabric_loss = loss_def(fabric_output, fabric_label)
        total_loss += (float(category_loss) + float(fabric_loss)) * input.size(0)

        #勾配の初期化
        optim.zero_grad()
        #勾配計算
        category_loss.backward()
        fabric_loss.backward()
        #パラメータの更新
        optim.step()

        #正解数を計算
        pred_label = category_output.argmax(dim=1)
        total_correct += int((pred_label == category_label).sum())

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
        for input, fabric_label, category_label in data_loader: #1イテレータ分
            input = input.to(device)
            category_label = category_label.to(device)
            fabric_label = fabric_label.to(device)

            category_output, fabric_output =model(input)

            category_loss = loss_def(category_output, category_label)
            fabric_loss = loss_def(fabric_output, fabric_label)
            total_loss += (float(category_loss) + float(fabric_loss)) * input.size(0)

            #正解数を計算
            pred_label = category_output.argmax(dim=1)
            total_correct += int((pred_label == category_label).sum())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)

    return avg_loss, accuracy

#実際の学習

#カスタムデータセットのインポート
from dataset import CategoryDataset, transformer, aug_transformer

category_dataset = CategoryDataset(csv_path="D:\\project_assignment\\deep_fashion_label\\final_label\\fabric_shuffled_label.csv", 
                           transform=transformer,
                           aug_transform=aug_transformer)

#データセット数削減
reduction = 0
if reduction == 1:
    dataset_size = 30000
    category_dataset, not_use_dataset = random_split(category_dataset, [dataset_size, len(category_dataset) - dataset_size])
    print('dataset reduction')

#データセット分割
dataset_size = len(category_dataset)
train_size = int(dataset_size*0.8)
train_dataset, val_dataset = random_split(category_dataset, [train_size, dataset_size-train_size])

print(len(train_dataset))

#データローダの作成
train_dl = DataLoader(dataset=train_dataset,
                    batch_size=128,
                    shuffle=True
                    )

val_dl = DataLoader(dataset=val_dataset,
                    batch_size=64,
                    shuffle=False)


#モデルを取得
from model import MultiTaskModel, criterion
print("criterion is: ", criterion)
#モデルをインスタンス化
multiTaskModel = MultiTaskModel(13, 14)
#モデルをGPUに転送
multiTaskModel = multiTaskModel.to(device)

lr = 0.001
optimizer = optim.Adam(multiTaskModel.parameters(), lr=lr)
print("lr: ", lr)

num_epochs = 8

history = defaultdict(list)
for epoch in range(num_epochs):
    #学習
    train_loss, train_accuracy = train(multiTaskModel, 
                                        device, 
                                        train_dl, 
                                        optimizer, 
                                        criterion)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    #評価
    validation_loss, validation_accuracy = validation(multiTaskModel,
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
torch.save(multiTaskModel, model_path)
