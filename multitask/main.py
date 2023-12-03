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

    category_total_loss=0 #1エポックの損失
    fabric_total_loss = 0
    category_total_correct=0 #1エポックの精度
    fabric_total_correct =0

    iter = 0
    for input, category_label, fabric_label in data_loader: #1イテレータ分
        input = input.to(device)
        category_label = category_label.to(device)
        fabric_label = fabric_label.to(device)

        category_output, fabric_output = model(input) #1イテレータの出力

        #損失関数の計算（lossは1イテレータの平均）
        category_loss = loss_def(category_output, category_label)
        fabric_loss = loss_def(fabric_output, fabric_label)
        category_total_loss += float(category_loss) * input.size(0)
        fabric_total_loss += float(fabric_loss)*input.size(0)

        #勾配の初期化
        optim.zero_grad()
        #勾配計算
        category_loss.backward(retain_graph=True) #計算グラフを保存
        fabric_loss.backward()
        #パラメータの更新
        optim.step()

        #正解数を計算
        category_pred_label = category_output.argmax(dim=1)
        category_total_correct += int((category_pred_label == category_label).sum())
        fabric_pred_label = fabric_output.argmax(dim=1)
        fabric_total_correct += int((fabric_pred_label == fabric_label).sum())

        print(iter)
        iter += 1

    #損失関数と正解率の計算
    category_avg_loss = category_total_loss / len(data_loader.dataset)
    category_accuracy = category_total_correct / len(data_loader.dataset)
    fabric_avg_loss = fabric_total_loss / len(data_loader.dataset)
    fabric_accuracy = fabric_total_correct / len(data_loader.dataset)

    return category_avg_loss, category_accuracy, fabric_avg_loss, fabric_accuracy

#検証を行う関数
def validation(model, device, data_loader, loss_def):
    model.eval()

    category_total_loss=0 #1エポックの損失
    fabric_total_loss = 0
    category_total_correct=0 #1エポックの精度
    fabric_total_correct =0

    iter = 0
    for input, category_label, fabric_label in data_loader: #1イテレータ分
        input = input.to(device)
        category_label = category_label.to(device)
        fabric_label = fabric_label.to(device)

        category_output, fabric_output = model(input) #1イテレータの出力

        #損失関数の計算（lossは1イテレータの平均）
        category_loss = loss_def(category_output, category_label)
        fabric_loss = loss_def(fabric_output, fabric_label)
        category_total_loss += float(category_loss) * input.size(0)
        fabric_total_loss += float(fabric_loss)*input.size(0)

        #正解数を計算
        category_pred_label = category_output.argmax(dim=1)
        category_total_correct += int((category_pred_label == category_label).sum())
        fabric_pred_label = fabric_output.argmax(dim=1)
        fabric_total_correct += int((fabric_pred_label == fabric_label).sum())

        #print(iter)
        iter += 1

    #損失関数と正解率の計算
    category_avg_loss = category_total_loss / len(data_loader.dataset)
    category_accuracy = category_total_correct / len(data_loader.dataset)
    fabric_avg_loss = fabric_total_loss / len(data_loader.dataset)
    fabric_accuracy = fabric_total_correct / len(data_loader.dataset)

    return category_avg_loss, category_accuracy, fabric_avg_loss, fabric_accuracy

#実際の学習

#カスタムデータセットのインポート
from dataset import CategoryDataset, transformer, rote_transformer

train_dataset = CategoryDataset(csv_path="D:\\project_assignment\\deep_fashion_label\\final_label\\multitask_train.csv", 
                           transform=transformer,
                           aug_transform= rote_transformer)

val_dataset = CategoryDataset(csv_path="D:\\project_assignment\\deep_fashion_label\\final_label\\multitask_val.csv",
                              transform=transformer,
                              aug_transform= rote_transformer)

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
from model import MultiTaskModel

#損失関数
criterion = nn.NLLLoss()

print("criterion is: ", criterion)
#モデルをインスタンス化
multiTaskModel = MultiTaskModel(12, 12)
#モデルをGPUに転送
multiTaskModel = multiTaskModel.to(device)

lr = 0.001
optimizer = optim.Adam(multiTaskModel.parameters(), lr=lr)
print("lr: ", lr)

num_epochs = 8

history = defaultdict(list)
for epoch in range(num_epochs):
    #学習
    category_train_loss, category_train_accuracy, fabric_train_loss, fabric_train_accuracy = train(multiTaskModel, 
                                        device, 
                                        train_dl, 
                                        optimizer, 
                                        criterion)
    
    #評価
    category_val_loss, category_val_accuracy, fabric_val_loss, fabric_val_accuracy = validation(multiTaskModel,
                                            device,
                                            val_dl,
                                            criterion)
    
    print(f"epoch {epoch + 1} ")
    print("<category>  ")
    print(
            f"[train] loss: {category_train_loss:.6f}, accuracy: {category_train_accuracy:.0%} |"
            f"[validation] loss: {category_val_loss:.6f}, accuracy: {category_val_accuracy:.0%}"
        )
    print("<fabric>  ")
    print(
            f"[train] loss: {fabric_train_loss:.6f}, accuracy: {fabric_train_accuracy:.0%} |"
            f"[validation] loss: {fabric_val_loss:.6f}, accuracy: {fabric_val_accuracy:.0%}"
        )
    print("------------------------------------------------")

#モデルの保存
model_path = 'D:\\project_assignment\\dee_fashion_model\\simple_category_model.pth'
torch.save(multiTaskModel, model_path)
