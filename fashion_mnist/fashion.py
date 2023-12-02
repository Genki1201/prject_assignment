from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_transform = transforms.ToTensor()

#トレーニング用データを読み込む
train_dataset = datasets.FashionMNIST(
    root="G:\マイドライブ\Colab Notebooks\datasets_of_FashionMNIST", train=True, transform=data_transform, download=True
)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
) #DataLoaderは指定したデータセットからデータを取得し、ミニバッチを作成して返す関数

#テスト用データを読み込む
test_dataset = datasets.FashionMNIST(
    root="G:\マイドライブ\Colab Notebooks\datasets_of_FashionMNIST", train=False, transform=data_transform, download=True
)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=True
)

from torch.nn.modules.activation import LogSoftmax
class Net(nn.Module): #torch.nn.Moduleクラスを継承
    def __init__(self): #コンストラクタ
        super().__init__() #親クラスnn.Moduleのインストラクタを実行
        self.features = nn.Sequential( #PyTorchレイヤー（畳み込み層、全結合層など）を受け取りそれらのレイヤーを順に適用する
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(), #デフォルトが0.5
            nn.Linear(in_features=64 * 7 * 7, out_features=128), #全結合層（入力チャネル数、出力チャネル数）
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) #平坦化
        x = self.classifier(x)

        return x

nll_loss = nn.NLLLoss()

#デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#モデルを計算を実行するデバイスに転送する
model = Net().to(device)

optim = torch.optim.Adam(model.parameters())

def train(model, device, data_loader, optim):
    #モデルを学習用モードに設定する
    model.train()

    total_loss = 0
    total_correct = 0
    ite=0
    for data, target in data_loader: #dataに画像データをtargetに正解ラベルを1バッチサイズ分付与
        #データおよびラベルを計算を実行するデバイスに転送する
        data, target = data.to(device), target.to(device)

        #1ミニバッチ（イテレータ）分順伝播した出力結果をoutputに入れる
        output = model(data)

        #損失関数の値を計算する

        loss = nll_loss(output, target) #損失関数は引数にモデルからの出力値と正解ラベルとる
        total_loss += float(loss) #データごとに損失を足し合わせていく

        #逆伝播を行う
        optim.zero_grad() #各パラメータの勾配を初期化
        loss.backward() #損失の平均？から勾配を計算

        #パラメータを更新する
        optim.step() #1ミニバッチにつき一回

        #確率の最も高いクラスを予測ラベルとする
        pred_target = output.argmax(dim=1) #各データにおいてそれぞれのクラスに対する出力の中で最も割合の高いクラス番号を入れる

        #予測ラベルと正解ラベルを比較し、一致する数正解数を計算
        total_correct += int((pred_target == target).sum())
        print(ite)
        ite += 1

    #損失関数の値の平均及び精度を計算する
    avg_loss = total_loss / len(data_loader.dataset) #一つのデータあたりの誤差
    accuracy = total_correct / len(data_loader.dataset) #正解率

    return avg_loss, accuracy

def test(model, device, data_loader):
    #モデルをテストモードに設定する
    model.eval

    with torch.no_grad(): #テストでは勾配を計算をする必要が無いため勾配計算に必要な情報をメモリ上に保存しなくする
        total_loss = 0
        total_correct = 0

        for data, target in data_loader:
            #データおよびラベルを計算を実行するデバイスに転送する
            data, target = data.to(device), target.to(device)

            #順伝播する
            output = model(data)

            #損失を計算する
            loss = nll_loss(output, target)
            total_loss += float(loss)

            #確率の最も高いクラスを予測ラベルとする
            pred_target = output.argmax(dim=1)

            #正解数を計算する
            total_correct += int((pred_target == target).sum())

    #損失の平均及び精度を計算する
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)

    return avg_loss, accuracy

n_epochs = 50

history = defaultdict(list) #具ラグとして描画するために結果をリストとして保存
for epoch in range(n_epochs):
    #1エポック分学習する
    train_loss, train_accuracy = train(model, device, train_data_loader, optim)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    #評価する
    test_loss, test_accuracy = test(model, device, test_data_loader)
    history["test_loss"].append(test_loss)
    history["test_accuracy"].append(test_accuracy)

    print(
        f"epoch {epoch + 1} "
        f"[train] loss: {train_loss:.6f}, accuracy: {train_accuracy:.0%} "
        f"[test] loss: {test_loss:.6f}, accuracy: {test_accuracy:.0%}"
    )


