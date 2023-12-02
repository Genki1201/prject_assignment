import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
import cv2
import random

#カスタムデータセットを作る
class CategoryDataset(Dataset):
    def __init__(self, csv_path, transform, aug_transform): #Noneはtransformを行わないときに備えている
        self.data = pd.read_csv(csv_path)
        self.data_size = len(self.data.index) #ファイルの長さ
        self.transform = transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.data.index)*2
    
    def __getitem__(self, idx): #idxとしてインデックス番号が呼ばれると入力ラベルを返す
        if idx < self.data_size:
            #idxの入力ラベル
            img_path = str(self.data.iloc[idx, 0])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.transform(image)

            #idxの正解ラベル
            nd_labels = self.data.iloc[idx, 1]
            labels = torch.tensor(nd_labels, dtype=torch.int64)

        else:
            #オーグメンテーション
            idx = idx - self.data_size
            img_path = str(self.data.iloc[idx, 0])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.aug_transform(image)

            nd_labels = self.data.iloc[idx, 1]
            labels = torch.tensor(nd_labels, dtype=torch.int64)
        
        return image, labels
    
#データの前処理の関数
def transformer(img):
    #モデルに入れるためにリサイズ
    img = cv2.resize(img, (224, 224))
    #画素値をnumpyのfloat型にしてからrgb値の最大255で割って正規化
    img = img.astype(np.float32) / 255.0
    #nddrayをtensorに変換
    img = torch.tensor(img)
    #HWCからCHWに変換
    img = torch.permute(img, (2, 0,1))
    return img

#回転オーグメンテーション込みの前処理
def aug_transformer(img):
    #画像を回転
    height = img.shape[0]
    width = img.shape[1]
    center=(int(width / 2), int(height / 2)) #中心
    angle = random.uniform(-45, 45)
    scale = 1.0
    trans = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    img = cv2.warpAffine(img, trans, (width, height)) #アフィン変換

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img)
    img = torch.permute(img, (2, 0,1))
    return img

#拡大オーグメンテーション込みの前処理
def scale_transformer(img):
    scale_factor = 2
    height = img.shape[0]
    width = img.shape[1]
    center=(int(width / 2), int(height / 2)) #中心
    trans = cv2.getRotationMatrix2D(center=center, angle=0, scale=scale_factor)
    img = cv2.warpAffine(img, trans, (width, height)) #アフィン変換

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img)
    img = torch.permute(img, (2, 0,1))
    return img


if __name__== "__main__":
    custom_dataset = CategoryDataset(csv_path="D:\\project_assignment\\label\\category_shuffled_label.csv", 
                           transform=transformer,
                           aug_transform=aug_transformer)

    img, label = custom_dataset[20000]
    print(img.size())
    print(label)  
    print(len(custom_dataset))



  
    



