import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
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
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(img_path)
            img = img.convert('RGB')

            image = self.transform(img)

            #idxの正解ラベル
            nd_labels = self.data.iloc[idx, 1]
            labels = torch.tensor(nd_labels, dtype=torch.int64)

        else:
            #オーグメンテーション
            idx = idx - self.data_size
            img_path = str(self.data.iloc[idx, 0])
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(img_path)
            img = img.convert('RGB')

            image = self.aug_transform(img)

            nd_labels = self.data.iloc[idx, 1]
            labels = torch.tensor(nd_labels, dtype=torch.int64)
        
        return image, labels
    
#データの前処理の関数
def transformer(img):
    #モデルに入れるためにリサイズ
    img = img.resize((224, 224), Image.LANCZOS)
    #画素値をnumpyのfloat型にしてからrgb値の最大255で割って正規化
    img = np.array(img) 
    img = img.astype(np.float32) / 255.0
    #nddrayをtensorに変換
    img = torch.tensor(img)
    #HWCからCHWに変換
    img = torch.permute(img, (2, 0,1))
    return img

#回転オーグメンテーション込みの前処理
def rote_transformer(img):
    #画像を回転
    img_rotate = img.rotate(30)
    # 画像をリサイズする
    img = img_rotate.resize((224, 224), Image.LANCZOS)
    img = np.array(img) 
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img)
    img = torch.permute(img, (2, 0,1))
    return img

#拡大オーグメンテーション込みの前処理
def scale_transformer(img):
    width, height = img.size
    # トリミングの範囲指定
    left = width // 4  # 1/4 from the left
    top = height // 4  # 1/4 from the top
    right = 3 * width // 4  # 3/4 from the left
    bottom = 3 * height // 4  # 3/4 from the top
    # トリミング
    cropped_img = img.crop((left, top, right, bottom))
    img = cropped_img.resize((224, 224), Image.LANCZOS)
    img = np.array(img) 
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img)
    img = torch.permute(img, (2, 0,1))
    return img


if __name__== "__main__":
    custom_dataset = CategoryDataset(csv_path="D:\\project_assignment\\label\\category_shuffled_label.csv", 
                           transform=transformer,
                           aug_transform=rote_transformer)

    img, label = custom_dataset[20000]
    print(img.size())
    print(label)  
    print(len(custom_dataset))



  
    



