import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
import cv2

#カスタムデータセットを作る
class CategoryDataset(Dataset):
    def __init__(self, csv_path, transform=None): #Noneはtransformを行わないときに備えている
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data.index)
    
    def __getitem__(self, idx): #idxとしてインデックス番号が呼ばれると入力ラベルを返す
        #idxの入力ラベル
        img_path = str(self.data.iloc[idx, 0])
        image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform: #画像の前処理
            image = self.transform(image)

        #idxの正解ラベル
        category_labels = self.data.iloc[idx, 1]
        category_labels = torch.tensor(category_labels, dtype=torch.int64)
        fabric_labels = self.data.iloc[idx, 2]
        fabric_labels = torch.tensor(fabric_labels, dtype=torch.int64)

        #カテゴリーと生地のリストで返すべし

        return image, fabric_labels, category_labels
    
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


if __name__== "__main__":
    custom_dataset = CategoryDataset(csv_path="D:\\project_assignment\\deep_fashion_label\\final_label\\category_label.csv", 
                           transform=transformer)

    img, label = custom_dataset[10]
    print(img.size())
    print(label)  
    print(len(custom_dataset))



  
    



