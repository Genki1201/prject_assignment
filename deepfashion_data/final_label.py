import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

concat_label = pd.read_csv("D:\\project_assignment\\deep_fashion_label\\concat_label.csv")

concat_label.set_index(concat_label.columns[0], inplace=True)

path = []
for idx in concat_label.index:
    path.append('D:/project_assignment/deep_fashion_image/img_highres/' + str(idx))

concat_label.set_index(pd.Index(path), inplace=True)

category_label = concat_label.iloc[:, :1]
fabric_label = concat_label.iloc[:, 1:]

#fabricのワンホットエンコーディングを解除
fabric_label = fabric_label.idxmax(axis=1).reset_index(name='fabric_label')
fabric_label.set_index(fabric_label.columns[0], inplace=True)

#マルチタスク用データセットを作成
concat_label = pd.concat([category_label, fabric_label], axis=1)
#データをシャッフル
concat_shuffled_label = concat_label.sample(frac=1)

#datasetをtrain用とvalidation用に分ける
dataset_size = len(concat_shuffled_label)
train_size = int(dataset_size*0.8)
label_train = concat_shuffled_label.head(train_size)
label_val = concat_shuffled_label.tail(dataset_size - train_size)
label_train.to_csv("D:\\project_assignment\\deep_fashion_label\\final_label\\multitask_train.csv", index=True)
label_val.to_csv("D:\\project_assignment\\deep_fashion_label\\final_label\\multitask_val.csv", index=True)

category_train = label_train.iloc[:, 0]
category_val = label_val.iloc[:, 0]
category_train.to_csv("D:\\project_assignment\\deep_fashion_label\\final_label\\category_train.csv", index=True)
category_val.to_csv("D:\\project_assignment\\deep_fashion_label\\final_label\\category_val.csv", index=True)

fabric_train = label_train.iloc[:, 1]
fabric_val = label_val.iloc[:, 1]
fabric_train.to_csv("D:\\project_assignment\\deep_fashion_label\\final_label\\fabric_train.csv", index=True)
fabric_val.to_csv("D:\\project_assignment\\deep_fashion_label\\final_label\\fabric_val.csv", index=True)

#csvファイルの前
img = Image.open(category_train.index[0])
img = img.resize((224, 224), Image.LANCZOS)
img.show()

#csvファイルに入れた後
label = pd.read_csv("D:\\project_assignment\\deep_fashion_label\\final_label\\fabric_val.csv")
file_path = str(label.iloc[10000, 0])
print("File Path:", file_path)

img = Image.open(file_path)
if img is not None:
    print("Image Loaded Successfully")
    width, height = img.size
    # トリミングの範囲指定
    left = width // 4  # 1/4 from the left
    top = height // 4  # 1/4 from the top
    right = 3 * width // 4  # 3/4 from the left
    bottom = 3 * height // 4  # 3/4 from the top
    # トリミング
    cropped_img = img.crop((left, top, right, bottom))
    img = cropped_img.resize((224, 224), Image.LANCZOS)
    img.show()

else:
    print("Failed to Load Image")
