import pandas as pd
from PIL import Image

"""
df = pd.read_csv("D:\\project_assignment\\temp_label\\data_label.csv")
df.set_index(df.columns[0], inplace=True)

print(len(df.index))

new_index = []
for i in range(len(df.index)):
    new_index.append("D:/project_assignment/temp_image/img_small/"+df.index[i])

df.index = new_index

df.to_csv("D:\\project_assignment\\temp_label\\temp_label_small.csv", index=True)

"""
file = pd.read_csv("D:\\project_assignment\\temp_label\\temp_label_small.csv")
file.set_index(file.columns[0], inplace=True)

#使えない画像のリスト
del_list = []
for i in range(len(file.index)):
    try:
        path = file.index[i]
        img = Image.open(path)
    except Exception as e:
        del_list.append(path)
print(del_list)

#ファイルから削除
for j in range(len(del_list)):
    file.drop(del_list[j], inplace=True)

file.to_csv("D:\\project_assignment\\temp_label\\temp_label_small.csv", index=True)



df = pd.read_csv("D:\project_assignment\temp_label\temp_label_small.csv")
df.set_index(df.columns[0], inplace=True)

#データをシャッフル
df = df.sample(frac=1)

#datasetをtrain用とvalidation用に分ける
dataset_size = len(df)
train_size = int(dataset_size*0.8)
label_train = df.head(train_size)
label_val = df.tail(dataset_size - train_size)
label_train.to_csv("D:\\project_assignment\\temp_label\\temp_label_small_train.csv", index=True)
label_val.to_csv("D:\\project_assignment\\temp_label\\temp_label_small_val.csv", index=True)


file_path = label_train.index[0]
print(file_path)
img = Image.open(file_path)
img.show()
