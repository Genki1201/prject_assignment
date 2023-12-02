import pandas as pd

with open("D:\project_assignment\deep_fashion_explain\list_attr_cloth.txt", 'r') as file_colum: #テキストファイルの読み込み
    lines_columns = file_colum.readlines()

colum = []
for line in lines_columns: 
    colum.append(line.strip()) # 列名の作成


with open('D:\project_assignment\deep_fashion_label\list_attr_img.txt', 'r') as file: #テキストファイルの読み込み
    lines = file.readlines()

data = []
for line in lines:
    data.append(line.strip().split()) #空白で分けたリストを作成

df = pd.DataFrame(data, columns=colum) #データフレームへ

#生地のデータだけを残す
fabric = []
fabric_columns = [df.columns[0]]
for index in range(len(colum)):
    if "2" in colum[index]: #文字列2を含むか確認
        fabric.append(colum[index])
        fabric_columns.append(df.columns[index])

# 生地の列だけを残す
fabric_df = df.loc[:, fabric_columns]

#fabric_df = pd.read_csv("D:\project_assignment\deep_fashion_label\label_fabric.csv")

fabric_df.set_index(fabric_df.columns[0], inplace=True)

fabric_mapping ={'lace                         2': 0,
                   'knit                         2': 1,
                   'mesh                         2': 2,
                   'denim                        2': 3,
                   'leather                      2': 4,
                   'cotton                       2': 5,
                   'chino                        2': 6,
                   'chiffon                      2': 7,
                   'corduroy                     2': 8,
                   'fur                          2': 9,
                   'nylon                        2': 10,
                   'metallic                     2': 11,
                   'stretch                      2': 12,
                   'suede                        2': 13
}
fabric_df.rename(columns=fabric_mapping, inplace=True)

#作成した列だけを抜き出す
reducted_label= fabric_df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]

del_list = []
fabric_row_sums = reducted_label.sum(axis=1) #生地の列において行方向の和を計算

#和が0の行をリストにいれる
for i in range(len(reducted_label)):
    if fabric_row_sums.iloc[i] == 0: 
        del_list.append(fabric_row_sums.index[i])
#del_list内の行を削除
reducted_label = reducted_label[~reducted_label.index.isin(del_list)]

#fabricのワンホットエンコーディングを解除
fabric_label = reducted_label.idxmax(axis=1).reset_index(name='fabric_label')
fabric_label.set_index(fabric_label.columns[0], inplace=True)

path = []
for idx in fabric_label.index:
    path.append('D:/project_assignment/deep_fashion_image/' + str(idx))

#インデックス名をファイルパスに
fabric_label.set_index(pd.Index(path), inplace=True)

fabric_label.to_csv("D:\\project_assignment\\label\\fabric_label.csv", index=True)

fabric_shuffled_label = fabric_label.sample(frac=1)
fabric_shuffled_label.to_csv("D:\\project_assignment\\label\\fabric_shuffled_label.csv", index=True)
print('data size is', len(fabric_label))
