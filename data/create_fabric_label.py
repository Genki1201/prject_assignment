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
    if "2" in colum[index]: #文字列3を含むか確認
        fabric.append(colum[index])
        fabric_columns.append(df.columns[index])

#df = pd.DataFrame(fabric) #データフレームへ
#df.to_csv('D:\project_assignment\deep_fashion_label\colum_fabric.csv', index=False)

# 生地の列だけを残す
df_fabric = df.loc[:, fabric_columns]

df_fabric = df_fabric.copy()

#文字列をint型に変換
df_fabric.replace("-1", -1, inplace=True)
df_fabric.replace("1", 1, inplace=True)

df_fabric.to_csv('D:\project_assignment\deep_fashion_label\label_fabric.csv', index=False)

"""
# 最初の列を除外して各行の和を計算
row_sums = df_fabric.iloc[:, 1:].sum(axis=1)

for i in range(len(row_sums)):
    if row_sums[i] == 0:
        row_sums[i] = 1

# 最初の列を除外して各列をその行の和で割る
df_normalized = df_fabric.iloc[:, 1:].div(row_sums, axis=0)

# 最初の列を結合して新しいデータフレームを作成
df_normalized = pd.concat([df_fabric.iloc[:, 0], df_normalized], axis=1)

df_normalized.to_csv('D:\project_assignment\deep_fashion_label\label_fabric_normalized.csv', index=False)
"""