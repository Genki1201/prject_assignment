import pandas as pd

fabric_label = pd.read_csv("D:\\project_assignment\\deep_fashion_label\\reducted_fabric.csv")
category_label = pd.read_csv("D:\\project_assignment\\deep_fashion_label\\reducted_category.csv")
fabric_label.set_index(fabric_label.columns[0], inplace=True)
category_label.set_index(category_label.columns[0], inplace=True)

#生地のラベルとカテゴリーのラベルを結合
concat_label = pd.concat([category_label, fabric_label], axis=1)

#削除対象の行のインデックスを格納するリスト
del_list = []
fabric_row_sums = fabric_label.sum(axis=1) #生地の列において行方向の和を計算

#どちらかの和が0の行を消していく
for i in range(len(concat_label)):
    if fabric_row_sums.iloc[i] == 0 or category_label.iloc[i, 0] == -1: 
        del_list.append(fabric_row_sums.index[i])

print('data size is', len(fabric_row_sums)-len(del_list))
concat_label = concat_label[~concat_label.index.isin(del_list)]

concat_label.to_csv('D:\project_assignment\deep_fashion_label\concat_label.csv', index=True)

print('category is', len(concat_label.columns[:1]), concat_label.columns[:1])
print('fabric is', len(concat_label.columns[1:]), concat_label.columns[1:])