import pandas as pd

with open("D:\project_assignment\deep_fashion_label\list_category_img.txt", 'r') as file: #テキストファイルの読み込み
    lines = file.readlines()

data = []
for line in lines:
    data.append(line.strip().split()) #空白で分けたリストを作成

df = pd.DataFrame(data, columns=['img_name', 'category']) #データフレームへ

"""
#カテゴリカルなデータ型へ変換
category_mapping ={'1': 'Anorak', '2': 'Blazer', '3': 'Blouse', '4': 'Bomber', '5': 'Button-Down',
                   '6': 'Cardigan', '7': 'Flannel', '8': 'Halter', '9': 'Henley', '10': 'Hoodie',
                   '11': 'Jacket', '12': 'Jersey', '13': 'Parka', '14': 'Peacoat', '15': 'Poncho',
                   '16': 'Sweater', '17': 'Tank', '18': 'Tee', '19': 'Top', '20': 'Turtleneck',
                   '21': 'Capris', '22': 'Chios', '23': 'Culottes', '24': 'Cutoffs', '25':'Gauchos',
                   '26': 'Jeans', '27': 'Jeggings', '28': 'Jodhpurs', '29': 'Joggers', '30': 'Leggings',
                   '31': 'Sarong', '32': 'Shorts', '33': 'Skirt', '34': 'Sweatpants', '35': 'Sweatshorts',
                   '36': 'Trunks', '37': 'Caftan', '38': 'Cape', '39': 'Coat', '40': 'Coverup', 
                   '41': 'Dress', '42': 'Jumpsuit', '43': 'Kaftan', '44': 'Kimono', '45': 'Nightdress',
                   '46': 'Onesie', '47': 'Robe', '48': 'Romper', '49': 'Shirtdress', '50': 'Sundress'}

df['category'] = df['category'].map(category_mapping).astype('category')

#one-hot-encoding
df = pd.get_dummies(df, columns=['category'])

df_str = df.iloc[:, 1:].astype(int) #最初の列以外をbool型をint型に

# 最初の列を結合して新しいデータフレームを作成
df = pd.concat([df.iloc[:, 0], df_str], axis=1)

#category_を削除
df.columns = df.columns.str.replace('category_', '')
"""
df.to_csv("D:\project_assignment\deep_fashion_label\label_category.csv", index=False)
