import pandas as pd

with open("D:\project_assignment\deep_fashion_label\list_category_img.txt", 'r') as file: #テキストファイルの読み込み
    lines = file.readlines()

data = []
for line in lines:
    data.append(line.strip().split()) #空白で分けたリストを作成

category_label = pd.DataFrame(data, columns=['img_name', 'category']) #データフレームへ

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

category_label['category'] = category_label['category'].replace(category_mapping).astype('category')

category_mapping ={'Anorak': 1, 'Blazer': -1, 'Blouse': -1, 'Bomber': 5, 'Button-Down': -1,
                   'Cardigan': 2, 'Flannel': 0, 'Halter': -1, 'Henley': -1, 'Hoodie': -1,
                   'Jacket': 3, 'Jersey': -1, 'Parka': 6, 'Peacoat': -1, 'Poncho': -1,
                   'Sweater': 8, 'Tank': -1, 'Tee': -1, 'Top': 4, 'Turtleneck': -1,
                   'Capris': -1, 'Chios': 10, 'Culottes': -1, 'Cutoffs': -1, 'Gauchos': -1,
                   'Jeans': 9, 'Jeggings': -1, 'Jodhpurs': -1, 'Joggers': 11, 'Leggings': -1,
                   'Sarong': -1, 'Shorts': 13, 'Skirt': 12, 'Sweatpants': -1, 'Sweatshorts': -1,
                   'Trunks': -1, 'Caftan': -1, 'Cape': -1, 'Coat': 7, 'Coverup': -1, 
                   'Dress': 14, 'Jumpsuit': -1, 'Kaftan': -1, 'Kimono': -1, 'Nightdress': -1,
                   'Onesie': -1, 'Robe': -1, 'Romper': -1, 'Shirtdress': -1, 'Sundress': -1}

category_label['category'] = category_label['category'].replace(category_mapping).astype('category')

category_label.set_index(category_label.columns[0], inplace=True)

#残す行のインデックスを取得
get_list = []
for i in range(len(category_label)):
    if category_label.iloc[i, 0] in range(15): 
        get_list.append(category_label.index[i])

print('data size is', len(get_list))
category_label = category_label[category_label.index.isin(get_list)]

path = []
for idx in category_label.index:
    path.append('D:/project_assignment/deep_fashion_image/' + str(idx))

category_label.set_index(pd.Index(path), inplace=True)

category_shuffled_label = category_label.sample(frac=1)
category_shuffled_label.to_csv("D:\project_assignment\label\category_shuffled_label.csv", index=True)

