import pandas as pd

category_label = pd.read_csv("D:\project_assignment\deep_fashion_label\label_category.csv")
category_label.set_index(category_label.columns[0], inplace=True)

#使わない生地を0にする
category_mapping = {8: -1, 15: -1, 31: -1, 47: -1, 44: -1, 45: -1, 42: -1, 38: -1, 40: -1, 50: -1, 37: -1, 43: -1, 48: -1, 46: -1, 49: -1}
category_label['category'] = category_label['category'].replace(category_mapping).astype('category')

#category_label.to_csv("D:\project_assignment\deep_fashion_label\deleted_category.csv", index=True)

#似ている生地をまとめる

reducted_label = category_label.copy()

category_mapping = {7: 'flannel',5: 'flannel', 
                    4: 'bomber', 13: 'bomber', 11: 'bomber',
                    1: 'blazer', 6: 'blazer', 2: 'blazer',
                    17: 'tank', 12: 'tank', 
                    3: 'light_top', 18: 'light_top', 9: 'light_top', 19: 'light_top',
                    10: 'parka',
                    39: 'coat', 14: 'coat',
                    16: 'sweater', 20: 'sweater',
                    21: 'long_bottom', 22: 'long_bottom', 30: 'long_bottom', 26: 'long_bottom', 27: 'long_bottom', 29: 'long_bottom', 34: 'long_bottom', 25: 'long_bottom', 28: 'long_bottom', 24: 'long_bottom',
                    35: 'short_bottom', 36: 'short_bottom', 23: 'short_bottom', 32: 'short_bottom',
                    33: 'skirt',
                    41: 'dress'}
category_label['category'] = category_label['category'].replace(category_mapping).astype('category')

category_label.to_csv("D:\project_assignment\deep_fashion_label\deleted_category.csv", index=True)

category_mapping = {'flannel':0,
                    'blazer': 1,
                    'tank': 2,
                    'light_top': 3,
                    'bomber': 4,
                    'parka': 5,
                    'coat': 6,
                    'sweater': 7,
                    'long_bottom': 8, 
                    'short_bottom': 9,
                    'skirt': 10,
                    'dress': 11}
category_label['category'] = category_label['category'].replace(category_mapping).astype('category')

category_label.to_csv("D:\\project_assignment\\deep_fashion_label\\reducted_category.csv")



"""
reducted_label['flannel'] = reducted_label['Flannel']
reducted_label['anorak'] = reducted_label['Anorak']
reducted_label['jacket'] = (
    reducted_label['Blazer']+
    reducted_label['Jacket']+
    reducted_label['Cardigan']
)

reducted_label['tank'] = reducted_label['Tee']+reducted_label['Tank']
reducted_label['light_top'] = (
    reducted_label['Blouse']+
    #reducted_label['Top']+
    reducted_label['Henley']+
    reducted_label['Button-Down']+
    reducted_label['Jersey']
)
reducted_label['bomber'] = reducted_label['Bomber']
reducted_label['parka'] = (
    reducted_label['Parka']+
    reducted_label['Hoodie']
)
reducted_label['coat'] = (
    reducted_label['Coat']+
    reducted_label['Peacoat']
)
reducted_label['sweater'] = (
    reducted_label['Sweater']+
    reducted_label['Turtleneck']
)
reducted_label['long_bottom'] = (
    reducted_label['Capris']+
    #reducted_label['Chinos']+
    reducted_label['Leggings']+
    reducted_label['Jeans']+
    reducted_label['Jeggings']+
    reducted_label['Joggers']+
    reducted_label['Sweatpants']+
    reducted_label['Gauchos']+
    reducted_label['Jodhpurs']+
    reducted_label['Cutoffs']
)
reducted_label['short_bottom'] = (
    reducted_label['Sweatshorts']+
    reducted_label['Trunks']+
    reducted_label['Culottes']+
    reducted_label['Shorts']
)
reducted_label['skirt'] = reducted_label['Skirt']
reducted_label['dress'] = (
    reducted_label['Caftan']+
    reducted_label['Kaftan']+
    #reducted_label['Shirtdress']+
    reducted_label['Dress']
    #reducted_label['Onesie']+
    #reducted_label['Romper']
    #reducted_label['Sundress']
)

#新しい列以外を消す（使わない種類も含めて）
reducted_label = reducted_label[['flannel', 'anorak', 'jacket', 'tank', 'light_top', 'bomber', 'parka', 'coat', 'sweater', 'long_bottom', 'short_bottom', 'skirt', 'dress']]

#0がtrueそれ以外がfalseとなる真偽値票を受け取りそれを反対にしてintにする
reducted_label = (~(reducted_label == 0)).astype(int)
reducted_label.to_csv("D:\\project_assignment\\deep_fashion_label\\reducted_category.csv")
"""