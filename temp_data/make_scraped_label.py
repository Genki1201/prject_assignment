#実行すると全部消える
import pandas as pd

#ラベルのファイルを作成
data = {
    'img_path': [],
    '最低気温': [],
    '最高気温': []
}
df = pd.DataFrame(data)
df.to_csv("/content/drive/MyDrive/データ/scraped_datas/data_label.csv", index=False)
