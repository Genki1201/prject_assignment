import requests
from bs4 import BeautifulSoup
import os
import pandas as pd

def scrape(page_url, category, under, top):
    #サイトに対する処理
    #htmlを取得
    r = requests.get(page_url)

    #解析できる形にする
    soup = BeautifulSoup(r.text, features="html.parser")

    print(len(soup.find_all("img")))

    #画像のリンク部分のみ取得
    img_tags = soup.find_all("img")
    img_links = []

    for img_tag in img_tags:
        link = img_tag.get('src') #src=画像のリンク
        if link != None:
            img_links.append(link)

    print(len(img_links))

    #各画像に対する処理
    label_file = pd.read_csv("/content/drive/MyDrive/データ/scraped_datas/data_label.csv")
    folder_path = "/content/drive/MyDrive/データ/scraped_datas/"+str(category)+"/"

    for index, link in enumerate(img_links):
        file_name = "{}.jpg".format(index)
        img_path = folder_path+file_name
        print(img_path)

        #画像を取得
        r = requests.get(link, stream=True)

        #画像を入れるファイルを作成
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        #書き込み
        with open(img_path, "wb") as f:
            f.write(r.content)

        #ラベルファイルに追加
        new_data = {
            'img_path': [img_path],
            '最低気温': [under],
            '最高気温': [top]
        }

        label_file = label_file.append(pd.DataFrame(new_data), ignore_index=True)

    label_file.to_csv("/content/drive/MyDrive/データ/scraped_datas/data_label.csv", index=False)

