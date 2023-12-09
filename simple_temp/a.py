import torch
from PIL import Image
import numpy as np

# 保存したモデルのパス
model_path = 'D:\\project_assignment\\dee_fashion_model\\simple_temp_model.pth'

# モデルを読み込む
loaded_model = torch.load(model_path)

img_path= 'D:/project_assignment/temp_image/img_small/short_top_nylon/125.jpg'
img = Image.open(img_path)
img = img.convert('RGB')
#モデルに入れるためにリサイズ
img = img.resize((224, 224), Image.LANCZOS)
img.show()
#画素値をnumpyのfloat型にしてからrgb値の最大255で割って正規化
img = np.array(img) 
img = img.astype(np.float32) / 255.0
#nddrayをtensorに変換
img = torch.tensor(img)
#HWCからCHWに変換
img = torch.permute(img, (2, 0,1))
input = img.unsqueeze(0) #バッチサイズが1のバッチを追加

output = loaded_model(input)
print(output)