import torch
from PIL import Image
import numpy as np

# 保存したモデルのパス
max_model_path = "D:\\project_assignment\\finetuning_model\\max_model.pth"

# モデルを読み込む
max_model = torch.jit.load(max_model_path, map_location='cpu')
max_model.eval()

min_model_path = "D:\\project_assignment\\finetuning_model\\min_model.pth"
min_model = torch.jit.load(min_model_path, map_location='cpu')
min_model.eval()

img_path= "D:/project_assignment/temp_image/img_small/skirt_creap/114.jpg"

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

output = max_model(input)
print("max is ", output)
output1 = min_model(input)
print("min is ", output1)