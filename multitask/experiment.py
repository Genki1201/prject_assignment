import torch
from PIL import Image
import numpy as np

# 保存したモデルのパス
multitask_model_path = "D:\project_assignment\deep_fashion_model\multitask.pth"

# モデルを読み込む
mutlitask_model_script = torch.jit.load(multitask_model_path, map_location='cpu')
#モデルを評価モードに
mutlitask_model_script.eval()
img_path= "D:/project_assignment/deep_fashion_image/img_highres/img/Pleated_Floral_Chiffon_Dress/img_00000011.jpg"

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

category_output, fabric_output = mutlitask_model_script(input)
print(category_output, fabric_output)

category_classes = torch.argmax(category_output, dim=1)
fabric_classes = torch.argmax(fabric_output, dim=1)
print("category class is ", category_classes)
print("fabric class is ", fabric_classes)
