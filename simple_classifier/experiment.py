from dataset1 import transformer
import cv2
import torch
from torchvision import models
from torch.nn.modules.activation import LogSoftmax

img_path= 'D:/project_assignment/deep_fashion_image/img/Hooded_Cotton_Canvas_Anorak/img_00000104.jpg'
image = cv2.imread(img_path)
input = transformer(image)
input = input.unsqueeze(0) #バッチサイズが1のバッチを追加

model_path = 'D:\project_assignment\dee_fashion_model\model.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
print(LogSoftmax(model(input)))
predicted_classes = torch.argmax(model(input), dim=1)
print(predicted_classes)