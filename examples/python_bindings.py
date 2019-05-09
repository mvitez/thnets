import thnets
from PIL import Image
import numpy as np

t = thnets.thnets()
t.LoadNetwork('resnet18.onnx')

img = Image.open('my_image.jpg')
img = img.resize((224,224))
img = np.ascontiguousarray(np.array(img).reshape(3,224,224).astype(np.float32) / 255)
result = t.ProcessFloat(img)
print(result)
