import thnets
from PIL import Image
import numpy as np

t = thnets.thnets()
t.LoadNetwork('lightcnn.onnx')
img = Image.open('bw128.jpg')
img = np.ascontiguousarray(np.array(img).reshape(1,128,128).astype(np.float32) / 255)
result = t.ProcessFloat(img)
print(result)
