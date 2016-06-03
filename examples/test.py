import time
import numpy as np
import tensorflow as tf
thnets = tf.load_op_library('thnets-tf.so')
kw = 9
kh = 9
w = 512
h = 512
c = 3
n = 16
b = 4
W = np.random.rand(n,c,kh,kw).astype(np.float32)
B = tf.fill([n], 0.0)
X = np.random.rand(b,c,h,w).astype(np.float32)
with tf.Session(''):
  t = time.time()
  y1 = thnets.conv(X,W,B,[1,1,1,1],"VALID").eval()
  print time.time() - t
# Put in [kh, kw, c, n] format
W1 = np.moveaxis(W, 0, 3)
W1 = np.moveaxis(W1, 0, 2)
# Put in [b, h, w, c] format
X1 = np.moveaxis(X, 1, 3)
with tf.Session(''):
  t = time.time()
  y2 = tf.nn.conv2d(X1,W1,[1,1,1,1],"VALID").eval()
  print time.time() - t
# Put back in [b, c, h, w] format
y2 = np.moveaxis(y2, 3, 1)
print "Difference",np.amax(y2-y1)
