import time
import tensorflow as tf
thnets = tf.load_op_library('thnets-tf.so')
kw = 9
kh = 9
w = 512
h = 512
c = 3
n = 16
b = 4
W = tf.fill([n,c,kh,kw], 1.0)
B = tf.fill([n], 0.0)
X = tf.fill([b,c,h,w], 2.0)
with tf.Session(''):
  t = time.time()
  thnets.conv(X,W,B,[1,1,1,1],"VALID").eval()
  print time.time() - t
W = tf.fill([kh,kw,c,n], 1.0)
B = tf.fill([n], 0.0)
X = tf.fill([b,h,w,c], 2.0)
with tf.Session(''):
  t = time.time()
  tf.nn.conv2d(X,W,[1,1,1,1],"VALID").eval()
  print time.time() - t
