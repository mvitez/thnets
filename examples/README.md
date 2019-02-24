# Python bindings

You will need protobuf to use ONNX models:

Installing protobuf from source can be accomplished this way:
		
```
wget https://github.com/google/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz
tar xf protobuf-all-3.6.1.tar.gz\
cd protobuf-3.6.1
./configure			
make -j4
sudo make install
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```



# Use in Tensorflow

This is an example using the low level thnets routines to build a convolution
layer for Tensorflow. It emulates the Tensorflow nn.conv2d function, with the
only difference of different ordering of the filter coefficients (thnets, as
Torch, uses the outlayers x inlayers x height x width organization, Tensorflow
uses height x width x inlayers x outlayers). The data is in the thnets (Torch)
order batch x layers x height x width, which is compatible with Tensorflow.

Since TensorFlow uses the same Toeplitz matrix technique as Torch to compute
convolutions and thnets creates this matrix on the fly as required by OpenBLAS,
this implementation has both the execution time and (especially) memory consumption
lower of the Tensorflow nn.conv2d.
