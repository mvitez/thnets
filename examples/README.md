# Use in Tensorflow

This is an example using the low level thnets routines to build a convolution
layer for Tensorflow. It emulates the Tensorflow nn.conv2d function, with the
only difference of different ordering of the filter coefficients (thnets, as
Torch, uses the outlayers x inlayers x height x width organization, Tensorflow
uses height x width x inlayers x outlayers). The data is in the thnets (Torch)
order batch x layers x height x width, which is compatible with Tensorflow.

Both the execution time and (especially) memory consumption will be lower, as
TensorFlow uses the same Toeplitz matrix technique as Torch to compute convolutions.
