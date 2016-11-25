# Stand-alone library for loading and running networks

## Installation

Requirements for library: OpenBLAS (already part of the library for ARM and x86_64),
CuDNN version 4 if compiled with the CUDNN=1 option, OpenCL if compiled with the OPENCL=1 option,
gemmlowp if compiled with the LOWP=1 option.
Requirements for test: libpng and libjpeg  
Check the CUDA and CUDNN directories in the Makefile if using CUDNN.
Make with "make".
Install with "(sudo) make install".
Make options are:
   * *DEBUG* 0 is off, 1 is on
   * *MEMORYDEBUG* 0 checks memory leaks, 1 generates full dump of allocations in memdump.txt
   * *CUDNN* 0 is off, 1 uses CuDNN
   * *OPENCL* 0 is off, 1 uses OpenCL
   * *LOWP* 0 is off, 1 uses gemmlowp

gemmlowp (https://github.com/google/gemmlowp) has to be installed into the lowp directory, if used.

## Test program

    export LD_LIBRARY_PATH=/usr/local/lib:/opt/OpenBLAS/lib (add CUDA and CUDNN lib directories, if using CUDNN)
    ./test -m <model_dir> -i <input_file>

The model directory must contain 2 files:
   * *model.net* the network file saved in .t7 format
   * *stat.t7* contains a table with a 'std' and 'mean' FloatTensor of dimension 3

Input file can be a .jpg or .png file, or a .t7 file containing a FloatTensor of dimension 3

## High level API description

### void THInit()

Initializes the library.

### THNETWORK *THLoadNetwork(const char *path)

Loads the network contained in the path directory and returns a THNETWORK object or 0, if the
network cannot be loaded. The reason can be obtained with THLastError().

### void THMakeSpatial(THNETWORK *network)

Makes the loaded network suitable for images bigger of the eye size.

### void THUseSpatialConvolutionMM(THNETWORK *network, int nn_type)

Changes every occurrence of SpatialConvolution in the network to SpatialConvolutionMM (nn_type>0) or viceversa (nn_type=0).
SpatialConvolutionMM modules with padW or padH different of 0 will not be changed to SpatialConvolution, as the
SpatialConvolution module does not support them. If nn_type=2 and supported (currently only on ARM), it will use virtual MM,
which takes much less memory.

### THNETWORK *THCreateCudaNetwork(THNETWORK *net)

Create a new network from the given network. The new network will use CuDNN.

### int THCudaHalfFloat(int enable)

Enables the use of 16 bit floats on CUDA.

### THNETWORK *THCreateOpenCLNetwork(THNETWORK *net)

Create a new network from the given network. The new network will use OpenCL.

### THNETWORK *THCreateLowpNetwork(THNETWORK *net, float range)

Create a new network from the given network. The new network will use gemmlowp (8 bit gemm). The output of
the convolutions is supposed to be between -range/2 and +range/2. *This feature is experimental.*

### int THOpenCLHalfFloat(int enable)

Enables the use of 16 bit floats on OpenCL.

### int THProcessFloat(THNETWORK *network, float *data, int batchsize, int width, int height, float **result, int *outwidth, int *outheight)

Runs the network on the float data. Float data is organized as a coniguous array of
size batchsize x 3 x height x width, where 3 is the number of color planes. The order
of the color components is expected to be red, green, blue.

Returns the number of categories in the output and the size of the output in outwidth and outheight.  
Result will point to the array with the data and *must* not be freed.  
The data is a contiguous array of size batchsize x number of categories x outheight x outwidth.

### int THProcessImages(THNETWORK *network, unsigned char **images, int batchsize, int width, int height, int stride, float **result, int *outwidth, int *outheight, int bgr)

Runs the network on the series of images. Images is an array with batchsize pointers and
each element points to the start of the image. Images are arrays of size
height x stride x 3, where only the first width of each line long stride contains data. If
bgr is zero, the order of the color components is expected to be red, green, blue, otherwise
it's expected to be blue, green, red.
 
Returns the number of categories in the output and the size of the output in outwidth and outheight.  
Result will point to the array with the data and *must* not be freed.  
The data is a contiguous array of size batchsize x number of categories x outheight x outwidth.

### int THProcessYUYV(THNETWORK *network, unsigned char *image, int width, int height, float **results, int *outwidth, int *outheight)

Runs the network on an image in the YUYV format. This function is useful when the image comes from a camera,
where the YUYV format is common.  

Returns the number of categories in the output and the size of the output in outwidth and outheight.  
Result will point to the array with the data and *must* not be freed.  
The data is a contiguous array of size batchsize x number of categories x outheight x outwidth.

### void THFreeNetwork(THNETWORK *network)

Frees the network and all associated data and outputs.

### int THLastError()

Returns an error code describing the reason of the last error. It is now used only for
THLoadNetwork and can give these results:

- 0 Ok
- -1 The file cannot be opened
- -2 The file cannot be read till the end
- -3 The file contains some elements, which were not implemented in this library
- -4 The file is corrupted
- -5 The file contains torch objects not expected in that file


### Demonstration application

See: https://github.com/teradeep/demo-apps.git. Please note that the application is
older of this library and will require some modifications in order to be compiled.

### Android instructions (for experts only)

I don't have the time to build a complete Android example, but just to give some hints, I created a
sample Android.mk and Application.mk file that will create an executable file that will run on Android
devices. If you don't have a rooted device, you can use SSHServer from Ice Cold Apps, ssh into the device
and copy the executable here: /data/data/com.icecoldapps.sshserver (the only accessible place where
you can set the executable flag on files). Before compiling, put the .c and .h files from
https://github.com/julienr/libpng-android/tree/stable/jni into a libpng subdirectory of thnets. I've
also removed every reference to JPEG from image.c. Otherwise you can try to compile libjpeg-turbo.

### Tegra TX1 results:

Forward times in seconds:

| Image Size | Direct   | MM       | cuDNN    | 16-bit cuDNN |
| :--------: | :------: | :------: | :------: | :----------: |
| 1280x720   | 2.461129 | 1.549271 | 0.088064 | 0.088064     |
| 1920x1080  | 7.196426 | 3.666830 | 0.195412 | 0.129979     |
