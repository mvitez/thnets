import torch
from torch import nn
from ctypes import *
from torchvision import models
from torch.autograd import Variable as V

def writetensor(f, name, t):
    f.write(bytearray(c_int8(3))) #Tensor id
    f.write(str.encode(name))    #Name
    f.write(bytearray(c_int8(0))) #Name terminator
    f.write(bytearray(c_int32(t.dim()))) #Number of dimensions
    for i in range(t.dim()):
        f.write(bytearray(c_int32(t.size(i)))) #Individual dimensions
    f.flush()
    t.contiguous().storage()._write_file(f)
    f.flush()

def writeint(f, name, v):
    f.write(bytearray(c_int8(4))) #int32param id
    f.write(str.encode(name))     #Param name
    f.write(bytearray(c_int8(0))) #Param name terminator
    f.write(bytearray(c_int32(v))) #Data

def writefloat(f, name, v):
    f.write(bytearray(c_int8(5))) #floatparam id
    f.write(str.encode(name))     #Param name
    f.write(bytearray(c_int8(0))) #Param name terminator
    f.write(bytearray(c_float(v))) #Data

def writeintvect(f, name, v):
    f.write(bytearray(c_int8(6))) #32tupleparam id
    f.write(str.encode(name))    #Param name
    f.write(bytearray(c_int8(0))) #Param name terminator
    f.write(bytearray(c_int32(len(v)))) #Tuple elements
    for i in range(len(v)):
        f.write(bytearray(c_int32(v[i])))

def writefunctionid(f, id):
    f.write(bytearray(c_int8(7)))  #Function id
    f.write(bytearray(c_int32(id))) #Data

def check_layer_class(obj):
    if (str(obj.__class__)=="<class 'torch.autograd.function.AddmmBackward'>"):
        return (True,'torch.nn._functions.linear.Linear')
    elif (str(obj.__class__)=="<class 'torch.autograd.function.ThresholdBackward'>"):
        return (True,'torch.nn._functions.thnn.auto.Threshold')
    elif (str(obj.__class__)=="<class 'torch.autograd.function.DropoutBackward'>"):
        return (True,'torch.nn._functions.dropout.Dropout')
    elif (str(obj.__class__)=="<class 'torch.autograd.function.ViewBackward'>"):
        return (True,'torch.autograd._functions.tensor.View')
    elif (str(obj.__class__)=="<class 'torch.autograd.function.MaxPool2dBackward'>"):
        return (True,'torch.nn._functions.thnn.pooling.MaxPool2d')
    elif (str(obj.__class__)=="<class 'ConvNdBackward'>"):
        return (True,'torch.nn._functions.conv.ConvNd')
    elif (str(obj.__class__)=="<class 'torch.autograd.function.AvgPool2dBackward'>"):
        return (True,'torch.nn._functions.thnn.pooling.AvgPool2d')
    elif (str(obj.__class__)=="<class 'torch.autograd.function.AddBackward'>"):
        obj.inplace=True
        return (True,'torch.autograd._functions.basic_ops.Add')
    elif (str(obj.__class__)=="<class 'BatchNormBackward'>"):
        return (True,'torch.nn._functions.batchnorm.BatchNorm')
    elif (str(obj.__class__)=="<class 'torch.autograd.function.ConcatBackward'>"):
        return (True,'torch.autograd._functions.tensor.Concat')
    return (False,'')

def check_parameter_class(obj):
    if (str(obj.__class__)=="<class 'AccumulateGrad'>"):
        return (True,obj.variable)
    return (False,'')

def check_if_linear_weight(obj):
    if (str(obj.__class__)=="<class 'torch.autograd.function.TransposeBackward'>"):
        return (True,obj.next_functions[0][0].variable)
    return (False,'')

class Exporter:
    def __init__(self, f):
        self.f = f
        self.output_id = 0
        self.objects = {}
        #Write 24 bytes header
        f.write(str.encode('PyTorch Graph Dump 1.00'))
        f.write(bytearray(c_int8(0))) #String terminator
    def end(self):
        self.f.write(bytearray(c_int8(0))) #End of function id
    def input(self):
        self.f.write(bytearray(c_int8(1))) #Input id
    def function(self, name, obj):
        self.f.write(bytearray(c_int8(2))) #Function id
        self.objects[obj] = self.output_id
        self.f.write(bytearray(c_int32(self.output_id))) #Unique ID of the output of this function
        self.output_id = self.output_id + 1
        self.f.write(str.encode(name))     #Function name
        self.f.write(bytearray(c_int8(0))) #Function name terminator
        if hasattr(obj, 'inplace'):
            writeint(self.f, 'inplace', obj.inplace)
        if hasattr(obj, 'ceil_mode'):
            writeint(self.f, 'ceil_mode', obj.ceil_mode)
        if hasattr(obj, 'kernel_size'):
            writeintvect(self.f, 'kernel_size', obj.kernel_size)
        if hasattr(obj, 'new_sizes'):
            writeintvect(self.f, 'sizes', obj.new_sizes)
        if hasattr(obj, 'stride'):
            writeintvect(self.f, 'stride', obj.stride)
        if hasattr(obj, 'padding'):
            writeintvect(self.f, 'padding', obj.padding)
        if hasattr(obj, 'eps'):
            writefloat(self.f, 'eps', obj.eps)
        #if hasattr(obj, 'threshold'):
        #    writefloat(self.f, 'threshold', obj.threshold)
        #if hasattr(obj, 'value'):
        #    writefloat(self.f, 'value', obj.value)
        if hasattr(obj, 'running_mean'):
            writetensor(self.f, 'running_mean', obj.running_mean)
        if hasattr(obj, 'running_var'):
            writetensor(self.f, 'running_var', obj.running_var)
        if hasattr(obj, 'dim'):
            writeint(self.f, 'dim', obj.dim)
    def tensor(self, t):
        writetensor(self.f, '', t.data)
    def write(self, obj):
        self.function(check_layer_class(obj)[1], obj)
        for o in obj.next_functions:
            if check_layer_class(o[0])[0]:
                if o[0] in self.objects:
                    writefunctionid(self.f, self.objects[o[0]])
                else:
                    self.write(o[0])
        if obj.next_functions[0][0] is None:
            self.input()
        for o in obj.next_functions:
            (check,param)=check_if_linear_weight(o[0])
            if check:
                self.tensor(param)
        for o in obj.next_functions:
            (check,param)=check_parameter_class(o[0])
            if check:
                self.tensor(param)
        self.end()

def save(path, output):
    with open(path, mode='wb') as f:
        e = Exporter(f)
        e.write(output.grad_fn)

#model=models.densenet201(pretrained=True).eval()
#out=model(V(torch.FloatTensor(1,3,227,227)))
#save('model.net',out)
