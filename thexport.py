import torch
from torch import nn
from ctypes import *

def writetensor(f, name, t):
    f.write(bytearray(c_int8(3))) #Tensor id
    f.write(str.encode(name))     #Name
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
    f.write(str.encode(name))     #Param name
    f.write(bytearray(c_int8(0))) #Param name terminator
    f.write(bytearray(c_int32(len(v)))) #Tuple elements
    for i in range(len(v)):
        f.write(bytearray(c_int32(v[i])))

def writefunctionid(f, id):
    f.write(bytearray(c_int8(7)))   #Function id
    f.write(bytearray(c_int32(id))) #Data

class Exporter:
    def __init__(self, f):
        self.f = f
        self.output_id = 0
        #Write 24 bytes header
        f.write(str.encode('PyTorch Graph Dump 1.00'))
        f.write(bytearray(c_int8(0))) #String terminator
    def end(self):
        self.f.write(bytearray(c_int8(0))) #End of function id
    def input(self):
        self.f.write(bytearray(c_int8(1))) #Input id
    def function(self, name, obj):
        self.f.write(bytearray(c_int8(2))) #Function id
        obj.output_id = self.output_id
        self.output_id = self.output_id + 1
        self.f.write(bytearray(c_int32(obj.output_id))) #Unique ID of the output of this function
        self.f.write(str.encode(name))     #Function name
        self.f.write(bytearray(c_int8(0))) #Function name terminator
        if hasattr(obj, 'inplace'):
            writeint(self.f, 'inplace', obj.inplace)
        if hasattr(obj, 'ceil_mode'):
            writeint(self.f, 'ceil_mode', obj.ceil_mode)
        if hasattr(obj, 'kernel_size'):
            writeintvect(self.f, 'kernel_size', obj.kernel_size)
        if hasattr(obj, 'sizes'):
            writeintvect(self.f, 'sizes', obj.sizes)
        if hasattr(obj, 'stride'):
            writeintvect(self.f, 'stride', obj.stride)
        if hasattr(obj, 'padding'):
            writeintvect(self.f, 'padding', obj.padding)
        if hasattr(obj, 'eps'):
            writefloat(self.f, 'eps', obj.eps)
        if hasattr(obj, 'threshold'):
            writefloat(self.f, 'threshold', obj.threshold)
        if hasattr(obj, 'value'):
            writefloat(self.f, 'value', obj.value)
        if hasattr(obj, 'running_mean'):
            writetensor(self.f, 'running_mean', obj.running_mean)
        if hasattr(obj, 'running_var'):
            writetensor(self.f, 'running_var', obj.running_var)
        if hasattr(obj, 'dim'):
            writeint(self.f, 'dim', obj.dim)
    def tensor(self, t):
        writetensor(self.f, '', t.data)
    def write(self, obj):
        s = str(obj.__class__)
        #print(s[s.find("'")+1:-2])
        self.function(s[s.find("'")+1:-2], obj)
        for o in obj.previous_functions:
            if isinstance(o[0], torch.autograd.Function):
                if hasattr(o[0], 'output_id'):
                    writefunctionid(self.f, o[0].output_id)
                else:
                    self.write(o[0])
            elif isinstance(o[0], torch.nn.parameter.Parameter):
                self.tensor(o[0])
            elif isinstance(o[0], torch.autograd.Variable):
                self.input()
        self.end()
        #print(s[s.find("'")+1:-2],'end')

def save(path, output):
    with open(path, mode='wb') as f:
        e = Exporter(f)
        e.write(output.creator)
