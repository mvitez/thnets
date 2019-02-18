import sys
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.ctypeslib import as_ctypes
from numpy.ctypeslib import as_array
f = CDLL("./libthnets.so")

class thnets:
    def __init__(self):
        f.THInit()
        self.handle = c_void_p()
        self.THLoadNetwork = f.THLoadNetwork
        self.THLoadNetwork.restype = c_void_p
        self.THFreeNetwork = f.THFreeNetwork
        self.THProcessImages = f.THProcessImages
        self.THProcessFloat = f.THProcessFloat
    def LoadNetwork(self, path):
        self.handle = self.THLoadNetwork(bytes(path, 'ascii'))
    def ProcessFloat(self, image):
        if self.handle  == 0:
            return
        result = c_float()
        presult = pointer(result)
        outwidth = c_int()
        outheight = c_int()
        c_image = as_ctypes(image)
        n = self.THProcessFloat(self.handle, c_image, 1, image.shape[2], image.shape[1], image.shape[0], byref(presult), byref(outwidth), byref(outheight))
        result = as_array(presult, [n])
        return result
    def FreeNetwork(self):
        if self.handle != 0:
            self.THFreeNetwork(self.handle)
        self.handle = 0
