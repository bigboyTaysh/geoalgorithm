cimport numpy 

cdef class Individual:
    #cdef public double real
    #cdef public int int_from_real
    cdef public binary
    #cdef public int int_from_bin
    #cdef public double real_from_int
    cdef public double fx
    
    def __init__(self, binary, fx):
        #self.real = real
        #self.int_from_real = int_from_real
        self.binary = binary
        #self.int_from_bin = int_from_bin
        #self.real_from_int = real_from_int
        self.fx = fx
    
    def __lt__(self, other):
        return self.qx < other

    def __gt__(self, other):
        return self.qx > other