import os
import struct
import numpy as np

# my functions
from . import const
from . import utils

# gas={pos:dblarr(3),dx:0.d,vel:fltarr(3),dum0:0.,density:0.d,temp:0.,metal:0.,h:0.,level:0L,mass:0.,dum1:0.,id:0LL,potential:0.d,f:dblarr(3)}
newdd_struct_hydro=[
    ("pos",       np.float64, 3), #dblarr(3)
    ("dx",        np.float64, 1), #0.d
    ("vel",       np.float32, 3), #fltarr(3)
    ("dum0",      np.float32, 1), #0.
    ("density",   np.float64, 1), #0.d
    ("temp",      np.float32, 1), #0.
    ("metal",     np.float32, 1), #0.
    ("H",         np.float32, 1), #0.
    ("level",     np.int32,   1), #0L
    ("mass",      np.float32, 1), #0.
    ("dum1",      np.float32, 1), #0.
    ("id",        np.int64,   1), #0LL
    ("potential", np.float64, 1), #0.d
    ("fgrav",     np.float64, 3)  #dblarr(3)
]

# part={pos:dblarr(3),vel:dblarr(3),mass:0.d,dum0:0.d,tp:0.d,zp:0.d,mass0:0.d,tpp:0.d, indtab:0.d,id:0LL,potential:0.d,level:0L,dum1:0.}
newdd_struct_part=[
    ("pos",       np.float64, 3), #dblarr(3)
    ("vel",       np.float64, 3), #dblarr(3)
    ("mass",      np.float64, 1), #0.d
    ("dum0",      np.float64, 1), #0.d
    ("tp",        np.float64, 1), #0.d
    ("zp",        np.float64, 1), #0.d
    ("mass0",     np.float64, 1), #0.d
    ("tpp",       np.float64, 1), #0.d
    ("indtab",    np.float64, 1), #0.d
    ("id",        np.int64,   1), #0LL
    ("potential", np.float64, 1), #0.d
    ("level",     np.int32,   1), #0L
    ("dum1",      np.float32, 1)  #0.
]

# sink={pos:dblarr(3),vel:dblarr(3),mass:0.d,tbirth:0.d,angm:dblarr(3),ang:dblarr(3),dmsmbh:dblarr(3),esave:0.d,smag:0.d,eps:0.d,id:0L,dum0:0L}
newdd_struct_sink=[
    ("pos",       np.float64, 3), #dblarr(3)
    ("vel",       np.float64, 3), #dblarr(3)
    ("mass",      np.float64, 1), #0.d
    ("tbirth",    np.float64, 1), #0.d
    ("angm",      np.float64, 3), #dblarr(3)
    ("ang",       np.float64, 3), #dblarr(3)
    ("dmsmbh",    np.float64, 3), #dblarr(3)
    ("esave",     np.float64, 1), #0.d
    ("smag",      np.float64, 1), #0.d
    ("eps",       np.float64, 1), #0.d
    ("id",        np.int32,   1), #0L
    ("dum1",      np.int32,   1)  #0L
]

def sizeof(struct_def):
    size = 0
    for name, dtype, shape in struct_def:
        count = np.prod(shape)
        size += np.dtype(dtype).itemsize * count
    return size