import os
import struct
import numpy as np

# my functions
from . import const
from . import utils
from .struct_def import newdd_struct_hydro, newdd_struct_part, newdd_struct_sink, sizeof

def loadInfoNewDD(basePath, snapNum):
    
    print("\n[Load Info (NewDD)]")
    
    fileName = f"NewDD.{snapNum:05d}/HR5.{snapNum:05d}.00000.info"
    filePath = os.path.join(basePath, fileName)
    
    if os.path.exists(filePath):
        print(f"  Found info: {filePath}")
    else:
        print(f"  Not found info: {filePath}")
    
    info = {}
    
    with open(filePath, "r") as file:
        for line in file:
            if "=" in line:
                key, value = line.split("=")
                
                key = key.replace('define', '').strip() # remove whitespace & 'define'
                value = value.strip()                   # remove whitespace
                
                try:
                    if "." in value:
                        value = float(value)
                        
                    else:
                        value = int(value)
                
                except ValueError:
                    pass # leave the value as a string
                
                info[key] = value    
    
    return info


def loadNewDD(ptype, basePath, snapNum):
    
    info_newdd = loadInfoNewDD(basePath, snapNum)
    ncpu = info_newdd['ncpu']
    
    if ptype=="gas":  
        newdd_struct = newdd_struct_hydro
    
    elif ptype=="sink": 
        newdd_struct = newdd_struct_sink
        
    elif ptype=="dm" or ptype=="star": 
        newdd_struct = newdd_struct_part
    
    else: print("...Something wrong! ptype should be ['gas', 'sink', 'dm', 'star']")
    
    
    fileSize_tot = 0
    for icpu in range(ncpu):
        fileName = f"NewDD.{snapNum:05d}/HR5.{snapNum:05d}.{ptype.upper()}.{icpu:05d}.dat"
        filePath = os.path.join(basePath, fileName)
        fileSize_tot += os.path.getsize(filePath)
    
    npart_tot = fileSize_tot // sizeof(newdd_struct)
    print(f"npart_tot = {npart_tot:,}")

    if fileSize_tot % sizeof(newdd_struct)!=0:
        print(fileSize_tot // sizeof(newdd_struct))
        print("...Something wrong! The particle number is not integer.")
    
    
    npart = [0]
    data_tot = np.zeros(npart_tot, dtype=newdd_struct)

    for icpu in range(ncpu):

        fileName = f"NewDD.{snapNum:05d}/HR5.{snapNum:05d}.{ptype.upper()}.{icpu:05d}.dat"
        filePath = os.path.join(basePath, fileName)

        with open(filePath, "rb") as file:

            file.seek(0, 2) # move to the end-point
            file_size = file.tell() # tell the file size (byte)
            file.seek(0) # move to the start-point

            struct_size = sizeof(newdd_struct)

            npart_this = file_size // struct_size
            print(f"icpu={icpu} \t npart_this={npart_this} \t npart_cum={np.sum(npart[:icpu+1])}")
            if file_size % struct_size != 0:
                print(file_size / struct_size)
                print("...Something wrong! The particle number is not integer.")

            npart.append(npart_this)

            i = int(np.sum(npart[:icpu+1]))
            j = i + npart_this

            data = np.fromfile(file, dtype=newdd_struct, count=npart_this)

            data_tot[i:j] = data

    print(f"last sum  = {np.sum(npart):,}")
    print(f"npart_tot = {npart_tot:,} (check)")
    
    return npart_tot, data_tot