import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# my functions
from . import const
from . import utils

def computeCSFR(cdata):
    
    table = utils.loadTimeTable(cdata['header']['cosmology'])
    
    # generating bins for histogram
    tp_face = np.linspace(0, 1.1, 101) # code unit
    tp_cent = 0.5 * (tp_face[:-1] + tp_face[1:])
    dt = np.diff(tp_face)[0]
    
    a_itp = np.interp(tp_cent, table['t'], table['a'])
    z_itp = 1/a_itp - 1
    
    # load necessary data
    star = np.where(cdata['ptype']==4) # we need only star particles
    dm = np.where(cdata['ptype']==1)
    bh = np.where(cdata['ptype']==5)
    
    tp_birth = -cdata['tp_birth'] # code unit
    masses = cdata['masses'] # code unit
    
    boxlen = cdata['header']['boxlen'] # code unit
    unit_t = cdata['header']['unit_t'] # sec
    unit_m = cdata['header']['unit_m'] # g
    unit_l = cdata['header']['unit_l'] # cm
    
    unit_l_in_Mpc = unit_l * const.cm_to_Mpc
    unit_t_in_yr = unit_t * const.sec_to_yr
    unit_m_in_Msun = unit_m * const.g_to_Msun
    
    Lbox_cMpc = cdata['header']['Lbox_cMpc'] # Mpc
    
    # Get a cosmic star formation rate
    hist = np.histogram(tp_birth[star], # code unit
                        bins = tp_face, # code unit
                        weights = masses[star]) # code unit
    csfr = hist[0] # code unit
    csfr *= unit_m_in_Msun / (dt * unit_t_in_yr) / Lbox_cMpc**3 # Msun/yr/cMpc^3
    
#     # for quick check the results
#     plt.figure(figsize=(5, 5))
#     plt.loglog(z_itp+1, csfr)
        
    print("\nReturns z, csfr")
    
    return z_itp, csfr


def writeOverdensity(file, header, array):
    
    # header
    header_format = 'iii ffff ffff'
    header_data = (
        header['n1'], header['n2'], header['n3'],
        header['dxini0'], header['xoff10'], header['xoff20'], header['xoff30'],
        header['astart0'], header['omega_m0'], header['omega_l0'], header['h00']
    )
    
    header_size = struct.calcsize(header_format)
    blksz = header_size
    print("block size for header: ", blksz)

    file.write(struct.pack('i', blksz))  # 블록 크기 기록
    file.write(struct.pack(header_format, *header_data))  # 헤더 데이터 기록
    file.write(struct.pack('i', blksz))  # 블록 크기 기록
    
    print("Writing header ... successfully done!")
    
    
    # data
    n3, n2, n1 = array.shape
    assert n1 == header['n1'] and n2 == header['n2'] and n3 == header['n3'], \
        "Array shape does not match header dimensions."
    
    for i in range(n3):
        slice_data = array[i, :, :].astype(np.float32)
        
        blksz = slice_data.size * 4  # float32 (4 bytes)
        print(f"Writing slice {i+1}/{n3}, block size: {blksz}")
        
        file.write(struct.pack('i', blksz))
        file.write(slice_data.tobytes())
        file.write(struct.pack('i', blksz))

    file.write(struct.pack('i', blksz))
    file.write(array.astype(np.float32).tobytes())
    file.write(struct.pack('i', blksz))

    print("Writing sliced array ... successfully done!")
    return None


def saveOverdensity(cdata):
    
    print("\n[Compute overdensity map]")
    
    cosmology  = cdata['header']['cosmology']
    snapNum    = cdata['header']['snapNum']
    lmin       = cdata['header']['levelmin']
    lmax       = cdata['header']['levelmax']
    unit_m     = cdata['header']['unit_m']
    unit_l     = cdata['header']['unit_l']
    
    dm   = np.where(cdata['ptype']==1)
    star = np.where(cdata['ptype']==4)
    bh   = np.where(cdata['ptype']==5)
    
#     # for quick check
#     plt.figure(figsize=(5, 5))
#     bins = np.logspace(-1, 10, 51)
#     plt.hist(masses[star], bins=bins, label='star')
#     plt.hist(masses[dm], bins=bins, label='dm')
#     plt.hist(masses[bh], bins=bins, label='bh')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.legend(fontsize=13)
#     plt.tight_layout(pad=0.3)
    
    Lbox_cMpc = cdata['header']['Lbox_cMpc']
    Lbox_pMpc = cdata['header']['Lbox_pMpc']
    grid_size = 2**lmin
    dgrid = Lbox_cMpc / grid_size # cMpc
    
    masses     = cdata['masses'] * unit_m * const.g_to_Msun # Msun
    positions  = cdata['positions'] * Lbox_cMpc # cMpc
#     positions  = cdata['positions'] * unit_l * const.cm_to_Mpc # pMpc  

    print(f"  Boxsize: {Lbox_pMpc:.2f} pMpc")
    print(f"  Boxsize: {Lbox_cMpc:.2f} cMpc")
    print(f"  levelmin: {lmin}")
    print(f"  Grid size: {grid_size}")
    print(f"  dxini: {dgrid:.6f} cMpc")
    print(f"  dxini: {Lbox_pMpc / grid_size:.6f} pMpc")
    
    grid = np.zeros((grid_size, grid_size, grid_size))
    indices = (positions / dgrid).astype(int) % grid_size
    for idx, mass in zip(indices[dm], masses[dm]):
        grid[tuple(idx)] += mass # Msun
        
    mean_density = masses[dm].sum() / (Lbox_cMpc**3) # Msun/Mpc^3
    
    overdensity = (grid / dgrid**3) / mean_density - 1 # rho_bin / rho_mean - 1
    
    omega_m = cdata['header']['omega_m']
    omega_l = cdata['header']['omega_l']
    h = cdata['header']['H0'] / 100
    
    header = {
        'n1': grid_size, 'n2': grid_size, 'n3': grid_size,
        'dxini0': dgrid, 'xoff10': 0.0, 'xoff20': 0.0, 'xoff30': 0.0,
        'astart0': 1/201, 'omega_m0': omega_m, 'omega_l0': omega_l, 'h00': h
    }
    
    print("header: ", header)
    print("overdensity: ", overdensity.shape)
    
    output_file = f"./deltac_{cosmology}_lmin{lmin:02d}_{snapNum:05d}"
    with open(output_file, 'wb') as file:
        writeOverdensity(file, header, overdensity)
        print(f"saved {output_file}")
    
    return None


def readOverdensity(filename):
    
    with open(filename, 'rb') as file:
        
        # Read header
        blksz = struct.unpack('i', file.read(4))[0]
        print("Block size for header: ", blksz)
        
        header_format = 'iii ffff ffff'
        header_size = struct.calcsize(header_format)
        
        header_data = struct.unpack(header_format, file.read(header_size))
        header = {
            'n1': header_data[0],
            'n2': header_data[1],
            'n3': header_data[2],
            'dxini0': header_data[3],
            'xoff10': header_data[4],
            'xoff20': header_data[5],
            'xoff30': header_data[6],
            'astart0': header_data[7],
            'omega_m0': header_data[8],
            'omega_l0': header_data[9],
            'h00': header_data[10]
        }
        
        blksz_check = struct.unpack('i', file.read(4))[0]
        if blksz != blksz_check:
            raise ValueError("Header block size mismatch!")

        print("Header successfully read!")
        
        # Prepare to read data
        n1, n2, n3 = header['n1'], header['n2'], header['n3']
        array_shape = (n3, n2, n1)
        data = np.empty(array_shape, dtype=np.float32)

        # Read data slice by slice
        for i in range(n3):
            blksz = struct.unpack('i', file.read(4))[0]
            
            slice_data = np.frombuffer(file.read(blksz), dtype=np.float32)
            data[i, :, :] = slice_data.reshape((n2, n1))
            
            blksz_check = struct.unpack('i', file.read(4))[0]
            if blksz != blksz_check:
                raise ValueError(f"Data block size mismatch at slice {i}!")

        print("Data successfully read!")
        return header, data
    