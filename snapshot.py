import os
import struct
import numpy as np

from . import const

def loadInfo(basePath, snapNum):
    
    print("\n[Load Info]")
    
    fileName = f"output_{snapNum:05d}/info_{snapNum:05d}.txt"
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
                key = key.strip()     # remove whitespace
                value = value.strip() # remove whitespace
                
                try:
                    if "." in value:
                        value = float(value)
                        
                    else:
                        value = int(value)
                
                except ValueError:
                    pass # leave the value as a string
                
                info[key] = value
    
    print(f"  keys in info: {info.keys()}")
    
    return info


def _readHead(file):
        
    header_size = struct.unpack('i', file.read(4))[0]
    
    if   header_size ==  4: value = struct.unpack('i', file.read(header_size))[0]
    elif header_size ==  8: value = struct.unpack('q', file.read(header_size))[0]
    elif header_size == 16: value = struct.unpack('4i', file.read(header_size))
    else:
        raise ValueError(f"Unexpected header size: {header_size} bytes")
    
    tail_size = struct.unpack('i', file.read(4))[0]
    
    if header_size != tail_size:
        raise ValueError(f"Size mismatch: header_size={header_size}, tail_size={tail_size}")
    
    return value


def _readDataArray(file, npart, dtype):
    
    header_size = struct.unpack('i', file.read(4))[0]
    data = np.fromfile(file, dtype=dtype, count=npart)
    tail_size = struct.unpack('i', file.read(4))[0]
    
    if header_size != tail_size:
        raise ValueError(f"Size mismatch: header_size={header_size}, tail_size={tail_size}")
    
    return data


def loadDataOne(basePath, snapNum, icpu, item="part", fields=None):
    
    if icpu==0:
        raise ValueError("Wrong icpu number (icpu=0). It should start from 1.")
    
    fileName = f"output_{snapNum:05d}/{item}_{snapNum:05d}.out{icpu:05d}"
    filePath = os.path.join(basePath, fileName)
    
    if os.path.exists(filePath):
        print(f"  Found data: {filePath}")
    else:
        print(f"  Not found data: {filePath}")
    
    data = {}
    
    with open(filePath, 'rb') as file:
        """
        Based on 'output_part.f90' in RAMSES:
        ! Write header
          write(ilun)ncpu
          write(ilun)ndim
          write(ilun)npart # particle number per MPI rank
          write(ilun)localseed
          write(ilun)nstar_tot # in total box
          write(ilun)mstar_tot # in total box
          write(ilun)mstar_lost
          write(ilun)nsink # bh
        (Note: Modify the order or data type of the header/data items as needed.)
        """

        # Read all header values
        ncpu = _readHead(file)
        ndim = _readHead(file)
        npart = _readHead(file)
        localseed = _readHead(file)
        nstar_tot = _readHead(file)
        mstar_tot = _readHead(file)
        mstar_lost = _readHead(file)
        nsink = _readHead(file)

        # Save header information
        data['header'] = {
            'ncpu': ncpu,
            'ndim': ndim,
            'npart': npart,
            'localseed': localseed,
            'nstar_tot': nstar_tot,
            'mstar_tot': mstar_tot,
            'mstar_lost': mstar_lost,
            'nsink': nsink
        }


        # Read data (array)
        """
        Based on 'output_part.f90' in RAMSES:
        ! Write position
        ! Write velocity
        ! Write mass
        ! Write identity
        ! Write level
        ! Write potential (#ifdef OUTPUT_PARTICLE_POTENTIAL)
        ! Write birth epoch
        ! Write metallicity
        ! Write birth epoch (proper time)
        ! Write indtab (checkpoint in yield table)
        (Note: Modify the order or data type of the header/data items as needed.)
        """
        positions = np.zeros((npart, ndim))
        for i in range(ndim):
            positions[:, i] = _readDataArray(file, npart, dtype=np.float64)
        data['positions'] = positions

        velocities = np.zeros((npart, ndim))
        for i in range(ndim):
            velocities[:, i] = _readDataArray(file, npart, dtype=np.float64)
        data['velocities'] = velocities

        data['masses']   = _readDataArray(file, npart, dtype=np.float64)
        data['ids']      = _readDataArray(file, npart, dtype=np.int64)
        data['levels']   = _readDataArray(file, npart, dtype=np.int32)
        data['phi']      = _readDataArray(file, npart, dtype=np.float64)
        data['tc_birth'] = _readDataArray(file, npart, dtype=np.float64)
        data['metal']    = _readDataArray(file, npart, dtype=np.float64)
        data['tp_birth'] = _readDataArray(file, npart, dtype=np.float64)
        data['masses0']  = _readDataArray(file, npart, dtype=np.float64)
        data['indtab']   = _readDataArray(file, npart, dtype=np.float64)

    # If specific fields are requested, filter data
    if fields:
        filtered_data = {field: data[field] for field in fields if field in data}
        return filtered_data
    
    return data # Note: All values are given in the code units.


def codeUnits_time(info): # utils.py?
    invH0_to_sec = const.Mpc_to_km # [1/H0] = Mpc*s/km --> km*s/km --> sec
    unit_t = 1/info['H0'] * invH0_to_sec
    return unit_t # sec


def codeUnits_mass(info): # utils.py?
    unit_m = info['unit_d']* info['unit_l']**3
    return unit_m # g


def loadDataAll(cosmology, basePath, snapNum, item="part", fields=None):
    
    print("\n[Load Data]")
    
    # Load info file
    info = loadInfo(basePath, snapNum)
    
    ## Save info data
    ncpu = info['ncpu']
    ndim = info['ndim']
    levelmin = info['levelmin']
    levelmax = info['levelmax']
    ngridmax = info['ngridmax']
    nstep_coarse = info['nstep_coarse']
    boxlen = info['boxlen'] # side length of a simulation box
    time = info['time']
    aexp = info['aexp']
    H0 = info['H0'] # km/s/Mpc
    h = H0 / 100
    omega_m = info['omega_m']
    omega_l = info['omega_l']
    omega_k = info['omega_k']
    omega_b = info['omega_b']
    unit_l = info['unit_l'] # side length of a simulation box in cm
    unit_d = info['unit_d'] # volume mean density in g/cm^3
    unit_t = info['unit_t'] # will be changed
    ordering = info['ordering type']
    
    ## Convert the code units in cgs
    unit_t = codeUnits_time(info)
    unit_m = codeUnits_mass(info)
    info['unit_t'] = unit_t # sec
    info['unit_m'] = unit_m # g
    
    # Load data file
    ## Count a total number of praticles 
    npart_tot = 0
    for icpu in range(1, ncpu+1): # icpu start from 1
        data = loadDataOne(basePath, snapNum, icpu, item, fields=['header'])
        npart_tot += data['header']['npart']
    print(f"  Total particle number = {npart_tot:,.0f}")
    
    ## Load all fields and combine them
    npart    = np.zeros(ncpu, dtype=np.int64) # particles per MPI rank
    pid      = np.zeros(npart_tot, dtype=np.int64) # particle id
    pos      = np.zeros((npart_tot, ndim), dtype=np.float64) # positions
    vel      = np.zeros((npart_tot, ndim), dtype=np.float64) # velocities
    levels   = np.zeros(npart_tot, dtype=np.int32) # AMR levels
    masses   = np.zeros(npart_tot, dtype=np.float64) # masses
    masses0  = np.zeros(npart_tot, dtype=np.float64) # initial masses
    tp_birth = np.zeros(npart_tot, dtype=np.float64) # proper birth time
    tc_birth = np.zeros(npart_tot, dtype=np.float64) # conformal birth time
    phi      = np.zeros(npart_tot, dtype=np.float64) # potential
    metal    = np.zeros(npart_tot, dtype=np.float64) # metallicity
    indtab   = np.zeros(npart_tot, dtype=np.float64) # checkpoint in tield table

    ## load all data and combine them
    for icpu in range(ncpu):

        data = loadDataOne(basePath, snapNum, icpu+1, item)

        npart[icpu] = data['header']['npart']

        i = np.sum(npart[:icpu])
        j = np.sum(npart[:icpu+1])

        pid[i:j] = data['ids']
        
        for k in range(ndim):
            pos[i:j, k] = data['positions'][:, k]
            vel[i:j, k] = data['velocities'][:, k]
        
        levels[i:j] = data['levels']
        masses[i:j]  = data['masses']
        masses0[i:j] = data['masses0']
        tp_birth[i:j] = data['tp_birth']
        tc_birth[i:j] = data['tc_birth']
        phi[i:j]    = data['phi']
        metal[i:j]  = data['metal']
        indtab[i:j] = data['indtab']

    ## flag for particle type
    dm   = np.where((tp_birth==0) & (pid>0))
    bh   = np.where(pid < 0)
    star = np.where(tp_birth != 0)
    
    ptype = -np.ones(npart_tot, dtype=np.int8)
    ptype[star] = 4
    ptype[dm]   = 1
    ptype[bh]   = 5
    
    ndm_tot   = np.sum((tp_birth==0) & (pid>0))
    nbh_tot   = np.sum(pid < 0)
    nstar_tot = np.sum(tp_birth != 0)
    print(f"  Star particle number = {nstar_tot:,.0f}")
    print(f"  DM particle number = {ndm_tot:,.0f}")
    print(f"  BH particle number = {nbh_tot:,.0f}")
    print(f"  Total particle number = {nstar_tot+ndm_tot+nbh_tot:,.0f} (check)")

    
    # Combine the data
    cdata = {}
    cdata['header'] = info
    cdata['npart'] = npart
    cdata['ids'] = pid
    cdata['positions'] = pos
    cdata['velocities'] = vel
    cdata['levels'] = levels
    cdata['masses'] = masses
    cdata['masses0'] = masses0
    cdata['tp_birth'] = tp_birth
    cdata['tc_birth'] = tc_birth
    cdata['phi'] = phi
    cdata['metal'] = metal
    cdata['indtab'] = indtab
    cdata['ptype'] = ptype
    
    cdata['header']['cosmology'] = cosmology
    cdata['header']['snapNum'] = snapNum

    # If specific fields are requested, filter data
    if fields:
        print(f"Available fields: {list(cdata.keys())}")
        print(f"Requested fields: {fields}")
        filtered_cdata = {field: cdata[field] for field in fields if field in cdata}
        return filtered_cdata
    
    return cdata
