import os
import numpy as np
import pandas as pd

# my functions
from . import const


def codeUnits_time(info):
    invH0_to_sec = const.Mpc_to_km # [1/H0] = Mpc*s/km --> km*s/km --> sec
    unit_t = 1/info['H0'] * invH0_to_sec
    return unit_t # sec


def codeUnits_mass(info):
    unit_m = info['unit_d']* info['unit_l']**3
    return unit_m # g


def loadTimeTable(cosmology):
    
    if "lcdm" in cosmology:
        w0 = -1.0
        wa = 0.0
    
    if cosmology == "cpl0":
        w0 = -1.2
        wa = 0.8
    
    if cosmology == "cpl1":
        w0 = -0.8
        wa = -0.8
    
    print("\n[Load Time Table]")
    print(f"  cosmology: {cosmology}")
    print(f"  (w0, wa) = ({w0}, {wa})")
    
    sign_w0 = '+' if w0 >= 0 else ''
    sign_wa = '+' if wa >= 0 else ''

    # Load time table value
    basePath_table = os.path.join(f"{os.path.dirname(os.path.abspath(__file__))}", "table/time_table")
    fileName_table = f"time_table_cpl{sign_w0}{w0:.1f}{sign_wa}{wa:.1f}.csv"
    filePath_table = os.path.join(basePath_table, fileName_table)
    
    if os.path.exists(filePath_table): 
        print(f"  Found time table: {filePath_table}")
    else:
        print(f"  Not found time table: {filePath_table}")

    table = pd.read_csv(filePath_table, skiprows=4, names=['t', 'tau0', 'tau1', 'a'])
    table = table.dropna()
    # t: proper time
    # tau0: conformal time (a dt)
    # tau1: conformal time in ramses (a*2 dt) (Here, we need tau1 rather than tau0)
    # a: scale factor
    
    return table
    
    