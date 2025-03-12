"""
The MBD force data generator for the three polymer melts (PE, PP, PVC).
It is an example to illustrate how to create MBD dataset for molecular systems.
The geo file (.gen) is not provide with the code.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
use_GPU = True
if not use_GPU:
    print('not using GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import time
from utils import Loadgen
from mbd_tf import MBDEvaluator

ang = 1/0.529177249  # 1 A = 1/0.529177249 bohr

polymer = 'E'
mbd = 'ts'  # 'ts' for the plain MBD

if polymer == 'E':
    n_group = 3
    Np = 24000
    file_path = 'dataset/polyethylene/'
elif polymer == 'P':
    n_group = 9
    Np = 11892
    file_path = 'dataset/polypropylene/'
elif polymer == 'VC':
    n_group = 6
    Np = 11887
    file_path = 'dataset/polyvinyl_chloride/'

folder_prefix = ''
file_path = folder_prefix + file_path

if mbd != 'ts':
    save_path = folder_prefix + 'dataset/mbd_mixed/'
else:
    save_path = file_path

Nat = 1000
core0_list = np.loadtxt(file_path+'core_list_Nat%d_Np%d.txt' % (Nat,Np))

mbd_evaluator = MBDEvaluator(hessians=False, method=mbd)
for i, core0 in enumerate(core0_list[:]):
    print('i = %d/%d' % (i+1, len(core0_list)),'Core0 =', core0, flush=True)
    F_save = np.zeros((n_group,3),dtype=np.float32)

    for j in range(n_group):
        _, coords, _, atom_type, _ = Loadgen(file_path + 'geo_cluster/geo_Nat%d_P%s_N%d%s%d_N0%d_m%d.gen'
                                             % (Nat, polymer,  core0[0], polymer, core0[1], core0[2], j+1),1)

        if polymer == 'VC':
            volume = [0.80*(t == 0) + 0.60*(t == 1) + 0.99*(t == 2) for t in atom_type]
        else:
            volume = [0.79*(t == 0) + 0.62*(t == 1) + 0.99*(t == 2) for t in atom_type]

        alpha_0 = [12.0*(t == 0) + 4.50*(t == 1) + 15.0*(t == 2) for t in atom_type]
        C6      = [46.6*(t == 0) + 6.50*(t == 1) + 94.6*(t == 2) for t in atom_type]
        R_vdW   = [3.59*(t == 0) + 3.10*(t == 1) + 3.71*(t == 2) for t in atom_type]

        ratio_list = np.array(volume)
        alpha_0 = alpha_0 * ratio_list
        R_vdW = R_vdW * ratio_list ** (1 / 3)
        C6 = C6 * ratio_list ** 2

        start = time.time()
        Embd, Fmbd = mbd_evaluator(coords*ang, alpha_0, C6, R_vdW)
        end = time.time()

        print('F cost =', end - start)
        F_save[j,:] = -Fmbd[0,:]

# Save F
    if mbd != 'ts':
        np.savetxt(save_path + 'F_cluster/F_%s_Nat%d_N%d%s%d_N0%d.txt' % (mbd, Nat, core0[0], polymer, core0[1], core0[2]), F_save)
    else:
        np.savetxt(save_path + 'F_cluster/F_Nat%d_N%d%s%d_N0%d.txt' % (Nat, core0[0], polymer, core0[1], core0[2]), F_save)




