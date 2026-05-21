import numpy as onp
from flax import nnx
import orbax.checkpoint as ocp
import jax
import jax.numpy as np
from jax import random
from jax import jit
from jax import lax, vmap, grad
from jax_tqdm import loop_tqdm
from model_lib_flax import trimmed_SchNet
import math
import time
from jax_md import space, smap, extra_energy, minimize, quantity, simulate, util, partition
from utils import Loadgen, SaveGeometry

np.set_printoptions(threshold=np.inf)
f32 = np.float32
i32 = np.int32

# random seed
seed = 0
key = random.PRNGKey(seed)

# Load initial configuration
geo_name = 'N6E250'
decay_rate = 6
t_resume = f32(0)
wrap = False
l0 = 44.73

if t_resume > 0:
    file_name = 'geo/nvt/seed%d/geo_' % seed + geo_name + '_wp%d_a%d_K300_t%.1f_nvt.gen' % (wrap, decay_rate, t_resume)
else:
    # Use seed0 LJ6-equilibrated config as initials for all.
    file_name = 'geo/nvt/seed0/geo_' + geo_name + '_wp%d_a6_K300_t20.0_b%.2f_nvt.gen' % (wrap, l0)

print('loading from ' + file_name)
_, R0, _, atom_type, unit_cell = Loadgen(file_name,1)
R0 = np.array(R0, dtype=f32)
atom_type_pure = np.array(atom_type)
# TraPPE distinguishes the cap H atoms in the polymer chain
atom_type = np.array([atom_type[i] + 2*int((atom_type[i] == 0) & (int(onp.sum(atom_type[i+1:i+4])) == 3)) for i in range(len(atom_type))],dtype=i32)
Natm = len(R0)

# Load preprocessed bond info.
# For the generation, see the help functions in geo/sort_fn.py
bond_info = np.load('geo/'+geo_name+'_bonds.npz')
bond_type_info = np.load('geo/'+geo_name+'_bond_type.npz')
bonds,bends,dihs = np.array(bond_info['bond'],dtype=i32),\
                   np.array(bond_info['bend'],dtype=i32),\
                   np.array(bond_info['dih'],dtype=i32)
bond_type,bend_type,dih_type = np.array(bond_type_info['bondT'],dtype=i32),\
                               np.array(bond_type_info['bendT'],dtype=i32),\
                               np.array(bond_type_info['dihT'],dtype=i32)

# Bond list for shifting H (used for MBD force projection)
id_C = []
id_CH = []
id_H = []
current_carbon = None
for i, atom in enumerate(atom_type_pure):
    if atom == 0:
        current_carbon = i   # update the current carbon index
        id_C.append(i)
    else:  # atom is hydrogen
        id_CH.append(current_carbon)
        id_H.append(i)
id_C = np.array(id_C,dtype=i32)
id_CH = np.array(id_CH,dtype=i32)
id_H = np.array(id_H,dtype=i32)


# parameters and unit convention (kj/mol, nm, ps, 16.6Bar)
A2nm = f32(0.1)
R0 = R0 * A2nm  # A to nm
unit_cell = np.array(unit_cell[1:],dtype=f32) * A2nm  # A to nm
K_B = f32(8.617*1e-5*96.485)  # kj/mol/K
kT = f32(300.00) * K_B
P = f32(1/16.6*1.013)

# set simulation condition
box = unit_cell
length = f32(box[0,0])
print('length = %.2f' % (length/A2nm))
displacement, shift = space.periodic(length, wrapped=wrap)
displacement_A, _ = space.periodic(length/A2nm, wrapped=wrap) # Angstrom used for trimmed-SchNet
_, shift_wrap = space.periodic(length, wrapped=True) # Wrapped configurations for neigbor search

# Harmonic TraPPE model for covalent interactions
print('initialize TraPPE models...')
# parameters from Trappe
atom_mass = np.array([int(l == 1)*1.0078+int(l == 0)*12 for l in atom_type_pure],dtype=f32)
atom_mass_3 = np.array([[int(l == 1)*1.0078+int(l == 0)*12,]*3 for l in atom_type_pure],dtype=f32)
r0 = np.array([1.5350, 0.55],dtype=f32) * A2nm
kr = np.array([1e6, 1e6],dtype=f32)
t0 = np.array([112.7,110.7,107.8],dtype=f32)*f32(np.pi/180)
kt = np.array([58765, 1e7, 1e7],dtype=f32)*K_B
c0 = np.array([0,0,0],dtype=f32)*K_B
c1 = np.array([355.03, 0,   0],dtype=f32)*K_B
c2 = np.array([-68.19, 0,   0],dtype=f32)*K_B
c3 = np.array([791.32, 0,   0],dtype=f32)*K_B
c4 = np.array([0,      854, 717],dtype=f32)*K_B

# Set energy functions
energy_fn_bond = energy.harmonic_bond(displacement, bonds, bond_type=bond_type, order=1, length=r0, k=kr)
energy_fn_bend = energy.harmonic_bond(displacement, bends, bond_type=bend_type, order=2, length=t0, k=kt)
energy_fn_dih = energy.trappe_dihedral_bond(displacement, dihs, bond_type=dih_type,c0=c0,c1=c1,c2=c2,c3=c3,c4=c4)

# LJ parameters
sigma = np.array([[3.650,  3.480,  3.475],
                  [3.480,  3.310,  3.305],
                  [3.475,  3.305,  3.300]],dtype=f32) * A2nm
epsilon = np.array([[5.000,  8.746,  4.472],
                    [8.746,  15.30,  7.823],
                    [4.472,  7.823,  4.000]],dtype=f32) * K_B
r_buffer = 0.5 * A2nm
# Three types of dispersion models. They couple to the same 12th order repulsive term as in LJ.
if decay_rate == 4:  # Effective decay
    alpha_46 = f32(0.6786)
    print('scaling factor: ', alpha_46)
    alpha = np.ones((3, 3),dtype=np.float32) * alpha_46
    r_cut = 14 * A2nm
    user_cut = None
    partition_format = partition.OrderedSparse
elif decay_rate == 6:  # LJ
    alpha = np.ones((3, 3), dtype=np.float32)
    r_cut = 10 * A2nm
    user_cut = None
    partition_format = partition.OrderedSparse
elif decay_rate == 0:  # Trimmed-SchNet
    alpha = np.ones((3, 3), dtype=np.float32)
    r_cut = 14 * A2nm
    user_cut = N_cut - 1
    partition_format = partition.Dense
else:
    raise ValueError('rate undefined')
# The user_cut of neigborlist is added to save memory and achieve fixed number of neighbor.
# Add the following script to jax_md/partition.py/neighbor_list(*add argument)/neighbor_list_fn()/neighbor_fn():
#     if user_cut is not None:
#         idx = idx[:, :user_cut]

# Build vdW energy function and neighbor function.
# Note that some close-neighbor interactions may be skipped, including intra-chain neighbor (required by TraPPE),
# and some close distance neighbor (to avoid singularity)
# To skip neighbors, insert the following mask operation to jax_md/smap.py/pair_neighbor_list(*extra arguments needed):
#     # skip some close neighbors in the chain (by index)
#     if partition.is_sparse(neighbor.format):
#       mask = jnp.abs(neighbor.idx[0] - neighbor.idx[1]) > skip_neighbor
#       mask = mask & (jnp.abs(neighbor.idx[0] - neighbor.idx[1]) != skip_tail)
#       out *= mask
#     else:
#       R_index = jnp.arange(neighbor.idx.shape[0])[:, None]
#       mask = jnp.abs(neighbor.idx - jnp.tile(R_index, (1, neighbor.idx.shape[1]))) > skip_neighbor
#       out *= mask
#
#     # damp some closest neighbors (by distance)
#     mask = (dR > skip_close)
#     out *= mask

_, energy_fn_vdW = energy.lj_low_rate_neighbor_list(
    displacement, length, species=atom_type, alpha=alpha, sigma=sigma, epsilon=epsilon,
    r_onset=r_cut-r_buffer, r_cutoff=r_cut, dr_threshold=r_buffer, skip_neighbor=11, skip_tail=0, rate=decay_rate,
    fractional_coordinates=False, skip_close=1.5*A2nm, format=partition_format, user_cut=user_cut)

# Wrapped neighbor function
# To meet the ordering requirement for the trimmed SchNet, the neighbor list need to be 'Dense' and sorted.
# The following sorting code needs to be added to partition.py/neighbor_list()/prune_neighbor_list_dense():
#     # sorting
#     sort_id = jnp.argsort(dR, axis=1)
#     dR = jnp.take_along_axis(dR, sort_id, axis=1)
#     idx = jnp.take_along_axis(idx, sort_id, axis=1)

neighbor_fn_wrap = partition.neighbor_list(displacement, length, r_cutoff=r_cut, dr_threshold=r_buffer, user_cut=user_cut,
    fractional_coordinates=False, format=partition_format)

# Trimmed-SchNet model
N_cut = 1000
ep0 = 200
num_interactions, embedding_dim, num_rbf, rbf_trainable = 1, 32, 100, True
N_extra = 50
idx_i = np.array([0,]*(N_cut-1) + list(range(1,N_cut)) + [1,]*N_extra + list(range(2,2+N_extra)) + [2,]*N_extra + list(range(3,3+N_extra)), dtype=np.int32)
idx_j = np.array(list(range(1,N_cut)) + [0,]*(N_cut-1) + list(range(2,2+N_extra)) + [1,]*N_extra + list(range(3,3+N_extra)) + [2,]*N_extra, dtype=np.int32)
seg_i = np.array([0,]*(N_cut-1) + list(range(1,N_cut)) + [1,]*N_extra + list(range(2,2+N_extra)) + [2,]*N_extra + list(range(3,3+N_extra)), dtype=np.int32)

sort_id = np.argsort(seg_i)
idx_i = np.array(idx_i[sort_id], dtype=np.int32)
idx_j = np.array(idx_j[sort_id], dtype=np.int32)
seg_i = np.array(seg_i[sort_id], dtype=np.int32)

schnet_flax = trimmed_SchNet(N_cut, num_interactions, embedding_dim, num_rbf, rbf_trainable, idx_i, idx_j, seg_i, displacement_A)

# LOAD: Pretrained on the trimmed-SchNet dataset, and fine-tuned
if decay_rate == 0 and ep0 > 0:
    print('loading pretrained model')
    current_path = '' # Need absolute path
    file_name_para = 'nnx_mse_gen_b%d_Nat%d_Np%d_BS%d_Ni%d_de%d_Nrbf%d_Trbf%d_ep%d' % \
                     (1, 1000, 6000, 18, num_interactions, embedding_dim, num_rbf, rbf_trainable, 200)
    checkpointer = ocp.StandardCheckpointer()
    graphdef, abstract_state = nnx.split(schnet_flax)
    state_restored = checkpointer.restore(current_path+'/training_log/check_points/' + file_name_para+'_ft_ep%d' % ep0, abstract_state)
    schnet_flax = nnx.merge(graphdef, state_restored)

# Function for MBD force computation.
# With the ML-surrogate, the inference is in a batched manner according to the VRAM.
batch_size = 2000
@jit
def Schnet_force_fn(R0, batch_size=batch_size, **kwargs):
    neighbor = kwargs['neighbor']
    # shift R
    distance_fn = vmap(displacement)
    dist_shift = distance_fn(R0[id_H], R0[id_CH])
    R = R0.at[id_H].add(dist_shift)

    # need A as input for real space displacement
    R = R / A2nm

    idx_full = np.concatenate((np.arange(Natm)[:, None], neighbor.idx), axis=1)
    R_neigh = R[idx_full]
    type_neigh = atom_type_pure[idx_full]
    force = np.zeros((Natm, 3), dtype=f32)
    N_steps = math.ceil(Natm / batch_size)
    for i in range(N_steps):
        fi = schnet_flax((R_neigh[batch_size * i:batch_size * i + batch_size],
                        type_neigh[batch_size * i:batch_size * i + batch_size]),**kwargs)
        force = force.at[batch_size * i:batch_size * i + batch_size, :].set(fi[:,0,:])
    force = force * f32(4.9615*1e4/1e3)  # Ha/Bohr to kj/mol/nm, 1e3 is the factor used in training

    ## redistribution, see the reference chapter for details.
    norm_dist = np.linalg.norm(dist_shift, axis=1, keepdims=True)
    e_CH = dist_shift / norm_dist
    F_H = force[id_H]
    F_L = np.sum(F_H * e_CH, axis=1, keepdims=True) * e_CH
    F_P = F_H - F_L

    F_Heff = F_L + 2 * F_P  # redistributed force for effective hydrogen
    F_Cadd = - F_P

    force = force.at[id_H].set(F_Heff)
    force = force.at[id_CH].add(F_Cadd)

    # remove rigid body motion
    force_residual = np.sum(force, axis=0) /np.sum(atom_mass)
    force = force.at[id_H].add(-force_residual * 1.0078)
    force = force.at[id_C].add(-force_residual * 12)

    # Residual torque correction, not yet implemented. See CORE2026-QuMuLS proposal.
    # R_com = np.sum(R0*atom_mass[:, np.newaxis], axis=0) /np.sum(atom_mass)
    # R_rel = R0 - R_com
    # tau = np.sum(np.cross(R_rel, force), axis=0)
    #
    # rr = np.sum(R_rel * R_rel, axis=1)  # (Ng,)
    # I_w = np.sum(atom_mass[:, None, None] * (rr[:, None, None] * np.eye(3) - R_rel[:, :, None] * R_rel[:, None, :]),axis=0)
    # omega = - np.linalg.solve(I_w+1e-6 * np.eye(3), tau)
    #
    # jax.debug.print('F: {}', np.linalg.norm(force, axis=0))
    # dF = atom_mass[:, None] * np.cross(omega[None, :], R_rel)
    # force = force + dF

    return force

# combine different interaction functions.
def combined_energy_fn(R,**kwargs):
    return energy_fn_bond(R,**kwargs) + energy_fn_bend(R,**kwargs) + energy_fn_dih(R,**kwargs) + energy_fn_vdW(R,**kwargs)

force_combined = quantity.force(combined_energy_fn)
def combined_force_fn(R,**kwargs):
    return force_combined(R,**kwargs) + Schnet_force_fn(R,**kwargs)

if decay_rate == 0:
    input_fn = combined_force_fn
else:
    input_fn = combined_energy_fn

# Initiate nvt
dt = 0.05e-3  # 1 fs = 1e-3 ps
steps = 100
init, apply = simulate.nvt_nose_hoover(input_fn, shift, dt, kT, tau=200*dt)
nbrs = neighbor_fn_wrap.allocate(shift_wrap(R0,0))

saved_name = geo_name + '_wp%d_a%d_K%d_t%.1f_b%.2f_nvt' % (wrap, decay_rate, 300, dt*steps+t_resume, length/A2nm)
print('simulating...'+saved_name)
saved_path = 'geo/nvt/seed%d/' % seed
print('will be saved to '+saved_path)

state = init(key, R0, mass=atom_mass, neighbor=nbrs)

write_every = int(0.2/dt)
@loop_tqdm(steps, print_rate=steps/10)
def step_fn(i, state_nbrs_log):
    state, nbrs, log = state_nbrs_log

    # Log information about the simulation.
    T = quantity.temperature(momentum=state.momentum, mass=atom_mass_3)
    log['kT'] = log['kT'].at[i].set(T)

    # Record positions every `write_every` steps.
    log['position'] = lax.cond((i+1) % write_every == 0,
                               lambda p: p.at[(i+1) // write_every-1].set(state.position),
                               lambda p: p,
                               log['position'])

    # Take a simulation step.
    state = apply(state, kT=kT, neighbor=nbrs)

    # update nbrs with wrapped geo.
    geo_wrapped = shift_wrap(state.position,0)
    nbrs = nbrs.update(geo_wrapped)
    return state, nbrs, log


log = {
    'kT': np.zeros((steps,)),
    'position': np.zeros((steps // write_every,) + R0.shape)

}

s = time.time()
state, nbrs, log = lax.fori_loop(0, steps, step_fn, (state, nbrs, log))
e = time.time()

# print(nbrs.did_buffer_overflow)
print(log['kT'][-100:]/K_B)
print(log['P'][-100:]*16.6)
print('%d steps loop cost:' % steps, e-s)

# save
onp.savetxt(saved_path + 'T_'+saved_name+'.txt',log['kT']/K_B)
onp.savez_compressed(saved_path + 'geo_'+saved_name,coord=log['position']/A2nm)
# #
unit_cell = onp.array(unit_cell)
SaveGeometry(state.position/A2nm, np.eye(3)*length/A2nm, atom_type_pure, file_name=saved_path + 'geo_'+saved_name+'.gen')
