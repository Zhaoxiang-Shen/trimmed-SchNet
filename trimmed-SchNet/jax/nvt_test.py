import jax.numpy as np
from jax import random
from jax import lax, vmap, grad
from jax_tqdm import loop_tqdm
from model_lib_flax import trimmed_SchNet
import math
from jax_md import simulate, partition, space
from utils import Loadgen

f32 = np.float32
i32 = np.int32

# Load an random PE structure
file_name = 'N6E250_nvt.gen'
_, R0, _, atom_type, unit_cell = Loadgen(file_name,1)
atom_type = np.array(atom_type)
atom_mass = np.array([int(l == 0)*1.0078+int(l == 1)*12 for l in atom_type],dtype=f32)
Natm = len(R0)

# parameters and unit convention (kj/mol, nm, ps, 16.6Bar)
A2nm = f32(0.1)
R0 = R0 * A2nm  # A to nm
unit_cell = np.array(unit_cell[1:],dtype=f32) * A2nm  # A to nm
K_B = f32(8.617*1e-5*96.485)  # kj/mol/K
kT = f32(300.00) * K_B

# set simulation condition
box = unit_cell
length = f32(box[0,0])
displacement, shift = space.periodic(np.array([length,length,length]))

# vdW cut off
N_cut = 1000  # this can be also defined as an input to the neighbor_list() to reduce memory usage.
r_buffer = 0.5 * A2nm
r_cut = 14 * A2nm  # neighbor search by cutoff distance

# To meet the ordering requirement for the trimmed SchNet, the neighbor list need to be 'Dense' and sorted.
# The following sorting code needs to be added to partition.py/neighbor_list()/prune_neighbor_list_dense():
## sort_id = jnp.argsort(dR, axis=1)
## dR = jnp.take_along_axis(dR, sort_id, axis=1)
## idx = jnp.take_along_axis(idx, sort_id, axis=1)

# Create the neighbor list function
neighbor_fn = partition.neighbor_list(displacement,length,r_cut,r_buffer,
                                      fractional_coordinates=False, format=partition.Dense)
nbrs = neighbor_fn.allocate(R0)

# call the trimmed SchNet model
num_interactions, embedding_dim, num_rbf, rbf_trainable = 1, 32, 100, True
N_extra = 50
idx_i = np.array([0,]*(N_cut-1) + list(range(1,N_cut)) + [1,]*N_extra + list(range(2,2+N_extra)) + [2,]*N_extra + list(range(3,3+N_extra)), dtype=np.int32)
idx_j = np.array(list(range(1,N_cut)) + [0,]*(N_cut-1) + list(range(2,2+N_extra)) + [1,]*N_extra + list(range(3,3+N_extra)) + [2,]*N_extra, dtype=np.int32)
seg_i = np.array([0,]*(N_cut-1) + list(range(1,N_cut)) + [1,]*N_extra + list(range(2,2+N_extra)) + [2,]*N_extra + list(range(3,3+N_extra)), dtype=np.int32)

sort_id = np.argsort(seg_i)
idx_i = np.array(idx_i[sort_id], dtype=np.int32)
idx_j = np.array(idx_j[sort_id], dtype=np.int32)
seg_i = np.array(seg_i[sort_id], dtype=np.int32)
MBD_trimmed_SchNet = trimmed_SchNet(N_cut, num_interactions, embedding_dim, num_rbf, rbf_trainable, idx_i, idx_j, seg_i, displacement)
# Load pretrained model here
# ...

# a for_loop function to compute MBD forces using trimmed SchNet
def MBD_force_fn(R, neighbor=nbrs, batch_size=1000, N_cut=N_cut, **kwargs):
    idx_full = np.concatenate((np.arange(Natm)[:, None], neighbor.idx[:,:(N_cut-1)]), axis=1)
    R_neigh = R[idx_full]
    type_neigh = atom_type[idx_full]
    force = np.zeros((Natm, 3), dtype=f32)
    N_steps = math.ceil(Natm / batch_size)
    for i in range(N_steps):
        fi = MBD_trimmed_SchNet((R_neigh[batch_size * i:batch_size * i + batch_size],
                          type_neigh[batch_size * i:batch_size * i + batch_size]),**kwargs)
        force = force.at[batch_size * i:batch_size * i + batch_size, :].set(fi[:,0,:])
    return force

# combine force fn
def combined_force_fn(R,**kwargs):
    return MBD_force_fn(R,**kwargs)  # plus other parts of the FF

# NVT set up
dt = 0.1e-3
steps = 100
init, apply = simulate.nvt_nose_hoover(combined_force_fn, shift, dt, kT, tau=200*dt)
print('simulating...')
state = init(random.PRNGKey(0), R0, mass=atom_mass, neighbor=nbrs)

@loop_tqdm(steps, print_rate=1)
def step_fn(i, state_nbrs_log):
    state, nbrs = state_nbrs_log
    # Take a simulation step.
    state = apply(state, kT=kT, neighbor=nbrs)
    return state, nbrs

state, nbrs = lax.fori_loop(0, steps, step_fn, (state, nbrs))
