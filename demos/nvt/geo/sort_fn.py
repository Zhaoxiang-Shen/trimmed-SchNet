'''
Help function for sorting bond info of polymer chains
'''

import jax.numpy as jnp
import numpy as onp
def calculate_bond_data(displacement_or_metric, R, dr_cutoff, species=None):
    if (not (species is None)):
      assert (False)

    N = jnp.shape(R)[0]
    metric = space.map_product(space.canonicalize_displacement_or_metric(displacement_or_metric))
    dr = metric(R, R)

    dr_include = jnp.triu(jnp.where(dr < dr_cutoff, 1, 0)) - jnp.eye(R.shape[0], dtype=jnp.int32)
    index_list = jnp.dstack(jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij'))

    i_s = jnp.where(dr_include == 1, index_list[:, :, 0], -1).flatten()
    j_s = jnp.where(dr_include == 1, index_list[:, :, 1], -1).flatten()
    ij_s = jnp.transpose(jnp.array([i_s, j_s]))


    bonds = ij_s[(ij_s != jnp.array([-1, -1]))[:, 1]]
    lengths = dr.flatten()[(ij_s != jnp.array([-1, -1]))[:, 1]]
    return bonds, lengths


def angle_bond(bond):
  bond = onp.array(bond)
  unique_atoms = onp.unique(bond)
  connections = {atom: [] for atom in unique_atoms}
  # Populate the connections dictionary
  for pair in bond:
    a, b = pair
    connections[a].append(b)
    connections[b].append(a)
    # Generate the bending array
  bending_array = []
  for center_atom, connected_atoms in connections.items():
    if len(connected_atoms) < 2:
      continue
    # Form all possible pairs of connected atoms
    for i in range(len(connected_atoms)):
      for j in range(i + 1, len(connected_atoms)):
        atom1 = connected_atoms[i]
        atom2 = connected_atoms[j]
        # Add the triplet (center_atom, atom1, atom2)
        bending_array.append([center_atom, atom1, atom2])

  return jnp.array(bending_array,dtype=jnp.int32)


def dih_bond(bond):
  bond = onp.array(bond)
  # Create a dictionary to store connections
  connections = {}
  for pair in bond:
    a, b = pair
    if a not in connections:
      connections[a] = []
    if b not in connections:
      connections[b] = []
    connections[a].append(b)
    connections[b].append(a)

  # Generate dihedral array
  dihedral_array = []
  dihedral_array_out = []

  # Helper function to find dihedrals recursively
  def find_dihedrals(atom1, atom2, path):
    if len(path) == 4:
      if path not in dihedral_array and [path[-1], path[-2], path[-3], path[-4]] not in dihedral_array:
        dihedral_array.append(path)
      return
    for neighbor in connections[atom2]:
      if neighbor != atom1 and neighbor not in path:
        find_dihedrals(atom2, neighbor, path + [neighbor])

  # Iterate over each bond to start forming dihedrals
  i = 0
  for pair in bond:
    a, b = pair
    find_dihedrals(a, b, [a, b])
    find_dihedrals(b, a, [b, a])

    i += 1
    if i % 1000 == 0:
      dihedral_array_out += dihedral_array
      dihedral_array = []
    if i % 10000 == 0:
      print('dih over bonds:', i, len(dihedral_array_out))

  dihedral_array_out += dihedral_array
  return jnp.array(dihedral_array_out,dtype=jnp.int32)


def bond_type_sort(bond, atom_type, order=1):
  bond_type = []
  if order == 1:
    # CC=[0,0]=0, CH=[0,1]=1
    for i in bond:
      bond_i = [int(atom_type[i[0]]), int(atom_type[i[1]])]
      bond_type.append(int(bond_i == [0, 1]))
  elif order == 2:
    # CCC=[0,0,0]=0,CHC=[0,1,0]=1,CHH=[0,1,1]=2
    for i in bond:
      bond_i = [int(atom_type[i[0]]), int(atom_type[i[1]]), int(atom_type[i[2]])]
      bond_type.append(int(bond_i == [0, 1, 0]) + int(bond_i == [0, 0, 1]) + int(bond_i == [0, 1, 1]) * 2)
  elif order == 3:
    # CCCC=[0,0,0,0]=0,CCCH=[0,0,0,1]=1,HCCH=[1,0,0,1]=2
    for i in bond:
      bond_i = [int(atom_type[i[0]]), int(atom_type[i[1]]), int(atom_type[i[2]]), int(atom_type[i[3]])]
      bond_type.append(int(bond_i == [0, 0, 0, 1]) + int(bond_i == [1, 0, 0, 0]) + int(bond_i == [1, 0, 0, 1]) * 2)
  else:
    raise ValueError('order has to be 1 or 2 or 3')
  return jnp.array(bond_type,dtype=jnp.int32)