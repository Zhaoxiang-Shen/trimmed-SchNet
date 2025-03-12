"""
Key modules for the trimmed SchNet architecture coded with FLAX_NNX
Coded to be compatible with the geometric description in JAX_MD

Limitations remains compared to TensorFlow, such as constraints to the rbf parameters
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any
from functools import partial

class ShiftedSoftplus(nnx.Module):
    def __call__(self, x):
        return nnx.softplus(x) - jnp.log(2.0)

class RBFExpansion(nnx.Module):
    num_rbf: int
    rbf_trainable: bool
    cutoff: float

    def __init__(self, num_rbf, rbf_trainable, cutoff=15.0):
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        centers_init = jnp.linspace(0, cutoff, num_rbf)
        gamma_init = jnp.ones(num_rbf) * 10.0
        self.centers = nnx.Param(centers_init, mutable=rbf_trainable)
        self.gamma = nnx.Param(gamma_init, mutable=rbf_trainable)

    def __call__(self, distances):
        # bounds setting is not available
        distances = jnp.expand_dims(distances, -1)
        return jnp.exp(-self.gamma * (distances - self.centers) ** 2)

# Continuous filter convolution layer with trainable RBF centers and gamma
class ContinuousFilterConvolution(nnx.Module):
    embedding_dim: int
    num_rbf: int
    rbf_trainable: bool

    def __init__(self, embedding_dim, Nat, num_rbf, rbf_trainable):
        self.rbf = RBFExpansion(num_rbf,rbf_trainable)
        self.dense1 = nnx.Linear(num_rbf, embedding_dim, rngs=nnx.Rngs(0))
        self.dense2 = nnx.Linear(embedding_dim, embedding_dim, rngs=nnx.Rngs(0))
        self.activation = ShiftedSoftplus()
        self.Nat = Nat
    def __call__(self, atom_features, distances, idx_j, seg_i):
        expanded_distances = self.rbf(distances)
        filters = self.dense1(expanded_distances)
        filters = self.activation(filters)
        filters = self.dense2(filters)
        filters = self.activation(filters)

        atom_features = atom_features[idx_j]
        x = atom_features * filters
        x = jax.ops.segment_sum(x, seg_i, num_segments=self.Nat)
        return x

# Interaction Block that uses Continuous Filter Convolution
class InteractionBlock(nnx.Module):
    embedding_dim: int
    dropout_rate: float
    num_rbf: int
    rbf_trainable: bool

    def __init__(self, embedding_dim, Nat, num_rbf, rbf_trainable):
        self.dense1 = nnx.Linear(embedding_dim,embedding_dim, rngs=nnx.Rngs(0))
        self.dense2 = nnx.Linear(embedding_dim,embedding_dim, rngs=nnx.Rngs(0))
        self.dense3 = nnx.Linear(embedding_dim,embedding_dim, rngs=nnx.Rngs(0))
        self.activation = ShiftedSoftplus()
        self.continuous_conv = ContinuousFilterConvolution(embedding_dim, Nat, num_rbf, rbf_trainable)

    def __call__(self, atom_features, distances, idx_j, seg_i):
        x = self.dense1(atom_features)
        x = self.continuous_conv(x, distances, idx_j, seg_i)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dense3(x)
        return atom_features + x

# Main trimmed_SchNet Model with Interaction Blocks
class trimmed_SchNet(nnx.Module):
    num_interactions: int
    embedding_dim: int
    num_rbf: int
    rbf_trainable: bool
    idx_i: Any
    idx_j: Any
    seg_i: Any

    def __init__(self, Nat, num_interactions, embedding_dim, num_rbf, rbf_trainable,  idx_i, idx_j, seg_i, displacement_fn):
        self.atom_embedding = nnx.Embed(10, embedding_dim, rngs=nnx.Rngs(0))
        self.interaction_blocks = [InteractionBlock(embedding_dim, Nat, num_rbf, rbf_trainable) for _ in range(num_interactions)]
        self.activation = ShiftedSoftplus()
        self.dense = nnx.Linear(embedding_dim, embedding_dim//2, rngs=nnx.Rngs(0))
        self.energy_output = nnx.Linear(embedding_dim//2, 1, use_bias=False, rngs=nnx.Rngs(0))
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.seg_i = seg_i
        self.displacement_fn = displacement_fn


    def __call__(self, inputs, **kwargs):
        # made to be compatible with JAX_MD
        positions, atom_types = inputs
        if 'box' in kwargs:
            distance_fn = jax.vmap(partial(self.displacement_fn, box=kwargs['box']))
        else:
            distance_fn = jax.vmap(self.displacement_fn)

        def total_energy(positions, atom_types):
            atom_features = self.atom_embedding(atom_types)

            # Using JAX_MD distance function with periodic boundary conditions
            distances = distance_fn(positions[self.idx_i], positions[self.idx_j])
            distances = jnp.sqrt(jnp.sum(distances ** 2, axis=-1) + 1e-8)

            for block in self.interaction_blocks:
                atom_features = block(atom_features, distances, self.idx_j, self.seg_i)

            atom_features = self.dense(atom_features)
            atom_features = self.activation(atom_features)

            energy = self.energy_output(atom_features)
            return jnp.sum(energy, axis=0)[0]
        forces = -jax.vmap(nnx.grad(total_energy))(positions, atom_types)
        return forces[:,:1,:]