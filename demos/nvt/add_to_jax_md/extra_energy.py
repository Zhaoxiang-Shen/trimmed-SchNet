"""
Extra energy functions to support TraPPE model.
Add/Import them to jax_md/energy.py
"""

def harmonic_bond(displacement_or_metric: DisplacementOrMetricFn,
                       bond: Array,
                       bond_type: Optional[Array]=None,
                       order: int=1,
                       length: Array=1,
                       global_length: Optional[Array] = None,
                       k: Array=1,
                       alpha: Array=2) -> Callable[[Array], Array]:
  """Convenience wrapper to compute energy of particles bonded by springs."""
  length = maybe_downcast(length)
  k = maybe_downcast(k)
  alpha = maybe_downcast(alpha)
  return smap.bond_general(
    simple_spring,
    displacement_or_metric,
    order,
    bond,
    bond_type,
    ignore_unused_parameters=True,
    length=length,
    global_length=global_length,
    epsilon=k,
    alpha=alpha)

def trappe_dihedral_bond(displacement_or_metric: DisplacementOrMetricFn,
                       bond: Array,
                       bond_type: Optional[Array]=None,
                       c0: Array=1,c1: Array=1,c2: Array=1,c3: Array=1,c4: Array=1) -> Callable[[Array], Array]:
  """Convenience wrapper to compute energy of particles bonded by springs."""
  c0 = maybe_downcast(c0)
  c1 = maybe_downcast(c1)
  c2 = maybe_downcast(c2)
  c3 = maybe_downcast(c3)
  c4 = maybe_downcast(c4)
  return smap.bond_trappe(
    dihedral_trappe,
    displacement_or_metric,
    bond,
    bond_type,
    ignore_unused_parameters=True,
    c0=c0,c1=c1,c2=c2,c3=c3,c4=c4)


def lennard_jones_low_rate(dr: Array,
                  sigma: Array=1,
                  epsilon: Array=1,
                  alpha: Array=1,
                  rate: int=4,
                  **unused_kwargs) -> Array:
  idr1 = (sigma / dr)
  idr2 = idr1 * idr1
  idr4 = idr2 * idr2
  idr6 = idr2 * idr2 * idr2
  idr12 = idr6 * idr6
  if rate == 4:
    return jnp.nan_to_num(f32(4) * epsilon * (idr12 - alpha * idr4))
  elif rate == 5:
    return jnp.nan_to_num(f32(4) * epsilon * (idr12 - alpha * idr4 * idr1))
  elif rate == 6:
    return jnp.nan_to_num(f32(4) * epsilon * (idr12 - alpha * idr6))
  elif rate == 0:
    return jnp.nan_to_num(f32(4) * epsilon * idr12)
  else:
    raise ValueError('rate has to be 4 or 5 or 6')

def lj_low_rate_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: Box,
    species: Optional[Array]=None,
    sigma: Array=1.0,
    epsilon: Array=1.0,
    r_onset: float=2.0,
    r_cutoff: float=2.5,
    dr_threshold: float=0.5,
    per_particle: bool=False,
    fractional_coordinates: bool=False,
    alpha: Array = 1,
    rate: int = 4,
    skip_neighbor: int=0,
    skip_tail: int=0,
    skip_close: float=0,
    user_cut: Optional[int]=None,
    format: partition.NeighborListFormat=partition.OrderedSparse,
    **neighbor_kwargs
    ) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
  """Convenience wrapper to compute :ref:`Lennard-Jones <lj-pot>` using a neighbor list."""

  sigma = maybe_downcast(sigma)
  epsilon = maybe_downcast(epsilon)
  r_onset = maybe_downcast(r_onset)
  r_cutoff = maybe_downcast(r_cutoff)
  dr_threshold = maybe_downcast(dr_threshold)
  alpha = maybe_downcast(alpha)
  skip_close = maybe_downcast(skip_close)

  neighbor_fn = partition.neighbor_list(
    displacement_or_metric,
    box_size,
    r_cutoff,
    dr_threshold,
    user_cut=user_cut,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **neighbor_kwargs)
  energy_fn = smap.pair_neighbor_list(
    multiplicative_isotropic_cutoff(lennard_jones_low_rate, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    skip_neighbor=skip_neighbor,
    skip_tail=skip_tail,
    ignore_unused_parameters=True,
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    alpha=alpha,
    rate=rate,
    skip_close=skip_close,
    reduce_axis=(1,) if per_particle else None)

  return neighbor_fn, energy_fn