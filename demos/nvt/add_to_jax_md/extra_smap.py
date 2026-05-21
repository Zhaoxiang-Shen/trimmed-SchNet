"""
Extra smap functions to support energy functions for TraPPE.
Add/Import them to jax_md/smap.py
"""

def bond_general(fn: Callable[..., Array],
         displacement_or_metric: DisplacementOrMetricFn,
         order: int=1,
         static_bonds: Optional[Array]=None,
         static_bond_types: Optional[Array]=None,
         global_length: Optional[Array]=None,
         ignore_unused_parameters: bool=False,
         **kwargs) -> Callable[..., Array]:

  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)

  def compute_fn(R, order, bonds, bond_types, static_kwargs, dynamic_kwargs):
    _kwargs = merge_dicts(static_kwargs, dynamic_kwargs)
    _kwargs = _kwargs_to_bond_parameters(bond_types, _kwargs)
    # NOTE(schsam): This pattern is needed due to JAX issue #912.

    if order == 1:  # spring
      Ra = R[bonds[:, 0]]
      Rb = R[bonds[:, 1]]
      d = vmap(partial(space.canonicalize_displacement_or_metric(displacement_or_metric), **dynamic_kwargs), 0, 0)
      dr = d(Ra, Rb)
    elif order == 2:  # bend
      R1 = R[bonds[:, 0]]
      R2 = R[bonds[:, 1]]
      R3 = R[bonds[:, 2]]
      d_vec = vmap(partial(displacement_or_metric, **dynamic_kwargs), 0, 0)
      angle = vmap(quantity.angle_between_two_vectors, 0, 0)
      # cosine_angle = vmap(quantity.cosine_angle_between_two_vectors, 0, 0)
      R12 = d_vec(R2, R1)
      R13 = d_vec(R3, R1)
      theta = angle(R12, R13)
      dr = theta
    elif order == 3:  # dihedral
      R1 = R[bonds[:, 0]]
      R2 = R[bonds[:, 1]]
      R3 = R[bonds[:, 2]]
      R4 = R[bonds[:, 3]]
      d_vec = vmap(partial(displacement_or_metric, **dynamic_kwargs), 0, 0)
      angle = vmap(quantity.dihedral_angle, 0, 0)
      R12 = d_vec(R2, R1)
      R23 = d_vec(R3, R2)
      R34 = d_vec(R4, R3)
      phi = angle(R12, R23, R34)
      dr = phi
    else:
      raise ValueError(
          'order has to be 1 or 2 or 3')
    # print(_kwargs)
    return jnp.sum(fn(dr, global_length=global_length, **_kwargs))

  def mapped_fn(R: Array,
                bonds: Optional[Array]=None,
                bond_types: Optional[Array]=None,
                **dynamic_kwargs) -> Array:
    accum = f32(0)

    if bonds is not None:
      accum = accum + compute_fn(R, order, bonds, bond_types, kwargs, dynamic_kwargs)

    if static_bonds is not None:
      accum = accum + compute_fn(
          R, order,static_bonds, static_bond_types, kwargs, dynamic_kwargs)

    return accum
  return mapped_fn

def bond_trappe(fn: Callable[..., Array],
         displacement_or_metric: DisplacementOrMetricFn,
         static_bonds: Optional[Array]=None,
         static_bond_types: Optional[Array]=None,
         global_length: Optional[Array]=None,
         ignore_unused_parameters: bool=False,
         **kwargs) -> Callable[..., Array]:

  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)

  def compute_fn(R, bonds, bond_types, static_kwargs, dynamic_kwargs):
    _kwargs = merge_dicts(static_kwargs, dynamic_kwargs)
    _kwargs = _kwargs_to_bond_parameters(bond_types, _kwargs)
    R1 = R[bonds[:, 0]]
    R2 = R[bonds[:, 1]]
    R3 = R[bonds[:, 2]]
    R4 = R[bonds[:, 3]]
    d_vec = vmap(partial(displacement_or_metric, **dynamic_kwargs), 0, 0)
    cos_angle = vmap(quantity.dihedral_cos_angle, 0, 0)
    R12 = d_vec(R2, R1)
    R23 = d_vec(R3, R2)
    R34 = d_vec(R4, R3)
    dr = cos_angle(R12, R23, R34)
    return jnp.sum(fn(dr, global_length=global_length, **_kwargs))

  def mapped_fn(R: Array,
                bonds: Optional[Array]=None,
                bond_types: Optional[Array]=None,
                **dynamic_kwargs) -> Array:
    accum = f32(0)

    if bonds is not None:
      accum = accum + compute_fn(R, bonds, bond_types, kwargs, dynamic_kwargs)

    if static_bonds is not None:
      accum = accum + compute_fn(
          R, static_bonds, static_bond_types, kwargs, dynamic_kwargs)

    return accum
  return mapped_fn