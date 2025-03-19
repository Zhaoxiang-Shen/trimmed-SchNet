# trimmed SchNet 
## Machine learning surrogate models of many-body dispersion interactions in polymer melts
Trimmed SchNet is a machine learning surrogate model designed to predict many-body dispersion (MBD) forces in polymer melts. Building on SchNet's core geometric encoding and continuous-filter convolutions, the model features a simplified architecture with trimmed atomic connections, significantly reducing computational costs while retaining essential many-body correlations. It also includes trainable radial basis function (rbf) encoding to minimize the number of encoding bases and adopts a unit-specific batching strategy inspired by polymer repeat units to enhance training convergence. The details of the architecture can be found in the paper.

The model takes an atomic cluster (vdW interaction cutoff) as input and predict the MBD force on the center atoms. It has been validated on datasets from polyethylene, polypropylene, and polyvinyl chloride melts, demonstrating high predictive accuracy and robust generalization across diverse polymer systems. The polymer melt datasets used for the training and testing are available in [ZENODO](https://doi.org/10.5281/zenodo.15012728). 

This repository contains the TensorFlow implementations for the model and equivalent [FLAX_NNX](https://github.com/google/flax) implementation for incorporation in [JAX_MD](https://github.com/jax-md/jax-md). A demo for JAX_MD NVT simulation using the model is provided. 


## Dependencies
- `tensorflow==2.15.0`
- `jax==0.4.30`
- `jax-md==0.2.8`
- `flax==0.8.5`


## Cite
```
@misc{trimmed-SchNet,
      title={Machine learning surrogate models of many-body dispersion interactions in polymer melts}, 
      author={Zhaoxiang Shen and Raúl I. Sosa and Jakub Lengiewicz and Alexandre Tkatchenko and Stéphane P. A. Bordas},
      year={2025},
}
```
