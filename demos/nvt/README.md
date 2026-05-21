# Towards large-scale quantum-informed molecular dynamics simulations: implementing a machine learning surrogate for many-body dispersion in polymer melts
## A JAX-MD NVT code using TraPPE and trimmed-SchNet

This repository provides a full molecular dynamics (MD) simulation framework for NVT simulations of polyethylene (PE) melts based on JAX-MD. The implementation incorporates the TraPPE force field for covalent interactions and vdW repulsion, while the dispersion contribution can be modeled using three different approaches:

- Classical Lennard-Jones \(r^{-6}\) dispersion (LJ-6),
- Effective Lennard-Jones \(r^{-4}\) dispersion (LJ-4),
- A trimmed-SchNet machine learning surrogate model for many-body dispersion (MBD).

This codebase was developed to support Chapter 4 of the doctoral thesis of Zhaoxiang Shen, as well as prospective studies on polymer dynamics, including analyses such as radial distribution functions (RDF), radius of gyration, and velocity power spectrum. The theoretical formulation, modeling assumptions, and simulation details implemented in this demo follow the descriptions provided in the thesis.

---
## Implementation details:
1. `add_to_jax_md` contains supplementary functions intended to be added to JAX-MD in order to support the complete TraPPE model and related customized simulation functionalities.
2. The main execution script `nvt_full.py` contains several short patches/modifications to JAX-MD required for full compatibility between JAX-MD, the TraPPE framework, and the trimmed-SchNet ML surrogate model.
3. Initial configurations: three equilibrated PE melt configurations are provided as initial states for simulations. These configurations can be used directly for reproduction of the reported results or for further dynamics analysis.
4. The trimmed-SchNet surrogate model provided in this repository was first pretrained using the full dataset introduced with trimmed-SchNet, and subsequently fine-tuned on the 9k-atom PE melt system used in this work. This is necessary because the original training dataset was generated using the CHARMM force field, which introduces discrepancies with the TraPPE-based model employed in the present simulations.
5. Several simulation trajectories and fine-tuned datasets have been open-sourced on [ZENODO](??) to support further analysis and benchmarking studies.

## Cite
```
@phdthesis{SHEN2025thesis,
	AUTHOR = {Shen, Z.},
	EPRINT = {https://hdl.handle.net/10993/66537},
	EPRINTTYPE = {hdl},
	TITLE = {Efficient Quantum-Informed Computational Frameworks for Mechanics of Materials: Many-Body Dispersion and Machine Learning Surrogate Modeling},
	LANGUAGE = {English},
	YEAR = {2025},
	SCHOOL = {University of Luxembourg}}
```
