# ACCM

**Advanced Computational Condensed Matter**

This repository contains Python code for simulating and analysing active polymer systems driven by molecular-motor-like forces using **HOOMD-blue**.

The project models polymers as chains of bonded particles and introduces a custom active force that acts along the local polymer tangent. The simulations are performed using Langevin molecular dynamics, allowing the behaviour and conformational dynamics of active polymers to be investigated computationally.

## Overview

The simulations describe polymers consisting of connected monomers confined within a circular boundary. Each polymer is represented by a chain of particles connected by harmonic bonds and bending interactions.

In addition to the passive molecular interactions, a custom `MolecularMotors` force is implemented. For each monomer, the local tangent of the polymer is calculated from neighbouring particles, and an active force is applied along this direction. This provides a simple model of molecular motors generating active motion along a polymer.

The simulations are two dimensional.

## Physical model

The simulation combines several components:

* **Harmonic bond potential** to maintain the polymer backbone.
* **Harmonic bending potential** to control the polymer's flexibility.
* **Lennard-Jones interactions** between particles.
* **Lennard-Jones interaction with a spherical confining boundary**.
* **Active molecular-motor force** acting along the local polymer tangent.
* **Langevin dynamics** to model thermal fluctuations and dissipative interactions with the surrounding medium.

The main simulation parameters can be controlled from the command line, including the number of polymers and monomers, active force, bond and bending stiffness, temperature, particle mass, simulation timestep, confinement radius, and simulation duration.

## Simulation

The main simulation is implemented in [`run.py`](run.py).

A simulation is initialized by constructing the polymer configuration and defining the interaction potentials. HOOMD-blue then integrates the equations of motion using Langevin dynamics.

The simulation consists of an initial equilibration phase followed by the production run. Particle configurations are periodically written to a **GSD** trajectory file for subsequent analysis.

The default simulation parameters include:

```text
Number of polymers:  1
Monomers per polymer: 100
Active force: 100
Bond stiffness: 50000
Bending stiffness: 300
Temperature: 1
Confinement radius: 500
Simulation time: 1e7
```

These values can be modified using command-line arguments.

### Example

```bash
python run.py --N_polymers 1 --N_monomers 100 --Fact 100 --kappa_bend 300 --simid test
```

The simulation produces a `.gsd` trajectory containing the particle positions throughout the simulation.

## Custom molecular-motor force

The custom active force is implemented in [`MolecularMotors.py`](MolecularMotors.py).

For each monomer, neighbouring particles are used to determine the local polymer tangent. The molecular-motor force is then applied in the direction of this tangent:

```text
r₁ ─── r₂ ─── r₃
          ↑
     active force
```

The implementation uses HOOMD-blue's custom-force interface and directly accesses the simulation state to determine particle positions and apply the resulting forces.

## Initialisation

[`utils.py`](utils.py) contains the routines used to construct the initial HOOMD-blue simulation frame.

## Analysis

The repository also contains Python routines for analysing the resulting GSD trajectories.

Implemented analysis includes:

### Polymer extension

The distance between the first and last monomer can be calculated to determine the polymer's head-to-tail length.

### Centre-of-mass motion

The centre of mass of the polymer is calculated for every recorded timestep. The displacement of the centre of mass between consecutive recorded frames can then be used to characterise the polymer's translational motion.

### Spiral number

The **spiral number** characterises the total angular rotation of the polymer backbone.

### Gyration radius

The code contains a calculation of the polymer's gyration radius as a measure of its spatial extension around its centre of mass.

### Tangent correlations

Both **static** and **dynamic tangent correlations** are implemented.

The static correlation measures how the orientation of polymer segments is correlated along the chain, while the dynamic correlation follows the evolution of tangent orientations over time.

These quantities can be used to investigate the conformational and dynamical behaviour of the active polymers.

## Repository structure

```text
ACCM/
├── MolecularMotors.py   # Custom molecular-motor force
├── run.py               # Main HOOMD-blue simulation
├── utils.py             # Simulation and polymer initialisation
├── load_data.py         # GSD trajectory analysis
└── load_file_new        # Extended trajectory-analysis routines
```

## Requirements

The simulation requires:

* Python
* Linux (or Ubuntu)
* [HOOMD-blue](https://hoomd-blue.readthedocs.io/)
* NumPy
* tqdm
* GSD
* Matplotlib

The analysis scripts additionally use GSD and Matplotlib for reading trajectories and generating plots.

## Methodology

The computational workflow is:

```text
Initial polymer configuration
            ↓
     Define interactions
            ↓
      Add active forces
            ↓
     Langevin dynamics
            ↓
       GSD trajectory
            ↓
        Data analysis
            ↓
Conformational & dynamical observables
```

This separation between simulation and post-processing allows the same trajectory data to be analysed using different physical observables without rerunning the simulation.

## Purpose

The project demonstrates the use of computational physics and molecular-dynamics simulations to investigate the behaviour of active polymer systems. It combines:

* physical modelling;
* numerical simulation;
* custom force implementation;
* parameterised computational experiments;
* trajectory analysis;
* statistical and geometrical analysis of polymer configurations.
