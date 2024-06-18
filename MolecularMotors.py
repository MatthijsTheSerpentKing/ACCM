import hoomd
import numpy as np
import argparse
from tqdm import tqdm
import gsd.hoomd
import numpy as np 

class MolecularMotors(hoomd.md.force.Custom):
   def __init__(self, Fact, N_monomers, N_polymers, box_size, dim):
       super().__init__()

       self.N_monomers = N_monomers
       self.N_polymers = N_polymers
       self.box_size = box_size
       self.Fact = Fact
       self.dim = dim

       self.motor_indices = np.zeros((N_monomers, 3), dtype=np.int64)
       self.motor_indices[1:,0] = np.arange(N_monomers-1)
       self.motor_indices[:,1] = np.arange(N_monomers)
       self.motor_indices[:-1,2] = np.arange(1, N_monomers)
       self.motor_indices[-1,2] = N_monomers - 1
       self.motor_indices = np.tile(self.motor_indices, (self.N_polymers, 1))

       self.polymer_indices = np.arange(0, N_polymers)
       self.polymer_indices = np.repeat(self.polymer_indices, N_monomers*3)
       self.polymer_indices = np.reshape(self.polymer_indices, (self.N_polymers*N_monomers, 3))
       self.polymer_indices *= N_monomers

       self.motor_indices += self.polymer_indices


   def set_forces(self, timestep):
       # Determine direction of forces
       with self._state.cpu_local_snapshot as snapshot:

           # Get particle indices in device database
           dev_indices = snapshot.particles.rtag[self.motor_indices]
           
           # Get particle locations
           r1 = snapshot.particles.position[dev_indices[:,0]]
           r1 += snapshot.particles.image[dev_indices[:,0]] * self.box_size # Unwrap box

           r2 = snapshot.particles.position[dev_indices[:,2]]
           r2 += snapshot.particles.image[dev_indices[:,2]] * self.box_size # Unwrap box

           # Determine tangent
           tangent = (r2 - r1) / 2
           
       # Apply forces
       with self.cpu_local_force_arrays as arrays:

           forces = np.zeros((self.N_polymers*self.N_monomers, 3))
           np.add.at(forces, dev_indices[:,1], self.Fact * tangent)
           arrays.force += forces