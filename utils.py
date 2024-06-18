import hoomd
import numpy as np
import argparse
from tqdm import tqdm
import gsd.hoomd
import numpy as np 

def initialize_frame(N_polymers, N_monomers, mass, r0, box_size, dim, radius): #added the radius here
   frame = gsd.hoomd.Frame()

   test = np.zeros(6)
   test[:dim] = box_size
   frame.configuration.box = test
   

   # Add particles
   frame.particles.N = N_polymers * N_monomers
   frame.particles.types = [f'p{i}' for i in range(N_polymers)]
   frame.particles.typeid = np.repeat(np.arange(N_polymers), N_monomers)
   frame.particles.mass = [mass] * N_polymers * N_monomers
   up = 1
   positions = np.zeros((N_monomers, 3))
   N_Rows = int(np.sqrt(N_monomers))
   N_in_row = N_monomers/N_Rows
   old_row = -1*N_Rows/2
   for j in range(0,N_monomers):
       specific_row = -1*N_Rows/2 + j//N_Rows
       x_pos = -1*N_in_row/2 + j%N_Rows
       positions[j][1] = specific_row
       if old_row != specific_row:
           up*= -1
       positions[j][0] = up*x_pos
       old_row = specific_row
   positions = np.tile(positions, (N_polymers,1))
   
   frame.particles.position = positions

   # Add bonds
   N_bonds = N_monomers - 1
   frame.bonds.N = N_polymers * N_bonds
   frame.bonds.types = ['backbone'] 
   frame.bonds.typeid = [0] * (N_polymers * N_bonds)

   bonds = np.zeros((N_bonds, 2))
   bonds[:,0] = np.arange(0, N_bonds)
   bonds[:,1] = np.arange(1, N_bonds+1)
   bonds = np.tile(bonds, (N_polymers, 1))

   polymer_indices = np.arange(0, N_polymers)
   polymer_indices = np.repeat(polymer_indices, N_bonds*2)
   polymer_indices = np.reshape(polymer_indices, (N_polymers*N_bonds, 2))
   polymer_indices *= N_monomers

   frame.bonds.group = bonds + polymer_indices

   # Add angles
   N_angles = N_monomers - 2
   frame.angles.N = N_polymers * N_angles
   frame.angles.types = ['bending']
   frame.angles.typeid = [0] * (N_polymers * N_angles)

   angles = np.zeros((N_angles, 3))
   angles[:,0] = np.arange(0, N_angles)
   angles[:,1] = np.arange(1, N_angles+1)
   angles[:,2] = np.arange(2, N_angles+2)
   angles = np.tile(angles, (N_polymers, 1))

   polymer_indices = np.arange(0, N_polymers)
   polymer_indices = np.repeat(polymer_indices, N_angles*3)
   polymer_indices = np.reshape(polymer_indices, (N_polymers*N_angles, 3))
   polymer_indices *= N_monomers

   frame.angles.group = angles + polymer_indices
   
   ### PERSONAL INPUT
   #frame.configuration.box = [50, 50, 0, 0, 0, 0]
   #Changed line above to fix the boundarties of the box at the size of the radius
   frame.configuration.box = [2*radius, 2*radius, 0, 0, 0, 0]
   return frame