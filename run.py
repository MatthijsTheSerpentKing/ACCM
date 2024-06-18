import hoomd
import numpy as np
import argparse
from tqdm import tqdm
from MolecularMotors import MolecularMotors
from utils import initialize_frame
parser = argparse.ArgumentParser()
parser.add_argument('--dt', type=str, default='1e-4') #Putting dt smaller makes particles jump to infty
parser.add_argument('--T_eq', type=str, default='1e6') #Put to 1e6, since T_sime 1e7. 1e6 is more than big enough (1e5 already big enoug for non edge states)
parser.add_argument('--T_sim', type=str, default='1e7')
parser.add_argument('--T_write', type=str, default='1e3')
parser.add_argument('--box_size', type=int, default=1000) 
parser.add_argument('--N_polymers', type=int, default=1)
parser.add_argument('--N_monomers', type=int, default=100) #100 like in papers
parser.add_argument('--Fact', type=float, default=100) #Tweak per simmulation, make sure particles don't jump to infty 

parser.add_argument('--kappa_bond', type=float, default=50000) #Here kappa_bond>>Fact/L_polymer. Making kappa_bond bigger makes the particle jump to infty. A big kappa_bond ensure that the actual bond lengths are close to r_0
parser.add_argument('--kappa_bend', type=float, default=300) #Vary per simulation
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--mass', type=float, default=0.001) #If mass is made smaller particle goes out of bounds
parser.add_argument('--kT', type=float, default=1)
parser.add_argument('--r0', type=float, default=1)
parser.add_argument('--sigma', type=float, default=1)    #Equals r0
parser.add_argument('--epsilon', type=float, default=1)  #Equals kT
parser.add_argument('--dim', type=int, choices=[2,3], default=2)
parser.add_argument('--radius', type=float, default=500) #This is the confinement radius

#Line below is not yet implemented
parser.add_argument('--radius_true', type = int,default = 1) #Variable to turn boundary on and of from the wsl prompt

parser.add_argument('--simid', type=str)

arguments = parser.parse_args()

dt = float(arguments.dt)
T_eq = float(arguments.T_eq)
T_sim = float(arguments.T_sim)
T_write = float(arguments.T_write)
 
box_size = arguments.box_size 

rad = arguments.radius
radius_true = arguments.radius_true

N_polymers = arguments.N_polymers
N_monomers = arguments.N_monomers
Fact = arguments.Fact

kappa_bond = arguments.kappa_bond
kappa_bend = arguments.kappa_bend
gamma = arguments.gamma
mass = arguments.mass
kT = arguments.kT
r0 = arguments.r0
sigma = arguments.sigma
epsilon = arguments.epsilon
dim = arguments.dim

simid = arguments.simid

output_file = f"{simid}"\
               f"-Fact={Fact}-N_polymers={N_polymers}-N_monomers={N_monomers}-dim={dim}"\
               f"-dt={arguments.dt}-T_eq={arguments.T_eq}-T_sim={arguments.T_sim}-T_write={arguments.T_write}-box_size={box_size}"\
               f"-kappa_bond={kappa_bond}-kappa_bend={kappa_bend}-gamma={gamma}-mass={mass}-kT={kT}-r0={r0}-sigma={sigma}-epsilon={epsilon}-radius={rad}.gsd"

#Shortening the file name to fix load data error
output_file = f"{simid}"\
               f"-F={Fact}"\
               f"-Te={arguments.T_eq}-Ts={arguments.T_sim}-Tw={arguments.T_write}"\
               f"-kbo={kappa_bond}-kbe={kappa_bend}-rad={rad}.gsd"


####################
# Initialize simulation
####################

print(f'Running {output_file}')

dev = hoomd.device.CPU()

frame = initialize_frame(N_polymers, N_monomers, mass, r0, box_size, dim, rad)

# Initialize simulation
simulation = hoomd.Simulation(device=dev, seed=1)
simulation.create_state_from_snapshot(frame)

# Define harmonic potential for the bonds.
bond_potential = hoomd.md.bond.Harmonic()
bond_potential.params['backbone'] = dict(k=kappa_bond, r0=r0)

# Define bending potential for the angles
bending_potential = hoomd.md.angle.Harmonic()
bending_potential.params['bending'] = dict(k=kappa_bend, t0=np.pi)

# Define Lennard-Jones potential
lj_potential = hoomd.md.pair.LJ(hoomd.md.nlist.Tree(1), default_r_cut=sigma*2**(1/6))
sphere = hoomd.wall.Sphere(radius=rad, inside=True)

#top = hoomd.wall.Plane(origin=(0,50, 0), normal=(0, -1, 0))
#bottom = hoomd.wall.Plane(origin=(0,-50, 0), normal=(0,1, 0))
#left = hoomd.wall.Plane(origin=(-50,0, 0), normal=(1, 0, 0))
#right = hoomd.wall.Plane(origin=(50,0, 0), normal=(-1, 0, 0))
#wall_p = hoomd.md.external.wall.LJ(walls=[top, bottom, left, right])
# add walls to interact with

wall = hoomd.md.external.wall.LJ(walls=[sphere])
for type1 in frame.particles.types:
   wall.params[(type1)] = {'sigma': sigma, 'epsilon': epsilon,'r_cut': 2.0 ** (1 / 6)}
   for type2 in frame.particles.types:
       
       if type1 == type2:
           lj_potential.params[(type1, type2)] = {'sigma': sigma, 'epsilon': epsilon}
       
       else:
           lj_potential.params[(type1, type2)] = {'sigma': 0, 'epsilon': 0}

# Define molecular motors
molecular_motors = MolecularMotors(Fact, N_monomers, N_polymers, box_size, dim)

# Add integrator
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT, default_gamma=gamma)




integrator = hoomd.md.Integrator(dt=dt,
                                methods=[langevin],
                                forces=[bond_potential, bending_potential, lj_potential, molecular_motors])


simulation.operations.integrator = integrator
simulation.operations.integrator.forces.append(wall)
#simulation.operations.integrator.external_potential = wall

####################
# Run simulation
####################


# Equilibriate simulation   
for _ in tqdm(range(100), desc='Equilibriation'):
   simulation.run(int(T_eq / 100))

# Add writer
gsd_writer = hoomd.write.GSD(filename=output_file,
                            trigger=hoomd.trigger.Periodic(int(T_write)),
                            mode='wb')
simulation.operations.writers.append(gsd_writer)

# Run actual simulation
for _ in tqdm(range(100), desc='Running simulation'):
   simulation.run(int(T_sim / 100))

# Save data
gsd_writer.flush()

