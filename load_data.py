import gsd
import gsd.hoomd
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

pi = np.pi

#File for analyzing GSD data

# Get the current directory
current_directory = os.getcwd()

# Specify the filename of the GSD file
gsd_filename = 'important-F=100-Te=1e6-Ts=1e7-Tw=1e3-kbo=50000-kbe=90.0-rad=20.0.gsd'

# # Combine the directory and filename to get the full path
gsd_filepath = os.path.join(current_directory, gsd_filename)

# # Open the GSD file
particle_positions = []
with gsd.hoomd.open(gsd_filename) as traj:
    # Access the frames in the file
    for frame in traj:
        # Example: Accessing particle positions
        #print("Frame:", frame.configuration.step)
        particle_positions.append(frame.particles.position)
     
#Function to find the distance between the head and the tail of the Polymer
def head_to_tail_length(polymer):
    L = ((polymer[0][0]-polymer[-1][0])**2+(polymer[0][1]-polymer[-1][1])**2)**0.5
    return L
    
Lengths = [head_to_tail_length(p) for p in particle_positions]
print('Mean head to tail length =',np.mean(Lengths))


#Check displacement of particles in between t_write
def displacements(positions):  #List of all the displacements for all the particles for all the time steps 
    displacements = []
    for p in range(len(positions)-1):
        for n in range(len(positions[0])):
            displacement = ((positions[p][n][0]-positions[p+1][n][0])**2+(positions[p][n][1]-positions[p+1][n][1])**2)**0.5
            displacements.append(displacement)
            if displacement>100:
                return displacements
            
    return displacements
            
ds = displacements(particle_positions)

#To check if particles really are staying in the convinement
def x_and_y_positions(positions):
    x_pos = []
    y_pos = []
    for a in positions:
        for b in a:
            x_pos.append(b[0])
            y_pos.append(b[1])
    return x_pos, y_pos

x_positions, y_positions = x_and_y_positions(particle_positions) #Particles stay in bound

#Checking the bond lengths
def bond_lengths(positions):
    bond_L = []
    for a in positions:
        for b in range(len(a)-1):
            bond_L.append(((a[b][0]-a[b+1][0])**2+(a[b][1]-a[b+1][1])**2)**0.5)
    
    return bond_L

bond_ls = bond_lengths(particle_positions)

def COM(polymer): #Center of mass of the polymer
    x_s = [p[0] for p in polymer]
    y_s = [p[1] for p in polymer]
    x_c = np.mean(x_s)
    y_c = np.mean(y_s)
    
    return x_c, y_c

x_coms = [] #x com positions
y_coms = [] #y com positions
for i in range(len(particle_positions)):
    x_com, y_com = COM(particle_positions[i])
    x_coms.append(x_com)
    y_coms.append(y_com)

#Below is sort of the speed of the COM
def d_COM(positions): #Absolute value of the displacement of the COM in consecutive time steps (t_write)
    x_coms = [] #x com positions
    y_coms = [] #y com positions
    for i in range(len(particle_positions)):
        x_com, y_com = COM(particle_positions[i])
        x_coms.append(x_com)
        y_coms.append(y_com)
    
    d_s = [((x_coms[i+1]-x_coms[i])**2+(y_coms[i+1]-y_coms[i])**2)**0.5 for i in range(len(x_coms)-1)]
    
    return d_s

Speed_COM = d_COM(particle_positions)
print('Mean speed COM =',np.mean(Speed_COM))


#Calculating the Spiral number
#Defining a function that gives an angle between 0 and 2pi
def angle(x,y): #Polar angle between 0 and 2 pi of a point in the x y plane
    pi = np.pi
    if x>0:
        if y>0:
            theta = np.arctan(y/x)
        if y<0:
            theta = 2*pi - np.arctan(abs(y/x))
    if x<0:
        if y>0:
            theta = pi -np.arctan(abs(y/x))
        if y<0:
            theta = pi +np.arctan(abs(y/x))
    return theta

#Needs to be updated to allow for negative angles
def spiral_number(polymer):
    pi = np.pi
    x_s = [p[0] for p in polymer]
    y_s = [p[1] for p in polymer]
    
    vectors = [[x_s[i+1]-x_s[i],y_s[i+1]-y_s[i]] for i in range(len(x_s)-1)]
    rot_vectors = [[-v[1],v[0]] for v in vectors]
    angles = [angle(r_v[0],r_v[1]) for r_v in rot_vectors]
    for i in range(1,len(angles)):
        while True:
            if abs(angles[i]-angles[i-1])>pi:
                if angles[i]-angles[i-1]>pi:
                    angles[i] = angles[i]-2*pi
                if angles[i]-angles[i-1]<-pi:
                    angles[i] = angles[i]+2*pi
            else:
                break
    s = (angles[-1]-angles[0])/(2*pi)
    return s

spiral_numbers = [spiral_number(p) for p in particle_positions]
print('Mean spiral number =',np.mean(spiral_numbers))

#Relevant parameter for distinguishing spiral regimes (see paper)
def kurtosis(positions): 
    s_values = [spiral_number(p) for p in positions]
    mean_s = np.mean(s_values)
    std_s = np.std(s_values)
    k = np.mean([((s-mean_s)/std_s)**4 for s in s_values])
    return k

#Square of the roations (cumulative angle) done by the factor stratching from the first to the last monomer in the polymer
def mean_square_rot(positions): 
    vectors = [[p[-1][0]-p[0][0],p[-1][1]-p[0][1]] for p in positions]
    angles = [angle(v[0],v[1]) for v in vectors]
    for i in range(1,len(angles)):
        while True:
            if abs(angles[i]-angles[i-1])>pi:
                if angles[i]-angles[i-1]>pi:
                    angles[i] = angles[i]-2*pi
                if angles[i]-angles[i-1]<-pi:
                    angles[i] = angles[i]+2*pi
            else:
                break
    angles = [a - angles[0] for a in angles]
    
    MSR =angles[-1]**2
    return MSR
    