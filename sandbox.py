from graphene_classes_deform import GrapheneSheet
from graphene_classes_deform import Simulation
import numpy as np

x_atoms = 60
y_atoms = 60
dist = 1.42
Lz = 3.4
if x_atoms % 2 == 0:
    Lx = ((x_atoms / 2) * dist) + (dist * 2 * (x_atoms / 2 - 1)) + dist
else:
    Lx = (((x_atoms - 1) / 2) * dist) + (dist * 2 * ((x_atoms + 1) / 2 - 1)) + (dist / 2)

Ly = y_atoms * dist * np.sin(np.deg2rad(60))
vol = Lx * Ly * Lz
print(vol)