# classes for graphene sample collection

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lammps import lammps
from scipy.signal import find_peaks
import os
from scipy.spatial import Delaunay
from datetime import timedelta
import time


class GrapheneSheet:
    def __init__(self, datafile_name, x_atoms, y_atoms):
        """
        Class to store information about a graphene sheet.
        
        Parameters:
        - datafile_name (str): The path to the data file.
        - x_atoms (int): The length of the sheet in the x direction (number of atoms).
        - y_atoms (int): The length of the sheet in the y direction (number of atoms).
        - makeStrengthSurface (bool): User specifies whether or not they want strength surface plots generated and saved (default True)
        """
        self.datafile_name = datafile_name
        self.x_atoms = x_atoms
        self.y_atoms = y_atoms
        self.volume = self.calcVolume()  # volume of sheet in angstroms cubed (note that this is unrelaxed)

    def __repr__(self):
        return f"GrapheneSheet(datafile_name='{self.datafile_name}', x_atoms={self.x_atoms}, y_atoms={self.y_atoms}, volume={self.volume})"
    
    # takes in the length and width of graphene sheet (in atoms) and gives back the volume in angstroms
    # assumes that the x-axis is the armchair edge and the y-axis is the zigzag edge
    def calcVolume(self):
        dist = 1.42
        Lz = 3.4
        if self.x_atoms % 2 == 0:
            Lx = ((self.x_atoms / 2) * dist) + (dist * 2 * (self.x_atoms / 2 - 1)) + dist
        else:
            Lx = (((self.x_atoms - 1) / 2) * dist) + (dist * 2 * ((self.x_atoms + 1) / 2 - 1)) + (dist / 2)

        Ly = self.y_atoms * dist * np.sin(np.deg2rad(60))
        vol = Lx * Ly * Lz

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        return vol


class Simulation:
    # we basically want the whole simulation to run on initialization, then we can pull whatever we want from it for postprocessing
    def __init__(self, comm, rank, sheet, x_erate=0, y_erate=0, z_erate=0, xy_erate=0, xz_erate=0, yz_erate=0, 
                 sim_length=100000, timestep=0.0005, thermo=1000, makeplots=False, fracture_window=10, storage_path='/data1/avb25/graphene_sim_data/deform_data'):
        """
        Class to execute one simulation and store information about it.
        This essentially loads the specimen to failure.
        
        Parameters:
        - comm (???): ???
        - rank (???): ???
        - sheet (GrapheneSheet): Object that stores all necessary info about the sheet being used.
        - x_erate (float): strain rate for fix deform (in 1/ps) (same for all other directions)
        - sim_length (int): Number of timesteps before simulation is killed (max timesteps) 
        - timestep (float): Size of one timestep in LAMMPS simulation (picoseconds)
        - thermo (int): Frequency of timesteps you output data (if thermo = 100, output every 100 timesteps)
        - makeplots (bool): User specifies whether or not they want plots of stress vs time generated and saved (default False)
        - fracture_window (int): Tunable parameter that says how much stress drop (GPa) is necessary to detect fracture (to eliminate noise). 10 GPa is default
        - storage_path (str): filepath to where we want to store the data
        """

        self.sheet = sheet
        self.x_erate = x_erate
        self.y_erate = y_erate
        self.z_erate = z_erate
        self.xy_erate = xy_erate
        self.xz_erate = xz_erate
        self.yz_erate = yz_erate
        self.sim_length = sim_length
        self.timestep = timestep
        self.thermo = thermo
        self.makeplots = makeplots
        self.fracture_window = fracture_window
        self.storage_path = storage_path

        self.starttime = time.perf_counter()  # start simulation timer

        self.erate_timestep = self.calculate_erateTimestep()  # used to calculate strain rate at any given timestep (just multiply this by whatever timestep you're on)

        self.big_csv = f'{storage_path}/all_simulations.csv'
        self.initialize_bigcsv()  # initialize csv file for data storage
        self.simulation_id = self.get_simid()  # returns string of simulation id (ex: '00001')

        self.simulation_directory = self.create_storage_directory()  # create data storage directory (storage_path/sim{simid})

        lmp = lammps(comm=comm)  # initialize lammps

        # import all variables to lammps
        lmp.command(f"variable vol equal {sheet.volume}")
        lmp.command(f"variable sheet_lx equal {sheet.Lx}")
        lmp.command(f"variable sheet_ly equal {sheet.Ly}")
        lmp.command(f"variable sheet_lz equal {sheet.Lz}")
        lmp.command(f"variable timestep equal {self.timestep}")

        lmp.command(f"variable conv_fact equal 0.0001")
        lmp.command(f"variable datafile string {sheet.datafile_name}")
        lmp.command(f"variable dump_output string {self.simulation_directory}/dump.sim{self.simulation_id}")

        lmp.file("in.deform_py")

        self.lmp = lmp
        self.comm = comm
        self.rank = rank

        self.apply_fix_deform()

        stress_tensor, pressure_tensor, step_vector, strain_tensor = self.run_simulation()
        principal_stresses = self.compute_principal_StressStrain(stress_tensor)
        principalAxes_strain = self.compute_principal_StressStrain(strain_tensor)

        self.stress_tensor = stress_tensor  # array of all six stress values at each outputted thermo
        self.pressure_tensor = pressure_tensor  # array of all three pressure values at each outputted thermo
        self.strain_tensor = strain_tensor  # array of all six strain values at each outputted thermo
        self.step_vector = step_vector  # vector of each thermo where data was collected
        self.principal_stresses = principal_stresses  # array of all three principal stress values at each outputted thermo
        self.principalAxes_strain = principalAxes_strain  # array of strain values along all three principal directions at each outputted thermo

        strength, crit_strain, fracture_time = self.find_fracture(principal_stresses, give_crit_strain=True)

        self.strength = strength  # vector of the three critical principal stresses at fracture (largest to smallest)
        self.crit_strain = crit_strain  # vector of the three critical strains in the principal sirections at fracture (largest to smallest)
        self.fracture_time = fracture_time  # thermo in which fracture was detected

        self.sim_duration = timedelta(seconds=time.perf_counter() - self.starttime)

        if self.rank == 0:
            print(f'Material Strength ({sheet.x_atoms}x{sheet.y_atoms} sheet): {strength} GPa. ')
            self.append_csv()  # append all data from the simulation to the csv file. 
            self.save_detailed_data()
            if sheet.makeStrengthSurface:
                sheet.populate_surface(strength)
            if makeplots:
                # right now we really only care about the principal stresses, so only output that
                self.plot_principalStress()
                # self.plot_pressure()
                # self.plot_stressTensor()

    # create the directory where we will be putting the detailed data for this simulation
    def create_storage_directory(self):
        path = f'{self.storage_path}/sim{self.simulation_id}'
        os.makedirs(path, exist_ok=True)
        return path

    # if the file doesn't exits, create one. if it does exist we'll just append to the old one so we don't overwrite data
    def initialize_bigcsv(self):
        if not os.path.exists(self.big_csv) or os.path.getsize(self.big_csv) == 0:
            df = pd.DataFrame(columns=['Simulation ID', 'Num Atoms x', 'Num Atoms y', 'Strength_1', 'Strength_2', 'Strength_3', 
                                       'CritStrain_1', 'CritStrain_2', 'CritStrain_3', 'Strain Rate x', 'Strain Rate y', 'Strain Rate z',
                                       'Strain Rate xy', 'Strain Rate xz', 'Strain Rate yz', 'Fracture Time', 'Max Sim Length', 
                                       'Output Timesteps', 'Fracture Window', 'Simulation Time'])
            df.to_csv(self.big_csv, index=False)

    def get_simid(self):
        if os.path.exists(self.big_csv) and os.path.getsize(self.big_csv) > 0:
            # Read the existing file and find the maximum simulation id
            df = pd.read_csv(self.big_csv)
            if "Simulation ID" in df.columns and not df.empty:
                return str(df["Simulation ID"].max() + 1).zfill(5)
        # Default to 1 if the file doesn't exist or is empty
        return str(1).zfill(5)

    # appends a new simulation to the csv file for storage
    def append_csv(self):
        new_row = pd.DataFrame({'Simulation ID':[self.simulation_id], 'Num Atoms x': [self.sheet.x_atoms], 'Num Atoms y': [self.sheet.y_atoms], 
                                'Strength_1': [self.strength[0]], 'Strength_2': [self.strength[1]], 'Strength_3': [self.strength[2]],
                                'CritStrain_1': [self.crit_strain[0]], 'CritStrain_2': [self.crit_strain[1]], 'CritStrain_3': [self.crit_strain[2]],
                                'Strain Rate x': [self.x_erate], 'Strain Rate y': [self.y_erate], 'Strain Rate z': [self.z_erate],
                                'Strain Rate xy': [self.xy_erate], 'Strain Rate xz': [self.xz_erate], 'Strain Rate yz': [self.yz_erate],
                                'Fracture Time': [self.fracture_time], 'Max Sim Length': [self.sim_length],
                                'Output Timesteps': [self.thermo], 'Fracture Window': [self.fracture_window], 'Simulation Time': [self.sim_duration]})
        new_row.to_csv(self.big_csv, mode="a", header=False, index=False)

    def save_detailed_data(self):
        df = pd.DataFrame({'Timestep': self.step_vector, 
                           'PrincipalStress_1': self.principal_stresses[:, 0], 'PrincipalStress_2': self.principal_stresses[:, 1], 
                           'PrincipalStress_3': self.principal_stresses[:, 2], 'Strain_1': self.principalAxes_strain[:, 0], 
                           'Strain_2': self.principalAxes_strain[:, 1], 'Strain_3': self.principalAxes_strain[:, 2], 
                           'Stress_xx': self.stress_tensor[:, 0], 'Stress_yy': self.stress_tensor[:, 1], 'Stress_zz': self.stress_tensor[:, 2], 
                           'Stress_xy': self.stress_tensor[:, 3], 'Stress_xz': self.stress_tensor[:, 4], 'Stress_yz': self.stress_tensor[:, 5],
                           'Strain_xx': self.strain_tensor[:, 0], 'Strain_yy': self.strain_tensor[:, 1], 'Strain_zz': self.strain_tensor[:, 2], 
                           'Strain_xy': self.strain_tensor[:, 3], 'Strain_xz': self.strain_tensor[:, 4], 'Strain_yz': self.strain_tensor[:, 5],
                           'Pressure_x': self.pressure_tensor[:, 0], 'Pressure_y': self.pressure_tensor[:, 1], 'Pressure_z': self.pressure_tensor[:, 2]})
        
        detailed_csv_file = f'{self.simulation_directory}/sim{self.simulation_id}.csv'
        df.to_csv(detailed_csv_file, index=False)

    # This writes the fix deform command for lammps and sends it to lammps
    def apply_fix_deform(self):
        # Start building the command
        fix_command = "fix 2 all deform 1"

        # Add deformation options dynamically
        if self.x_erate != 0:
            fix_command += f" x erate {self.x_erate}"
        if self.y_erate != 0:
            fix_command += f" y erate {self.y_erate}"
        if self.z_erate != 0:
            fix_command += f" z erate {self.z_erate}"
        if self.xy_erate != 0:
            fix_command += f" xy erate {self.xy_erate}"
        if self.xz_erate != 0:
            fix_command += f" xz erate {self.xz_erate}"
        if self.yz_erate != 0:
            fix_command += f" yz erate {self.yz_erate}"

        # Ensure remap is included
        fix_command += " remap x"
        self.lmp.command(fix_command)

    # takes the strain rates and the simulation timestep and gives a vector that holds the strain per timestep for the simulation
    def calculate_erateTimestep(self):
        xx = self.x_erate * self.timestep
        yy = self.y_erate * self.timestep
        zz = self.z_erate * self.timestep
        xy = self.xy_erate * self.timestep
        xz = self.xz_erate * self.timestep
        yz = self.yz_erate * self.timestep
        return np.array([xx, yy, zz, xy, xz, yz])

    # runs lammps simulation and returns array with every stress tensor entry over time and vector of every timestep at which data was taken
    def run_simulation(self):
        stress_tensor = np.zeros((int(self.sim_length/self.thermo + 1), 6))
        step_vector = np.zeros(int(self.sim_length/self.thermo + 1))
        pressure_tensor = np.zeros((int(self.sim_length/self.thermo + 1), 3))
        strain_tensor = np.zeros((int(self.sim_length/self.thermo + 1), 6))

        # initialize stress at time = 0
        self.lmp.command(f"run 0 pre yes post yes")
        stress_tensor[0] = self.extract_stress()
        pressure_tensor[0] = self.extract_pressure()
        strain_tensor[0] = self.compute_strain(0)

        iters = 0
        # run and store for desired timesteps
        for step in range(0, self.sim_length, self.thermo):
            iters, stress_tensor, step_vector, pressure_tensor, strain_tensor = self.run_step(step, iters, stress_tensor, step_vector, pressure_tensor, strain_tensor)

            column_sums = np.sum(np.abs(stress_tensor), axis=0)
            dominant_direction = np.argmax(column_sums)

            # checks when dominant direction stress drops to detect fracture - this is just an on the fly check because we don't have principal stresses calculated
            strength, _ = self.find_fracture(stress_tensor[:(iters+1), dominant_direction].reshape(-1, 1))

            # if we get a strength value, that means fracture was detected and we can leave the loop
            if strength[0] is not None:
                # run the simulation for a few more thermos to visualize fracture
                for i in range(5):
                    step += self.thermo  # update thermo (cuz we are no longer in the big loop)
                    if step == self.sim_length:
                        break
                    iters, stress_tensor, step_vector, pressure_tensor, strain_tensor = self.run_step(step, iters, stress_tensor, step_vector, pressure_tensor, strain_tensor)
                    
                stress_tensor = stress_tensor[:(iters+1)]
                step_vector = step_vector[:(iters+1)]
                pressure_tensor = pressure_tensor[:(iters+1)]
                strain_tensor = strain_tensor[:(iters+1)]

                return stress_tensor, pressure_tensor, step_vector, strain_tensor

        return stress_tensor, pressure_tensor, step_vector, strain_tensor
    
    # returns a vector of length 3 representing the pressure in each direction (imposed by npt)
    def extract_pressure(self):
        xx = self.lmp.extract_variable("p_x", None, 0)
        yy = self.lmp.extract_variable("p_y", None, 0)
        zz = self.lmp.extract_variable("p_z", None, 0)
        return [xx, yy, zz]
    
    # returns a flattened np array of the stress tensor
    def extract_stress(self):
        xx = self.lmp.extract_variable("stress_xx", None, 0)
        yy = self.lmp.extract_variable("stress_yy", None, 0)
        zz = self.lmp.extract_variable("stress_zz", None, 0)
        xy = self.lmp.extract_variable("stress_xy", None, 0)
        xz = self.lmp.extract_variable("stress_xz", None, 0)
        yz = self.lmp.extract_variable("stress_yz", None, 0)

        arr = np.array([xx, yy, zz, xy, xz, yz])
        return arr
    
    # pass through a timestep number, (ex. timestep 200) and it will give you the strain for that timestep
    def compute_strain(self, current_timestep):
        x_strain = self.erate_timestep[0] * current_timestep
        y_strain = self.erate_timestep[1] * current_timestep
        z_strain = self.erate_timestep[2] * current_timestep
        xy_strain = self.erate_timestep[3] * current_timestep
        xz_strain = self.erate_timestep[4] * current_timestep
        yz_strain = self.erate_timestep[5] * current_timestep
        return np.array([x_strain, y_strain, z_strain, xy_strain, xz_strain, yz_strain])
    
    # helper function for run_simulation... runs one thermo
    def run_step(self, step, iters, stress_tensor, step_vector, pressure_tensor, strain_tensor):
        iters += 1

        self.lmp.command(f"run {self.thermo} pre yes post no")
        step_vector[iters] = step + self.thermo
        stress = self.extract_stress()  # extract stress tensor
        pressure = self.extract_pressure()
        strain = self.compute_strain(step_vector[iters])
        stress_tensor[iters] = stress
        pressure_tensor[iters] = pressure
        strain_tensor[iters] = strain
        return iters, stress_tensor, step_vector, pressure_tensor, strain_tensor
    
    # input vector of principal stresses so far, outputs strength and fracture timestep (of it occurred), otherwise None for both
    # idea here is to have the previous 5 thermos be our testcase, and the previous 10 before that be what we're testing against
    # if the average of the previous 5 is lower than the average of the 10 before that, fracture has occurred and we should find peaks
    # we know that principal_stresses[:, 0] will always be the dominant direction
    def find_fracture(self, principal_stresses, give_crit_strain=False):
        # Not enough thermos to actually check
        if len(principal_stresses[:, 0]) < 25:
            strength = [None, None, None]  # must be a list of None's for later
            fracture_timestep = None
        # Now there's enough timesteps to check
        else:
            mean_last_10 = sum(principal_stresses[:, 0][-10:]) / 10
            mean_15_before = sum(principal_stresses[:, 0][-25:-10]) / 15

            # if we are still increasing on average, no fracture yet
            if mean_last_10 >= mean_15_before:
                strength = [None, None, None]
                fracture_timestep = None
            
            # we have detected a significant drop on average (fracture)
            else:
                peaks, _ = find_peaks(principal_stresses[:, 0])
                if len(peaks) == 0:
                    # No peaks found (idt this could ever happen but who knows lol)
                    strength = [None, None, None]
                    fracture_timestep = None
                
                # Now we are confident there is fracture, let's return the proper values
                else:
                    fracture_index = peaks[np.argmax(principal_stresses[:, 0][peaks])]  # find the index of the highest peak

                    # see if this "fracture" is high enough to be considered (or is it just noise?)
                    # for fracture, the (peak - window) must be greater than the minimum point
                    if (np.max(principal_stresses[fracture_index]) - self.fracture_window < np.min(principal_stresses[:, 0][-10:])):
                        strength = [None, None, None]
                        fracture_timestep = None
                    else:
                        strength = principal_stresses[fracture_index]
                        fracture_timestep = fracture_index * self.thermo

                        if give_crit_strain:
                            crit_strain = self.principalAxes_strain[fracture_index]
                            return strength, crit_strain, fracture_timestep
                        
                        print("FRACTUREEEEEE")

        return strength, fracture_timestep
    
    # pass through length six vector that contains all stresses or strains in each direction, computes the eigvals so you have the principal stresses,
    # or the strain along the principal axes, and returns - ordered from which direction experienced max stress (or strain) to min stress (or strain)
    def compute_principal_StressStrain(self, tensor):
        num_timesteps = tensor.shape[0]
        principal_values = np.zeros((num_timesteps, 3))

        for i in range(num_timesteps):
            # get whole stress tensor for this timestep
            matrix = np.array([[tensor[i, 0], tensor[i, 3], tensor[i, 4]],
                                    [tensor[i, 3], tensor[i, 1], tensor[i, 5]],
                                    [tensor[i, 4], tensor[i, 5], tensor[i, 2]]])
            
            # compute principal stresses and principal directions - this is sorted from lowest to highest eigval every time, 
            eigvals = np.linalg.eigvalsh(matrix)
            principal_values[i] = eigvals[::-1]

        return principal_values
    
    # plots value of each principal stress over time
    def plot_principalStress(self):
        princ_fig, princ_ax = plt.subplots()

        princ_ax.plot(self.step_vector, self.principal_stresses[:, 0], label=r'$\sigma_1$')
        princ_ax.plot(self.step_vector, self.principal_stresses[:, 1], label=r'$\sigma_2$')
        princ_ax.plot(self.step_vector, self.principal_stresses[:, 2], label=r'$\sigma_3$')
        if self.fracture_time is not None:
            princ_ax.axvline(x=self.fracture_time, color='r', linestyle='--', linewidth=1, label='Fracture Time')

        princ_ax.set_title(f"Principal Stresses vs. Time ({self.sheet.x_atoms}x{self.sheet.y_atoms}, Strain Rate = {self.x_erate})")
        princ_ax.set_xlabel("Timesteps")
        princ_ax.set_ylabel("Stress (GPa)")
        princ_ax.legend()
        princ_ax.grid()
        princ_fig.savefig(f'{self.simulation_directory}/sim{self.simulation_id}_princ_vs_time.png')

    # plots value of each pressure value imposed by npt over time
    def plot_pressure(self):
        pressure_fig, pressure_ax = plt.subplots()

        pressure_ax.plot(self.step_vector, self.pressure_tensor[:, 0], label=r'$p_x$')
        pressure_ax.plot(self.step_vector, self.pressure_tensor[:, 1], label=r'$p_y$')
        pressure_ax.plot(self.step_vector, self.pressure_tensor[:, 2], label=r'$p_z$')
        if self.fracture_time is not None:
            pressure_ax.axvline(x=self.fracture_time, color='r', linestyle='--', linewidth=1, label='Fracture Time')

        pressure_ax.set_title(f"Applied Pressure vs. Time ({self.sheet.x_atoms}x{self.sheet.y_atoms})")
        pressure_ax.set_xlabel("Timesteps")
        pressure_ax.set_ylabel("Pressure (Bar)")
        pressure_ax.legend()
        pressure_ax.grid()
        pressure_fig.savefig(f'{self.simulation_directory}_press_vs_time.png')

    # plots value of each stress tensor direction over time
    def plot_stressTensor(self):
        stress_tensor_fig, stress_tensor_ax = plt.subplots()

        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 0], label=r'$\sigma_{xx}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 1], label=r'$\sigma_{yy}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 2], label=r'$\sigma_{zz}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 3], label=r'$\sigma_{xy}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 4], label=r'$\sigma_{xz}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 5], label=r'$\sigma_{yz}$')
        if self.fracture_time is not None:
            stress_tensor_ax.axvline(x=self.fracture_time, color='r', linestyle='--', linewidth=1, label='Fracture Time')

        stress_tensor_ax.set_title(f"Stress Components vs. Time ({self.sheet.x_atoms}x{self.sheet.y_atoms})")
        stress_tensor_ax.set_xlabel("Timesteps")
        stress_tensor_ax.set_ylabel("Stress (GPa)")
        stress_tensor_ax.legend()
        stress_tensor_ax.grid()
        stress_tensor_fig.savefig(f'{self.simulation_directory}_tensor_vs_time.png')


class Relaxation:
    def __init__(self, comm, rank, sheet):
        """
        Class to execute one relaxation simulation and store information about it.
        
        Parameters:
        - comm (???): ???
        - rank (???): ???
        - sheet (GrapheneSheet): Object that stores all necessary info about the sheet being used.
        """

        lmp = lammps(comm=comm)

        lmp.command(f"variable Volume equal {sheet.volume}")
        lmp.command(f"variable conv_fact equal 0.0001")
        lmp.command(f"variable datafile string {sheet.datafile_name}")
        lmp.command(f"variable dump_output string dump.{sheet.x_atoms}_{sheet.y_atoms}")


        lmp.file("in.relaxation")

        self.lmp = lmp
        self.comm = comm
        self.rank = rank
        self.sheet = sheet